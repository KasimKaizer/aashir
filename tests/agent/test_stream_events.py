from collections.abc import Iterator

import httpx2
import pytest
from openai import APIError

from aashir.agent import (
    AgentEvent,
    AgentRequest,
    AnswerDelta,
    ProviderError,
    ProviderErrorCategory,
    ReasoningDelta,
    ReasoningSummary,
    TurnCompleted,
    run_agent,
)
from tests.helpers import make_openai_client, sse_chunk


def test_run_agent_yields_answer_deltas_before_provider_stream_finishes() -> None:
    # Given: a chunk-counting stream that makes buffering observable.
    consumed: list[str] = []

    class StreamingBody(httpx2.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            consumed.append("first")
            yield sse_chunk({"content": "Hel"})
            consumed.append("second")
            yield sse_chunk({"content": "lo"}, finish_reason="stop")
            yield b"data: [DONE]\n\n"

    # When: the consumer pulls only the first event.
    events = run_agent(
        make_openai_client(StreamingBody()),
        AgentRequest(model="test-model", messages=[{"role": "user", "content": "hi"}]),
    )
    first = next(events)

    # Then: answer text is observable before the stream finishes.
    assert isinstance(first, AnswerDelta)
    assert first.text == "Hel"
    assert consumed == ["first"]

    rest = list(events)
    assert (
        "".join(
            event.text for event in [first, *rest] if isinstance(event, AnswerDelta)
        )
        == "Hello"
    )
    assert isinstance(rest[-1], TurnCompleted)


@pytest.mark.parametrize("reasoning_effort", [None, "medium"])
def test_run_agent_does_not_fabricate_reasoning_when_provider_omits_it(
    reasoning_effort: str | None,
) -> None:
    # Given: a plain answer stream with no reasoning payload.
    body = sse_chunk({"content": "Hello"}, finish_reason="stop")
    body += b"data: [DONE]\n\n"

    # When: the agent consumes the provider stream.
    events = list(
        run_agent(
            make_openai_client(httpx2.ByteStream(body)),
            AgentRequest(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                reasoning_effort=reasoning_effort,
            ),
        )
    )

    # Then: no reasoning events are invented and the answer is preserved.
    assert not any(isinstance(event, ReasoningDelta) for event in events)
    assert not any(isinstance(event, ReasoningSummary) for event in events)
    assert (
        "".join(event.text for event in events if isinstance(event, AnswerDelta))
        == "Hello"
    )
    assert isinstance(events[-1], TurnCompleted)


def test_run_agent_yields_partial_deltas_then_raises_normalized_provider_error() -> (
    None
):
    # Given: partial answer text followed by an error completion chunk.
    body = sse_chunk({"content": "Partial"}) + sse_chunk(
        {},
        finish_reason="error",
        error={"message": "stream failed", "type": "server_error"},
    )
    events = run_agent(
        make_openai_client(httpx2.ByteStream(body)),
        AgentRequest(model="test-model", messages=[{"role": "user", "content": "hi"}]),
    )

    # When: the consumer iterates the stream.
    seen: list[AnswerDelta | ReasoningDelta | ReasoningSummary | TurnCompleted] = []
    with pytest.raises(ProviderError, match="stream failed") as error:
        for event in events:
            seen.append(event)  # noqa: PERF402 -- retain partial events before the exception

    # Then: provider context and already-yielded output survive normalization.
    assert error.value.category == "provider_unavailable"
    assert error.value.error_type == "server_error"
    assert error.value.message == "stream failed"
    assert isinstance(error.value.__cause__, APIError)
    assert [event.text for event in seen if isinstance(event, AnswerDelta)] == [
        "Partial"
    ]
    assert not any(isinstance(event, TurnCompleted) for event in seen)


@pytest.mark.parametrize(
    ("provider_type", "provider_code", "expected_category"),
    [
        ("rate_limit_error", None, "rate_limit"),
        ("authentication_error", None, "auth"),
        ("invalid_request_error", "context_length_exceeded", "context_length"),
        ("server_error", None, "provider_unavailable"),
        ("invalid_request_error", "content_policy_violation", "content_policy"),
        ("unrecognized_error", None, "unknown"),
    ],
)
def test_run_agent_normalizes_provider_error_categories(
    provider_type: str,
    provider_code: str | None,
    expected_category: str,
) -> None:
    # Given: a provider error payload with a known or unrecognized type/code.
    error_payload = {"message": "provider failed", "type": provider_type}
    if provider_code is not None:
        error_payload["code"] = provider_code
    body = sse_chunk({}, error=error_payload) + b"data: [DONE]\n\n"
    events = run_agent(
        make_openai_client(httpx2.ByteStream(body)),
        AgentRequest(model="test-model", messages=[{"role": "user", "content": "hi"}]),
    )

    # When: the consumer iterates the provider stream.
    with pytest.raises(ProviderError) as error:
        list(events)

    # Then: the normalized error retains its category and provider type.
    assert error.value.category == expected_category
    assert error.value.error_type == provider_type


@pytest.mark.parametrize(
    ("finish_reason", "expected_category"),
    [("length", "context_length"), ("content_filter", "content_policy")],
)
def test_run_agent_normalizes_length_and_content_filter_finish_reasons(
    finish_reason: str,
    expected_category: ProviderErrorCategory,
) -> None:
    # Given: partial output followed by a length or content-filter finish reason.
    body = sse_chunk({"content": "Partial"}, finish_reason=finish_reason)
    body += b"data: [DONE]\\n\\n"
    events = run_agent(
        make_openai_client(httpx2.ByteStream(body)),
        AgentRequest(model="test-model", messages=[{"role": "user", "content": "hi"}]),
    )
    seen: list[AgentEvent] = []

    # When: the consumer iterates the provider stream.
    with pytest.raises(ProviderError) as error:
        seen.extend(events)

    # Then: the reason has its normalized category without losing partial output.
    assert error.value.category == expected_category
    assert error.value.error_type is None
    assert (
        "".join(event.text for event in seen if isinstance(event, AnswerDelta))
        == "Partial"
    )
    assert not any(isinstance(event, TurnCompleted) for event in seen)


def test_run_agent_normalizes_unknown_finish_reason() -> None:
    # Given: partial answer text followed by an unrecognized completion reason.
    body = sse_chunk({"content": "Partial"}, finish_reason="future_reason")
    body += b"data: [DONE]\n\n"
    events = run_agent(
        make_openai_client(httpx2.ByteStream(body)),
        AgentRequest(model="test-model", messages=[{"role": "user", "content": "hi"}]),
    )
    seen: list[AnswerDelta | ReasoningDelta | ReasoningSummary | TurnCompleted] = []

    # When: the consumer iterates the provider stream.
    with pytest.raises(ProviderError) as error:
        for event in events:
            seen.append(event)  # noqa: PERF402 -- retain partial events before the exception

    # Then: the unknown reason is categorized without reporting completion.
    assert error.value.category == "unknown"
    assert [event.text for event in seen if isinstance(event, AnswerDelta)] == [
        "Partial"
    ]
    assert not any(isinstance(event, TurnCompleted) for event in seen)


def test_run_agent_preserves_event_types_and_provider_order() -> None:
    # Given: interleaved raw reasoning, summaries, answer text, and unknown details.
    body = (
        sse_chunk({"reasoning_details": [{"type": "reasoning.text", "text": "r1"}]})
        + sse_chunk(
            {"reasoning_details": [None, {"type": "reasoning.opaque", "data": "x"}]}
        )
        + sse_chunk({"content": "A"})
        + sse_chunk(
            {"reasoning_details": [{"type": "reasoning.summary", "summary": "s1"}]}
        )
        + sse_chunk({"reasoning": "top-level reasoning"})
        + sse_chunk(
            {"reasoning_details": [{"type": "reasoning.text", "text": "hidden"}]}
        )
        + sse_chunk(
            {"reasoning_details": [{"type": "reasoning.summary", "summary": "s2"}]}
        )
        + sse_chunk({"reasoning_details": [{"type": "reasoning.text", "text": "r2"}]})
        + sse_chunk({"content": "B"}, finish_reason="stop")
        + sse_chunk({})
        + b"data: [DONE]\n\n"
    )

    # When: the agent consumes the interleaved provider stream.
    events = list(
        run_agent(
            make_openai_client(httpx2.ByteStream(body)),
            AgentRequest(
                model="test-model", messages=[{"role": "user", "content": "hi"}]
            ),
        )
    )

    # Then: unknown details are ignored; event types remain distinct and ordered.
    ordered = [
        (type(event), event.text)
        for event in events
        if isinstance(event, (AnswerDelta, ReasoningDelta, ReasoningSummary))
    ]
    assert ordered == [
        (ReasoningDelta, "r1"),
        (AnswerDelta, "A"),
        (ReasoningSummary, "s1"),
        (ReasoningDelta, "top-level reasoning"),
        (ReasoningDelta, "hidden"),
        (ReasoningSummary, "s2"),
        (ReasoningDelta, "r2"),
        (AnswerDelta, "B"),
    ]
    assert isinstance(events[-1], TurnCompleted)
