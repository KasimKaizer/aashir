"""Events produced from a provider stream, and the request that starts it."""

import json
from collections.abc import Iterator
from typing import override

import httpx2
import pytest
from openai import APIError

from aashir.agent import (
    AgentEvent,
    AnswerDelta,
    ProviderError,
    ProviderErrorCategory,
    ReasoningDelta,
    ReasoningSummary,
    TurnCompleted,
    run_agent,
)
from tests.helpers import (
    answer_text,
    make_agent_request,
    make_openai_client,
    make_round_trip_client,
    sse_body,
    sse_chunk,
    tool_call,
)


def test_run_agent_yields_answer_deltas_before_provider_stream_finishes() -> None:
    # Given: a chunk-counting stream that makes buffering observable.
    consumed: list[str] = []

    class StreamingBody(httpx2.SyncByteStream):
        @override
        def __iter__(self) -> Iterator[bytes]:
            consumed.append("first")
            yield sse_chunk({"content": "Hel"})
            consumed.append("second")
            yield sse_chunk({"content": "lo"}, finish_reason="stop")
            yield sse_body()

    # When: the consumer pulls only the first event.
    with make_openai_client(StreamingBody()) as client:
        events = run_agent(client, make_agent_request())
        first = next(events)

        # Then: answer text is observable before the stream finishes.
        assert isinstance(first, AnswerDelta)
        assert first.text == "Hel"
        assert consumed == ["first"]

        rest = list(events)
    assert answer_text([first, *rest]) == "Hello"
    assert isinstance(rest[-1], TurnCompleted)


def test_run_agent_does_not_fabricate_reasoning_when_provider_omits_it() -> None:
    # Given: a plain answer stream with no reasoning payload.
    body = sse_body(sse_chunk({"content": "Hello"}, finish_reason="stop"))

    # When: the agent consumes the provider stream.
    with make_openai_client(httpx2.ByteStream(body)) as client:
        events = list(run_agent(client, make_agent_request()))

    # Then: no reasoning events are invented and the answer is preserved.
    assert not any(isinstance(event, ReasoningDelta) for event in events)
    assert not any(isinstance(event, ReasoningSummary) for event in events)
    assert answer_text(events) == "Hello"
    assert events[-1] == TurnCompleted(produced_answer=True)


def test_run_agent_omits_tools_and_reasoning_effort_from_the_request_when_unset() -> (
    None
):
    # Given: a request with no tools and no reasoning effort.
    client, requests = make_round_trip_client(
        sse_body(sse_chunk({"content": "Hello"}, finish_reason="stop"))
    )

    # When: the agent sends it to the provider.
    with client:
        list(run_agent(client, make_agent_request()))

    # Then: the provider request carries neither field.
    payload = json.loads(requests[0].content)
    assert "tools" not in payload
    assert "reasoning_effort" not in payload


def test_run_agent_yields_partial_deltas_then_raises_normalized_provider_error() -> (
    None
):
    # Given: partial answer text followed by an error completion chunk.
    body = sse_chunk({"content": "Partial"}) + sse_chunk(
        {},
        finish_reason="error",
        error={"message": "stream failed", "type": "server_error"},
    )
    seen: list[AgentEvent] = []

    # When: the consumer iterates the stream.
    with (
        make_openai_client(httpx2.ByteStream(body)) as client,
        pytest.raises(ProviderError, match="stream failed") as error,
    ):
        seen.extend(run_agent(client, make_agent_request()))

    # Then: provider context and already-yielded output survive normalization.
    assert error.value.category == "provider_unavailable"
    assert error.value.error_type == "server_error"
    assert error.value.message == "stream failed"
    assert isinstance(error.value.__cause__, APIError)
    assert answer_text(seen) == "Partial"
    assert not any(isinstance(event, TurnCompleted) for event in seen)


def test_run_agent_reports_that_a_turn_without_answer_text_produced_none() -> None:
    # Given: a provider stream that only reasons and then stops.
    body = sse_body(
        sse_chunk({"reasoning": "thinking"}),
        sse_chunk({}, finish_reason="stop"),
    )

    # When: the agent consumes it.
    with make_openai_client(httpx2.ByteStream(body)) as client:
        events = list(run_agent(client, make_agent_request()))

    # Then: the turn still completes, and says it produced no answer.
    assert events[-1] == TurnCompleted(produced_answer=False)


def test_run_agent_counts_answer_text_from_an_earlier_round() -> None:
    # Given: a first round with text and a tool call, then an empty final round.
    first_round = sse_body(
        sse_chunk({"content": "Checking."}),
        sse_chunk(
            {"tool_calls": [tool_call(0, "call_1", "no_such_tool", "{}")]},
            finish_reason="tool_calls",
        ),
    )
    empty_final_round = sse_body(sse_chunk({}, finish_reason="stop"))
    client, _requests = make_round_trip_client(first_round, empty_final_round)

    # When: the agent runs both rounds.
    with client:
        events = list(run_agent(client, make_agent_request(with_tools=True)))

    # Then: the run produced an answer even though its last round did not.
    assert events[-1] == TurnCompleted(produced_answer=True)


def test_run_agent_normalizes_a_stream_event_that_is_not_json() -> None:
    # Given: partial answer text followed by a data line that is not JSON.
    body = sse_chunk({"content": "Partial"}) + b"data: {not json}\n\n"
    seen: list[AgentEvent] = []

    # When: the consumer iterates the provider stream.
    with (
        make_openai_client(httpx2.ByteStream(body)) as client,
        pytest.raises(ProviderError) as error,
    ):
        seen.extend(run_agent(client, make_agent_request()))

    # Then: the failure is a protocol error that keeps its cause and earlier output.
    assert error.value.category == "protocol"
    assert error.value.message == "provider stream contained an event that is not JSON"
    assert isinstance(error.value.__cause__, json.JSONDecodeError)
    assert answer_text(seen) == "Partial"
    assert not any(isinstance(event, TurnCompleted) for event in seen)


@pytest.mark.parametrize(
    ("finish_reason", "expected_category"),
    [
        ("length", "context_length"),
        ("content_filter", "content_policy"),
        ("future_reason", "unknown"),
        ("function_call", "protocol"),
        (None, "protocol"),
        ("tool_calls", "protocol"),
    ],
)
def test_run_agent_normalizes_failure_finish_reasons(
    finish_reason: str | None, expected_category: ProviderErrorCategory
) -> None:
    # Given: partial output followed by a failure, unrecognized or unusable finish
    # reason (tool_calls here has no calls to run).
    body = sse_body(sse_chunk({"content": "Partial"}, finish_reason=finish_reason))
    seen: list[AgentEvent] = []

    # When: the consumer iterates the provider stream.
    with (
        make_openai_client(httpx2.ByteStream(body)) as client,
        pytest.raises(ProviderError) as error,
    ):
        seen.extend(run_agent(client, make_agent_request()))

    # Then: the reason has its normalized category without losing partial output.
    assert error.value.category == expected_category
    assert error.value.error_type is None
    assert answer_text(seen) == "Partial"
    assert not any(isinstance(event, TurnCompleted) for event in seen)


def test_run_agent_preserves_event_types_and_provider_order() -> None:
    # Given: interleaved raw reasoning, summaries, answer text, and unknown details.
    body = sse_body(
        sse_chunk({"reasoning_details": [{"type": "reasoning.text", "text": "r1"}]}),
        sse_chunk(
            {"reasoning_details": [None, {"type": "reasoning.opaque", "data": "x"}]}
        ),
        sse_chunk({"content": "A"}),
        sse_chunk(
            {"reasoning_details": [{"type": "reasoning.summary", "summary": "s1"}]}
        ),
        sse_chunk({"reasoning": "top-level reasoning"}),
        sse_chunk(
            {"reasoning_details": [{"type": "reasoning.text", "text": "late-detail"}]}
        ),
        sse_chunk(
            {"reasoning_details": [{"type": "reasoning.summary", "summary": "s2"}]}
        ),
        sse_chunk({"reasoning_details": [{"type": "reasoning.text", "text": "r2"}]}),
        sse_chunk({"content": "B"}, finish_reason="stop"),
        sse_chunk({}),
    )

    # When: the agent consumes the interleaved provider stream.
    with make_openai_client(httpx2.ByteStream(body)) as client:
        events = list(run_agent(client, make_agent_request()))

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
        (ReasoningDelta, "late-detail"),
        (ReasoningSummary, "s2"),
        (ReasoningDelta, "r2"),
        (AnswerDelta, "B"),
    ]
    assert isinstance(events[-1], TurnCompleted)
