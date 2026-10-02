import json
from pathlib import Path

import httpx2
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from aashir.agent import (
    AgentRequest,
    AnswerDelta,
    ProviderError,
    ReasoningDelta,
    ToolCallStarted,
    ToolResult,
    TurnCompleted,
    run_agent,
)
from aashir.tools import AGENT_TOOLS
from tests.helpers import (
    JsonValue,
    make_failing_openai_client,
    make_openai_client,
    sse_chunk,
)


@pytest.mark.parametrize(
    ("reasoning_fields", "expected"),
    [
        ({"reasoning_content": "alias reasoning"}, "alias reasoning"),
        (
            {
                "reasoning": "canonical reasoning",
                "reasoning_content": "alias reasoning",
            },
            "canonical reasoning",
        ),
        (
            {
                "reasoning": {"unexpected": "value"},
                "reasoning_content": "alias reasoning",
            },
            "alias reasoning",
        ),
        (
            {
                "reasoning": "thinking",
                "reasoning_details": [{"type": "reasoning.text", "text": "thinking"}],
            },
            "thinking",
        ),
    ],
)
def test_run_agent_normalizes_documented_raw_reasoning_fields(
    reasoning_fields: dict[str, JsonValue], expected: str
) -> None:
    # Given: canonical, alias-only, duplicate, or invalid canonical reasoning data.
    body = sse_chunk({"content": "answer", **reasoning_fields}, finish_reason="stop")
    body += b"data: [DONE]\n\n"

    # When: the agent parses the provider chunk.
    events = list(
        run_agent(
            make_openai_client(httpx2.ByteStream(body)),
            AgentRequest(
                model="test-model", messages=[{"role": "user", "content": "hi"}]
            ),
        )
    )

    # Then: one reasoning delta is emitted and answer text stays separate.
    assert [event.text for event in events if isinstance(event, ReasoningDelta)] == [
        expected
    ]
    assert (
        "".join(event.text for event in events if isinstance(event, AnswerDelta))
        == "answer"
    )


@pytest.mark.parametrize(
    ("finish_reason",),
    [("function_call",), (None,), ("tool_calls",)],
)
def test_run_agent_rejects_nonterminal_or_incomplete_completions(
    finish_reason: str | None,
) -> None:
    # Given: partial answer text without a successful finish or required tool calls.
    body = sse_chunk({"content": "Partial"}, finish_reason=finish_reason)
    body += b"data: [DONE]\n\n"
    events = []

    # When: the agent consumes the incomplete completion.
    with pytest.raises(ProviderError) as error:
        events.extend(
            run_agent(
                make_openai_client(httpx2.ByteStream(body)),
                AgentRequest(
                    model="test-model", messages=[{"role": "user", "content": "hi"}]
                ),
            )
        )

    # Then: the protocol error preserves partial output without reporting completion.
    assert error.value.category == "protocol"
    assert (
        "".join(event.text for event in events if isinstance(event, AnswerDelta))
        == "Partial"
    )
    assert not any(isinstance(event, TurnCompleted) for event in events)


def test_run_agent_does_not_execute_tool_calls_without_tool_calls_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a complete tool call followed by an ordinary answer finish reason.
    arguments = json.dumps({"file_path": "must-not-be-written.txt", "content": "no"})
    body = sse_chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_write",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": arguments},
                }
            ]
        }
    )
    body += sse_chunk({}, finish_reason="stop")
    body += b"data: [DONE]\n\n"
    events = []

    def fail_on_execution(name: str, arguments: str) -> str:
        pytest.fail(f"unexpected tool execution: {name}({arguments})")

    monkeypatch.setattr("aashir.agent.tool_calls.execute_tool_call", fail_on_execution)

    # When: the agent consumes the stream.
    with pytest.raises(ProviderError) as error:
        events.extend(
            run_agent(
                make_openai_client(httpx2.ByteStream(body)),
                AgentRequest(
                    model="test-model",
                    messages=[{"role": "user", "content": "do not write"}],
                    tools=AGENT_TOOLS,
                ),
            )
        )

    # Then: the call is rejected before it is announced or executed.
    assert error.value.category == "protocol"
    assert not any(
        isinstance(event, (ToolCallStarted, ToolResult, TurnCompleted))
        for event in events
    )


def test_run_agent_handles_call_errors_and_continues_the_batch(
    tmp_path: Path,
) -> None:
    # Given: a valid write, an incomplete call, and a repeated call ID.
    written_path = tmp_path / "written.txt"
    duplicate_path = tmp_path / "duplicate.txt"
    calls = [
        {
            "index": 0,
            "id": "call_write",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": json.dumps(
                    {"file_path": str(written_path), "content": "saved"}
                ),
            },
        },
        {
            "index": 1,
            "id": "call_incomplete",
            "type": "function",
            "function": {"name": "read_file", "arguments": ""},
        },
        {
            "index": 2,
            "id": "call_write",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": json.dumps(
                    {"file_path": str(duplicate_path), "content": "ignored"}
                ),
            },
        },
    ]
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if len(requests) == 1:
            body = sse_chunk({"tool_calls": calls}, finish_reason="tool_calls")
        else:
            body = sse_chunk({"content": "Done"}, finish_reason="stop")
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body + b"data: [DONE]\n\n",
        )

    client = OpenAI(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    )

    # When: the agent processes the provider's tool-call batch.
    events = list(
        run_agent(
            client,
            AgentRequest(
                model="test-model",
                messages=[{"role": "user", "content": "use tools"}],
                tools=AGENT_TOOLS,
            ),
        )
    )

    # Then: each call yields a result, and the duplicate write is prevented.
    assert written_path.read_text(encoding="utf-8") == "saved"
    assert not duplicate_path.exists()
    results = [event for event in events if isinstance(event, ToolResult)]
    assert [(event.call_id, event.output) for event in results] == [
        ("call_write", f"success: written 5 characters in {written_path}"),
        ("call_incomplete", "error: incomplete tool call in response"),
        ("call_write", "error: duplicate tool call id in response"),
    ]
    assert isinstance(events[-1], TurnCompleted)

    # Then: all per-call outcomes are returned as tool messages.
    assert len(requests) == 2
    messages = json.loads(requests[1].content)["messages"]
    assert messages[-4]["role"] == "assistant"
    assert messages[-3:] == [
        {
            "role": "tool",
            "tool_call_id": "call_write",
            "content": f"success: written 5 characters in {written_path}",
        },
        {
            "role": "tool",
            "tool_call_id": "call_incomplete",
            "content": "error: incomplete tool call in response",
        },
        {
            "role": "tool",
            "tool_call_id": "call_write",
            "content": "error: duplicate tool call id in response",
        },
    ]


@pytest.mark.parametrize(
    ("error_class", "status", "expected_category"),
    [
        (RateLimitError, 429, "rate_limit"),
        (APIStatusError, 429, "rate_limit"),
        (AuthenticationError, 401, "auth"),
        (PermissionDeniedError, 403, "auth"),
        (APIStatusError, 401, "auth"),
        (APIStatusError, 403, "auth"),
        (InternalServerError, 500, "provider_unavailable"),
        (APIStatusError, 500, "provider_unavailable"),
        (APIStatusError, 503, "provider_unavailable"),
    ],
)
def test_run_agent_normalizes_sdk_status_error_classes(
    error_class: type[APIStatusError],
    status: int,
    expected_category: str,
) -> None:
    # Given: a provider transport that fails with an SDK status error.
    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    error = error_class(
        "provider failed",
        response=httpx2.Response(status, request=request),
        body=None,
    )

    # When: the agent attempts the provider request.
    events = run_agent(
        make_failing_openai_client(error),
        AgentRequest(model="test-model", messages=[{"role": "user", "content": "hi"}]),
    )
    with pytest.raises(ProviderError) as raised:
        list(events)

    # Then: the error is normalized with its message and cause preserved.
    assert raised.value.category == expected_category
    assert raised.value.error_type is None
    assert raised.value.message == "provider failed"
    assert raised.value.__cause__ is error


@pytest.mark.parametrize("error_class", [APIConnectionError, APITimeoutError])
def test_run_agent_normalizes_sdk_connection_error_classes(
    error_class: type[APIConnectionError],
) -> None:
    # Given: a provider transport that fails with an SDK connection error.
    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    error = error_class(request=request)

    # When: the agent attempts the provider request.
    events = run_agent(
        make_failing_openai_client(error),
        AgentRequest(model="test-model", messages=[{"role": "user", "content": "hi"}]),
    )
    with pytest.raises(ProviderError) as raised:
        list(events)

    # Then: the error is normalized with its message and cause preserved.
    assert raised.value.category == "provider_unavailable"
    assert raised.value.error_type is None
    assert raised.value.message == error.message
    assert raised.value.__cause__ is error
