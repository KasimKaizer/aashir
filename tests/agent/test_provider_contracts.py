import json
from pathlib import Path

import httpx2
import pytest

from aashir.agent import (
    AgentEvent,
    AgentRequest,
    AnswerDelta,
    ProviderError,
    ReasoningDelta,
    ToolCallReceived,
    ToolResult,
    TurnCompleted,
    run_agent,
)
from aashir.tools import AGENT_TOOLS
from tests.helpers import JsonValue, make_openai_client, sse_chunk


def test_run_agent_rejects_length_truncated_tool_calls_before_execution(
    tmp_path: Path,
) -> None:
    # Given: a complete write call followed by a length-truncated finish reason.
    output_path = tmp_path / "must-not-be-written.txt"
    arguments = json.dumps({"file_path": str(output_path), "content": "no"})
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
    body += sse_chunk({}, finish_reason="length")
    body += b"data: [DONE]\n\n"
    request = AgentRequest(
        model="test-model",
        messages=[{"role": "user", "content": "do not write"}],
        tools=AGENT_TOOLS,
    )
    events: list[AgentEvent] = []

    # When: the agent consumes the truncated completion.
    client = make_openai_client(httpx2.ByteStream(body))
    with client, pytest.raises(ProviderError) as error:
        events.extend(run_agent(client, request))

    # Then: truncation is reported before the tool is announced or executed.
    assert error.value.category == "context_length"
    assert not output_path.exists()
    assert not any(
        isinstance(event, (ToolCallReceived, ToolResult, TurnCompleted))
        for event in events
    )


@pytest.mark.parametrize(
    ("reasoning_fields", "expected"),
    [
        ({"reasoning_content": "alias reasoning"}, "alias reasoning"),
        (
            {
                "reasoning": "",
                "reasoning_content": "alias reasoning",
            },
            "alias reasoning",
        ),
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
    with make_openai_client(httpx2.ByteStream(body)) as client:
        events = list(
            run_agent(
                client,
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
    with (
        make_openai_client(httpx2.ByteStream(body)) as client,
        pytest.raises(ProviderError) as error,
    ):
        events.extend(
            run_agent(
                client,
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
    with (
        make_openai_client(httpx2.ByteStream(body)) as client,
        pytest.raises(ProviderError) as error,
    ):
        events.extend(
            run_agent(
                client,
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
        isinstance(event, (ToolCallReceived, ToolResult, TurnCompleted))
        for event in events
    )
