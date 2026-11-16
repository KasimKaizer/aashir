"""Provider stream contracts: finish reasons and raw reasoning fields."""

import json
from pathlib import Path

import httpx2
import pytest
from pydantic import JsonValue

from aashir.agent import (
    AgentEvent,
    ProviderError,
    ProviderErrorCategory,
    ReasoningDelta,
    ToolCallReceived,
    ToolResult,
    TurnCompleted,
    run_agent,
)
from tests.helpers import (
    answer_text,
    make_agent_request,
    make_openai_client,
    sse_body,
    sse_chunk,
    tool_call,
)


@pytest.mark.parametrize(
    ("finish_reason", "expected_category"),
    [("length", "context_length"), ("stop", "protocol")],
)
def test_run_agent_rejects_tool_calls_without_a_tool_calls_finish_reason(
    tmp_path: Path, finish_reason: str, expected_category: ProviderErrorCategory
) -> None:
    # Given: a complete write call followed by a finish reason other than tool_calls.
    output_path = tmp_path / "must-not-be-written.txt"
    arguments = json.dumps({"file_path": str(output_path), "content": "no"})
    body = sse_body(
        sse_chunk(
            {"tool_calls": [tool_call(0, "call_write", "write_file", arguments)]}
        ),
        sse_chunk({}, finish_reason=finish_reason),
    )
    events: list[AgentEvent] = []

    # When: the agent consumes the completion.
    with (
        make_openai_client(httpx2.ByteStream(body)) as client,
        pytest.raises(ProviderError) as error,
    ):
        events.extend(run_agent(client, make_agent_request(with_tools=True)))

    # Then: the call is rejected before it is announced or executed.
    assert error.value.category == expected_category
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
            {"reasoning": "", "reasoning_content": "alias reasoning"},
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
    body = sse_body(
        sse_chunk({"content": "answer", **reasoning_fields}, finish_reason="stop")
    )

    # When: the agent parses the provider chunk.
    with make_openai_client(httpx2.ByteStream(body)) as client:
        events = list(run_agent(client, make_agent_request()))

    # Then: one reasoning delta is emitted and answer text stays separate.
    assert [event.text for event in events if isinstance(event, ReasoningDelta)] == [
        expected
    ]
    assert answer_text(events) == "answer"
