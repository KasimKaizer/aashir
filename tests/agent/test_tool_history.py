"""Tool calls and results in the conversation history sent back to the model."""

import json
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletionMessageParam

from aashir.agent import (
    AgentRequest,
    ToolCallReceived,
    ToolResult,
    TurnCompleted,
    run_agent,
)
from aashir.tools import AGENT_TOOLS
from tests.helpers import (
    make_round_trip_client,
    run_tool_round_trip,
    sse_body,
    sse_chunk,
    tool_call,
)


@pytest.fixture
def read_and_unknown_turn(tmp_path: Path) -> tuple[str, bytes]:
    """A provider turn calling read_file on a real file, then an unknown tool.

    Returns:
        The file path the read_file call uses, and the provider's SSE body.
    """
    path = tmp_path / "note.txt"
    path.write_text("café\n", encoding="utf-8", newline="")
    calls = [
        tool_call(0, "call_tool", "read_file", json.dumps({"file_path": str(path)})),
        tool_call(1, "call_unknown", "unknown_tool", json.dumps({"unused": "value"})),
    ]
    delta = {"role": "assistant", "content": "Checking.", "tool_calls": calls}
    return str(path), sse_body(sse_chunk(delta, finish_reason="tool_calls"))


def test_run_agent_emits_each_tool_call_before_its_result_and_then_completes(
    read_and_unknown_turn: tuple[str, bytes],
) -> None:
    # Given: a provider turn with a real file read and an unknown tool call.
    path, first_turn = read_and_unknown_turn

    # When: the agent completes the tool round trip.
    round_trip = run_tool_round_trip(first_turn)

    # Then: each call is announced, then answered, in order, before completion.
    tool_events = [
        event
        for event in round_trip.events
        if isinstance(event, (ToolCallReceived, ToolResult))
    ]
    assert tool_events == [
        ToolCallReceived(
            call_id="call_tool",
            name="read_file",
            arguments=json.dumps({"file_path": path}),
        ),
        ToolResult(call_id="call_tool", name="read_file", output="1:café"),
        ToolCallReceived(
            call_id="call_unknown",
            name="unknown_tool",
            arguments=json.dumps({"unused": "value"}),
        ),
        ToolResult(
            call_id="call_unknown",
            name="unknown_tool",
            output="error: unknown tool called",
        ),
    ]
    assert isinstance(round_trip.events[-1], TurnCompleted)


def test_run_agent_sends_both_tool_calls_and_their_results_in_the_next_request(
    read_and_unknown_turn: tuple[str, bytes],
) -> None:
    # Given: a provider turn with a real file read and an unknown tool call.
    _, first_turn = read_and_unknown_turn

    # When: the agent completes the tool round trip.
    round_trip = run_tool_round_trip(first_turn)

    # Then: the follow-up request carries the assistant calls and both results.
    messages = json.loads(round_trip.requests[1].content)["messages"]
    assistant_message = messages[-3]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == "Checking."
    assert [call["id"] for call in assistant_message["tool_calls"]] == [
        "call_tool",
        "call_unknown",
    ]
    assert messages[-2:] == [
        {"role": "tool", "tool_call_id": "call_tool", "content": "1:café"},
        {
            "role": "tool",
            "tool_call_id": "call_unknown",
            "content": "error: unknown tool called",
        },
    ]


def test_run_agent_does_not_change_the_callers_message_list(
    read_and_unknown_turn: tuple[str, bytes],
) -> None:
    # Given: a request whose message list the caller keeps.
    _, first_turn = read_and_unknown_turn
    initial_messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "use a tool"}
    ]
    request = AgentRequest(
        model="test-model", messages=initial_messages, tools=AGENT_TOOLS
    )
    client, _ = make_round_trip_client(first_turn)

    # When: the agent runs a tool round trip with it.
    with client:
        list(run_agent(client, request))

    # Then: the caller's list is untouched.
    assert initial_messages == [{"role": "user", "content": "use a tool"}]
