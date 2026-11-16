"""Assembly of streamed tool-call fragments into ordered calls."""

import json
from pathlib import Path

import pytest

from aashir.agent import ToolCallReceived, ToolResult
from tests.helpers import run_tool_round_trip, sse_body, sse_chunk


@pytest.fixture
def fragmented_turn(tmp_path: Path) -> tuple[str, bytes]:
    """A provider turn with two read_file calls arriving as interleaved fragments.

    Call 0 starts with an empty id and no function. A late fragment then offers a
    different id and function name for it, which must not replace the first
    non-empty ones.

    Returns:
        The complete arguments of call 0, and the provider's SSE body.
    """
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8", newline="")
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8", newline="")
    args_a = json.dumps({"file_path": str(tmp_path / "a.txt")})
    args_b = json.dumps({"file_path": str(tmp_path / "b.txt")})
    split_a = len(args_a) // 2
    split_b = len(args_b) // 2
    deltas = [
        {
            "role": "assistant",
            "tool_calls": [{"index": 0, "id": "", "type": "function"}],
        },
        {
            "tool_calls": [
                {
                    "index": 1,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": args_b[:split_b]},
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_0",
                    "function": {"name": "read_file", "arguments": args_a[:split_a]},
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_0",
                    "function": {"name": "read_file", "arguments": args_a[split_a:]},
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_replacement",
                    "function": {"name": "write_file"},
                }
            ]
        },
        {"tool_calls": [{"index": 1, "function": {"arguments": args_b[split_b:]}}]},
    ]
    chunks = [sse_chunk(delta) for delta in deltas]
    chunks.append(sse_chunk({}, finish_reason="tool_calls"))
    return args_a, sse_body(*chunks)


def test_run_agent_runs_fragmented_calls_in_index_order_with_their_first_identity(
    fragmented_turn: tuple[str, bytes],
) -> None:
    # Given: interleaved fragments of two calls, one that starts without a function.
    args_a, first_turn = fragmented_turn

    # When: the agent assembles and runs the completed calls.
    round_trip = run_tool_round_trip(first_turn)

    # Then: calls run in index order, each announced before its result.
    tool_events = [
        event
        for event in round_trip.events
        if isinstance(event, (ToolCallReceived, ToolResult))
    ]
    assert [(type(event), event.call_id) for event in tool_events] == [
        (ToolCallReceived, "call_0"),
        (ToolResult, "call_0"),
        (ToolCallReceived, "call_1"),
        (ToolResult, "call_1"),
    ]
    first_call = tool_events[0]
    assert isinstance(first_call, ToolCallReceived)
    assert (first_call.name, first_call.arguments) == ("read_file", args_a)
    assert [event.output for event in tool_events if isinstance(event, ToolResult)] == [
        "1:alpha",
        "1:beta",
    ]
