"""Recovery from incomplete, unknown, invalid, and duplicate tool calls."""

import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from aashir.agent import ToolResult
from tests.helpers import (
    ToolRoundTrip,
    run_tool_round_trip,
    sse_body,
    sse_chunk,
    tool_call,
)


def _write_arguments(path: Path, content: str) -> str:
    return json.dumps({"file_path": str(path), "content": content})


def _tool_results(round_trip: ToolRoundTrip) -> list[tuple[str, str]]:
    return [
        (event.call_id, event.output)
        for event in round_trip.events
        if isinstance(event, ToolResult)
    ]


def _tool_calls_turn(*calls: dict[str, JsonValue]) -> bytes:
    delta = {"tool_calls": list(calls)}
    return sse_body(sse_chunk(delta, finish_reason="tool_calls"))


@pytest.fixture
def incomplete_call_turn(tmp_path: Path) -> bytes:
    """A provider turn with valid writes around a call whose arguments are empty."""
    return _tool_calls_turn(
        tool_call(
            0, "call_first", "write_file", _write_arguments(tmp_path / "a.txt", "first")
        ),
        tool_call(1, "call_incomplete", "read_file", ""),
        tool_call(
            2,
            "call_recovered",
            "write_file",
            _write_arguments(tmp_path / "b.txt", "recovered"),
        ),
    )


def test_incomplete_call_gets_an_error_and_the_valid_calls_around_it_still_run(
    tmp_path: Path, incomplete_call_turn: bytes
) -> None:
    # Given: a batch with an incomplete call between two valid writes.
    # When: the agent runs the batch.
    round_trip = run_tool_round_trip(incomplete_call_turn)

    # Then: every call has one result in call order, and both valid writes ran.
    assert _tool_results(round_trip) == [
        ("call_first", f"success: written 5 characters in {tmp_path / 'a.txt'}"),
        ("call_incomplete", "error: incomplete tool call in response"),
        ("call_recovered", f"success: written 9 characters in {tmp_path / 'b.txt'}"),
    ]
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "first"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "recovered"


def test_incomplete_call_results_are_sent_back_to_the_model_in_call_order(
    tmp_path: Path, incomplete_call_turn: bytes
) -> None:
    # Given: a batch with an incomplete call between two valid writes.
    # When: the agent runs the batch and sends its follow-up request.
    round_trip = run_tool_round_trip(incomplete_call_turn)

    # Then: all results, including the error, follow the assistant message.
    messages = json.loads(round_trip.requests[1].content)["messages"]
    assert messages[-4]["role"] == "assistant"
    assert messages[-3:] == [
        {
            "role": "tool",
            "tool_call_id": "call_first",
            "content": f"success: written 5 characters in {tmp_path / 'a.txt'}",
        },
        {
            "role": "tool",
            "tool_call_id": "call_incomplete",
            "content": "error: incomplete tool call in response",
        },
        {
            "role": "tool",
            "tool_call_id": "call_recovered",
            "content": f"success: written 9 characters in {tmp_path / 'b.txt'}",
        },
    ]


@pytest.mark.parametrize(
    ("call_id", "name"),
    [("", "write_file"), ("call_nameless", "")],
    ids=("missing-id", "missing-name"),
)
def test_call_without_an_id_or_name_is_not_executed(
    tmp_path: Path, call_id: str, name: str
) -> None:
    # Given: a call with valid write arguments but no id or no tool name.
    path = tmp_path / "must-not-be-written.txt"
    first_turn = _tool_calls_turn(
        tool_call(0, call_id, name, _write_arguments(path, "no"))
    )

    # When: the agent runs the batch.
    round_trip = run_tool_round_trip(first_turn)

    # Then: it returns the incomplete-call error and the side effect never happens.
    assert _tool_results(round_trip) == [
        (call_id, "error: incomplete tool call in response")
    ]
    assert not path.exists()


def test_unknown_invalid_and_duplicate_calls_each_get_one_result_and_do_not_run(
    tmp_path: Path,
) -> None:
    # Given: valid writes around unknown, invalid, and duplicate-ID calls.
    first_turn = _tool_calls_turn(
        tool_call(
            0, "call_write", "write_file", _write_arguments(tmp_path / "a.txt", "saved")
        ),
        tool_call(1, "call_unknown", "unknown_tool", json.dumps({"unused": "value"})),
        tool_call(
            2,
            "call_invalid",
            "write_file",
            json.dumps({"file_path": str(tmp_path / "invalid.txt")}),
        ),
        tool_call(
            3,
            "call_write",
            "write_file",
            _write_arguments(tmp_path / "duplicate.txt", "ignored"),
        ),
        tool_call(
            4,
            "call_after_errors",
            "write_file",
            _write_arguments(tmp_path / "b.txt", "done"),
        ),
    )

    # When: the agent runs the batch.
    round_trip = run_tool_round_trip(first_turn)

    # Then: every call has one result in call order, and only valid writes ran.
    assert _tool_results(round_trip) == [
        ("call_write", f"success: written 5 characters in {tmp_path / 'a.txt'}"),
        ("call_unknown", "error: unknown tool called"),
        ("call_invalid", "error: content must be a string"),
        ("call_write", "error: duplicate tool call id in response"),
        ("call_after_errors", f"success: written 4 characters in {tmp_path / 'b.txt'}"),
    ]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.txt", "b.txt"]
