"""End-to-end CLI workflows that call advertised tools."""

import json
import sys
from collections.abc import Callable
from pathlib import Path

import httpx2
import pytest
from openai import OpenAI

from aashir import cli
from aashir.tools import ToolCatalog, edit_file, read_file, write_file
from tests.helpers import (
    ClientArguments,
    make_round_trip_client,
    sse_body,
    sse_chunk,
    tool_call,
)


@pytest.fixture
def run_main_with_tool_call(
    configured_cli: None, serve_provider: Callable[[OpenAI], list[ClientArguments]]
) -> Callable[[str, str], list[httpx2.Request]]:
    """Return a function that runs ``cli.main()`` against one provider tool call.

    The function takes a tool name and its raw JSON arguments, and returns the
    requests the provider received: the initial one, then the follow-up.
    """

    def run(tool_name: str, arguments: str) -> list[httpx2.Request]:
        delta = {
            "role": "assistant",
            "tool_calls": [tool_call(0, "call_tool", tool_name, arguments)],
        }
        first_turn = sse_body(sse_chunk(delta, finish_reason="tool_calls"))
        client, requests = make_round_trip_client(first_turn)
        serve_provider(client)
        cli.main()
        return requests

    return run


def test_main_sends_the_configured_request_to_the_provider(
    run_main_with_tool_call: Callable[[str, str], list[httpx2.Request]],
) -> None:
    # Given: a configured CLI invocation.
    # When: the CLI completes a provider tool round trip.
    initial_request, _ = run_main_with_tool_call("read_file", "{}")

    # Then: the request uses the configured settings and prompts, and sends
    # exactly the advertised tools.
    payload = json.loads(initial_request.content)
    assert payload["model"] == "test-model"
    assert payload["reasoning_effort"] == "medium"
    assert payload["messages"] == [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "hello"},
    ]
    assert payload["tools"] == list(
        ToolCatalog((read_file, write_file, edit_file)).provider_tools()
    )


def test_main_returns_a_read_file_result_to_the_provider(
    tmp_path: Path,
    run_main_with_tool_call: Callable[[str, str], list[httpx2.Request]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a UTF-8 file that the advertised read_file tool can access.
    path = tmp_path / "note.txt"
    path.write_text("café\n", encoding="utf-8", newline="")

    # When: the CLI completes a read_file round trip.
    _, follow_up_request = run_main_with_tool_call(
        "read_file", json.dumps({"file_path": str(path)})
    )

    # Then: the provider receives the file text and the assistant can finish.
    messages = json.loads(follow_up_request.content)["messages"]
    assert messages[-2]["role"] == "assistant"
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_tool",
        "content": "1:café",
    }
    assert capsys.readouterr().out == "Done"


def test_main_applies_the_tool_argument_byte_cap_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_main_with_tool_call: Callable[[str, str], list[httpx2.Request]],
) -> None:
    # Given: a 1-byte tool argument cap chosen by flag and a valid write call.
    monkeypatch.setattr(
        sys, "argv", ["aashir", "-p", "hello", "--tool-argument-byte-cap", "1"]
    )
    path = tmp_path / "blocked.txt"
    arguments = json.dumps({"file_path": str(path), "content": "text"})

    # When: the CLI completes the round trip.
    _, follow_up_request = run_main_with_tool_call("write_file", arguments)

    # Then: the call is rejected without writing, and the provider is told why.
    assert not path.exists()
    assert json.loads(follow_up_request.content)["messages"][-1]["content"] == (
        "error: tool arguments exceed the 1-byte limit"
    )


def test_main_returns_a_tool_error_to_the_provider(
    tmp_path: Path,
    run_main_with_tool_call: Callable[[str, str], list[httpx2.Request]],
) -> None:
    # Given: a read_file call for a file that does not exist.
    path = tmp_path / "missing.txt"

    # When: the CLI completes the round trip.
    _, follow_up_request = run_main_with_tool_call(
        "read_file", json.dumps({"file_path": str(path)})
    )

    # Then: the provider receives an error that names the missing file. (Only the
    # name is compared: OSError doubles the backslashes of a Windows path.)
    tool_message = json.loads(follow_up_request.content)["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_tool"
    assert tool_message["content"].startswith("error:")
    assert path.name in tool_message["content"]
