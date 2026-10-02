"""End-to-end CLI workflows that call advertised tools."""

import json
import sys
from collections.abc import Callable
from pathlib import Path

import httpx2
import pytest
from openai import OpenAI

from aashir import cli
from tests.helpers import sse_chunk


@pytest.fixture
def run_tool_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[str, str], tuple[httpx2.Request, httpx2.Request]]:
    monkeypatch.chdir(tmp_path)
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("You are a test assistant.", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["aashir", "-p", "use a tool"])
    monkeypatch.setattr(cli, "API_KEY", "test-key")
    monkeypatch.setattr(cli, "BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(cli, "MODEL_NAME", "test-model")
    monkeypatch.setattr(cli, "SYSTEM_PROMPT", str(prompt_file))
    monkeypatch.setattr(cli, "REASONING_EFFORT", "medium")

    def run(tool_name: str, arguments: str) -> tuple[httpx2.Request, httpx2.Request]:
        tool_delta = {
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_tool",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": arguments},
                }
            ],
        }
        requests: list[httpx2.Request] = []

        def respond(request: httpx2.Request) -> httpx2.Response:
            requests.append(request)
            body = (
                sse_chunk(tool_delta, finish_reason="tool_calls")
                if len(requests) == 1
                else sse_chunk({"content": "Done"}, finish_reason="stop")
            )
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body + b"data: [DONE]\n\n",
            )

        def make_client(*, api_key: str, base_url: str) -> OpenAI:
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=0,
                http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
            )

        monkeypatch.setattr(cli, "OpenAI", make_client)
        cli.main()
        initial_request, follow_up_request = requests
        return initial_request, follow_up_request

    return run


def test_main_sends_configured_request_and_advertises_tools(
    run_tool_round_trip: Callable[[str, str], tuple[httpx2.Request, httpx2.Request]],
) -> None:
    # Given: a configured CLI invocation that requests a missing file.
    # When: the CLI completes the provider tool round-trip.
    initial_request, follow_up_request = run_tool_round_trip(
        "read_file", json.dumps({"file_path": "missing.txt"})
    )
    initial_payload = json.loads(initial_request.content)

    # Then: the request uses configured settings and advertises the tools.
    assert initial_payload["model"] == "test-model"
    assert initial_payload["reasoning_effort"] == "medium"
    initial_messages = initial_payload["messages"]
    assert initial_messages[0] == {
        "role": "system",
        "content": "You are a test assistant.",
    }
    assert initial_messages[-1] == {"role": "user", "content": "use a tool"}

    edit_function = next(
        tool["function"]
        for tool in initial_payload["tools"]
        if tool["function"]["name"] == "edit_file"
    )
    parameters = edit_function["parameters"]
    assert {"file_path", "edits"} <= set(parameters["required"])
    assert parameters["properties"]["file_path"]["type"] == "string"
    edits_schema = parameters["properties"]["edits"]
    assert edits_schema["type"] == "array"
    edit_schema = edits_schema["items"]
    assert {"oldText", "newText"} <= set(edit_schema["required"])
    assert edit_schema["properties"]["oldText"]["type"] == "string"
    assert edit_schema["properties"]["newText"]["type"] == "string"

    # Then: a missing-file tool error is returned to the provider.
    tool_result = json.loads(follow_up_request.content)["messages"][-1]
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == "call_tool"
    assert tool_result["content"].startswith("error:")


def test_main_returns_tool_result_to_provider_after_round_trip(
    tmp_path: Path,
    run_tool_round_trip: Callable[[str, str], tuple[httpx2.Request, httpx2.Request]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a UTF-8 file that the advertised read_file tool can access.
    (tmp_path / "note.txt").write_text("café\n", encoding="utf-8")

    # When: the CLI completes a read_file round-trip.
    _, follow_up_request = run_tool_round_trip(
        "read_file", json.dumps({"file_path": "note.txt"})
    )

    # Then: the provider receives the tool result and the assistant can finish.
    messages = json.loads(follow_up_request.content)["messages"]
    assistant_message = messages[-2]
    tool_result = messages[-1]
    assert assistant_message["role"] == "assistant"
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == assistant_message["tool_calls"][0]["id"]
    assert tool_result["content"] == "café\n"
    assert "Done" in capsys.readouterr().out
