import json
import sys
from pathlib import Path

import httpx2
import pytest
from openai import OpenAI

from aashir import cli
from aashir.tools import execute_tool_call


def test_read_tool_returns_text_from_requested_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "note.txt").write_text("café\n", encoding="utf-8")

    result = execute_tool_call("Read", json.dumps({"file_path": "note.txt"}))

    assert result == "café\n"


def test_agent_reads_file_and_sends_result_back_to_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "note.txt").write_text("café\n", encoding="utf-8")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("You are a test assistant.", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["aashir", "-p", "Read note.txt"])
    monkeypatch.setattr(cli, "API_KEY", "test-key")
    monkeypatch.setattr(cli, "BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(cli, "MODEL_NAME", "test-model")
    monkeypatch.setattr(cli, "SYSTEM_PROMPT", str(prompt_file))
    monkeypatch.setattr(cli, "REASONING_EFFORT", "medium")
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if len(requests) == 1:
            deltas = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_read",
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "arguments": json.dumps({"file_path": "note.txt"}),
                            },
                        }
                    ],
                },
                {"role": "assistant"},
            ]
            finish_reason = "tool_calls"
        else:
            deltas = [{"content": "The note says café."}]
            finish_reason = "stop"

        chunks = [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason
                        if i == len(deltas) - 1
                        else None,
                    }
                ],
            }
            for i, delta in enumerate(deltas)
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        body += "data: [DONE]\n\n"
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, text=body
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

    assert len(requests) == 2
    messages = json.loads(requests[1].content)["messages"]
    assert messages[-2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_read",
                "type": "function",
                "function": {
                    "name": "Read",
                    "arguments": json.dumps({"file_path": "note.txt"}),
                },
            }
        ],
    }
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_read",
        "content": "café\n",
    }
    assert "The note says café." in capsys.readouterr().out
