import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx2
import pytest
from openai import APIError, OpenAI

from aashir import cli


@pytest.mark.parametrize("key", [None, ""])
def test_main_reports_missing_api_key_when_unset_or_empty(
    monkeypatch: pytest.MonkeyPatch, key: str | None
) -> None:
    monkeypatch.setattr(sys, "argv", ["aashir", "-p", "hello"])
    monkeypatch.setattr(cli, "API_KEY", key)
    monkeypatch.setattr(cli, "BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(cli, "MODEL_NAME", "test-model")
    monkeypatch.setattr(cli, "SYSTEM_PROMPT", "prompt.txt")
    monkeypatch.setattr(cli, "REASONING_EFFORT", "medium")

    with pytest.raises(
        RuntimeError, match="^Missing required variables: OPENAI_API_KEY$"
    ):
        cli.main()


def test_main_lists_all_missing_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["aashir", "-p", "hello"])
    monkeypatch.setattr(cli, "API_KEY", None)
    monkeypatch.setattr(cli, "BASE_URL", "")
    monkeypatch.setattr(cli, "MODEL_NAME", "test-model")
    monkeypatch.setattr(cli, "SYSTEM_PROMPT", "prompt.txt")
    monkeypatch.setattr(cli, "REASONING_EFFORT", "medium")

    with pytest.raises(
        RuntimeError,
        match="^Missing required variables: OPENAI_API_KEY, OPENAI_BASE_URL$",
    ):
        cli.main()


def _chunk(
    content: str | None = None,
    *,
    finish_reason: str | None = None,
    empty_choices: bool = False,
) -> bytes:
    choices = (
        []
        if empty_choices
        else [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": finish_reason,
            }
        ]
    )
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": choices,
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


@pytest.fixture
def mock_streaming_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[httpx2.SyncByteStream], None]:
    monkeypatch.setattr(sys, "argv", ["aashir", "-p", "hello"])
    monkeypatch.setattr(cli, "API_KEY", "test-key")
    monkeypatch.setattr(cli, "BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(cli, "MODEL_NAME", "test-model")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("You are a test assistant.", encoding="utf-8")
    monkeypatch.setattr(cli, "SYSTEM_PROMPT", str(prompt_file))
    monkeypatch.setattr(cli, "REASONING_EFFORT", "medium")

    def install(stream: httpx2.SyncByteStream) -> None:
        def respond(_request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )

        def make_client(*, api_key: str, base_url: str) -> OpenAI:
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=0,
                http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
            )

        monkeypatch.setattr(cli, "OpenAI", make_client)

    return install


def test_main_prints_text_before_stream_finishes(
    mock_streaming_api: Callable[[httpx2.SyncByteStream], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed_before_finish: list[str] = []

    class StreamingBody(httpx2.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield _chunk("Hel")
            observed_before_finish.append(capsys.readouterr().out)
            assert observed_before_finish[-1] == "Hel"
            yield _chunk(empty_choices=True)
            yield _chunk("lo", finish_reason="stop")
            yield b"data: [DONE]\n\n"

    mock_streaming_api(StreamingBody())

    cli.main()

    output = "".join(observed_before_finish) + capsys.readouterr().out
    assert output in {"Hello", "Hello\n"}


def test_main_reports_stream_with_no_choices(
    mock_streaming_api: Callable[[httpx2.SyncByteStream], None],
) -> None:
    mock_streaming_api(
        httpx2.ByteStream(_chunk(empty_choices=True) + b"data: [DONE]\n\n")
    )

    with pytest.raises(RuntimeError, match="no choices in response"):
        cli.main()


def test_main_preserves_error_after_partial_output(
    mock_streaming_api: Callable[[httpx2.SyncByteStream], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    body = _chunk("Partial") + (
        b'data: {"error": {"message": "stream failed", "type": "server_error"}}\n\n'
    )
    mock_streaming_api(httpx2.ByteStream(body))

    with pytest.raises(APIError, match="stream failed"):
        cli.main()

    assert capsys.readouterr().out == "Partial"


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
        tool_chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_tool",
                                "type": "function",
                                "function": {"name": tool_name, "arguments": arguments},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        requests: list[httpx2.Request] = []

        def respond(request: httpx2.Request) -> httpx2.Response:
            requests.append(request)
            body = (
                f"data: {json.dumps(tool_chunk)}\n\n".encode()
                if len(requests) == 1
                else _chunk("Done", finish_reason="stop")
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
        assert len(requests) == 2
        return requests[0], requests[1]

    return run


def test_main_returns_read_file_result_to_model(
    tmp_path: Path,
    run_tool_round_trip: Callable[[str, str], tuple[httpx2.Request, httpx2.Request]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "note.txt").write_text("café\n", encoding="utf-8")

    _, request = run_tool_round_trip("read_file", json.dumps({"file_path": "note.txt"}))

    messages = json.loads(request.content)["messages"]
    assert messages[-2]["role"] == "assistant"
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_tool",
        "content": "café\n",
    }
    assert "Done" in capsys.readouterr().out


def test_main_returns_write_file_result_to_model(
    tmp_path: Path,
    run_tool_round_trip: Callable[[str, str], tuple[httpx2.Request, httpx2.Request]],
) -> None:
    arguments = json.dumps({"file_path": "notes/note.txt", "content": "café\n"})

    _, request = run_tool_round_trip("write_file", arguments)

    messages = json.loads(request.content)["messages"]
    assert (tmp_path / "notes" / "note.txt").read_text(encoding="utf-8") == "café\n"
    assert messages[-2]["role"] == "assistant"
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "call_tool"
    assert messages[-1]["content"].startswith("success:")


def test_main_advertises_edit_file_tool_to_model(
    tmp_path: Path,
    run_tool_round_trip: Callable[
        [str, str], tuple[httpx2.Request, httpx2.Request]
    ],
) -> None:
    (tmp_path / "note.txt").write_text("note", encoding="utf-8")

    initial_request, _ = run_tool_round_trip(
        "read_file", json.dumps({"file_path": "note.txt"})
    )

    initial_payload = json.loads(initial_request.content)
    edit_function = next(
        tool["function"]
        for tool in initial_payload["tools"]
        if tool["function"]["name"] == "edit_file"
    )
    parameters = edit_function["parameters"]
    assert set(parameters["required"]) == {"file_path", "edits"}
    assert parameters["properties"]["file_path"]["type"] == "string"
    edits_schema = parameters["properties"]["edits"]
    assert edits_schema["type"] == "array"
    edit_schema = edits_schema["items"]
    assert set(edit_schema["required"]) == {"oldText", "newText"}
    assert edit_schema["properties"]["oldText"]["type"] == "string"
    assert edit_schema["properties"]["newText"]["type"] == "string"


def test_main_returns_edit_file_result_to_model(
    tmp_path: Path,
    run_tool_round_trip: Callable[
        [str, str], tuple[httpx2.Request, httpx2.Request]
    ],
) -> None:
    path = tmp_path / "note.txt"
    path.write_text("title=old\nstatus=ready\n", encoding="utf-8")
    arguments = json.dumps(
        {
            "file_path": "note.txt",
            "edits": [
                {"oldText": "title=old", "newText": "title=new"},
                {"oldText": "status=ready", "newText": "status=done"},
            ],
        }
    )

    _, result_request = run_tool_round_trip("edit_file", arguments)

    messages = json.loads(result_request.content)["messages"]
    assert path.read_text(encoding="utf-8") == "title=new\nstatus=done\n"
    assert messages[-2]["role"] == "assistant"
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "call_tool"
    assert messages[-1]["content"].startswith("success:")
