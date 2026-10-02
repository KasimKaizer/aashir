"""CLI entrypoint behavior: validation, streaming, and answer presentation."""

import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx2
import pytest
from openai import OpenAI

from aashir import cli
from aashir.agent import (
    AnswerDelta,
    ProviderError,
    ReasoningDelta,
    ReasoningSummary,
    ToolCallStarted,
    ToolResult,
    TurnCompleted,
)


@pytest.mark.parametrize(
    ("api_key", "base_url", "model_name", "reasoning_effort", "expected_missing"),
    [
        (None, "https://example.invalid/v1", "test-model", "medium", "OPENAI_API_KEY"),
        ("", "https://example.invalid/v1", "test-model", "medium", "OPENAI_API_KEY"),
        (
            None,
            "",
            None,
            "",
            "OPENAI_API_KEY, OPENAI_BASE_URL, BASE_MODEL_NAME, REASONING_EFFORT",
        ),
    ],
)
def test_main_reports_missing_settings_when_required_values_are_unset_or_empty(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str | None,
    base_url: str,
    model_name: str | None,
    reasoning_effort: str | None,
    expected_missing: str,
) -> None:
    # Given: incomplete required settings and a valid prompt argument.
    monkeypatch.setattr(sys, "argv", ["aashir", "-p", "hello"])
    monkeypatch.setattr(cli, "API_KEY", api_key)
    monkeypatch.setattr(cli, "BASE_URL", base_url)
    monkeypatch.setattr(cli, "MODEL_NAME", model_name)
    monkeypatch.setattr(cli, "SYSTEM_PROMPT", "prompt.txt")
    monkeypatch.setattr(cli, "REASONING_EFFORT", reasoning_effort)

    # When: the CLI validates its configuration.
    with pytest.raises(RuntimeError) as error:
        cli.main()

    # Then: all missing settings are named in configuration order.
    assert str(error.value) == f"Missing required variables: {expected_missing}"


@pytest.mark.parametrize(
    ("prompt_path", "prompt_content", "warning_expected"),
    [
        (None, None, False),
        ("missing.txt", None, True),
        ("empty.txt", "", False),
    ],
)
def test_configure_request_uses_default_prompt_without_nonempty_custom_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    prompt_path: str | None,
    prompt_content: str | None,
    warning_expected: bool,
) -> None:
    # Given: valid settings with no prompt, a missing file, or an empty file.
    monkeypatch.setattr(sys, "argv", ["aashir", "-p", "hello"])
    monkeypatch.setattr(cli, "API_KEY", "test-key")
    monkeypatch.setattr(cli, "BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(cli, "MODEL_NAME", "test-model")
    monkeypatch.setattr(cli, "REASONING_EFFORT", "medium")
    prompt_file = None if prompt_path is None else tmp_path / prompt_path
    if prompt_file is not None and prompt_content is not None:
        prompt_file.write_text(prompt_content, encoding="utf-8")
    system_prompt = None if prompt_file is None else str(prompt_file)
    monkeypatch.setattr(cli, "SYSTEM_PROMPT", system_prompt)

    # When: the CLI builds its agent request.
    request = cli._configure_request()

    # Then: it uses the built-in prompt and warns only when reading fails.
    assert request.messages[0] == {
        "role": "system",
        "content": "You are a helpful assistant.",
    }
    warning = capsys.readouterr().out
    assert warning.startswith("warn:") is warning_expected


def test_main_requires_prompt_argument_when_option_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: command-line arguments without the required prompt option.
    monkeypatch.setattr(sys, "argv", ["aashir"])

    # When: the CLI parses its arguments.
    with pytest.raises(SystemExit) as error:
        cli.main()

    # Then: argparse reports the missing option.
    assert error.value.code == 2
    assert "-p" in capsys.readouterr().err


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


def test_main_prints_answer_before_stream_finishes_when_provider_sends_chunks(
    mock_streaming_api: Callable[[httpx2.SyncByteStream], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a provider stream that yields answer text in separate chunks.
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

    # When: the CLI consumes the stream.
    cli.main()

    # Then: the first fragment was already visible and all answer text was printed.
    output = "".join(observed_before_finish) + capsys.readouterr().out
    assert output in {"Hello", "Hello\n"}


def test_main_rejects_provider_stream_without_choices_when_chunk_is_empty(
    mock_streaming_api: Callable[[httpx2.SyncByteStream], None],
) -> None:
    # Given: a provider stream with no completion choices.
    mock_streaming_api(
        httpx2.ByteStream(_chunk(empty_choices=True) + b"data: [DONE]\n\n")
    )

    # When: the CLI consumes the invalid stream.
    with pytest.raises(ProviderError) as error:
        cli.main()

    # Then: the invalid provider response is reported as a protocol error.
    assert error.value.category == "protocol"


def test_main_preserves_partial_answer_when_provider_returns_error(
    mock_streaming_api: Callable[[httpx2.SyncByteStream], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a provider stream that emits answer text before an API error.
    body = _chunk("Partial") + (
        b'data: {"error": {"message": "stream failed", "type": "server_error"}}\n\n'
    )
    mock_streaming_api(httpx2.ByteStream(body))

    # When: the CLI consumes the failed stream.
    with pytest.raises(ProviderError, match="stream failed") as error:
        cli.main()

    # Then: the error is normalized and already streamed answer text remains visible.
    assert error.value.category == "provider_unavailable"
    assert capsys.readouterr().out == "Partial"


def test_render_events_prints_only_answer_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: answer, reasoning, tool lifecycle, and completion events.
    events = [
        AnswerDelta(text="Hel"),
        ReasoningDelta(text="raw reasoning"),
        ReasoningSummary(text="should-hide"),
        ToolCallStarted(call_id="c1", name="example_tool", arguments="{}"),
        ToolResult(call_id="c1", name="example_tool", output="file-bytes"),
        AnswerDelta(text="lo"),
        TurnCompleted(),
    ]

    # When: the CLI renders the event stream without executing tools.
    cli._render_events(events)

    # Then: answer text is printed while all other event types stay hidden.
    assert capsys.readouterr().out == "Hello"
