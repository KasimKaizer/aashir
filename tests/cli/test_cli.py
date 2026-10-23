"""CLI entrypoint behavior: validation, streaming, and answer presentation."""

import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import override

import httpx2
import pytest
from openai import OpenAI

from aashir import cli
from aashir.agent import (
    AnswerDelta,
    ReasoningDelta,
    ReasoningSummary,
    ToolCallReceived,
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
    capsys: pytest.CaptureFixture[str],
    api_key: str | None,
    base_url: str,
    model_name: str | None,
    reasoning_effort: str | None,
    expected_missing: str,
) -> None:
    # Given: incomplete required settings and a valid prompt argument.
    monkeypatch.setattr(sys, "argv", ["aashir", "-p", "hello"])
    monkeypatch.setenv("OPENAI_API_KEY", api_key or "")
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    monkeypatch.setenv("BASE_MODEL_NAME", model_name or "")
    monkeypatch.setenv("SYSTEM_PROMPT_FILE", "prompt.txt")
    monkeypatch.setenv("REASONING_EFFORT", reasoning_effort or "")

    # When: the CLI validates its configuration.
    with pytest.raises(SystemExit) as error:
        cli.main()

    # Then: it reports all missing settings and exits with a usage-error status.
    assert error.value.code == 2
    assert capsys.readouterr().err == (
        f"error: Missing required variables: {expected_missing}\n"
    )


def test_main_reports_unsupported_reasoning_effort_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: complete settings with an unsupported reasoning-effort value.
    monkeypatch.setattr(sys, "argv", ["aashir", "-p", "hello"])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("BASE_MODEL_NAME", "test-model")
    monkeypatch.setenv("SYSTEM_PROMPT_FILE", "")
    monkeypatch.setenv("REASONING_EFFORT", "unsupported")

    # When: the CLI validates its configuration.
    with pytest.raises(SystemExit) as error:
        cli.main()

    # Then: it reports the invalid setting without a traceback.
    assert error.value.code == 2
    assert capsys.readouterr().err == (
        "error: Invalid REASONING_EFFORT 'unsupported'; expected one of: "
        "none, minimal, low, medium, high, xhigh, or max.\n"
    )


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
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("BASE_MODEL_NAME", "test-model")
    monkeypatch.setenv("REASONING_EFFORT", "medium")
    prompt_file = None if prompt_path is None else tmp_path / prompt_path
    if prompt_file is not None and prompt_content is not None:
        prompt_file.write_text(prompt_content, encoding="utf-8")
    system_prompt = "" if prompt_file is None else str(prompt_file)
    monkeypatch.setenv("SYSTEM_PROMPT_FILE", system_prompt)

    # When: the CLI builds its agent request from the current settings.
    request = cli._configure_request(cli._load_settings())

    # Then: it uses the built-in prompt and warns only when reading fails.
    assert request.messages[0] == {
        "role": "system",
        "content": "You are a helpful assistant.",
    }
    captured = capsys.readouterr()
    assert captured.out == ""
    if warning_expected:
        assert captured.err.startswith(
            "warn: custom system prompt could not be loaded because of the following: "
        )
        assert captured.err.endswith(", using default prompt\n")
    else:
        assert captured.err == ""


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
    assert "arguments are required: -p" in capsys.readouterr().err


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
) -> Callable[[httpx2.SyncByteStream], httpx2.Client]:
    monkeypatch.setattr(sys, "argv", ["aashir", "-p", "hello"])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("BASE_MODEL_NAME", "test-model")
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("You are a test assistant.", encoding="utf-8")
    monkeypatch.setenv("SYSTEM_PROMPT_FILE", str(prompt_file))
    monkeypatch.setenv("REASONING_EFFORT", "medium")

    def install(stream: httpx2.SyncByteStream) -> httpx2.Client:
        def respond(_request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=stream,
            )

        http_client = httpx2.Client(transport=httpx2.MockTransport(respond))

        def make_client(*, api_key: str, base_url: str) -> OpenAI:
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=0,
                http_client=http_client,
            )

        monkeypatch.setattr(cli, "OpenAI", make_client)
        return http_client

    return install


def test_main_closes_http_client_after_streaming(
    mock_streaming_api: Callable[[httpx2.SyncByteStream], httpx2.Client],
) -> None:
    # Given: a successful provider stream and its underlying HTTP client.
    http_client = mock_streaming_api(
        httpx2.ByteStream(_chunk("Done", finish_reason="stop") + b"data: [DONE]\n\n")
    )

    # When: the CLI consumes the complete stream.
    try:
        cli.main()
    finally:
        client_was_closed = http_client.is_closed
        http_client.close()

    # Then: the CLI closes the HTTP client after the run.
    assert client_was_closed


def test_main_prints_answer_before_stream_finishes_when_provider_sends_chunks(
    mock_streaming_api: Callable[[httpx2.SyncByteStream], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a provider stream that yields answer text in separate chunks.
    observed_before_finish: list[str] = []

    class StreamingBody(httpx2.SyncByteStream):
        @override
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
    assert output == "Hello"


def test_main_reports_provider_stream_without_choices_cleanly(
    mock_streaming_api: Callable[[httpx2.SyncByteStream], None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a provider stream with no completion choices.
    mock_streaming_api(
        httpx2.ByteStream(_chunk(empty_choices=True) + b"data: [DONE]\n\n")
    )

    # When: the CLI consumes the invalid stream.
    with pytest.raises(SystemExit) as error:
        cli.main()
    captured = capsys.readouterr()

    # Then: it exits unsuccessfully and reports a concise protocol error.
    assert error.value.code == 1
    assert (
        captured.err
        == "error: protocol: provider stream contained no completion choices\n"
    )


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
    with pytest.raises(SystemExit) as error:
        cli.main()
    captured = capsys.readouterr()

    # Then: the CLI reports the provider error and preserves already-streamed text.
    assert error.value.code == 1
    assert captured.err == "error: provider_unavailable: stream failed\n"
    assert captured.out == "Partial"


def test_render_events_prints_only_answer_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: answer, reasoning, tool lifecycle, and completion events.
    events = [
        AnswerDelta(text="Hel"),
        ReasoningDelta(text="raw reasoning"),
        ReasoningSummary(text="should-hide"),
        ToolCallReceived(call_id="c1", name="example_tool", arguments="{}"),
        ToolResult(call_id="c1", name="example_tool", output="file-bytes"),
        AnswerDelta(text="lo"),
        TurnCompleted(),
    ]

    # When: the CLI renders the event stream without executing tools.
    cli._render_events(events)

    # Then: answer text is printed while all other event types stay hidden.
    assert capsys.readouterr().out == "Hello"
