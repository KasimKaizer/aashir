"""CLI entrypoint behavior: validation, streaming, and answer presentation."""

import io
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
from tests.helpers import (
    ClientSettings,
    make_openai_client,
    sse_body,
    sse_chunk,
    sse_chunk_without_choices,
)

_ALL_REQUIRED_VARIABLES = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "BASE_MODEL_NAME")


@pytest.mark.parametrize(
    ("blanked_variables", "unset", "expected_missing"),
    [
        (("OPENAI_API_KEY",), True, "OPENAI_API_KEY"),
        (("OPENAI_API_KEY",), False, "OPENAI_API_KEY"),
        (
            _ALL_REQUIRED_VARIABLES,
            False,
            "OPENAI_API_KEY, OPENAI_BASE_URL, BASE_MODEL_NAME",
        ),
    ],
    ids=("unset-key", "empty-key", "all-empty"),
)
@pytest.mark.usefixtures("configured_cli")
def test_main_reports_missing_settings_when_required_values_are_unset_or_empty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    blanked_variables: tuple[str, ...],
    unset: bool,
    expected_missing: str,
) -> None:
    # Given: valid settings except for the blanked required values.
    for name in blanked_variables:
        if unset:
            monkeypatch.delenv(name)
        else:
            monkeypatch.setenv(name, "")

    # When: the CLI validates its configuration.
    with pytest.raises(SystemExit) as error:
        cli.main()

    # Then: it reports all missing settings and exits with a usage-error status.
    assert error.value.code == 2
    assert capsys.readouterr().err == (
        f"error: Missing required variables: {expected_missing}\n"
    )


@pytest.mark.usefixtures("configured_cli")
def test_main_reports_unsupported_reasoning_effort_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: complete settings with an unsupported reasoning-effort value.
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


@pytest.mark.parametrize(("value",), [(None,), ("",)], ids=("unset", "empty"))
def test_parse_reasoning_effort_returns_none_when_unset_or_empty(
    value: str | None,
) -> None:
    # Given: a reasoning effort that is unset or empty.
    # When: the CLI parses the setting.
    result = cli._parse_reasoning_effort(value)

    # Then: no reasoning effort is requested.
    assert result is None


@pytest.mark.parametrize(
    ("prompt_file_name", "warning_expected"),
    [(None, False), ("missing.txt", True), ("empty.txt", False)],
    ids=("no-file", "missing-file", "empty-file"),
)
@pytest.mark.usefixtures("configured_cli")
def test_configure_request_uses_default_prompt_without_nonempty_custom_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    prompt_file_name: str | None,
    warning_expected: bool,
) -> None:
    # Given: valid settings with no prompt, a missing file, or an empty file.
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    if prompt_file_name is None:
        monkeypatch.delenv("SYSTEM_PROMPT_FILE")
    else:
        monkeypatch.setenv("SYSTEM_PROMPT_FILE", str(tmp_path / prompt_file_name))

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


@pytest.mark.usefixtures("configured_cli")
def test_main_closes_http_client_after_streaming(
    serve_provider: Callable[[OpenAI], list[ClientSettings]],
) -> None:
    # Given: a successful provider stream and the client that serves it.
    body = sse_body(sse_chunk({"content": "Done"}, finish_reason="stop"))
    client = make_openai_client(httpx2.ByteStream(body))
    serve_provider(client)

    # When: the CLI consumes the complete stream.
    cli.main()

    # Then: the CLI closed the client after the run.
    assert client.is_closed()


@pytest.mark.usefixtures("configured_cli")
def test_main_creates_the_provider_client_from_its_settings(
    serve_provider: Callable[[OpenAI], list[ClientSettings]],
) -> None:
    # Given: valid settings and a provider that answers immediately.
    body = sse_body(sse_chunk({"content": "Done"}, finish_reason="stop"))
    created_with = serve_provider(make_openai_client(httpx2.ByteStream(body)))

    # When: the CLI runs.
    cli.main()

    # Then: the client was created from the configured key and base URL.
    assert created_with == [("test-key", "https://example.invalid/v1")]


@pytest.mark.usefixtures("configured_cli")
def test_main_flushes_each_answer_chunk_before_the_stream_finishes(
    monkeypatch: pytest.MonkeyPatch,
    serve_provider: Callable[[OpenAI], list[ClientSettings]],
) -> None:
    # Given: a stdout that keeps text buffered until flushed (like a real pipe),
    # and a provider stream that records what reached the pipe after its first chunk.
    pipe = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(pipe, encoding="utf-8"))
    flushed_after_first_chunk: list[bytes] = []

    class StreamingBody(httpx2.SyncByteStream):
        @override
        def __iter__(self) -> Iterator[bytes]:
            yield sse_chunk({"content": "Hel"})
            yield sse_chunk_without_choices()
            flushed_after_first_chunk.append(pipe.getvalue())
            yield sse_chunk({"content": "lo"}, finish_reason="stop")
            yield sse_body()

    serve_provider(make_openai_client(StreamingBody()))

    # When: the CLI consumes the stream.
    cli.main()

    # Then: the first fragment had already been flushed, and all text arrived.
    assert flushed_after_first_chunk == [b"Hel"]
    sys.stdout.flush()
    assert pipe.getvalue() == b"Hello"


@pytest.mark.usefixtures("configured_cli")
def test_main_reports_provider_stream_without_choices_cleanly(
    serve_provider: Callable[[OpenAI], list[ClientSettings]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a provider stream with no completion choices.
    body = sse_body(sse_chunk_without_choices())
    serve_provider(make_openai_client(httpx2.ByteStream(body)))

    # When: the CLI consumes the invalid stream.
    with pytest.raises(SystemExit) as error:
        cli.main()

    # Then: it exits unsuccessfully and reports a concise protocol error.
    assert error.value.code == 1
    assert (
        capsys.readouterr().err
        == "error: protocol: provider stream contained no completion choices\n"
    )


@pytest.mark.usefixtures("configured_cli")
def test_main_preserves_partial_answer_when_provider_returns_error(
    serve_provider: Callable[[OpenAI], list[ClientSettings]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a provider stream that emits answer text before an API error.
    body = sse_chunk({"content": "Partial"}) + sse_chunk_without_choices(
        error={"message": "stream failed", "type": "server_error"}
    )
    serve_provider(make_openai_client(httpx2.ByteStream(body)))

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
