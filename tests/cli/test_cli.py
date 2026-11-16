"""CLI entrypoint behavior: validation, streaming, and answer presentation."""

import io
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final, override

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
from aashir.settings import _SETTING_SOURCES, SettingsOverrides, load_settings
from tests.helpers import (
    ClientArguments,
    make_openai_client,
    sse_body,
    sse_chunk,
    sse_chunk_without_choices,
)

_ALL_REQUIRED_VARIABLES = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL")
_DEFAULT_SYSTEM_MESSAGE: Final = {
    "role": "system",
    "content": "You are a helpful assistant.",
}


@pytest.mark.parametrize(
    ("blanked_variables", "expected_missing"),
    [
        (("OPENAI_API_KEY",), "OPENAI_API_KEY"),
        (
            _ALL_REQUIRED_VARIABLES,
            "OPENAI_API_KEY, --base-url (or OPENAI_BASE_URL), --model (or OPENAI_MODEL)",
        ),
    ],
    ids=("empty-key", "all-empty"),
)
@pytest.mark.usefixtures("configured_cli")
def test_main_reports_missing_settings_when_required_values_are_empty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    blanked_variables: tuple[str, ...],
    expected_missing: str,
) -> None:
    # Given: valid settings except for the blanked required values.
    for name in blanked_variables:
        monkeypatch.setenv(name, "")

    # When: the CLI validates its configuration.
    with pytest.raises(SystemExit) as error:
        cli.main()

    # Then: it reports all missing settings and exits with a usage-error status.
    assert error.value.code == 2
    assert capsys.readouterr().err == (
        f"error: Missing required settings: {expected_missing}\n"
    )


@pytest.mark.usefixtures("configured_cli")
def test_main_reports_unsupported_reasoning_effort_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: complete settings with a unique invalid environment value.
    private_value = "fixture-secret-invalid-setting-2a7d"
    monkeypatch.setenv("REASONING_EFFORT", private_value)

    # When: the CLI validates its configuration.
    with pytest.raises(SystemExit) as error:
        cli.main()

    # Then: it reports the invalid field without echoing its input.
    assert error.value.code == 2
    diagnostic = capsys.readouterr().err
    assert diagnostic.startswith(
        "error: Invalid --reasoning-effort (or REASONING_EFFORT): "
    )
    assert private_value not in diagnostic


@pytest.mark.usefixtures("configured_cli")
def test_configure_request_uses_the_default_prompt_without_a_prompt_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: valid settings with no system prompt file.
    monkeypatch.delenv("SYSTEM_PROMPT_FILE")

    # When: the CLI builds its agent request from the current settings.
    request = cli._configure_request(load_settings(SettingsOverrides()), "hello")

    # Then: it uses the built-in prompt without a warning.
    assert request.messages[0] == _DEFAULT_SYSTEM_MESSAGE
    assert capsys.readouterr().err == ""


@pytest.mark.usefixtures("configured_cli")
def test_configure_request_warns_and_uses_the_default_prompt_for_a_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: valid settings naming a system prompt file that does not exist.
    monkeypatch.setenv("SYSTEM_PROMPT_FILE", str(tmp_path / "missing.txt"))

    # When: the CLI builds its agent request from the current settings.
    request = cli._configure_request(load_settings(SettingsOverrides()), "hello")

    # Then: it falls back to the built-in prompt and says why.
    assert request.messages[0] == _DEFAULT_SYSTEM_MESSAGE
    warning = capsys.readouterr().err
    assert warning.startswith(
        "warn: custom system prompt could not be loaded because of the following: "
    )
    assert warning.endswith(", using default prompt\n")


@pytest.mark.usefixtures("configured_cli")
def test_configure_request_warns_and_uses_the_default_prompt_for_an_empty_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: valid settings naming an empty system prompt file.
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("SYSTEM_PROMPT_FILE", str(empty_file))

    # When: the CLI builds its agent request from the current settings.
    request = cli._configure_request(load_settings(SettingsOverrides()), "hello")

    # Then: it falls back to the built-in prompt and says why.
    assert request.messages[0] == _DEFAULT_SYSTEM_MESSAGE
    assert capsys.readouterr().err == (
        "warn: custom system prompt file is empty, using default prompt\n"
    )


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
    assert "arguments are required: -p/--prompt" in capsys.readouterr().err


def test_parse_arguments_accepts_the_short_prompt_and_env_file_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: the short forms of the prompt and env-file options.
    dotenv_path = tmp_path / "credentials.env"
    monkeypatch.setattr(sys, "argv", ["aashir", "-p", "hello", "-e", str(dotenv_path)])

    # When: the CLI parses its arguments.
    arguments = cli._parse_arguments()

    # Then: both values reach their typed argument fields.
    assert arguments.prompt == "hello"
    assert arguments.dotenv_path == dotenv_path


def test_parse_arguments_maps_each_long_flag_to_its_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: every long flag with a distinct value.
    dotenv_path = tmp_path / "credentials.env"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aashir",
            "--prompt", "hello",
            "--env-file", str(dotenv_path),
            "--base-url", "https://flag.invalid/v1",
            "--model", "flag-model",
            "--system-prompt-file", "flag-prompt.txt",
            "--reasoning-effort", "low",
            "--tool-argument-byte-cap", "123",
        ],
    )  # fmt: skip

    # When: the CLI parses its arguments.
    arguments = cli._parse_arguments()

    # Then: each value lands in the field of the setting it names.
    assert arguments.prompt == "hello"
    assert arguments.dotenv_path == dotenv_path
    assert arguments.settings_overrides == SettingsOverrides(
        base_url="https://flag.invalid/v1",
        model="flag-model",
        system_prompt_file=Path("flag-prompt.txt"),
        reasoning_effort="low",
        tool_argument_byte_cap=123,
    )


def test_help_names_every_flag_and_environment_variable_used_in_error_messages(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Given: a wide terminal, so help text is not wrapped between words.
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr(sys, "argv", ["aashir", "--help"])

    # When: the CLI prints its help.
    with pytest.raises(SystemExit):
        cli._parse_arguments()

    # Then: each flag and variable that settings errors mention is documented.
    help_text = capsys.readouterr().out
    documented_settings = [
        (source.flag, source.environment_variable)
        for source in _SETTING_SOURCES.values()
        if source.flag is not None
    ]
    for flag, environment_variable in documented_settings:
        assert flag in help_text
        assert f"({environment_variable})" in help_text


def test_parse_arguments_treats_an_empty_system_prompt_file_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an empty system prompt file flag, which must not become the path ".".
    monkeypatch.setattr(
        sys, "argv", ["aashir", "-p", "hello", "--system-prompt-file", ""]
    )

    # When: the CLI parses its arguments.
    arguments = cli._parse_arguments()

    # Then: no prompt file was chosen, so the environment variable can apply.
    assert arguments.settings_overrides.system_prompt_file is None


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--reasoning-effort", "unsupported"), ("--tool-argument-byte-cap", "many")],
)
def test_parse_arguments_rejects_flag_values_of_the_wrong_type(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
) -> None:
    # Given: a setting flag whose value is not an accepted choice or integer.
    monkeypatch.setattr(sys, "argv", ["aashir", "-p", "hello", flag, value])

    # When: the CLI parses its arguments.
    with pytest.raises(SystemExit) as error:
        cli._parse_arguments()

    # Then: argparse rejects the value with a usage error.
    assert error.value.code == 2
    assert capsys.readouterr().err.startswith("usage:")


@pytest.mark.usefixtures("configured_cli")
def test_main_closes_http_client_after_streaming(
    serve_provider: Callable[[OpenAI], list[ClientArguments]],
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
def test_main_flushes_each_answer_chunk_before_the_stream_finishes(
    monkeypatch: pytest.MonkeyPatch,
    serve_provider: Callable[[OpenAI], list[ClientArguments]],
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
    serve_provider: Callable[[OpenAI], list[ClientArguments]],
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
    serve_provider: Callable[[OpenAI], list[ClientArguments]],
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
        TurnCompleted(produced_answer=True),
    ]

    # When: the CLI renders the event stream without executing tools.
    cli._render_events(events)

    # Then: answer text is printed while all other event types stay hidden.
    captured = capsys.readouterr()
    assert captured.out == "Hello"
    assert captured.err == ""


def test_render_events_warns_when_the_turn_produced_no_answer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a turn that only reasoned before completing.
    events = [ReasoningDelta(text="thinking"), TurnCompleted(produced_answer=False)]

    # When: the CLI renders the event stream.
    cli._render_events(events)

    # Then: stdout stays empty and stderr says why.
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "warn: the model returned no answer\n"


@pytest.mark.usefixtures("configured_cli")
def test_main_warns_and_exits_normally_when_the_model_returns_no_answer(
    serve_provider: Callable[[OpenAI], list[ClientArguments]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a provider that reasons and stops without any answer text.
    body = sse_body(
        sse_chunk({"reasoning": "thinking"}),
        sse_chunk({}, finish_reason="stop"),
    )
    serve_provider(make_openai_client(httpx2.ByteStream(body)))

    # When: the CLI runs the turn.
    cli.main()

    # Then: it returns normally, with a warning instead of silence.
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "warn: the model returned no answer\n"
