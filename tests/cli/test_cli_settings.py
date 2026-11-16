"""CLI handling of settings: flags, credentials, and secrecy."""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest
from openai import OpenAI

from aashir import cli
from tests.helpers import (
    ClientArguments,
    make_round_trip_client,
    sse_body,
    sse_chunk,
)

_STOP_TURN: Final = sse_body(sse_chunk({"content": "Done"}, finish_reason="stop"))


@pytest.mark.usefixtures("configured_cli")
def test_main_does_not_create_a_provider_client_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: valid settings except for the missing API key.
    monkeypatch.delenv("OPENAI_API_KEY")

    def fail_if_called(*, api_key: str, base_url: str) -> OpenAI:
        pytest.fail("Missing credentials must fail before client construction")

    monkeypatch.setattr(cli, "OpenAI", fail_if_called)

    # When: the CLI validates its configuration.
    with pytest.raises(SystemExit) as error:
        cli.main()

    # Then: it exits with a usage error before constructing a provider client.
    assert error.value.code == 2


def test_main_runs_from_flags_alone_apart_from_the_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    serve_provider: Callable[[OpenAI], list[ClientArguments]],
) -> None:
    # Given: every setting chosen by flag and only the API key in the environment.
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Flag system prompt.", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "flag-run-key")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aashir",
            "--prompt", "hello",
            "--base-url", "https://flag.invalid/v1",
            "--model", "flag-model",
            "--system-prompt-file", str(prompt_file),
            "--reasoning-effort", "low",
        ],
    )  # fmt: skip
    client, requests = make_round_trip_client(_STOP_TURN)
    created_with = serve_provider(client)

    # When: the CLI makes one provider request.
    cli.main()

    # Then: the client and request reflect the flags.
    assert created_with == [
        ClientArguments(api_key="flag-run-key", base_url="https://flag.invalid/v1")
    ]
    payload = json.loads(requests[0].content)
    assert payload["model"] == "flag-model"
    assert payload["reasoning_effort"] == "low"
    assert payload["messages"][0] == {
        "role": "system",
        "content": "Flag system prompt.",
    }


def test_main_uses_an_explicit_dotenv_credential_without_exposing_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    serve_provider: Callable[[OpenAI], list[ClientArguments]],
) -> None:
    # Given: a selected dotenv file containing only a fixture API key.
    api_key = "fixture-secret-from-dotenv-7e2a"
    dotenv_path = tmp_path / "credentials.env"
    dotenv_path.write_text(f"OPENAI_API_KEY={api_key}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["aashir", "-p", "hello", "--env-file", str(dotenv_path)],
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    client, requests = make_round_trip_client(_STOP_TURN)
    created_with = serve_provider(client)

    # When: the CLI makes one provider request using the selected dotenv file.
    cli.main()
    captured = capsys.readouterr()

    # Then: the key reaches only the client, not the request body or output, and
    # the file's use is reported without its contents.
    assert created_with == [
        ClientArguments(api_key=api_key, base_url="https://example.invalid/v1")
    ]
    assert api_key not in requests[0].content.decode("utf-8")
    assert api_key not in captured.out
    assert api_key not in captured.err
    assert "info: read selected credential file" in captured.err


@pytest.mark.usefixtures("configured_cli")
def test_main_rejects_non_secret_values_in_the_selected_dotenv_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given: a selected dotenv file that tries to set the model.
    private_value = "model-value-that-must-not-be-echoed"
    dotenv_path = tmp_path / "credentials.env"
    dotenv_path.write_text(f"OPENAI_MODEL={private_value}\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["aashir", "-p", "hello", "--env-file", str(dotenv_path)],
    )

    def fail_if_called(*, api_key: str, base_url: str) -> OpenAI:
        pytest.fail("A rejected dotenv file must fail before client construction")

    monkeypatch.setattr(cli, "OpenAI", fail_if_called)

    # When: the CLI loads the selected dotenv file.
    with pytest.raises(SystemExit) as error:
        cli.main()

    # Then: it reports only the entry name and exits before the provider.
    captured = capsys.readouterr()
    assert error.value.code == 2
    assert "OPENAI_MODEL" in captured.err
    assert private_value not in captured.err
