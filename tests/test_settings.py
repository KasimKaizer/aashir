"""Settings model validation, secrecy, and source precedence."""

from pathlib import Path
from typing import get_args

import pytest
from openai.types.chat import ChatCompletionReasoningEffort
from pydantic import SecretStr, ValidationError

from aashir.settings import (
    REASONING_EFFORT_CHOICES,
    Settings,
    SettingsError,
    SettingsOverrides,
    load_settings,
)


@pytest.fixture
def required_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide every required setting through the process environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")


@pytest.fixture
def full_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide every setting through the process environment."""
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.invalid/v1")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    monkeypatch.setenv("SYSTEM_PROMPT_FILE", "env-prompt.txt")
    monkeypatch.setenv("REASONING_EFFORT", "high")
    monkeypatch.setenv("TOOL_ARGUMENT_BYTE_CAP", "20")


def test_settings_string_representation_redacts_the_api_key() -> None:
    # Given: settings containing a unique fixture API key.
    api_key = "fixture-secret-settings-3ca8"
    settings = Settings(
        api_key=SecretStr(api_key), base_url="https://example.invalid", model="m"
    )

    # When: the settings object is converted to a string.
    representation = str(settings)

    # Then: the API key is redacted.
    assert api_key not in representation
    assert "**********" in representation


@pytest.mark.parametrize(
    ("field_name",), [("base_url",), ("reasoning_effort",), ("tool_argument_byte_cap",)]
)
def test_settings_validation_error_does_not_echo_raw_input(field_name: str) -> None:
    # Given: an invalid value that contains a unique fixture secret.
    secret = "fixture-secret-validation-41bc"
    values = {
        "api_key": "valid-key",
        "base_url": "https://example.invalid",
        "model": "valid-model",
        field_name: secret,
    }

    # When: Pydantic validates the settings input.
    with pytest.raises(ValidationError) as error:
        Settings.model_validate(values)

    # Then: its rendered validation error does not contain the secret.
    assert secret not in str(error.value)


def test_reasoning_effort_choices_match_the_sdk_type() -> None:
    # Given: the values of the SDK's reasoning-effort type, an optional Literal.
    sdk_literal = get_args(ChatCompletionReasoningEffort)[0]

    # When: they are listed.
    sdk_values = get_args(sdk_literal)

    # Then: they are exactly the choices the CLI offers.
    assert sdk_values == REASONING_EFFORT_CHOICES


@pytest.mark.usefixtures("required_environment")
def test_load_settings_uses_defaults_for_unset_optional_settings() -> None:
    # Given: only the required settings in the environment.
    # When: settings are loaded.
    settings = load_settings(SettingsOverrides())

    # Then: optional settings take their documented defaults.
    assert settings.system_prompt_file is None
    assert settings.reasoning_effort is None
    assert settings.tool_argument_byte_cap == 1_048_576


@pytest.mark.usefixtures("full_environment")
def test_load_settings_reads_every_setting_from_the_environment() -> None:
    # Given: every setting in the environment and no flags.
    # When: settings are loaded.
    settings = load_settings(SettingsOverrides())

    # Then: each setting has the environment's value.
    assert settings == Settings(
        api_key=SecretStr("environment-key"),
        base_url="https://env.invalid/v1",
        model="env-model",
        system_prompt_file=Path("env-prompt.txt"),
        reasoning_effort="high",
        tool_argument_byte_cap=20,
    )


@pytest.mark.usefixtures("full_environment")
def test_load_settings_prefers_every_flag_to_its_environment_variable() -> None:
    # Given: a flag for every flag-settable setting, each different from the
    # environment's value.
    overrides = SettingsOverrides(
        base_url="https://flag.invalid/v1",
        model="flag-model",
        system_prompt_file=Path("flag-prompt.txt"),
        reasoning_effort="low",
        tool_argument_byte_cap=10,
    )

    # When: settings are loaded.
    settings = load_settings(overrides)

    # Then: each flag's value wins.
    assert settings == Settings(
        api_key=SecretStr("environment-key"),
        base_url="https://flag.invalid/v1",
        model="flag-model",
        system_prompt_file=Path("flag-prompt.txt"),
        reasoning_effort="low",
        tool_argument_byte_cap=10,
    )


@pytest.mark.usefixtures("required_environment")
def test_load_settings_treats_an_empty_flag_as_unset() -> None:
    # Given: an empty model flag and a model in the environment.
    overrides = SettingsOverrides(model="")

    # When: settings are loaded.
    settings = load_settings(overrides)

    # Then: the environment's model is used.
    assert settings.model == "test-model"


def test_load_settings_reads_the_api_key_from_a_selected_dotenv_file(
    tmp_path: Path,
) -> None:
    # Given: a selected dotenv key and no process key.
    dotenv_path = tmp_path / "credentials.env"
    dotenv_path.write_text("OPENAI_API_KEY=dotenv-key\n", encoding="utf-8")
    overrides = SettingsOverrides(base_url="https://example.invalid", model="m")

    # When: settings are loaded from that file.
    settings = load_settings(overrides, dotenv_path)

    # Then: the file supplies the key.
    assert settings.api_key == SecretStr("dotenv-key")


def test_load_settings_treats_an_empty_process_api_key_as_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: a selected dotenv key and an empty process key.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    dotenv_path = tmp_path / "credentials.env"
    dotenv_path.write_text("OPENAI_API_KEY=dotenv-key\n", encoding="utf-8")
    overrides = SettingsOverrides(base_url="https://example.invalid", model="m")

    # When: settings are loaded from that file.
    settings = load_settings(overrides, dotenv_path)

    # Then: the file supplies the key.
    assert settings.api_key == SecretStr("dotenv-key")


@pytest.mark.usefixtures("required_environment")
def test_load_settings_prefers_the_process_api_key_to_the_dotenv_key(
    tmp_path: Path,
) -> None:
    # Given: a selected dotenv key and a different explicit process key.
    dotenv_path = tmp_path / "credentials.env"
    dotenv_path.write_text("OPENAI_API_KEY=dotenv-key\n", encoding="utf-8")

    # When: settings are loaded.
    settings = load_settings(SettingsOverrides(), dotenv_path)

    # Then: the process environment takes precedence.
    assert settings.api_key == SecretStr("environment-key")


def test_load_settings_does_not_discover_a_dotenv_file_implicitly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: a current directory with an unselected dotenv file holding the key.
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=unselected-key\n", encoding="utf-8")
    overrides = SettingsOverrides(base_url="https://example.invalid", model="m")

    # When: settings are loaded without selecting a dotenv file.
    with pytest.raises(SettingsError) as error:
        load_settings(overrides)

    # Then: the key is not discovered.
    assert str(error.value) == "Missing required settings: OPENAI_API_KEY"


def test_load_settings_does_not_interpolate_dotenv_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Given: a dotenv key that references another environment variable.
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-read")
    dotenv_path = tmp_path / "credentials.env"
    dotenv_path.write_text("OPENAI_API_KEY=${UNRELATED_SECRET}\n", encoding="utf-8")
    overrides = SettingsOverrides(base_url="https://example.invalid", model="m")

    # When: settings are loaded from that file.
    settings = load_settings(overrides, dotenv_path)

    # Then: the reference is kept literally rather than expanded.
    assert settings.api_key == SecretStr("${UNRELATED_SECRET}")


@pytest.mark.parametrize(
    ("field_name",),
    [
        ("OPENAI_BASE_URL",),
        ("OPENAI_MODEL",),
        ("SYSTEM_PROMPT_FILE",),
        ("REASONING_EFFORT",),
        ("TOOL_ARGUMENT_BYTE_CAP",),
        ("UNRECOGNIZED_NAME",),
    ],
)
@pytest.mark.usefixtures("required_environment")
def test_load_settings_rejects_other_entries_in_the_dotenv_file_by_name(
    tmp_path: Path, field_name: str
) -> None:
    # Given: a selected dotenv file with an entry other than the API key.
    private_value = "entry-value-that-must-not-be-echoed"
    dotenv_path = tmp_path / "credentials.env"
    dotenv_path.write_text(f"{field_name}={private_value}\n", encoding="utf-8")

    # When: settings are loaded from that file.
    with pytest.raises(SettingsError) as error:
        load_settings(SettingsOverrides(), dotenv_path)

    # Then: the diagnostic names the entry but not its value.
    assert field_name in str(error.value)
    assert private_value not in str(error.value)


@pytest.mark.parametrize(
    ("file_name", "content", "expected_message"),
    [
        ("missing.env", None, "Selected dotenv file is not a readable file"),
        (
            "invalid.env",
            b"OPENAI_API_KEY=\xff\xfe\n",
            "Could not read selected dotenv file",
        ),
    ],
    ids=("missing-file", "undecodable-file"),
)
@pytest.mark.usefixtures("required_environment")
def test_load_settings_reports_an_unusable_dotenv_file(
    tmp_path: Path, file_name: str, content: bytes | None, expected_message: str
) -> None:
    # Given: a selected dotenv path that is missing or not valid UTF-8.
    dotenv_path = tmp_path / file_name
    if content is not None:
        dotenv_path.write_bytes(content)

    # When: settings are loaded from that path.
    with pytest.raises(SettingsError) as error:
        load_settings(SettingsOverrides(), dotenv_path)

    # Then: the error describes the file problem.
    assert str(error.value) == expected_message


def test_load_settings_reports_every_missing_required_setting() -> None:
    # Given: no required setting in any source.
    # When: settings are loaded.
    with pytest.raises(SettingsError) as error:
        load_settings(SettingsOverrides())

    # Then: one message lists them all by flag and environment variable.
    assert str(error.value) == (
        "Missing required settings: OPENAI_API_KEY, "
        "--base-url (or OPENAI_BASE_URL), --model (or OPENAI_MODEL)"
    )


@pytest.mark.parametrize(
    ("environment_variable", "private_value", "label", "hint"),
    [
        (
            "REASONING_EFFORT",
            "fixture-secret-effort-9c1d",
            "--reasoning-effort (or REASONING_EFFORT)",
            "'max'",
        ),
        (
            "OPENAI_BASE_URL",
            "fixture-secret-url-77aa",
            "--base-url (or OPENAI_BASE_URL)",
            "valid URL",
        ),
        (
            "TOOL_ARGUMENT_BYTE_CAP",
            "fixture-secret-cap-5b2e",
            "--tool-argument-byte-cap (or TOOL_ARGUMENT_BYTE_CAP)",
            "valid integer",
        ),
    ],
)
@pytest.mark.usefixtures("required_environment")
def test_load_settings_reports_invalid_values_without_echoing_them(
    monkeypatch: pytest.MonkeyPatch,
    environment_variable: str,
    private_value: str,
    label: str,
    hint: str,
) -> None:
    # Given: an environment value that is not valid for its setting.
    monkeypatch.setenv(environment_variable, private_value)

    # When: settings are loaded.
    with pytest.raises(SettingsError) as error:
        load_settings(SettingsOverrides())

    # Then: the message names the setting and what is acceptable, not the value.
    message = str(error.value)
    assert message.startswith(f"Invalid {label}: ")
    assert hint in message
    assert private_value not in message


@pytest.mark.parametrize(("value",), [(0,), (-5,)])
@pytest.mark.usefixtures("required_environment")
def test_load_settings_rejects_a_non_positive_tool_argument_byte_cap(
    value: int,
) -> None:
    # Given: a cap flag that would reject every tool call.
    overrides = SettingsOverrides(tool_argument_byte_cap=value)

    # When: settings are loaded.
    with pytest.raises(SettingsError) as error:
        load_settings(overrides)

    # Then: the setting is reported as invalid.
    assert str(error.value).startswith("Invalid --tool-argument-byte-cap")


@pytest.mark.usefixtures("required_environment")
def test_load_settings_rejects_a_blank_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a model name made only of whitespace.
    monkeypatch.setenv("OPENAI_MODEL", "   ")

    # When: settings are loaded.
    with pytest.raises(SettingsError) as error:
        load_settings(SettingsOverrides())

    # Then: the setting is reported as invalid instead of reaching the provider.
    assert str(error.value).startswith("Invalid --model (or OPENAI_MODEL): ")
