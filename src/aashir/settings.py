"""Application settings: the validated model and its loader."""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Final, override

from dotenv import dotenv_values
from openai.types.chat import ChatCompletionReasoningEffort
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    ValidationError,
)

from aashir.tools import DEFAULT_TOOL_ARGUMENT_BYTE_CAP

type SettingValue = str | int | Path

_API_KEY_VARIABLE: Final = "OPENAI_API_KEY"

# Must list the values of the OpenAI SDK's ChatCompletionReasoningEffort; a test
# fails when the SDK's list changes.
REASONING_EFFORT_CHOICES: Final = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


class Settings(BaseModel):
    """Validated settings for one run, with a redacted provider credential."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    # TODO: Validate the key before the first model request, ideally with a provider
    # call that consumes no tokens. Until then, a key with whitespace or non-ASCII
    # characters reaches the HTTP library, which reports a misleading connection
    # error or raises UnicodeEncodeError.
    api_key: SecretStr
    base_url: AnyHttpUrl
    model: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    system_prompt_file: Path | None = None
    reasoning_effort: ChatCompletionReasoningEffort = None
    tool_argument_byte_cap: int = Field(default=DEFAULT_TOOL_ARGUMENT_BYTE_CAP, gt=0)


@dataclass(frozen=True, slots=True)
class SettingsOverrides:
    """Setting values chosen on the command line; ``None`` means not chosen."""

    base_url: str | None = None
    model: str | None = None
    system_prompt_file: Path | None = None
    reasoning_effort: ChatCompletionReasoningEffort = None
    tool_argument_byte_cap: int | None = None


@dataclass(frozen=True, slots=True)
class SettingsError(Exception):
    """A setting or credential file that cannot be used to start a run."""

    message: str

    @override
    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class _SettingSource:
    """The environment variable and optional command-line flag that supply a setting."""

    environment_variable: str
    flag: str | None = None

    @property
    def label(self) -> str:
        """Name the sources the way a user would type them."""
        if self.flag is None:
            return self.environment_variable
        return f"{self.flag} (or {self.environment_variable})"


# Keys are the names of the ``Settings`` and ``SettingsOverrides`` fields. The API
# key has no flag, so it stays out of shell history and the process list.
_SETTING_SOURCES: Final = {
    "api_key": _SettingSource(_API_KEY_VARIABLE),
    "base_url": _SettingSource("OPENAI_BASE_URL", "--base-url"),
    "model": _SettingSource("OPENAI_MODEL", "--model"),
    "system_prompt_file": _SettingSource("SYSTEM_PROMPT_FILE", "--system-prompt-file"),
    "reasoning_effort": _SettingSource("REASONING_EFFORT", "--reasoning-effort"),
    "tool_argument_byte_cap": _SettingSource(
        "TOOL_ARGUMENT_BYTE_CAP", "--tool-argument-byte-cap"
    ),
}


def load_settings(
    overrides: SettingsOverrides, dotenv_path: Path | None = None
) -> Settings:
    """Resolve and validate settings from flags, the environment, and a dotenv file.

    Precedence is command-line flag, then process environment, then the selected
    dotenv file. The dotenv file may supply only the API key, and the API key has no
    flag, so it never appears in shell history or the process list. An empty value
    counts as unset.

    Args:
        overrides: Setting values chosen on the command line.
        dotenv_path: A credential-only dotenv file selected explicitly by the user.

    Returns:
        The validated settings.

    Raises:
        SettingsError: If the dotenv file is unusable, a required setting is missing,
            or a setting value is invalid. Messages name settings, never their values.
    """
    dotenv_credentials: dict[str, str | None] = {}
    if dotenv_path is not None:
        dotenv_credentials = _read_dotenv_credentials(dotenv_path)

    flag_values: dict[str, SettingValue | None] = asdict(overrides)

    values: dict[str, SettingValue] = {}
    for field_name, source in _SETTING_SOURCES.items():
        # The dotenv file can only hold the API key, so this lookup finds nothing
        # for every other setting.
        value = _first_set(
            flag_values.get(field_name),
            os.environ.get(source.environment_variable),
            dotenv_credentials.get(source.environment_variable),
        )
        if value is not None:
            values[field_name] = value

    try:
        return Settings.model_validate(values)
    except ValidationError as error:
        raise SettingsError(_describe_validation_error(error)) from error


def _first_set(*candidates: SettingValue | None) -> SettingValue | None:
    """Return the first candidate that is neither ``None`` nor an empty string."""
    for candidate in candidates:
        if candidate is not None and candidate != "":
            return candidate
    return None


def _read_dotenv_credentials(dotenv_path: Path) -> dict[str, str | None]:
    """Read the API key from a dotenv file and reject every other entry.

    Values are not interpolated, so ``${NAME}`` cannot pull another environment
    variable into the key. Rejecting unknown entries keeps the file from silently
    changing execution settings.
    """
    if not dotenv_path.is_file():
        raise SettingsError("Selected dotenv file is not a readable file")
    try:
        entries = dotenv_values(dotenv_path, encoding="utf-8", interpolate=False)
    except (OSError, UnicodeError) as error:
        raise SettingsError("Could not read selected dotenv file") from error

    unsupported_names = [name for name in entries if name != _API_KEY_VARIABLE]
    if unsupported_names:
        raise SettingsError(
            f"Selected dotenv file may only set {_API_KEY_VARIABLE}; "
            f"found: {', '.join(unsupported_names)}"
        )
    return entries


def _describe_validation_error(error: ValidationError) -> str:
    """Summarize missing and invalid settings without echoing any input value."""
    missing_labels: list[str] = []
    invalid_descriptions: list[str] = []
    for detail in error.errors(include_input=False, include_url=False):
        label = _SETTING_SOURCES[str(detail["loc"][0])].label
        if detail["type"] == "missing":
            missing_labels.append(label)
        else:
            invalid_descriptions.append(f"Invalid {label}: {detail['msg']}")

    problems: list[str] = []
    if missing_labels:
        problems.append(f"Missing required settings: {', '.join(missing_labels)}")
    problems.extend(invalid_descriptions)
    return "; ".join(problems)
