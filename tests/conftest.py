"""Fixtures shared by every test."""

from typing import Final

import pytest

_SETTINGS_ENVIRONMENT_VARIABLES: Final = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "SYSTEM_PROMPT_FILE",
    "REASONING_EFFORT",
    "TOOL_ARGUMENT_BYTE_CAP",
)


@pytest.fixture(autouse=True)
def isolated_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the developer's settings from the environment for every test."""
    for name in _SETTINGS_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
