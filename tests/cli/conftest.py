"""Fixtures shared by the command-line interface tests."""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest
from openai import OpenAI

from aashir import cli
from tests.helpers import ClientArguments

_SYSTEM_PROMPT: Final = "You are a test assistant."


@pytest.fixture
def configured_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Give the CLI complete, valid settings and the prompt ``hello``."""
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(_SYSTEM_PROMPT, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["aashir", "-p", "hello"])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("SYSTEM_PROMPT_FILE", str(prompt_file))
    monkeypatch.setenv("REASONING_EFFORT", "medium")


@pytest.fixture
def serve_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[OpenAI], list[ClientArguments]]:
    """Return a function that makes ``cli.main()`` talk to the given client.

    The function returns a list that records the ``(api_key, base_url)`` pair
    ``cli.main()`` passes each time it creates a client.
    """

    def install(client: OpenAI) -> list[ClientArguments]:
        created_with: list[ClientArguments] = []

        def make_client(*, api_key: str, base_url: str) -> OpenAI:
            created_with.append(ClientArguments(api_key, base_url))
            return client

        monkeypatch.setattr(cli, "OpenAI", make_client)
        return created_with

    return install
