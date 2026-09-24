import sys

import pytest

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
