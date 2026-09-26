import json
from pathlib import Path

import pytest

from aashir.tools import execute_tool_call


def test_read_tool_returns_text_from_requested_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "note.txt").write_text("café\n", encoding="utf-8")

    result = execute_tool_call("Read", json.dumps({"file_path": "note.txt"}))

    assert result == "café\n"


def test_write_creates_file_in_missing_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    content = "café\n"

    result = execute_tool_call(
        "Write", json.dumps({"file_path": "notes/today.txt", "content": content})
    )

    assert (tmp_path / "notes" / "today.txt").read_text(encoding="utf-8") == content
    assert result.startswith("success:")


def test_write_replaces_existing_content_with_empty_string(tmp_path: Path) -> None:
    path = tmp_path / "existing.txt"
    path.write_text("old content", encoding="utf-8")

    execute_tool_call("Write", json.dumps({"file_path": str(path), "content": ""}))

    assert path.read_text(encoding="utf-8") == ""


def test_write_rejects_missing_content_without_creating_directories(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes" / "today.txt"

    result = execute_tool_call("Write", json.dumps({"file_path": str(path)}))

    assert result == "error: content must be a string"
    assert not path.parent.exists()
