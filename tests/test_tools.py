import json
from pathlib import Path

import pytest

from aashir.tools import execute_tool_call


def test_read_file_tool_returns_text_from_requested_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "note.txt").write_text("café\n", encoding="utf-8")

    result = execute_tool_call("read_file", json.dumps({"file_path": "note.txt"}))

    assert result == "café\n"


def test_read_file_preserves_crlf_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_bytes(b"first\r\nsecond\r\n")

    result = execute_tool_call("read_file", json.dumps({"file_path": str(path)}))

    assert result == "first\r\nsecond\r\n"


def test_write_file_creates_file_in_missing_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    content = "café\n"

    result = execute_tool_call(
        "write_file", json.dumps({"file_path": "notes/today.txt", "content": content})
    )

    assert (tmp_path / "notes" / "today.txt").read_text(encoding="utf-8") == content
    assert result.startswith("success:")


def test_write_file_preserves_crlf_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    content = "first\r\nsecond\r\n"

    execute_tool_call(
        "write_file",
        json.dumps({"file_path": str(path), "content": content}),
    )

    assert path.read_bytes() == b"first\r\nsecond\r\n"


def test_write_file_replaces_existing_content_with_empty_string(tmp_path: Path) -> None:
    path = tmp_path / "existing.txt"
    path.write_text("old content", encoding="utf-8")

    execute_tool_call("write_file", json.dumps({"file_path": str(path), "content": ""}))

    assert path.read_text(encoding="utf-8") == ""


def test_write_file_rejects_missing_content_without_creating_directories(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes" / "today.txt"

    result = execute_tool_call("write_file", json.dumps({"file_path": str(path)}))

    assert result == "error: content must be a string"
    assert not path.parent.exists()


def test_edit_file_applies_multiple_exact_edits_and_allows_deletion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "note.txt"
    path.write_text("title=café\nstatus=ready\nkeep this\n", encoding="utf-8")
    arguments = json.dumps(
        {
            "file_path": str(path),
            "edits": [
                {"oldText": "title=café", "newText": "title=naïve"},
                {"oldText": "status=ready", "newText": ""},
            ],
        }
    )

    result = execute_tool_call("edit_file", arguments)

    assert result.startswith("success:")
    assert path.read_text(encoding="utf-8") == "title=naïve\n\nkeep this\n"


def test_edit_file_preserves_crlf_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_bytes(b"before\r\ntarget=old\r\nafter\r\n")
    arguments = json.dumps(
        {
            "file_path": str(path),
            "edits": [{"oldText": "target=old", "newText": "target=new"}],
        }
    )

    result = execute_tool_call("edit_file", arguments)

    assert result.startswith("success:")
    assert path.read_bytes() == b"before\r\ntarget=new\r\nafter\r\n"


def test_edit_file_matches_each_edit_against_original_content(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("first second", encoding="utf-8")
    arguments = json.dumps(
        {
            "file_path": str(path),
            "edits": [
                {"oldText": "first", "newText": "second"},
                {"oldText": "second", "newText": "third"},
            ],
        }
    )

    result = execute_tool_call("edit_file", arguments)

    assert result.startswith("success:")
    assert path.read_text(encoding="utf-8") == "second third"


def test_edit_file_rejects_missing_match_without_partially_writing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "note.txt"
    original = "alpha beta"
    path.write_text(original, encoding="utf-8")
    arguments = json.dumps(
        {
            "file_path": str(path),
            "edits": [
                {"oldText": "alpha", "newText": "A"},
                {"oldText": "missing", "newText": "M"},
            ],
        }
    )

    result = execute_tool_call("edit_file", arguments)

    assert result.startswith("error:")
    assert path.read_text(encoding="utf-8") == original


def test_edit_file_rejects_repeated_match_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    original = "repeat repeat"
    path.write_text(original, encoding="utf-8")
    arguments = json.dumps(
        {
            "file_path": str(path),
            "edits": [{"oldText": "repeat", "newText": "once"}],
        }
    )

    result = execute_tool_call("edit_file", arguments)

    assert result.startswith("error:")
    assert path.read_text(encoding="utf-8") == original


def test_edit_file_rejects_overlapping_occurrences_of_one_match(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    original = "banana"
    path.write_text(original, encoding="utf-8")
    arguments = json.dumps(
        {
            "file_path": str(path),
            "edits": [{"oldText": "ana", "newText": "X"}],
        }
    )

    result = execute_tool_call("edit_file", arguments)

    assert result.startswith("error:")
    assert path.read_text(encoding="utf-8") == original


def test_edit_file_rejects_overlapping_matches_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    original = "abcdef"
    path.write_text(original, encoding="utf-8")
    arguments = json.dumps(
        {
            "file_path": str(path),
            "edits": [
                {"oldText": "abc", "newText": "abc"},
                {"oldText": "cde", "newText": "Y"},
            ],
        }
    )

    result = execute_tool_call("edit_file", arguments)

    assert result.startswith("error:")
    assert path.read_text(encoding="utf-8") == original


def test_edit_file_does_not_create_a_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"
    arguments = json.dumps(
        {
            "file_path": str(path),
            "edits": [{"oldText": "before", "newText": "after"}],
        }
    )

    result = execute_tool_call("edit_file", arguments)

    assert result.startswith("error:")
    assert not path.exists()


def test_edit_file_rejects_empty_edit_list_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    original = "unchanged"
    path.write_text(original, encoding="utf-8")
    arguments = json.dumps({"file_path": str(path), "edits": []})

    result = execute_tool_call("edit_file", arguments)

    assert result.startswith("error:")
    assert path.read_text(encoding="utf-8") == original


def test_edit_file_rejects_empty_search_text_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    original = "abc"
    path.write_text(original, encoding="utf-8")
    arguments = json.dumps(
        {
            "file_path": str(path),
            "edits": [{"oldText": "", "newText": "insert"}],
        }
    )

    result = execute_tool_call("edit_file", arguments)

    assert result.startswith("error:")
    assert path.read_text(encoding="utf-8") == original
