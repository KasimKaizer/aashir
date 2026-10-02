import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from aashir.tools import execute_tool_call


@pytest.mark.parametrize("raw_arguments", ["{", "[]", "{}"])
def test_execute_tool_call_rejects_malformed_or_non_object_arguments(
    raw_arguments: str,
) -> None:
    # Given: malformed JSON, an array, or an empty argument object.
    # When: the tool dispatcher parses the arguments.
    result = execute_tool_call("read_file", raw_arguments)

    # Then: the invalid request is returned as a tool error.
    assert result.startswith("error:")


def test_read_file_tool_returns_text_from_requested_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a UTF-8 file in the tool's working directory.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "note.txt").write_text("café\n", encoding="utf-8")

    # When: the read tool opens the file.
    result = execute_tool_call("read_file", json.dumps({"file_path": "note.txt"}))

    # Then: it returns the original text.
    assert result == "café\n"


def test_read_file_preserves_crlf_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    # Given: a file with CRLF line endings.
    path.write_bytes(b"first\r\nsecond\r\n")

    # When: the read tool opens the file.
    result = execute_tool_call("read_file", json.dumps({"file_path": str(path)}))

    # Then: the line endings are returned unchanged.
    assert result == "first\r\nsecond\r\n"


def test_write_file_creates_file_in_missing_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a new nested path and UTF-8 content.
    monkeypatch.chdir(tmp_path)
    content = "café\n"

    # When: the write tool creates the file.
    result = execute_tool_call(
        "write_file", json.dumps({"file_path": "notes/today.txt", "content": content})
    )

    # Then: parent directories are created and the content is written.
    assert (tmp_path / "notes" / "today.txt").read_text(encoding="utf-8") == content
    assert result.startswith("success:")


def test_write_file_preserves_crlf_line_endings(tmp_path: Path) -> None:
    # Given: content containing CRLF line endings.
    path = tmp_path / "note.txt"
    content = "first\r\nsecond\r\n"

    # When: the write tool writes the content.
    execute_tool_call(
        "write_file",
        json.dumps({"file_path": str(path), "content": content}),
    )

    # Then: the bytes on disk preserve those line endings.
    assert path.read_bytes() == b"first\r\nsecond\r\n"


def test_write_file_replaces_existing_content_with_empty_string(tmp_path: Path) -> None:
    # Given: an existing file and valid empty content.
    path = tmp_path / "existing.txt"
    path.write_text("old content", encoding="utf-8")

    # When: the write tool replaces the contents.
    execute_tool_call("write_file", json.dumps({"file_path": str(path), "content": ""}))

    # Then: the existing contents are cleared.
    assert path.read_text(encoding="utf-8") == ""


def test_write_file_rejects_missing_content_without_creating_directories(
    tmp_path: Path,
) -> None:
    # Given: a destination path but no content value.
    path = tmp_path / "notes" / "today.txt"

    # When: the write tool receives the invalid request.
    result = execute_tool_call("write_file", json.dumps({"file_path": str(path)}))

    # Then: it reports the validation error before creating directories.
    assert result == "error: content must be a string"
    assert not path.parent.exists()


def test_edit_file_applies_original_matches_and_allows_deletion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "note.txt"
    # Given: two unique edits, including a deletion, against the original file.
    path.write_text("title=café\nstatus=ready\nkeep this\n", encoding="utf-8")
    arguments = json.dumps(
        {
            "file_path": str(path),
            "edits": [
                {"oldText": "title=café", "newText": "status=ready"},
                {"oldText": "status=ready", "newText": ""},
            ],
        }
    )

    # When: the edit tool applies both replacements.
    result = execute_tool_call("edit_file", arguments)

    # Then: the edits are applied sequentially to the original matches.
    assert result.startswith("success:")
    assert path.read_text(encoding="utf-8") == "status=ready\n\nkeep this\n"


def test_edit_file_preserves_crlf_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    # Given: a target line between CRLF-separated lines.
    path.write_bytes(b"before\r\ntarget=old\r\nafter\r\n")
    arguments = json.dumps(
        {
            "file_path": str(path),
            "edits": [{"oldText": "target=old", "newText": "target=new"}],
        }
    )

    # When: the edit tool replaces the target line.
    result = execute_tool_call("edit_file", arguments)

    # Then: the changed file keeps its original line endings.
    assert result.startswith("success:")
    assert path.read_bytes() == b"before\r\ntarget=new\r\nafter\r\n"


def test_edit_file_rejects_missing_match_without_partially_writing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "note.txt"
    # Given: one matching edit followed by a missing match.
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

    # When: the edit tool validates the full edit list.
    result = execute_tool_call("edit_file", arguments)

    # Then: it reports the error without partially changing the file.
    assert result.startswith("error:")
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("original", "edits"),
    [
        ("repeat repeat", [{"oldText": "repeat", "newText": "once"}]),
        ("banana", [{"oldText": "ana", "newText": "X"}]),
        (
            "abcdef",
            [
                {"oldText": "abc", "newText": "abc"},
                {"oldText": "cde", "newText": "Y"},
            ],
        ),
    ],
)
def test_edit_file_rejects_ambiguous_or_overlapping_matches_without_writing(
    tmp_path: Path, original: str, edits: list[dict[str, str]]
) -> None:
    # Given: edits with repeated or overlapping match regions.
    path = tmp_path / "note.txt"
    path.write_text(original, encoding="utf-8")
    arguments = json.dumps({"file_path": str(path), "edits": edits})

    # When: the edit tool checks their match regions.
    result = execute_tool_call("edit_file", arguments)

    # Then: it rejects the ambiguous request without changing the file.
    assert result.startswith("error:")
    assert path.read_text(encoding="utf-8") == original


def test_edit_file_returns_write_error_without_changing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a valid edit for an existing file and a write failure at the I/O boundary.
    path = tmp_path / "note.txt"
    original = "before"
    path.write_text(original, encoding="utf-8")
    arguments = json.dumps(
        {
            "file_path": str(path),
            "edits": [{"oldText": "before", "newText": "after"}],
        }
    )

    # When: the edit tool attempts to write the validated replacement.
    with monkeypatch.context() as scoped:
        scoped.setattr(
            Path,
            "write_text",
            Mock(side_effect=OSError("simulated write failure")),
        )
        result = execute_tool_call("edit_file", arguments)

    # Then: the failure is returned as an error and the file remains unchanged.
    assert result == "error: simulated write failure"
    assert path.read_text(encoding="utf-8") == original


def test_edit_file_does_not_create_a_missing_file(tmp_path: Path) -> None:
    # Given: an edit request targeting a file that does not exist.
    path = tmp_path / "missing.txt"
    arguments = json.dumps(
        {
            "file_path": str(path),
            "edits": [{"oldText": "before", "newText": "after"}],
        }
    )

    # When: the edit tool tries to read the target.
    result = execute_tool_call("edit_file", arguments)

    # Then: it reports the I/O error without creating a file.
    assert result.startswith("error:")
    assert not path.exists()


@pytest.mark.parametrize(
    ("edits", "expected_error"),
    [
        ([], "error: edits must be a non-empty list"),
        ([None], "error: edit at index 0 must be a dict"),
        (
            [{"oldText": "before", "newText": None}],
            "error: edit at index 0 - 'newText' must be a string",
        ),
        (
            [{"oldText": "", "newText": "insert"}],
            "error: edit at index 0 - 'oldText' must be a non-empty string",
        ),
    ],
)
def test_edit_file_rejects_invalid_edits_without_writing(
    tmp_path: Path,
    edits: list[dict[str, str | None] | None],
    expected_error: str,
) -> None:
    # Given: an existing file and an edit list with an invalid shape or value.
    path = tmp_path / "note.txt"
    original = "before text"
    path.write_text(original, encoding="utf-8")
    arguments = json.dumps({"file_path": str(path), "edits": edits})

    # When: the edit tool validates the request.
    result = execute_tool_call("edit_file", arguments)

    # Then: it returns the validation error without changing the file.
    assert result == expected_error
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("read_file", {"file_path": ""}),
        ("write_file", {"file_path": "", "content": "ignored"}),
        (
            "edit_file",
            {"file_path": "", "edits": [{"oldText": "before", "newText": "after"}]},
        ),
    ],
)
def test_file_tools_reject_empty_paths(
    tool_name: str, arguments: dict[str, str | list[dict[str, str]]]
) -> None:
    # Given: a tool request with an empty file path.
    raw_arguments = json.dumps(arguments)

    # When: the request is dispatched to the file tool.
    result = execute_tool_call(tool_name, raw_arguments)

    # Then: the file tool rejects it before any filesystem access.
    assert result == "error: file_path must be a non-empty string"


def test_write_file_returns_error_when_parent_path_is_a_file(tmp_path: Path) -> None:
    # Given: a regular file at the path where the target's parent directory belongs.
    parent = tmp_path / "not-a-directory"
    parent.write_text("keep", encoding="utf-8")
    target = parent / "child.txt"

    # When: the write tool attempts to create the child file.
    result = execute_tool_call(
        "write_file",
        json.dumps({"file_path": str(target), "content": "new"}),
    )

    # Then: it returns the I/O error and leaves the existing file unchanged.
    assert result.startswith("error:")
    assert parent.read_text(encoding="utf-8") == "keep"
