"""The ``edit_file`` tool."""

from pathlib import Path
from typing import Annotated

from pydantic import ConfigDict, Field, ValidationError

from aashir.tools.base import (
    _generic_validation_error_message,
    _ToolArguments,
    _ToolDefinition,
)
from aashir.tools.file_replacement import (
    _FileReplacementError,
    _observe_file,
    _replace_file,
)


class _TextEdit(_ToolArguments):
    """One exact text replacement within a file."""

    model_config = ConfigDict(title="TextEdit")

    old_text: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Exact text to replace. It must be unique in the original file and "
                "must not overlap with any other edits[].old_text in the same call."
            ),
        ),
    ]
    new_text: Annotated[
        str,
        Field(description="Replacement text for this targeted edit."),
    ]


class _EditFileArguments(_ToolArguments):
    """Arguments for applying exact text replacements to a file."""

    model_config = ConfigDict(title="EditFileArguments")

    file_path: Annotated[
        str,
        Field(
            min_length=1,
            description="Path to the file to edit (relative or absolute)",
        ),
    ]
    edits: Annotated[
        list[_TextEdit],
        Field(
            min_length=1,
            description=(
                "Non-empty list of edits to apply. Each edit is matched against the "
                "original file, not incrementally. Do not include overlapping or "
                "nested edits. If two changes touch the same block or nearby lines, "
                "merge them into one edit instead."
            ),
        ),
    ]


def _edit_validation_error_message(error: ValidationError) -> str:
    """Preserve concise validation messages for ``edit_file``."""
    issue = error.errors(include_input=False)[0]
    match issue["loc"]:
        case ("file_path",):
            return "error: file_path must be a non-empty string"
        case ("edits",):
            return "error: edits must be a non-empty list"
        case ("edits", index):
            return f"error: edit at index {index} must be a dict"
        case ("edits", index, "old_text"):
            return (
                f"error: edit at index {index} - 'old_text' must be a non-empty string"
            )
        case ("edits", index, "new_text"):
            return f"error: edit at index {index} - 'new_text' must be a string"
        case _:
            return _generic_validation_error_message(error)


def _execute_edit(arguments: _EditFileArguments) -> str:
    """Apply exact, non-overlapping replacements to one UTF-8 file.

    Every ``old_text`` must occur exactly once in the original content. An empty
    ``new_text`` deletes a match. The file is replaced only after all edits pass
    validation, existing newline sequences are preserved, and the original bytes
    stay in place if the replacement fails or is refused.

    Args:
        arguments: Validated file path and edit list.

    Returns:
        A success message or error text if validation or the file operation fails.
    """
    try:
        file_path = Path(arguments.file_path)
        # Observe before reading, so a change between the read and the rename is
        # detected.
        expected_state = _observe_file(file_path)
        content = file_path.read_text(encoding="utf-8", newline="")
    except (UnicodeError, OSError, _FileReplacementError) as error:
        return f"error: {error}"

    spans: list[tuple[int, int, str]] = []
    for index, edit in enumerate(arguments.edits):
        old_text, new_text = edit.old_text, edit.new_text
        start = content.find(old_text)
        if start == -1:
            return f"error: edit at index {index} - 'old_text' not found in the file"
        if content.find(old_text, start + 1) != -1:
            return (
                f"error: edit at index {index} - 'old_text' is found more than once "
                "in the file"
            )
        end = start + len(old_text)
        spans.append((start, end, new_text))

    # Sort edits by their start position in the original content.
    ordered_spans = sorted(spans, key=lambda span: span[0])
    pointer = 0
    parts: list[str] = []
    for start, end, new_text in ordered_spans:
        if pointer > start:
            return f"error: overlapping edit at position {pointer}"
        parts.extend((content[pointer:start], new_text))
        pointer = end
    content = "".join((*parts, content[pointer:]))
    try:
        _replace_file(file_path, content, expected_state)
    except (UnicodeError, OSError, _FileReplacementError) as error:
        return f"error: {error}"

    return (
        f"success: edited {len(arguments.edits)} text blocks in {arguments.file_path}"
    )


_EDIT_FILE_TOOL = _ToolDefinition(
    name="edit_file",
    description=(
        "Edit a single file using exact text replacement. Every "
        "edits[].old_text must match a unique, non-overlapping region of the "
        "original file. If two changes affect the same block or nearby lines, "
        "merge them into one edit instead of emitting overlapping edits. Do "
        "not include large unchanged regions just to connect distant changes."
    ),
    arguments_model=_EditFileArguments,
    execute=_execute_edit,
    validation_error_formatter=_edit_validation_error_message,
)
