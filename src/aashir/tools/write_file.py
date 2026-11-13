"""The ``write_file`` tool."""

from pathlib import Path
from typing import Annotated

from pydantic import ConfigDict, Field, ValidationError

from aashir.tools.base import (
    _generic_validation_error_message,
    _ToolArguments,
    _ToolDefinition,
)


class _WriteFileArguments(_ToolArguments):
    """Arguments for writing a UTF-8 text file."""

    model_config = ConfigDict(title="WriteFileArguments")

    file_path: Annotated[
        str,
        Field(
            min_length=1,
            description="Path to the file to write (relative or absolute)",
        ),
    ]
    content: Annotated[str, Field(description="The content to write to the file")]


# TODO: Revisit atomic replacement for both file-writing tools (this one and
# edit_file) after deciding how permissions, symlinks, hard links, and inode
# metadata should behave.


def _write_validation_error_message(error: ValidationError) -> str:
    """Preserve concise validation messages for ``write_file``."""
    issue = error.errors(include_input=False)[0]
    match issue["loc"]:
        case ("file_path",):
            return "error: file_path must be a non-empty string"
        case ("content",):
            return "error: content must be a string"
        case _:
            return _generic_validation_error_message(error)


def _execute_write(arguments: _WriteFileArguments) -> str:
    """Create parent directories and replace a UTF-8 file's contents.

    Empty content is valid, and newline sequences are preserved.

    Args:
        arguments: Validated file path and content.

    Returns:
        A success message or error text if validation or the file operation fails.
    """
    try:
        file_path = Path(arguments.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        character_count = file_path.write_text(
            arguments.content, encoding="utf-8", newline=""
        )
        message = f"written {character_count} characters in {arguments.file_path}"
        return f"success: {message}"
    except (UnicodeError, OSError) as error:
        return f"error: {error}"


_WRITE_FILE_TOOL = _ToolDefinition(
    name="write_file",
    description=(
        "Write content to a file. Creates the file if it doesn't exist, "
        "overwrites if it does. Automatically creates parent directories."
    ),
    arguments_model=_WriteFileArguments,
    execute=_execute_write,
    validation_error_formatter=_write_validation_error_message,
)
