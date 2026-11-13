"""The ``read_file`` tool."""

import itertools
import sys
from pathlib import Path
from typing import Annotated

from pydantic import ConfigDict, Field, ValidationError

from aashir.tools.base import (
    _generic_validation_error_message,
    _ToolArguments,
    _ToolDefinition,
)


class _ReadFileArguments(_ToolArguments):
    """Arguments for reading a UTF-8 text file."""

    model_config = ConfigDict(title="ReadFileArguments")

    file_path: Annotated[
        str,
        Field(
            min_length=1,
            description="Path to the file to read (relative or absolute)",
        ),
    ]
    offset: Annotated[
        int | None,
        Field(
            ge=1,
            description="First line to return, counting from 1. Defaults to line 1.",
        ),
    ] = None
    limit: Annotated[
        int | None,
        Field(
            ge=1,
            description="Maximum number of lines to return. Defaults to all lines.",
        ),
    ] = None


def _read_validation_error_message(error: ValidationError) -> str:
    """Preserve concise validation messages for ``read_file``."""
    issue = error.errors(include_input=False)[0]
    match issue["loc"]:
        case ("file_path",):
            return "error: file_path must be a non-empty string"
        case ("offset",) | ("limit",):
            return f"error: {issue['loc'][0]} must be a positive integer"
        case _:
            return _generic_validation_error_message(error)


def _execute_read(arguments: _ReadFileArguments) -> str:
    """Read a span of a UTF-8 file as ``N:text`` lines.

    Only the lines up to the end of the requested span are read. Line endings
    (``\\n``, ``\\r\\n`` or ``\\r``) are not part of the returned text.

    Args:
        arguments: Validated file path with optional 1-based ``offset`` and ``limit``.

    Returns:
        The selected lines joined by newlines, empty text when the span has no
        lines, or error text if the path or file cannot be read.
    """
    first_line_number = 1 if arguments.offset is None else arguments.offset
    start_index = first_line_number - 1
    stop_index = None if arguments.limit is None else start_index + arguments.limit
    try:
        with Path(arguments.file_path).open(encoding="utf-8") as file:
            # islice rejects indexes above sys.maxsize, a line count no file reaches.
            selected_lines = itertools.islice(
                file,
                min(start_index, sys.maxsize),
                None if stop_index is None else min(stop_index, sys.maxsize),
            )
            numbered_lines = [
                f"{line_number}:{line.removesuffix('\n')}"
                for line_number, line in enumerate(
                    selected_lines, start=first_line_number
                )
            ]
    except (UnicodeError, OSError) as error:
        return f"error: {error}"
    return "\n".join(numbered_lines)


_READ_FILE_TOOL = _ToolDefinition(
    name="read_file",
    description=(
        "Read a file. Every returned line is formatted as N:text, "
        "where N is its 1-based line number. Use offset and limit to read a "
        "range of lines instead of the whole file."
    ),
    arguments_model=_ReadFileArguments,
    execute=_execute_read,
    validation_error_formatter=_read_validation_error_message,
)
