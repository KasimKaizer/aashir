"""The ``read_file`` tool."""

import itertools
import stat
import sys
from pathlib import Path
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aashir.tools.base import (
    ToolDefinition,
    generic_validation_error_message,
)


# Public on purpose: the class name is also the schema title the provider sees.
class ReadFileArguments(BaseModel):
    """Arguments for reading a UTF-8 text file."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        title="ReadFileArguments",
    )

    file_path: Annotated[
        str,
        Field(
            min_length=1,
            description="Path to the file to read (relative or absolute)",
        ),
    ]
    offset: Annotated[
        int,
        Field(
            ge=1,
            description="First line to return, counting from 1. Defaults to line 1.",
        ),
    ] = 1
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
            return generic_validation_error_message(error)


def _execute_read(arguments: ReadFileArguments) -> str:
    """Read a span of a UTF-8 file as ``N:text`` lines.

    Only the lines up to the end of the requested span are read. Line endings
    (``\\n``, ``\\r\\n`` or ``\\r``) are not part of the returned text.

    Args:
        arguments: Validated file path with optional 1-based ``offset`` and ``limit``.

    Returns:
        The selected lines joined by newlines, empty text when the span has no
        lines, or error text if the path is not a regular file or cannot be read.
    """
    start_index = arguments.offset - 1
    stop_index = None if arguments.limit is None else start_index + arguments.limit
    file_path = Path(arguments.file_path)
    try:
        # Opening a FIFO blocks until a writer appears, and a device such as
        # /dev/zero has no end, so both are refused before opening. A symbolic
        # link to a regular file is still read.
        if not stat.S_ISREG(file_path.stat().st_mode):
            return f"error: {file_path} is not a regular file"
        with file_path.open(encoding="utf-8") as file:
            # islice rejects indexes above sys.maxsize, a line count no file reaches.
            selected_lines = itertools.islice(
                file,
                min(start_index, sys.maxsize),
                None if stop_index is None else min(stop_index, sys.maxsize),
            )
            numbered_lines = [
                f"{line_number}:{line.removesuffix('\n')}"
                for line_number, line in enumerate(
                    selected_lines, start=arguments.offset
                )
            ]
    except (UnicodeError, OSError) as error:
        return f"error: {error}"
    return "\n".join(numbered_lines)


read_file: Final[ToolDefinition[ReadFileArguments]] = ToolDefinition(
    name="read_file",
    description=(
        "Read a file. Every returned line is formatted as N:text, "
        "where N is its 1-based line number. Use offset and limit to read a "
        "range of lines instead of the whole file."
    ),
    arguments_model=ReadFileArguments,
    execute=_execute_read,
    validation_error_formatter=_read_validation_error_message,
)
