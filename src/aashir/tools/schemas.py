"""Validated argument schemas for agent tools."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class _ToolArguments(BaseModel):
    """Base configuration for validated model tool arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


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
