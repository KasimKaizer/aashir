"""Validated argument schemas and provider definitions for agent tools."""

from openai.types.chat.chat_completion_tool_union_param import (
    ChatCompletionToolUnionParam,
)
from pydantic import BaseModel, ConfigDict, Field


class ToolArguments(BaseModel):
    """Base configuration for validated model tool arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReadFileArguments(ToolArguments):
    """Arguments for reading a UTF-8 text file."""

    file_path: str = Field(
        min_length=1,
        description="Path to the file to read (relative or absolute)",
    )


class WriteFileArguments(ToolArguments):
    """Arguments for writing a UTF-8 text file."""

    file_path: str = Field(
        min_length=1,
        description="Path to the file to write (relative or absolute)",
    )
    content: str = Field(description="The content to write to the file")


class TextEdit(ToolArguments):
    """One exact text replacement within a file."""

    old_text: str = Field(
        min_length=1,
        description=(
            "Exact text to replace. It must be unique in the original file and "
            "must not overlap with any other edits[].old_text in the same call."
        ),
    )
    new_text: str = Field(description="Replacement text for this targeted edit.")


class EditFileArguments(ToolArguments):
    """Arguments for applying exact text replacements to a file."""

    file_path: str = Field(
        min_length=1,
        description="Path to the file to edit (relative or absolute)",
    )
    edits: list[TextEdit] = Field(
        min_length=1,
        description=(
            "Non-empty list of edits to apply. Each edit is matched against the "
            "original file, not incrementally. Do not include overlapping or "
            "nested edits. If two changes touch the same block or nearby lines, "
            "merge them into one edit instead."
        ),
    )


def _function_tool(
    name: str,
    description: str,
    arguments_model: type[ToolArguments],
) -> ChatCompletionToolUnionParam:
    """Build a function tool using its argument model's JSON Schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": arguments_model.model_json_schema(),
        },
    }


AGENT_TOOLS: list[ChatCompletionToolUnionParam] = [
    _function_tool(
        "read_file",
        "Read and return the contents of a file",
        ReadFileArguments,
    ),
    _function_tool(
        "write_file",
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
        WriteFileArguments,
    ),
    _function_tool(
        "edit_file",
        "Edit a single file using exact text replacement. Every edits[].old_text must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.",
        EditFileArguments,
    ),
]
