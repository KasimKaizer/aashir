"""Tool dispatch and local file operations for the agent."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from openai.types.chat.chat_completion_tool_union_param import (
    ChatCompletionToolUnionParam,
)
from pydantic import ValidationError

from aashir.tools.schemas import (
    _EditFileArguments,
    _ReadFileArguments,
    _ToolArguments,
    _WriteFileArguments,
)

_TOOL_ARGUMENT_BYTE_CAP: Final = 1_048_576


def _generic_validation_error_message(error: ValidationError) -> str:
    """Format validation errors without echoing the invalid input."""
    issue = error.errors(include_input=False)[0]
    if issue["type"] == "extra_forbidden":
        # Pydantic includes the unrecognized input key in the error location.
        return f"error: invalid arguments: {issue['msg']}"

    location_text = ".".join(str(part) for part in issue["loc"]) or "arguments"
    return f"error: invalid arguments at {location_text}: {issue['msg']}"


def _file_validation_error_message(error: ValidationError) -> str:
    """Preserve concise validation messages for the built-in file tools."""
    issue = error.errors(include_input=False)[0]
    match issue["loc"]:
        case ("file_path",):
            return "error: file_path must be a non-empty string"
        case ("content",):
            return "error: content must be a string"
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


@dataclass(frozen=True, slots=True)
class _ToolDefinition[ArgumentsT: _ToolArguments]:
    """Describe one model-visible tool and its validated executor."""

    name: str
    description: str
    arguments_model: type[ArgumentsT]
    execute: Callable[[ArgumentsT], str]
    validation_error_formatter: Callable[[ValidationError], str] = (
        _generic_validation_error_message
    )

    def provider_tool(self) -> ChatCompletionToolUnionParam:
        """Build the provider schema from this definition's argument model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments_model.model_json_schema(),
            },
        }

    def execute_json(self, raw_arguments: str) -> str:
        """Validate raw arguments, then execute and normalize tool failures."""
        try:
            arguments = self.arguments_model.model_validate_json(raw_arguments)
        except ValidationError as error:
            return self.validation_error_formatter(error)

        # TODO: Add permission hook here; advertised tools are not authorization.
        # TODO: Enforce workspace containment in the filesystem-tool boundary.
        try:
            return self.execute(arguments)
        except Exception:
            # Keep unexpected internal exception details out of model-visible output.
            return "error: tool execution failed"


def _execute_read(arguments: _ReadFileArguments) -> str:
    """Read UTF-8 text while preserving the file's newline sequences.

    Args:
        arguments: Validated tool arguments containing ``file_path``.

    Returns:
        File contents, or error text if the path or file cannot be read.
    """
    try:
        return Path(arguments.file_path).read_text(encoding="utf-8", newline="")
    except (UnicodeError, OSError) as error:
        return f"error: {error}"


# TODO: Revisit atomic replacement for both file-writing tools after deciding how
# permissions, symlinks, hard links, and inode metadata should behave.


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


def _execute_edit(arguments: _EditFileArguments) -> str:
    """Apply exact, non-overlapping replacements to one UTF-8 file.

    Every ``old_text`` must occur exactly once in the original content. An empty
    ``new_text`` deletes a match. The file is written only after all edits pass
    validation, and existing newline sequences are preserved.

    Args:
        arguments: Validated file path and edit list.

    Returns:
        A success message or error text if validation or the file operation fails.
    """
    try:
        file_path = Path(arguments.file_path)
        content = file_path.read_text(encoding="utf-8", newline="")
    except (UnicodeError, OSError) as error:
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
        file_path.write_text(content, encoding="utf-8", newline="")
        return f"success: edited {len(arguments.edits)} text blocks in {arguments.file_path}"
    except (UnicodeError, OSError) as error:
        return f"error: {error}"


_TOOL_DEFINITIONS: Final = (
    _ToolDefinition(
        name="read_file",
        description="Read and return the contents of a file",
        arguments_model=_ReadFileArguments,
        execute=_execute_read,
        validation_error_formatter=_file_validation_error_message,
    ),
    _ToolDefinition(
        name="write_file",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, "
            "overwrites if it does. Automatically creates parent directories."
        ),
        arguments_model=_WriteFileArguments,
        execute=_execute_write,
        validation_error_formatter=_file_validation_error_message,
    ),
    _ToolDefinition(
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
        validation_error_formatter=_file_validation_error_message,
    ),
)

AGENT_TOOLS: Final[list[ChatCompletionToolUnionParam]] = [
    definition.provider_tool() for definition in _TOOL_DEFINITIONS
]
_TOOL_DEFINITIONS_BY_NAME: Final = {
    definition.name: definition for definition in _TOOL_DEFINITIONS
}


def execute_tool_call(tool_name: str, raw_arguments: str) -> str:
    """Resolve, bound, validate, and execute one built-in tool call.

    Args:
        tool_name: Name of the requested tool.
        raw_arguments: JSON-encoded tool arguments from the model.

    Returns:
        Tool output or error text suitable for the model.
    """
    definition = _TOOL_DEFINITIONS_BY_NAME.get(tool_name)
    if definition is None:
        return "error: unknown tool called"

    argument_byte_count = len(raw_arguments.encode("utf-8", errors="surrogatepass"))
    if argument_byte_count > _TOOL_ARGUMENT_BYTE_CAP:
        return f"error: tool arguments exceed the {_TOOL_ARGUMENT_BYTE_CAP}-byte limit"

    return definition.execute_json(raw_arguments)
