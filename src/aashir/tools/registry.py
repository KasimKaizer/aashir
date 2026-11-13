"""Tool registry and dispatch for the agent."""

from typing import Final

from openai.types.chat.chat_completion_tool_union_param import (
    ChatCompletionToolUnionParam,
)

from aashir.tools.edit_file import _EDIT_FILE_TOOL
from aashir.tools.read_file import _READ_FILE_TOOL
from aashir.tools.write_file import _WRITE_FILE_TOOL

_TOOL_ARGUMENT_BYTE_CAP: Final = 1_048_576

_TOOL_DEFINITIONS: Final = (_READ_FILE_TOOL, _WRITE_FILE_TOOL, _EDIT_FILE_TOOL)

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
