import json
from pathlib import Path

from openai.types.chat.chat_completion_tool_union_param import (
    ChatCompletionToolUnionParam,
)

AGENT_TOOLS: list[ChatCompletionToolUnionParam] = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read and return the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
]


def execute_tool_call(tool_name: str, raw_arguments: str) -> str:
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as e:
        return f"error: {e}"

    if not isinstance(arguments, dict) or not arguments:
        return "error: invalid arguments"

    match tool_name:
        case "Read":
            print("Using Read Tool")
            return execute_read(arguments)
        case "Write":
            pass
        case "Edit":
            pass
            # return execute_edit(arguments)
        case "Bash":
            pass
            # return execute_bash(arguments)
        case _:
            return "error: unknown tool called"


def execute_read(arguments) -> str:
    raw_file_path = arguments.get("file_path")

    if not isinstance(raw_file_path, str) or not raw_file_path:
        return "error: file_path must be a non-empty string"

    try:
        content = Path(raw_file_path).read_text()
        print(f"read {raw_file_path}")
        return content
    except (UnicodeError, OSError) as e:
        return f"error: {e}"
