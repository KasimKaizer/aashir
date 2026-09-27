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
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
            "parameters": {
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to write (relative or absolute)",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
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
            print("Using Write Tool")
            return execute_write(arguments)
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


def execute_write(arguments) -> str:
    raw_file_path = arguments.get("file_path")
    if not isinstance(raw_file_path, str) or not raw_file_path:
        return "error: file_path must be a non-empty string"

    content = arguments.get("content")
    if not isinstance(content, str):
        return "error: content must be a string"

    try:
        file_path = Path(raw_file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        chr_count = file_path.write_text(content, encoding="utf-8")
        msg = f" written {chr_count} characters in {raw_file_path}"
        print(msg)
        return f"success: {msg}"
    except (UnicodeError, OSError) as e:
        return f"error: {e}"
