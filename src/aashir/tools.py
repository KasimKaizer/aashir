import json
from pathlib import Path

from openai.types.chat.chat_completion_tool_union_param import (
    ChatCompletionToolUnionParam,
)

AGENT_TOOLS: list[ChatCompletionToolUnionParam] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
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
            "name": "write_file",
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
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to edit (relative or absolute)",
                    },
                    "edits": {
                        "type": "array",
                        "description": "Non-empty list of edits to apply. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {
                                    "type": "string",
                                    "description": "Exact text to replace. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call.",
                                },
                                "newText": {
                                    "type": "string",
                                    "description": "Replacement text for this targeted edit.",
                                },
                            },
                            "required": ["oldText", "newText"],
                        },
                    },
                },
                "required": ["file_path", "edits"],
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
        case "read_file":
            print("Using read_file Tool")
            return execute_read(arguments)
        case "write_file":
            print("Using write_file Tool")
            return execute_write(arguments)
        case "edit_file":
            print("Using edit_file Tool")
            return execute_edit(arguments)
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
        content = Path(raw_file_path).read_text(encoding="utf-8", newline="")
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
        chr_count = file_path.write_text(content, encoding="utf-8", newline="")
        msg = f" written {chr_count} characters in {raw_file_path}"
        print(msg)
        return f"success: {msg}"
    except (UnicodeError, OSError) as e:
        return f"error: {e}"


def execute_edit(arguments) -> str:
    raw_file_path = arguments.get("file_path")
    if not isinstance(raw_file_path, str) or not raw_file_path:
        return "error: file_path must be a non-empty string"

    edits = arguments.get("edits")
    if not isinstance(edits, list) or not edits:
        return "error: edits must be a non-empty list"

    try:
        file_path = Path(raw_file_path)
        content = file_path.read_text(encoding="utf-8", newline="")
    except (UnicodeError, OSError) as e:
        return f"error: {e}"

    spans: list[tuple[int, int, str]] = []
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return f"error: edit at index {i} must be a dict"

        old_text, new_text = edit.get("oldText"), edit.get("newText")
        if not isinstance(old_text, str) or not old_text:
            return f"error: edit at index {i} - oldText must be a non-empty string"
        if not isinstance(new_text, str):
            return f"error: edit at index {i} - newText must be a string"

        start = content.find(old_text)
        if start == -1:
            return f"error: edit at index {i} - 'oldText' not found in the file"
        if content.find(old_text, start + 1) != -1:
            return f"error: edit at index {i} - 'oldText' is found more than once in the file"
        end = start + len(old_text)
        spans.append((start, end, new_text))

    ordered_spans = sorted(spans, key=lambda s: s[0])
    pointer: int = 0
    parts: list[str] = []
    for start, end, new_text in ordered_spans:
        if pointer > start:
            return f"error: overlapping edit at position {pointer}"
        parts.extend((content[pointer:start], new_text))
        pointer = end
    parts.append(content[pointer:])
    content = "".join(parts)

    try:
        file_path.write_text(content, encoding="utf-8", newline="")
        msg = f" edited {len(edits)} text blocks in {raw_file_path}"
        print(msg)
        return f"success: {msg}"
    except (UnicodeError, OSError) as e:
        return f"error: {e}"
