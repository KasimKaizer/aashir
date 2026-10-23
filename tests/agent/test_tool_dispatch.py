import json
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletionMessageParam

from aashir.agent.contracts import ToolResult
from aashir.agent.tool_calls import _execute_tool_calls, _ToolCallAccumulator
from aashir.tools import AGENT_TOOLS, execute_tool_call
from aashir.tools.execution import _TOOL_DEFINITIONS, _ToolDefinition
from aashir.tools.schemas import _EditFileArguments, _ToolArguments


class SearchArguments(_ToolArguments):
    """Arguments for the generic formatter test."""

    content: str


@pytest.mark.parametrize(
    ("tool_name",),
    [("computer_use",), ("git",), ("unknown_tool",)],
)
def test_execute_tool_calls_rejects_unknown_tools(tool_name: str) -> None:
    # Given: a call using an unregistered tool name.
    call = _ToolCallAccumulator(
        call_id="call_unknown",
        name=tool_name,
        arguments="{}",
    )
    history: list[ChatCompletionMessageParam] = []

    # When: the tool call is dispatched without starting a provider.
    events = list(_execute_tool_calls((call,), history))

    # Then: it produces one unknown-tool result for the original call id.
    results = [event for event in events if isinstance(event, ToolResult)]
    assert results == [
        ToolResult(
            call_id="call_unknown",
            name=tool_name,
            output="error: unknown tool called",
        )
    ]


@pytest.mark.parametrize(
    ("raw_arguments", "expected_output"),
    [
        (
            '{"content": 42}',
            "error: invalid arguments at content: Input should be a valid string",
        ),
        (
            '{"content": "query", "unexpected": "value"}',
            "error: invalid arguments: Extra inputs are not permitted",
        ),
    ],
    ids=("invalid-field-value", "unrecognized-field"),
)
def test_tool_definition_uses_generic_validation_error_formatter_by_default(
    raw_arguments: str, expected_output: str
) -> None:
    # Given: a generic tool definition with no custom formatter.
    definition = _ToolDefinition(
        name="search",
        description="Search for information",
        arguments_model=SearchArguments,
        execute=lambda arguments: arguments.content,
    )

    # When: the dispatcher validates invalid arguments.
    result = definition.execute_json(raw_arguments)

    # Then: it returns the generic error for that validation issue.
    assert result == expected_output


def test_execute_tool_calls_rejects_invalid_arguments_before_writing(
    tmp_path: Path,
) -> None:
    # Given: a write call missing its required content argument.
    path = tmp_path / "missing" / "note.txt"
    call = _ToolCallAccumulator(
        call_id="call_invalid",
        name="write_file",
        arguments=json.dumps({"file_path": str(path)}),
    )
    history: list[ChatCompletionMessageParam] = []

    # When: the tool call is dispatched without starting a provider.
    events = list(_execute_tool_calls((call,), history))

    # Then: it produces one validation error and leaves the target directory absent.
    results = [event for event in events if isinstance(event, ToolResult)]
    assert results == [
        ToolResult(
            call_id="call_invalid",
            name="write_file",
            output="error: content must be a string",
        )
    ]
    assert not path.parent.exists()


def test_execute_tool_calls_rejects_arguments_over_the_utf8_byte_cap(
    tmp_path: Path,
) -> None:
    # Given: a JSON write call whose character count is under the limit but UTF-8
    # byte count exceeds the frozen 1 MiB cap.
    path = tmp_path / "oversized.txt"
    tool_argument_byte_cap = 1_048_576
    content = "é" * (tool_argument_byte_cap // 2 + 1)
    raw_arguments = json.dumps(
        {"file_path": str(path), "content": content},
        ensure_ascii=False,
    )
    assert len(raw_arguments) < tool_argument_byte_cap
    assert len(raw_arguments.encode("utf-8")) > tool_argument_byte_cap
    call = _ToolCallAccumulator(
        call_id="call_oversized",
        name="write_file",
        arguments=raw_arguments,
    )
    history: list[ChatCompletionMessageParam] = []

    # When: the tool call is dispatched without starting a provider.
    events = list(_execute_tool_calls((call,), history))

    # Then: it returns one error result and does not write the file.
    results = [event for event in events if isinstance(event, ToolResult)]
    assert len(results) == 1
    assert results[0].call_id == "call_oversized"
    assert results[0].name == "write_file"
    assert (
        results[0].output
        == f"error: tool arguments exceed the {tool_argument_byte_cap}-byte limit"
    )
    assert not path.exists()


def test_execute_tool_calls_converts_tool_exceptions_to_one_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a read tool call and an unexpected failure at the file-read boundary.
    def fail_read(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> str:
        raise RuntimeError("unexpected tool failure")

    monkeypatch.setattr(Path, "read_text", fail_read)
    call = _ToolCallAccumulator(
        call_id="call_failed",
        name="read_file",
        arguments=json.dumps({"file_path": "note.txt"}),
    )
    history: list[ChatCompletionMessageParam] = []

    # When: the tool call is dispatched without starting a provider.
    events = list(_execute_tool_calls((call,), history))

    # Then: the exception becomes one non-sensitive ToolResult for that call id.
    results = [event for event in events if isinstance(event, ToolResult)]
    assert results == [
        ToolResult(
            call_id="call_failed",
            name="read_file",
            output="error: tool execution failed",
        )
    ]
    assert history == [
        {
            "role": "tool",
            "tool_call_id": "call_failed",
            "content": "error: tool execution failed",
        }
    ]


def test_advertised_schema_and_dispatch_use_the_same_tool_definition(
    tmp_path: Path,
) -> None:
    # Given: the write_file definition and its advertised provider schema.
    definition = next(
        definition
        for definition in _TOOL_DEFINITIONS
        if definition.name == "write_file"
    )
    advertised_function = next(
        tool["function"]
        for tool in AGENT_TOOLS
        if tool["type"] == "function" and tool["function"]["name"] == definition.name
    )
    path = tmp_path / "missing" / "note.txt"
    raw_arguments = json.dumps({"file_path": str(path)})

    # When: the dispatcher validates the same definition's missing required field.
    result = execute_tool_call(definition.name, raw_arguments)

    # Then: advertised schema comes from the definition's model and validation
    # prevents the executor from creating the target directory.
    assert (
        advertised_function["parameters"]
        == definition.arguments_model.model_json_schema()
    )
    assert result == "error: content must be a string"
    assert not path.parent.exists()


@pytest.mark.parametrize(
    ("tool_name", "expected_title"),
    [
        ("read_file", "ReadFileArguments"),
        ("write_file", "WriteFileArguments"),
        ("edit_file", "EditFileArguments"),
    ],
)
def test_advertised_schema_keeps_descriptive_model_title(
    tool_name: str, expected_title: str
) -> None:
    # Given: the advertised schema for a built-in file tool.
    function_schema = next(
        tool["function"]
        for tool in AGENT_TOOLS
        if tool["type"] == "function" and tool["function"]["name"] == tool_name
    )

    # When: the provider-visible parameter schema is inspected.
    parameters_schema = function_schema["parameters"]

    # Then: its descriptive schema title is stable.
    assert parameters_schema["title"] == expected_title


def test_edit_file_model_schema_keeps_descriptive_text_edit_title() -> None:
    # Given: the internal edit-file argument model.
    parameters_schema = _EditFileArguments.model_json_schema()

    # When: the edit-item schema reference is resolved.
    text_edit_reference = parameters_schema["properties"]["edits"]["items"]["$ref"]
    definition_name = text_edit_reference.rsplit("/", maxsplit=1)[-1]
    text_edit_schema = parameters_schema["$defs"][definition_name]

    # Then: the provider sees the descriptive model title.
    assert text_edit_schema["title"] == "TextEdit"
