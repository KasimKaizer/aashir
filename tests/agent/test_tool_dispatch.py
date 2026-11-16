"""Tool dispatch limits and the schemas advertised to the provider."""

import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import FunctionDefinition

from aashir.agent import AgentRequest, ToolResult, run_agent
from aashir.agent.tool_calls import _execute_tool_calls, _ToolCallAccumulator
from aashir.tools import AGENT_TOOLS, DEFAULT_TOOL_ARGUMENT_BYTE_CAP
from aashir.tools.edit_file import EditFileArguments
from tests.helpers import (
    make_agent_request,
    make_round_trip_client,
    sse_body,
    sse_chunk,
    tool_call,
)


def _advertised_function(tool_name: str) -> FunctionDefinition:
    """Return the provider-visible definition of one built-in tool."""
    return next(
        tool["function"]
        for tool in AGENT_TOOLS
        if tool["type"] == "function" and tool["function"]["name"] == tool_name
    )


def test_execute_tool_calls_rejects_unknown_tools() -> None:
    # Given: a call using an unregistered tool name.
    call = _ToolCallAccumulator(
        call_id="call_unknown",
        name="unknown_tool",
        arguments="{}",
    )
    history: list[ChatCompletionMessageParam] = []

    # When: the tool call is dispatched without starting a provider.
    events = list(_execute_tool_calls((call,), history, DEFAULT_TOOL_ARGUMENT_BYTE_CAP))

    # Then: it produces one unknown-tool result for the original call id.
    results = [event for event in events if isinstance(event, ToolResult)]
    assert results == [
        ToolResult(
            call_id="call_unknown",
            name="unknown_tool",
            output="error: unknown tool called",
        )
    ]


def test_execute_tool_calls_rejects_arguments_over_the_utf8_byte_cap(
    tmp_path: Path,
) -> None:
    # Given: a JSON write call whose character count is under the limit but UTF-8
    # byte count exceeds the default 1 MiB cap.
    path = tmp_path / "oversized.txt"
    content = "é" * (DEFAULT_TOOL_ARGUMENT_BYTE_CAP // 2 + 1)
    raw_arguments = json.dumps(
        {"file_path": str(path), "content": content},
        ensure_ascii=False,
    )
    assert len(raw_arguments) < DEFAULT_TOOL_ARGUMENT_BYTE_CAP
    assert len(raw_arguments.encode("utf-8")) > DEFAULT_TOOL_ARGUMENT_BYTE_CAP
    call = _ToolCallAccumulator(
        call_id="call_oversized",
        name="write_file",
        arguments=raw_arguments,
    )
    history: list[ChatCompletionMessageParam] = []

    # When: the tool call is dispatched with the default cap.
    events = list(_execute_tool_calls((call,), history, DEFAULT_TOOL_ARGUMENT_BYTE_CAP))

    # Then: it returns one error result and does not write the file.
    results = [event for event in events if isinstance(event, ToolResult)]
    assert len(results) == 1
    assert results[0].call_id == "call_oversized"
    assert results[0].name == "write_file"
    assert (
        results[0].output
        == f"error: tool arguments exceed the {DEFAULT_TOOL_ARGUMENT_BYTE_CAP}-byte limit"
    )
    assert not path.exists()


def test_agent_request_defaults_the_tool_argument_byte_cap_to_one_mebibyte() -> None:
    # Given: a request that does not choose a tool argument cap.
    # When: the request is built.
    request = AgentRequest(model="test-model", messages=[])

    # Then: it carries the documented default.
    assert request.tool_argument_byte_cap == 1_048_576


@pytest.mark.parametrize(("cap",), [(0,), (-1,)])
def test_agent_request_rejects_a_non_positive_tool_argument_byte_cap(cap: int) -> None:
    # Given: a cap that would reject every tool call.
    # When: the request is built.
    with pytest.raises(ValueError) as error:
        AgentRequest(model="test-model", messages=[], tool_argument_byte_cap=cap)

    # Then: the request is refused instead of silently disabling every tool.
    assert str(error.value) == f"tool_argument_byte_cap must be positive, got {cap}"


def test_run_agent_enforces_the_requests_tool_argument_byte_cap(
    tmp_path: Path,
) -> None:
    # Given: a provider turn with a valid write call and a request with a 1-byte cap.
    path = tmp_path / "too-large.txt"
    arguments = json.dumps({"file_path": str(path), "content": "text"})
    first_turn = sse_body(
        sse_chunk(
            {"tool_calls": [tool_call(0, "call_custom_cap", "write_file", arguments)]},
            finish_reason="tool_calls",
        )
    )
    client, _ = make_round_trip_client(first_turn)
    request = replace(make_agent_request(with_tools=True), tool_argument_byte_cap=1)

    # When: the public agent loop dispatches the provider's tool call.
    with client:
        events = list(run_agent(client, request))

    # Then: dispatch rejects the call before writing the file.
    results = [event for event in events if isinstance(event, ToolResult)]
    assert results == [
        ToolResult(
            call_id="call_custom_cap",
            name="write_file",
            output="error: tool arguments exceed the 1-byte limit",
        )
    ]
    assert not path.exists()


def test_execute_tool_calls_converts_tool_exceptions_to_one_error_result() -> None:
    # Given: a read call whose path makes the real tool raise an unexpected
    # ValueError (an embedded NUL byte is not a valid path).
    call = _ToolCallAccumulator(
        call_id="call_failed",
        name="read_file",
        arguments=json.dumps({"file_path": "bad\u0000name"}),
    )
    history: list[ChatCompletionMessageParam] = []

    # When: the tool call is dispatched without starting a provider.
    events = list(_execute_tool_calls((call,), history, DEFAULT_TOOL_ARGUMENT_BYTE_CAP))

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


def test_execute_tool_calls_logs_the_traceback_of_a_failed_tool(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Given: a read call whose path makes the real tool raise an unexpected
    # ValueError (an embedded NUL byte is not a valid path).
    call = _ToolCallAccumulator(
        call_id="call_failed",
        name="read_file",
        arguments=json.dumps({"file_path": "bad\u0000name"}),
    )
    caplog.set_level(logging.ERROR)

    # When: the tool call is dispatched without starting a provider.
    list(_execute_tool_calls((call,), [], DEFAULT_TOOL_ARGUMENT_BYTE_CAP))

    # Then: one error record names the tool and carries the original exception.
    [record] = caplog.records
    assert record.getMessage() == "tool read_file failed unexpectedly"
    assert record.exc_info is not None
    assert record.exc_info[0] is ValueError


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
    function_schema = _advertised_function(tool_name)

    # When: the provider-visible parameter schema is inspected.
    parameters_schema = function_schema["parameters"]

    # Then: its descriptive schema title is stable.
    assert parameters_schema["title"] == expected_title


def test_edit_file_schema_names_the_nested_edit_model_text_edit() -> None:
    # Given: the edit-file argument model.
    # When: its JSON Schema is generated.
    parameters_schema = EditFileArguments.model_json_schema()

    # Then: the provider sees TextEdit as the reference key and title, not a
    # private Python class name.
    assert parameters_schema["properties"]["edits"]["items"]["$ref"] == (
        "#/$defs/TextEdit"
    )
    assert parameters_schema["$defs"]["TextEdit"]["title"] == "TextEdit"


def test_advertised_read_file_schema_requires_a_path_and_positive_range_values() -> (
    None
):
    # Given: the provider-visible read_file schema, as the provider receives it.
    function_schema = _advertised_function("read_file")

    # When: its parameters are read back as plain JSON.
    parameters = json.loads(json.dumps(function_schema["parameters"]))

    # Then: the path is required and non-empty, and offset and limit are positive.
    assert parameters["required"] == ["file_path"]
    assert parameters["properties"]["file_path"]["minLength"] == 1
    offset = parameters["properties"]["offset"]
    assert (offset["type"], offset["minimum"], offset["default"]) == ("integer", 1, 1)
    limit_options = parameters["properties"]["limit"]["anyOf"]
    assert {"type": "integer", "minimum": 1} in limit_options


def test_advertised_edit_file_schema_requires_a_path_and_strict_non_empty_edits() -> (
    None
):
    # Given: the provider-visible edit_file schema, as the provider receives it.
    function_schema = _advertised_function("edit_file")

    # When: its parameters are read back as plain JSON.
    parameters = json.loads(json.dumps(function_schema["parameters"]))

    # Then: it needs a path and at least one edit with non-empty old_text, and
    # unknown fields are rejected at both levels.
    assert parameters["additionalProperties"] is False
    assert {"file_path", "edits"} <= set(parameters["required"])
    assert parameters["properties"]["file_path"]["minLength"] == 1
    edits = parameters["properties"]["edits"]
    assert edits["type"] == "array"
    assert edits["minItems"] == 1
    definition_name = edits["items"]["$ref"].rsplit("/", maxsplit=1)[-1]
    text_edit = parameters["$defs"][definition_name]
    assert text_edit["additionalProperties"] is False
    assert {"old_text", "new_text"} <= set(text_edit["required"])
    assert text_edit["properties"]["old_text"]["minLength"] == 1
