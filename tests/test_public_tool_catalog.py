"""Caller-owned tool catalogs."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from aashir.tools import (
    TextEdit,
    ToolCatalog,
    ToolCatalogError,
    ToolDefinition,
    UncertainToolEffect,
    edit_file,
    generic_validation_error_message,
    read_file,
    write_file,
)


class HandbookQuery(BaseModel):
    """Arguments for the fixture handbook lookup."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    topic: str


class ScoreQuery(BaseModel):
    """Arguments for the fixture score tool."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    points: int


def _handbook_tool(
    execute: Callable[[HandbookQuery], str],
    name: str = "retrieve_handbook",
) -> ToolDefinition[HandbookQuery]:
    return ToolDefinition(
        name=name,
        description="Look up a handbook section by topic.",
        arguments_model=HandbookQuery,
        execute=execute,
    )


def test_public_catalog_runs_a_caller_tool_with_its_typed_arguments() -> None:
    # Given: a handbook lookup defined outside Aashir's implementation modules.
    assert HandbookQuery.__bases__ == (BaseModel,)
    seen: list[HandbookQuery] = []

    def retrieve_handbook(arguments: HandbookQuery) -> str:
        seen.append(arguments)
        return f"{arguments.topic}: section 2"

    catalog = ToolCatalog((_handbook_tool(retrieve_handbook),))

    # When: the caller dispatches a real JSON argument object.
    result = catalog.execute(
        "retrieve_handbook",
        json.dumps({"topic": "advising"}),
    )

    # Then: the executor receives the typed model and its text is the result.
    assert result == "advising: section 2"
    assert seen == [HandbookQuery(topic="advising")]


def test_advertised_schema_comes_from_the_selected_definition() -> None:
    # Given: a catalog whose only tool is the handbook lookup.
    def retrieve_handbook(arguments: HandbookQuery) -> str:
        return arguments.topic

    catalog = ToolCatalog((_handbook_tool(retrieve_handbook),))

    # When: the caller reads the schemas the catalog would advertise.
    advertised = catalog.provider_tools()

    # Then: the schema is that definition's argument model, not a file tool.
    assert len(advertised) == 1
    tool = advertised[0]
    assert tool["type"] == "function"
    function = tool["function"]
    assert function["name"] == "retrieve_handbook"
    assert function["description"] == "Look up a handbook section by topic."
    parameters = json.loads(json.dumps(function["parameters"]))
    assert parameters["title"] == "HandbookQuery"
    assert parameters["type"] == "object"
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["topic"]
    assert parameters["properties"]["topic"]["type"] == "string"
    assert "file_path" not in parameters["properties"]


def test_catalog_does_not_run_a_tool_owned_by_another_catalog() -> None:
    # Given: two catalogs, each with a different tool.
    score_calls: list[int] = []

    def retrieve_handbook(arguments: HandbookQuery) -> str:
        return arguments.topic

    def calculate_score(arguments: ScoreQuery) -> str:
        score_calls.append(arguments.points)
        return str(arguments.points)

    handbook = ToolCatalog((_handbook_tool(retrieve_handbook),))
    scores = ToolCatalog(
        (
            ToolDefinition(
                name="calculate_score",
                description="Calculate a fixture score.",
                arguments_model=ScoreQuery,
                execute=calculate_score,
            ),
        )
    )

    # When: the handbook catalog is asked for the score tool.
    result = handbook.execute("calculate_score", json.dumps({"points": 4}))

    # Then: the other catalog's executor does not run.
    assert result == "error: unknown tool called"
    assert score_calls == []
    assert scores.execute("calculate_score", json.dumps({"points": 4})) == "4"


def test_catalog_does_not_fall_back_to_unselected_builtin_file_tools(
    tmp_path: Path,
) -> None:
    # Given: a handbook catalog and a write_file call it did not select.
    def retrieve_handbook(arguments: HandbookQuery) -> str:
        return arguments.topic

    catalog = ToolCatalog((_handbook_tool(retrieve_handbook),))
    path = tmp_path / "note.txt"
    raw_arguments = json.dumps({"file_path": str(path), "content": "secret"})

    # When: the caller asks that catalog to write a file.
    result = catalog.execute("write_file", raw_arguments)

    # Then: the built-in file tool does not run.
    assert result == "error: unknown tool called"
    assert not path.exists()


def test_empty_catalog_advertises_nothing_and_runs_nothing(tmp_path: Path) -> None:
    # Given: a catalog with no selected tools.
    catalog = ToolCatalog(())
    path = tmp_path / "note.txt"

    # When: the caller reads its schemas and requests a built-in file tool.
    advertised = catalog.provider_tools()
    result = catalog.execute(
        "read_file",
        json.dumps({"file_path": str(path)}),
    )

    # Then: nothing is advertised and the built-in tool does not run.
    assert advertised == ()
    assert result == "error: unknown tool called"
    assert not path.exists()


def test_catalog_rejects_duplicate_names_before_execution() -> None:
    # Given: two tools that advertise the same name.
    calls: list[str] = []

    def retrieve_handbook(arguments: HandbookQuery) -> str:
        calls.append(arguments.topic)
        return arguments.topic

    first = _handbook_tool(retrieve_handbook)
    second = _handbook_tool(retrieve_handbook)

    # When: the caller builds a catalog from both.
    with pytest.raises(ToolCatalogError) as error:
        ToolCatalog((first, second))

    # Then: selection fails before either callback runs.
    assert error.value.reason == "duplicate_name"
    assert error.value.tool_name == "retrieve_handbook"
    assert str(error.value) == "duplicate tool name: retrieve_handbook"
    assert calls == []


@pytest.mark.parametrize(
    ("name",),
    [("",), ("retrieve handbook",), ("retrieve.handbook",), ("x" * 65,)],
)
def test_catalog_rejects_invalid_names_before_execution(name: str) -> None:
    # Given: a tool whose name is not a provider function name.
    calls: list[str] = []

    def retrieve_handbook(arguments: HandbookQuery) -> str:
        calls.append(arguments.topic)
        return arguments.topic

    tool = _handbook_tool(retrieve_handbook, name=name)

    # When: the caller tries to select it.
    with pytest.raises(ToolCatalogError) as error:
        ToolCatalog((tool,))

    # Then: the catalog is refused before the callback runs.
    assert error.value.reason == "invalid_name"
    assert error.value.tool_name == name
    assert calls == []


class _OpenArguments(BaseModel):
    topic: str


class _CoercingArguments(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)

    topic: str


class _MutableArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    topic: str


class _ReopenedSchemaArguments(BaseModel):
    """Closed in config, but the advertised schema allows extra properties."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        json_schema_extra={"additionalProperties": True},
    )

    topic: str


@pytest.mark.parametrize(
    ("model", "detail"),
    [
        (_OpenArguments, "strict=True"),
        (_CoercingArguments, "extra='forbid'"),
        (_MutableArguments, "frozen=True"),
        (_ReopenedSchemaArguments, "closed object"),
    ],
)
def test_catalog_rejects_argument_models_that_are_not_strict_and_closed(
    model: type[BaseModel],
    detail: str,
) -> None:
    # Given: an argument model that does not use the file tools' validation rules.
    def retrieve(arguments: BaseModel) -> str:
        return str(arguments)

    tool = ToolDefinition(
        name="retrieve_handbook",
        description="Look up a handbook section by topic.",
        arguments_model=model,
        execute=retrieve,
    )

    # When: the caller tries to select it.
    with pytest.raises(ToolCatalogError) as error:
        ToolCatalog((tool,))

    # Then: the schema is refused before execution.
    assert error.value.reason == "unsupported_schema"
    assert error.value.tool_name == "retrieve_handbook"
    assert detail in str(error.value)


@pytest.mark.parametrize(
    ("raw_arguments", "expected"),
    [
        (
            "{",
            "error: invalid arguments at arguments: Invalid JSON: "
            "EOF while parsing an object at line 1 column 1",
        ),
        ("[]", "error: invalid arguments at arguments: Input should be an object"),
        (
            '{"topic": 1}',
            "error: invalid arguments at topic: Input should be a valid string",
        ),
    ],
)
def test_invalid_arguments_do_not_run_the_selected_tool(
    raw_arguments: str,
    expected: str,
) -> None:
    # Given: a handbook tool and arguments that fail validation.
    calls: list[HandbookQuery] = []

    def retrieve_handbook(arguments: HandbookQuery) -> str:
        calls.append(arguments)
        return arguments.topic

    catalog = ToolCatalog((_handbook_tool(retrieve_handbook),))

    # When: the caller dispatches those arguments.
    result = catalog.execute("retrieve_handbook", raw_arguments)

    # Then: the tool reports the validation failure and does not run.
    assert result == expected
    assert calls == []


def test_extra_fields_are_rejected_without_echoing_their_values() -> None:
    # Given: a valid topic plus an extra field carrying a secret.
    calls: list[HandbookQuery] = []

    def retrieve_handbook(arguments: HandbookQuery) -> str:
        calls.append(arguments)
        return arguments.topic

    catalog = ToolCatalog((_handbook_tool(retrieve_handbook),))
    raw_arguments = json.dumps({"topic": "advising", "token": "session-token-secret"})

    # When: the caller dispatches the object.
    result = catalog.execute("retrieve_handbook", raw_arguments)

    # Then: the extra field is rejected and its value stays out of the result.
    assert result == "error: invalid arguments: Extra inputs are not permitted"
    assert "session-token-secret" not in result
    assert calls == []


def test_caller_uses_the_public_validation_formatter() -> None:
    # Given: a caller tool that selects the shared validation formatter.
    def retrieve_handbook(arguments: HandbookQuery) -> str:
        return arguments.topic

    catalog = ToolCatalog(
        (
            ToolDefinition(
                name="retrieve_handbook",
                description="Look up a handbook section by topic.",
                arguments_model=HandbookQuery,
                execute=retrieve_handbook,
                validation_error_formatter=generic_validation_error_message,
            ),
        )
    )
    raw_arguments = json.dumps({"topic": "advising", "token": "session-token-secret"})

    # When: the caller dispatches an object with an extra field.
    result = catalog.execute("retrieve_handbook", raw_arguments)

    # Then: the shared formatter rejects the field and does not echo the secret.
    assert result == "error: invalid arguments: Extra inputs are not permitted"
    assert "session-token-secret" not in result


def test_oversized_arguments_do_not_run_the_selected_tool() -> None:
    # Given: a handbook request larger than the caller's byte cap.
    calls: list[HandbookQuery] = []

    def retrieve_handbook(arguments: HandbookQuery) -> str:
        calls.append(arguments)
        return arguments.topic

    catalog = ToolCatalog((_handbook_tool(retrieve_handbook),))
    raw_arguments = json.dumps({"topic": "advising"})
    assert len(raw_arguments.encode("utf-8")) > 8

    # When: the caller dispatches it under that cap.
    result = catalog.execute("retrieve_handbook", raw_arguments, 8)

    # Then: the tool does not run and the cap is named in the result.
    assert result == "error: tool arguments exceed the 8-byte limit"
    assert calls == []


def test_executor_exception_after_a_mutation_is_uncertain_and_is_not_retried() -> None:
    # Given: a callback that records an effect, then raises with a secret.
    secret = "session-token-secret"
    counter = 0

    def retrieve_handbook(_arguments: HandbookQuery) -> str:
        nonlocal counter
        counter += 1
        raise RuntimeError(secret)

    catalog = ToolCatalog((_handbook_tool(retrieve_handbook),))

    # When: the caller dispatches arguments that pass validation.
    with pytest.raises(UncertainToolEffect) as error:
        catalog.execute("retrieve_handbook", json.dumps({"topic": "advising"}))

    # Then: the effect is uncertain, the secret stays private, and it ran once.
    assert error.value.tool_name == "retrieve_handbook"
    assert str(error.value) == (
        "tool retrieve_handbook ended with an uncertain effect"
    )
    assert secret not in str(error.value)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == secret
    assert counter == 1


def test_caller_selects_the_public_file_tools() -> None:
    # Given: the public file tools, selected by the caller.
    catalog = ToolCatalog((read_file, write_file, edit_file))

    # When: the caller reads the provider schemas.
    advertised = catalog.provider_tools()

    # Then: the schemas are those tools, in the caller's order.
    names: list[str] = []
    for tool in advertised:
        assert tool["type"] == "function"
        names.append(tool["function"]["name"])
    assert names == ["read_file", "write_file", "edit_file"]


def test_public_file_tools_use_the_same_definition_type_as_caller_tools() -> None:
    # Given: the public file tools and one caller-defined tool.
    def retrieve_handbook(arguments: HandbookQuery) -> str:
        return arguments.topic

    caller_tool = _handbook_tool(retrieve_handbook)
    catalog = ToolCatalog((caller_tool, read_file, write_file, edit_file))

    # When: the combined catalog is dispatched like any other catalog.
    result = catalog.execute("retrieve_handbook", json.dumps({"topic": "advising"}))

    # Then: built-in and caller tools are the same definition type in one catalog.
    assert result == "advising"
    assert all(
        isinstance(definition, ToolDefinition) for definition in catalog.definitions
    )
    assert [definition.name for definition in catalog.definitions] == [
        "retrieve_handbook",
        "read_file",
        "write_file",
        "edit_file",
    ]
    argument_models = [
        definition.arguments_model for definition in (read_file, write_file, edit_file)
    ]
    for model in (*argument_models, TextEdit):
        assert model.__bases__ == (BaseModel,)
