"""Caller-selected tool sets and the one dispatcher that runs them."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Protocol

from openai.types.chat.chat_completion_tool_union_param import (
    ChatCompletionToolUnionParam,
)
from pydantic import BaseModel

from aashir.tools.errors import ToolCatalogError

DEFAULT_TOOL_ARGUMENT_BYTE_CAP: Final = 1_048_576

# Provider function names are a single token of at most 64 characters.
_TOOL_NAME_PATTERN: Final = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class RunnableTool(Protocol):
    """The catalog view of a ``ToolDefinition``.

    The protocol exists so one catalog can hold tools with different argument
    models. Every value is still a ``ToolDefinition``. Attributes are
    read-only, matching a frozen definition.
    """

    @property
    def name(self) -> str: ...

    @property
    def arguments_model(self) -> type[BaseModel]: ...

    def provider_tool(self) -> ChatCompletionToolUnionParam:
        """Build the provider schema for this tool."""
        ...

    def execute_json(self, raw_arguments: str) -> str:
        """Validate raw arguments and run this tool's executor."""
        ...


def _schema_problem(model: type[BaseModel]) -> str | None:
    """Return why an argument model cannot be selected, or None when it can."""
    config = model.model_config
    if config.get("strict") is not True:
        return "argument model must set strict=True"
    if config.get("extra") != "forbid":
        return "argument model must set extra='forbid'"
    if config.get("frozen") is not True:
        return "argument model must set frozen=True"
    schema = model.model_json_schema()
    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
    ):
        return "argument schema must be a closed object"
    return None


def _reject_unusable_definition(definition: RunnableTool) -> None:
    """Refuse a definition whose name or argument schema cannot be selected."""
    if _TOOL_NAME_PATTERN.fullmatch(definition.name) is None:
        raise ToolCatalogError(
            reason="invalid_name",
            tool_name=definition.name,
            detail=f"invalid tool name: {definition.name!r}",
        )
    problem = _schema_problem(definition.arguments_model)
    if problem is not None:
        raise ToolCatalogError(
            reason="unsupported_schema",
            tool_name=definition.name,
            detail=f"unsupported tool schema for {definition.name}: {problem}",
        )


@dataclass(frozen=True, slots=True)
class ToolCatalog:
    """The immutable tools selected for one caller.

    An empty catalog advertises and runs no tools. It is not filled from the
    built-in file tools.
    """

    definitions: tuple[RunnableTool, ...] = ()
    _by_name: Mapping[str, RunnableTool] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_name: dict[str, RunnableTool] = {}
        for definition in self.definitions:
            _reject_unusable_definition(definition)
            if definition.name in by_name:
                raise ToolCatalogError(
                    reason="duplicate_name",
                    tool_name=definition.name,
                    detail=f"duplicate tool name: {definition.name}",
                )
            by_name[definition.name] = definition
        object.__setattr__(self, "_by_name", MappingProxyType(by_name))

    def provider_tools(self) -> tuple[ChatCompletionToolUnionParam, ...]:
        """Return provider schemas for the selected tools, in selection order."""
        return tuple(definition.provider_tool() for definition in self.definitions)

    def execute(
        self,
        tool_name: str,
        raw_arguments: str,
        argument_byte_cap: int = DEFAULT_TOOL_ARGUMENT_BYTE_CAP,
    ) -> str:
        """Bound, validate, and run one selected tool call.

        Args:
            tool_name: Name of the requested tool.
            raw_arguments: JSON-encoded tool arguments from the model.
            argument_byte_cap: Largest accepted size of ``raw_arguments`` in
                UTF-8 bytes.

        Returns:
            The tool's text, or an error string when the name was not selected,
            the arguments exceed the cap, or they fail validation. Those three
            outcomes happen before the executor runs.

        Raises:
            UncertainToolEffect: The selected executor started and then raised.
                This call does not retry it.
        """
        definition = self._by_name.get(tool_name)
        if definition is None:
            return "error: unknown tool called"

        argument_byte_count = len(raw_arguments.encode("utf-8", errors="surrogatepass"))
        if argument_byte_count > argument_byte_cap:
            return f"error: tool arguments exceed the {argument_byte_cap}-byte limit"

        return definition.execute_json(raw_arguments)
