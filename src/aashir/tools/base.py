"""Shared building blocks for defining agent tools."""

from collections.abc import Callable
from dataclasses import dataclass

from openai.types.chat.chat_completion_tool_union_param import (
    ChatCompletionToolUnionParam,
)
from pydantic import BaseModel, ConfigDict, ValidationError


class _ToolArguments(BaseModel):
    """Base configuration for validated model tool arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _generic_validation_error_message(error: ValidationError) -> str:
    """Format validation errors without echoing the invalid input."""
    issue = error.errors(include_input=False)[0]
    if issue["type"] == "extra_forbidden":
        # Pydantic includes the unrecognized input key in the error location.
        return f"error: invalid arguments: {issue['msg']}"

    location_text = ".".join(str(part) for part in issue["loc"]) or "arguments"
    return f"error: invalid arguments at {location_text}: {issue['msg']}"


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
