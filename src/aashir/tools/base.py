"""Shared building blocks for defining agent tools."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from openai.types.chat.chat_completion_tool_union_param import (
    ChatCompletionToolUnionParam,
)
from pydantic import BaseModel, ValidationError

from aashir.tools.errors import UncertainToolEffect

_LOGGER: Final = logging.getLogger(__name__)


def generic_validation_error_message(error: ValidationError) -> str:
    """Format a validation error for the model without the rejected input.

    Pass this function as ``validation_error_formatter`` on a
    ``ToolDefinition``. It is also the default formatter.

    Args:
        error: Validation error for one tool call's arguments.

    Returns:
        The first issue as ``error: invalid arguments: {message}`` when the
        issue is an extra field. Every other issue is
        ``error: invalid arguments at {location}: {message}``.
    """
    issue = error.errors(include_input=False)[0]
    if issue["type"] == "extra_forbidden":
        # Pydantic includes the unrecognized input key in the error location.
        return f"error: invalid arguments: {issue['msg']}"

    location_text = ".".join(str(part) for part in issue["loc"]) or "arguments"
    return f"error: invalid arguments at {location_text}: {issue['msg']}"


@dataclass(frozen=True, slots=True)
class ToolDefinition[ArgumentsT: BaseModel]:
    """One tool: its name, argument model, and typed executor.

    Built-in tools and caller-supplied tools are both values of this type.
    The argument model is the caller's own Pydantic model.
    """

    name: str
    description: str
    arguments_model: type[ArgumentsT]
    execute: Callable[[ArgumentsT], str]
    validation_error_formatter: Callable[[ValidationError], str] = (
        generic_validation_error_message
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
        """Validate raw arguments, then run the executor.

        Args:
            raw_arguments: JSON object supplied for this tool call.

        Returns:
            The executor's text, or a validation error when the arguments do
            not match the model. Validation failure does not call the executor.

        Raises:
            UncertainToolEffect: The executor started and then raised. The
                original exception is preserved as ``__cause__`` and is not
                included in this error's text.
        """
        try:
            arguments = self.arguments_model.model_validate_json(raw_arguments)
        except ValidationError as error:
            return self.validation_error_formatter(error)

        # TODO: Add permission hook here; advertised tools are not authorization.
        # TODO: Enforce workspace containment in the filesystem-tool boundary.
        try:
            return self.execute(arguments)
        except Exception as error:
            # Any exception after the callback starts can follow an effect that
            # already happened. Keep that exception as the private cause and do
            # not turn it into a model-visible error string.
            _LOGGER.exception("tool %s failed unexpectedly", self.name)
            raise UncertainToolEffect(tool_name=self.name) from error
