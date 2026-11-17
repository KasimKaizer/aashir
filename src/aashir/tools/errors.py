"""Failures while selecting or running a tool."""

from dataclasses import dataclass
from typing import Literal, override

type ToolCatalogRejection = Literal[
    "duplicate_name",
    "invalid_name",
    "unsupported_schema",
]


@dataclass(frozen=True, slots=True)
class ToolCatalogError(Exception):
    """A catalog was refused before any tool callback ran."""

    reason: ToolCatalogRejection
    tool_name: str
    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class UncertainToolEffect(Exception):
    """A tool callback started, then failed without a known outcome.

    The original exception is the private ``__cause__``. Its text stays off
    the model-visible result: the effect may already have happened.
    """

    tool_name: str

    @override
    def __str__(self) -> str:
        return f"tool {self.tool_name} ended with an uncertain effect"
