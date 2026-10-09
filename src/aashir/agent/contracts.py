"""Public request and event types for agent runs."""

from dataclasses import dataclass

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionReasoningEffort,
    ChatCompletionToolUnionParam,
)


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Model input and starting conversation history for one agent run."""

    model: str
    messages: list[ChatCompletionMessageParam]
    tools: list[ChatCompletionToolUnionParam] | None = None
    reasoning_effort: ChatCompletionReasoningEffort = None


@dataclass(frozen=True, slots=True)
class AnswerDelta:
    """A fragment of assistant response text."""

    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    """A raw reasoning fragment explicitly returned by the provider."""

    text: str


@dataclass(frozen=True, slots=True)
class ReasoningSummary:
    """A provider-returned reasoning summary, separate from raw reasoning."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallReceived:
    """A provider tool call received before validation or execution."""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Text returned by a tool after its execution, including tool errors."""

    call_id: str
    name: str
    output: str


@dataclass(frozen=True, slots=True)
class TurnCompleted:
    """Marks a successful assistant completion with no further tool calls."""


type AgentEvent = (
    TurnCompleted
    | ToolResult
    | ToolCallReceived
    | ReasoningSummary
    | AnswerDelta
    | ReasoningDelta
)
