"""Public request and event types for agent runs."""

from dataclasses import dataclass, field

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionReasoningEffort,
)

from aashir.tools import DEFAULT_TOOL_ARGUMENT_BYTE_CAP, ToolCatalog


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """Model input, starting conversation history, and limits for one agent run.

    ``tools`` is the catalog selected for the run. An empty catalog advertises
    no tools and runs no tools.

    Raises:
        ValueError: If ``tool_argument_byte_cap`` is not positive, since every
            tool call would then be rejected.
    """

    model: str
    messages: list[ChatCompletionMessageParam]
    tools: ToolCatalog = field(default_factory=ToolCatalog)
    reasoning_effort: ChatCompletionReasoningEffort = None
    tool_argument_byte_cap: int = DEFAULT_TOOL_ARGUMENT_BYTE_CAP

    def __post_init__(self) -> None:
        if self.tool_argument_byte_cap <= 0:
            raise ValueError(
                "tool_argument_byte_cap must be positive, "
                f"got {self.tool_argument_byte_cap}"
            )


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
    """Marks a successful assistant completion with no further tool calls.

    ``produced_answer`` is false when no round of the run streamed any answer
    text, for example when the model only reasoned and then stopped.
    """

    produced_answer: bool


type AgentEvent = (
    TurnCompleted
    | ToolResult
    | ToolCallReceived
    | ReasoningSummary
    | AnswerDelta
    | ReasoningDelta
)
