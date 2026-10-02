"""Provider stream parsing and event conversion."""

from collections.abc import Generator, Iterator
from dataclasses import dataclass

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from aashir.agent.contracts import (
    AgentEvent,
    AgentRequest,
    AnswerDelta,
    ReasoningDelta,
    ReasoningSummary,
)
from aashir.agent.provider_errors import ProviderError
from aashir.agent.tool_calls import _ToolCallAccumulator


@dataclass(frozen=True, slots=True)
class _StreamedCompletion:
    """Assembled content, finish reason, and tool calls from one response."""

    content: str | None
    finish_reason: str | None
    tool_calls: tuple[_ToolCallAccumulator, ...]


def _events_from_delta(delta: ChoiceDelta) -> Iterator[AgentEvent]:
    """Convert one provider delta into events, ignoring unknown reasoning types.

    Args:
        delta: One streamed provider delta.

    Yields:
        Answer and reasoning events represented by the delta.
    """
    if delta.content:
        yield AnswerDelta(text=delta.content)

    if not delta.model_extra:
        return

    reasoning = delta.model_extra.get("reasoning")
    has_raw_reasoning: bool = False
    if not isinstance(reasoning, str) or not reasoning:
        reasoning = delta.model_extra.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        has_raw_reasoning = True
        yield ReasoningDelta(text=reasoning)

    reasoning_details = delta.model_extra.get("reasoning_details")
    if not isinstance(reasoning_details, list):
        return

    for reasoning_detail in reasoning_details:
        if not isinstance(reasoning_detail, dict):
            continue

        match reasoning_detail:
            case {"type": "reasoning.text", "text": str() as text}:
                if not has_raw_reasoning:
                    yield ReasoningDelta(text=text)
            case {"type": "reasoning.summary", "summary": str() as summary}:
                yield ReasoningSummary(text=summary)
            case _:
                continue


def _stream_completion(
    client: OpenAI,
    request: AgentRequest,
    history: list[ChatCompletionMessageParam],
) -> Generator[AgentEvent, None, _StreamedCompletion]:
    """Yield events live, then return the assembled fields for the agent loop.

    Args:
        client: OpenAI client used to request the model completion.
        request: Model settings and optional tools for the completion.
        history: Conversation messages sent with the request.

    Yields:
        Answer, reasoning, and tool-related events from provider chunks.

    Returns:
        The assembled completion after the stream ends.

    Raises:
        ProviderError: If the provider stream contains no completion choices.

    `run_agent` uses `yield from` to forward these events and receive the
    `_StreamedCompletion` returned when this stream ends. SDK errors propagate for
    normalization by `run_agent`.
    """
    params = {}
    if request.reasoning_effort:
        params["reasoning_effort"] = request.reasoning_effort
    if request.tools:
        params["tools"] = request.tools

    tool_calls: dict[int, _ToolCallAccumulator] = {}
    answer_parts: list[str] = []
    has_choice = False
    finish_reason: str | None = None

    with client.chat.completions.create(
        model=request.model, messages=history, stream=True, **params
    ) as stream:
        for event in stream:
            if not event.choices:
                continue

            has_choice = True
            choice = event.choices[0]
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta.content:
                answer_parts.append(delta.content)
            yield from _events_from_delta(delta)

            for tool_call_delta in delta.tool_calls or []:
                accumulator = tool_calls.setdefault(
                    tool_call_delta.index, _ToolCallAccumulator()
                )
                if tool_call_delta.id:
                    accumulator.call_id += tool_call_delta.id
                if tool_call_delta.function:
                    if tool_call_delta.function.name:
                        accumulator.name += tool_call_delta.function.name
                    if tool_call_delta.function.arguments:
                        accumulator.arguments += tool_call_delta.function.arguments

    if not has_choice:
        raise ProviderError(
            category="protocol",
            error_type=None,
            message="provider stream contained no completion choices",
        )

    return _StreamedCompletion(
        content="".join(answer_parts) or None,
        finish_reason=finish_reason,
        tool_calls=tuple(tool_calls[index] for index in sorted(tool_calls)),
    )
