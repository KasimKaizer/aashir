"""Tool-call state and execution within an agent turn."""

from collections.abc import Iterator
from dataclasses import dataclass

from openai.types.chat import ChatCompletionMessageParam

from aashir.agent.contracts import AgentEvent, ToolCallStarted, ToolResult
from aashir.tools import execute_tool_call


@dataclass(slots=True)
class _ToolCallAccumulator:
    """Mutable fields for joining one tool call's streamed fragments."""

    call_id: str = ""
    name: str = ""
    arguments: str = ""


def _execute_tool_calls(
    tool_calls: tuple[_ToolCallAccumulator, ...],
    history: list[ChatCompletionMessageParam],
) -> Iterator[AgentEvent]:
    """Process tool calls in order and append each result to conversation history.

    Each call is checked before execution. Malformed calls produce error results
    but do not raise, allowing the agent to continue with their feedback.

    Args:
        tool_calls: Calls assembled from the provider's streamed response.
        history: Conversation history to append tool results to.

    Yields:
        A start event and result event for each tool call, including malformed calls.
    """
    seen_call_ids: set[str] = set()
    for tool_call in tool_calls:
        yield ToolCallStarted(
            call_id=tool_call.call_id,
            name=tool_call.name,
            arguments=tool_call.arguments,
        )
        if not tool_call.call_id or not tool_call.name or not tool_call.arguments:
            content = "error: incomplete tool call in response"
        elif tool_call.call_id in seen_call_ids:
            content = "error: duplicate tool call id in response"
        else:
            content = execute_tool_call(tool_call.name, tool_call.arguments)
        yield ToolResult(
            call_id=tool_call.call_id,
            name=tool_call.name,
            output=content,
        )
        history.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.call_id,
                "content": content,
            }
        )
        seen_call_ids.add(tool_call.call_id)
