"""Tool-call state and execution within an agent turn."""

from collections.abc import Iterator
from dataclasses import dataclass

from openai.types.chat import ChatCompletionMessageParam

from aashir.agent.contracts import AgentEvent, ToolCallReceived, ToolResult
from aashir.tools.catalog import ToolCatalog


@dataclass(slots=True)
class _ToolCallAccumulator:
    """Mutable fields for joining one tool call's streamed fragments."""

    call_id: str = ""
    name: str = ""
    arguments: str = ""


def _execute_tool_calls(
    catalog: ToolCatalog,
    tool_calls: tuple[_ToolCallAccumulator, ...],
    history: list[ChatCompletionMessageParam],
    argument_byte_cap: int,
) -> Iterator[AgentEvent]:
    """Process tool calls in order and append each known result to history.

    Each call is checked before execution. A call that is incomplete, repeated,
    unknown, oversized, or invalid produces an error result and does not run a
    tool. Once a selected callback starts, an unexpected exception stops the
    batch: later calls are not run, and no tool result is recorded for the
    failed call.

    Args:
        catalog: Tools selected for this batch. Only these definitions can run.
        tool_calls: Calls assembled from the provider's streamed response.
        history: Conversation history to append tool results to.
        argument_byte_cap: Largest accepted tool argument size in UTF-8 bytes.

    Yields:
        A start event and, for a call that finishes with a known result, a
        result event.

    Raises:
        UncertainToolEffect: A selected callback started and then raised. Later
            calls in ``tool_calls`` are not run.
    """
    seen_call_ids: set[str] = set()
    for tool_call in tool_calls:
        yield ToolCallReceived(
            call_id=tool_call.call_id,
            name=tool_call.name,
            arguments=tool_call.arguments,
        )
        # TODO: Verify how supported providers handle missing or duplicate tool-call IDs.
        # Adjust follow-up handling if any provider rejects this history.
        if not tool_call.call_id or not tool_call.name or not tool_call.arguments:
            content = "error: incomplete tool call in response"
        elif tool_call.call_id in seen_call_ids:
            content = "error: duplicate tool call id in response"
        else:
            content = catalog.execute(
                tool_call.name,
                tool_call.arguments,
                argument_byte_cap,
            )
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
