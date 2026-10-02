"""The event-producing agent loop."""

from collections.abc import Iterator

from openai import APIError, OpenAI
from openai.types.chat import ChatCompletionMessageParam

from aashir.agent.contracts import AgentEvent, AgentRequest, TurnCompleted
from aashir.agent.provider_errors import ProviderError, _normalize_provider_error
from aashir.agent.stream import _stream_completion
from aashir.agent.tool_calls import _execute_tool_calls


def run_agent(client: OpenAI, request: AgentRequest) -> Iterator[AgentEvent]:
    """Run model/tool rounds and yield events for a CLI, TUI, or other consumer.

    Args:
        client: OpenAI client used to request each model round.
        request: Model, starting messages, and optional tools and reasoning effort.

    Yields:
        AgentEvent: Answer and reasoning events, tool-call lifecycle events, and
            turn-completion markers.

    Raises:
        ProviderError: If the provider returns an API error, an incomplete or filtered
            completion, an unknown finish reason, or an invalid response protocol.

    The request history is copied before tool messages are appended. Provider
    errors are normalized after any events already yielded from the stream.
    """
    history: list[ChatCompletionMessageParam] = list(request.messages)

    while True:
        try:
            completion = yield from _stream_completion(client, request, history)
        except APIError as error:
            raise _normalize_provider_error(error) from error

        # Provider finish reasons are open-ended, so unknown values get normalized.
        match completion.finish_reason:
            case "stop":
                if completion.tool_calls:
                    raise ProviderError(
                        category="protocol",
                        error_type=None,
                        message="stop completion contained tool calls",
                    )
                yield TurnCompleted()
                return
            case "tool_calls":
                if not completion.tool_calls:
                    raise ProviderError(
                        category="protocol",
                        error_type=None,
                        message="tool_calls completion contained no tool calls",
                    )
            case "length":
                raise ProviderError(
                    category="context_length",
                    error_type=None,
                    message="completion was truncated at the model's token limit",
                )
            case "content_filter":
                raise ProviderError(
                    category="content_policy",
                    error_type=None,
                    message="completion was blocked by the provider's content filter",
                )
            case "function_call" | None:
                raise ProviderError(
                    category="protocol",
                    error_type=None,
                    message=f"completion did not finish successfully: {completion.finish_reason!r}",
                )
            case unknown_finish_reason:
                raise ProviderError(
                    category="unknown",
                    error_type=None,
                    message=f"completion did not finish successfully: {unknown_finish_reason!r}",
                )

        # Reasoning stays on the event stream; only tool protocol messages enter history.
        history.append(
            {
                "role": "assistant",
                "content": completion.content,
                "tool_calls": [
                    {
                        "id": tool_call.call_id,
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                        "type": "function",
                    }
                    for tool_call in completion.tool_calls
                ],
            }
        )
        yield from _execute_tool_calls(completion.tool_calls, history)
