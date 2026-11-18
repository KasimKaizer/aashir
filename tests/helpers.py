"""Shared provider-stream helpers for agent and CLI tests."""

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Final

import httpx2
from openai import APIError, OpenAI
from openai.types.chat import ChatCompletionReasoningEffort
from pydantic import JsonValue, TypeAdapter

from aashir.agent import AgentEvent, AgentRequest, AnswerDelta, run_agent
from aashir.tools import ToolCatalog, edit_file, read_file, write_file

_JSON_OBJECT: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(
    dict[str, JsonValue]
)
_STREAM_END: Final = b"data: [DONE]\n\n"


@dataclass(frozen=True, slots=True)
class ClientArguments:
    """The arguments the CLI passed when it created a provider client."""

    api_key: str
    base_url: str


def sse_chunk[DeltaValue, ErrorValue](
    delta: Mapping[str, DeltaValue],
    *,
    finish_reason: str | None = None,
    error: Mapping[str, ErrorValue] | None = None,
) -> bytes:
    """Encode one chat-completion SSE chunk with an optional provider error."""
    json_delta = _JSON_OBJECT.validate_python(dict(delta), strict=True)
    choice: JsonValue = {
        "index": 0,
        "delta": json_delta,
        "finish_reason": finish_reason,
    }
    return _encode_chunk([choice], error)


def sse_chunk_without_choices[ErrorValue](
    *, error: Mapping[str, ErrorValue] | None = None
) -> bytes:
    """Encode one chat-completion SSE chunk that carries no completion choices."""
    return _encode_chunk([], error)


def _encode_chunk[ErrorValue](
    choices: list[JsonValue], error: Mapping[str, ErrorValue] | None
) -> bytes:
    payload: dict[str, JsonValue] = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": choices,
    }
    if error is not None:
        payload["error"] = _JSON_OBJECT.validate_python(dict(error), strict=True)
    return f"data: {json.dumps(payload)}\n\n".encode()


def sse_body(*chunks: bytes) -> bytes:
    """Join SSE chunks into one complete provider stream."""
    return b"".join(chunks) + _STREAM_END


def answer_text(events: Iterable[AgentEvent]) -> str:
    """Join the text of every answer delta, ignoring other event types."""
    return "".join(event.text for event in events if isinstance(event, AnswerDelta))


def tool_call(
    index: int, call_id: str, name: str, arguments: str
) -> dict[str, JsonValue]:
    """Build one complete tool-call fragment for an SSE delta."""
    return {
        "index": index,
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def make_agent_request(
    *,
    with_tools: bool = False,
    reasoning_effort: ChatCompletionReasoningEffort = None,
) -> AgentRequest:
    """Build a one-message request, selecting the file tools on demand."""
    return AgentRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        tools=ToolCatalog((read_file, write_file, edit_file))
        if with_tools
        else ToolCatalog(),
        reasoning_effort=reasoning_effort,
    )


def make_openai_client(stream: httpx2.SyncByteStream) -> OpenAI:
    """Build an OpenAI client served by an in-memory SSE stream."""

    def respond(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    return _client_for(respond)


def make_round_trip_client(
    first_turn: bytes,
    final_turn: bytes | None = None,
) -> tuple[OpenAI, list[httpx2.Request]]:
    """Build a client that replies ``first_turn``, then a final answer.

    Args:
        first_turn: Complete SSE body for the first request, usually tool calls.
        final_turn: Complete SSE body for every later request. Defaults to a
            ``Done`` answer.

    Returns:
        The client and the list that records every request it receives.
    """
    requests: list[httpx2.Request] = []
    if final_turn is None:
        final_turn = sse_body(sse_chunk({"content": "Done"}, finish_reason="stop"))

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=first_turn if len(requests) == 1 else final_turn,
        )

    return _client_for(respond), requests


@dataclass(frozen=True, slots=True)
class ToolRoundTrip:
    """Everything observed while the agent ran one provider tool round trip."""

    events: list[AgentEvent]
    requests: list[httpx2.Request]


def run_tool_round_trip(first_turn: bytes) -> ToolRoundTrip:
    """Run the agent against a provider that replies ``first_turn``, then ``Done``.

    Args:
        first_turn: Complete SSE body for the first request, usually tool calls.

    Returns:
        The agent's events and every request the provider received.
    """
    client, requests = make_round_trip_client(first_turn)
    with client:
        events = list(run_agent(client, make_agent_request(with_tools=True)))
    return ToolRoundTrip(events=events, requests=requests)


def make_failing_openai_client(error: APIError) -> OpenAI:
    """Build an OpenAI client whose transport raises an SDK error."""

    def respond(_request: httpx2.Request) -> httpx2.Response:
        raise error

    return _client_for(respond)


def _client_for(respond: Callable[[httpx2.Request], httpx2.Response]) -> OpenAI:
    return OpenAI(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    )
