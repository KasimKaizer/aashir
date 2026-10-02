"""Shared provider-stream helpers for agent and CLI tests."""

import json
from collections.abc import Mapping, Sequence

import httpx2
from openai import APIError, OpenAI

type JsonValue = (
    str | int | float | bool | None | Sequence[JsonValue] | Mapping[str, JsonValue]
)


def sse_chunk(
    delta: Mapping[str, JsonValue],
    *,
    finish_reason: str | None = None,
    error: Mapping[str, JsonValue] | None = None,
) -> bytes:
    """Encode one chat-completion SSE chunk with an optional provider error."""
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if error is not None:
        payload["error"] = error
    return f"data: {json.dumps(payload)}\n\n".encode()


def make_openai_client(stream: httpx2.SyncByteStream) -> OpenAI:
    """Build an OpenAI client served by an in-memory SSE stream."""
    def respond(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=stream,
        )

    return OpenAI(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    )


def make_failing_openai_client(error: APIError) -> OpenAI:
    """Build an OpenAI client whose transport raises an SDK error."""
    def respond(_request: httpx2.Request) -> httpx2.Response:
        raise error

    return OpenAI(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    )
