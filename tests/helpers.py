"""Shared provider-stream helpers for agent and CLI tests."""

import json
from collections.abc import Mapping
from typing import Final

import httpx2
from openai import APIError, OpenAI
from pydantic import JsonValue, TypeAdapter

_JSON_OBJECT: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(
    dict[str, JsonValue]
)


def sse_chunk[DeltaValue, ErrorValue](
    delta: Mapping[str, DeltaValue],
    *,
    finish_reason: str | None = None,
    error: Mapping[str, ErrorValue] | None = None,
) -> bytes:
    """Encode one chat-completion SSE chunk with an optional provider error."""
    json_delta = _JSON_OBJECT.validate_python(dict(delta), strict=True)
    payload: dict[str, JsonValue] = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "delta": json_delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if error is not None:
        payload["error"] = _JSON_OBJECT.validate_python(dict(error), strict=True)
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
