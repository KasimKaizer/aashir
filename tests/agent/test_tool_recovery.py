import json
from pathlib import Path

import httpx2
import pytest
from openai import OpenAI

from aashir.agent import (
    AgentEvent,
    AgentRequest,
    ProviderError,
    ToolCallReceived,
    ToolResult,
    TurnCompleted,
    run_agent,
)
from aashir.tools import AGENT_TOOLS
from tests.helpers import make_openai_client, sse_chunk


def test_run_agent_continues_after_an_incomplete_tool_call(
    tmp_path: Path,
) -> None:
    # Given: valid writes on either side of an incomplete tool call.
    first_path = tmp_path / "first.txt"
    recovered_path = tmp_path / "recovered.txt"
    calls = [
        {
            "index": 0,
            "id": "call_first",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": json.dumps(
                    {"file_path": str(first_path), "content": "first"}
                ),
            },
        },
        {
            "index": 1,
            "id": "call_incomplete",
            "type": "function",
            "function": {"name": "read_file", "arguments": ""},
        },
        {
            "index": 2,
            "id": "call_recovered",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": json.dumps(
                    {"file_path": str(recovered_path), "content": "recovered"}
                ),
            },
        },
    ]
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if len(requests) == 1:
            body = sse_chunk({"tool_calls": calls}, finish_reason="tool_calls")
        else:
            body = sse_chunk({"content": "Done"}, finish_reason="stop")
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body + b"data: [DONE]\n\n",
        )

    with OpenAI(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as client:
        # When: the agent processes the tool-call batch.
        events = list(
            run_agent(
                client,
                AgentRequest(
                    model="test-model",
                    messages=[{"role": "user", "content": "use tools"}],
                    tools=AGENT_TOOLS,
                ),
            )
        )

    # Then: the valid calls execute and the incomplete call returns an error result.
    assert first_path.read_text(encoding="utf-8") == "first"
    assert recovered_path.read_text(encoding="utf-8") == "recovered"
    results = [event for event in events if isinstance(event, ToolResult)]
    assert [(event.call_id, event.output) for event in results] == [
        ("call_first", f"success: written 5 characters in {first_path}"),
        ("call_incomplete", "error: incomplete tool call in response"),
        ("call_recovered", f"success: written 9 characters in {recovered_path}"),
    ]
    incomplete_events = [
        event
        for event in events
        if isinstance(event, (ToolCallReceived, ToolResult))
        and event.call_id == "call_incomplete"
    ]
    assert incomplete_events == [
        ToolCallReceived(
            call_id="call_incomplete",
            name="read_file",
            arguments="",
        ),
        ToolResult(
            call_id="call_incomplete",
            name="read_file",
            output="error: incomplete tool call in response",
        ),
    ]
    assert isinstance(events[-1], TurnCompleted)

    # Then: all results, including the error, are sent to the model in call order.
    assert len(requests) == 2
    messages = json.loads(requests[1].content)["messages"]
    assert messages[-4]["role"] == "assistant"
    assert messages[-3:] == [
        {
            "role": "tool",
            "tool_call_id": "call_first",
            "content": f"success: written 5 characters in {first_path}",
        },
        {
            "role": "tool",
            "tool_call_id": "call_incomplete",
            "content": "error: incomplete tool call in response",
        },
        {
            "role": "tool",
            "tool_call_id": "call_recovered",
            "content": f"success: written 9 characters in {recovered_path}",
        },
    ]


def test_run_agent_rejects_length_truncated_tool_calls_before_execution(
    tmp_path: Path,
) -> None:
    # Given: a complete write call followed by a length-truncated finish reason.
    output_path = tmp_path / "must-not-be-written.txt"
    arguments = json.dumps({"file_path": str(output_path), "content": "no"})
    body = sse_chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_write",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": arguments},
                }
            ]
        }
    )
    body += sse_chunk({}, finish_reason="length")
    body += b"data: [DONE]\n\n"
    request = AgentRequest(
        model="test-model",
        messages=[{"role": "user", "content": "do not write"}],
        tools=AGENT_TOOLS,
    )
    events: list[AgentEvent] = []

    # When: the agent consumes the truncated completion.
    client = make_openai_client(httpx2.ByteStream(body))
    with client, pytest.raises(ProviderError) as error:
        events.extend(run_agent(client, request))

    # Then: truncation is reported before the tool is announced or executed.
    assert error.value.category == "context_length"
    assert not output_path.exists()
    assert not any(
        isinstance(event, (ToolCallReceived, ToolResult, TurnCompleted))
        for event in events
    )
