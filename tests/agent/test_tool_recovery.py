import json
from pathlib import Path

import httpx2
from openai import OpenAI

from aashir.agent import (
    AgentRequest,
    ToolCallReceived,
    ToolResult,
    TurnCompleted,
    run_agent,
)
from aashir.tools import AGENT_TOOLS
from tests.helpers import sse_chunk


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


def test_run_agent_handles_call_errors_and_continues_the_batch(
    tmp_path: Path,
) -> None:
    # Given: valid writes around unknown, invalid, and duplicate-ID calls.
    written_path = tmp_path / "written.txt"
    invalid_path = tmp_path / "invalid.txt"
    duplicate_path = tmp_path / "duplicate.txt"
    recovered_path = tmp_path / "recovered.txt"
    calls = [
        {
            "index": 0,
            "id": "call_write",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": json.dumps(
                    {"file_path": str(written_path), "content": "saved"}
                ),
            },
        },
        {
            "index": 1,
            "id": "call_unknown",
            "type": "function",
            "function": {
                "name": "unknown_tool",
                "arguments": json.dumps({"unused": "value"}),
            },
        },
        {
            "index": 2,
            "id": "call_invalid",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": json.dumps({"file_path": str(invalid_path)}),
            },
        },
        {
            "index": 3,
            "id": "call_write",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": json.dumps(
                    {"file_path": str(duplicate_path), "content": "ignored"}
                ),
            },
        },
        {
            "index": 4,
            "id": "call_after_errors",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": json.dumps(
                    {"file_path": str(recovered_path), "content": "done"}
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
        # When: the agent processes the provider's tool-call batch.
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

    # Then: every call has one result, and the final valid call executes.
    assert written_path.read_text(encoding="utf-8") == "saved"
    assert not invalid_path.exists()
    assert not duplicate_path.exists()
    assert recovered_path.read_text(encoding="utf-8") == "done"
    results = [event for event in events if isinstance(event, ToolResult)]
    assert [(event.call_id, event.output) for event in results] == [
        ("call_write", f"success: written 5 characters in {written_path}"),
        ("call_unknown", "error: unknown tool called"),
        ("call_invalid", "error: content must be a string"),
        ("call_write", "error: duplicate tool call id in response"),
        ("call_after_errors", f"success: written 4 characters in {recovered_path}"),
    ]
    assert isinstance(events[-1], TurnCompleted)
    assert len(requests) == 2
