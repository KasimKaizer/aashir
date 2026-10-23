import json
from pathlib import Path

import httpx2
import pytest
from openai import OpenAI

from aashir.agent import (
    AgentRequest,
    AnswerDelta,
    ToolCallReceived,
    ToolResult,
    TurnCompleted,
    run_agent,
)
from aashir.tools import AGENT_TOOLS
from tests.helpers import sse_chunk


def test_run_agent_assembles_fragmented_interleaved_tool_calls_in_completion_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: interleaved file calls arrive in fragments, and one starts without a function.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
    args_a = json.dumps({"file_path": "a.txt"})
    args_b = json.dumps({"file_path": "b.txt"})
    split_a = len(args_a) // 2
    split_b = len(args_b) // 2
    first_chunks = [
        {
            "role": "assistant",
            "tool_calls": [
                {"index": 0, "id": "", "type": "function"},
            ],
        },
        {
            "tool_calls": [
                {
                    "index": 1,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": args_b[:split_b]},
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_0",
                    "function": {"name": "read_file", "arguments": args_a[:split_a]},
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_0",
                    "function": {
                        "name": "read_file",
                        "arguments": args_a[split_a:],
                    },
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_replacement",
                    "function": {"name": "write_file"},
                }
            ]
        },
        {"tool_calls": [{"index": 1, "function": {"arguments": args_b[split_b:]}}]},
    ]
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        if len(requests) == 1:
            body = b"".join(sse_chunk(delta) for delta in first_chunks)
            body += sse_chunk({}, finish_reason="tool_calls")
        else:
            body = sse_chunk({"content": "Done"}, finish_reason="stop")
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body + b"data: [DONE]\n\n",
        )

    # When: the agent assembles and runs the completed calls.
    with OpenAI(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as client:
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

    # Then: calls execute in index order, with each start before its result.
    tool_events = [
        event for event in events if isinstance(event, (ToolCallReceived, ToolResult))
    ]
    assert [(type(event), event.call_id) for event in tool_events] == [
        (ToolCallReceived, "call_0"),
        (ToolResult, "call_0"),
        (ToolCallReceived, "call_1"),
        (ToolResult, "call_1"),
    ]
    first_call_event = tool_events[0]
    assert isinstance(first_call_event, ToolCallReceived)
    assert (
        first_call_event.call_id,
        first_call_event.name,
        first_call_event.arguments,
    ) == ("call_0", "read_file", args_a)
    assert isinstance(events[-1], TurnCompleted)

    results = [event for event in events if isinstance(event, ToolResult)]
    assert [(event.call_id, event.output) for event in results] == [
        ("call_0", "alpha\n"),
        ("call_1", "beta\n"),
    ]
    assert (
        "".join(event.text for event in events if isinstance(event, AnswerDelta))
        == "Done"
    )

    # Then: the next request includes the complete assistant call batch and results.
    assert len(requests) == 2
    messages = json.loads(requests[1].content)["messages"]
    assert messages[-3]["role"] == "assistant"
    assert [call["id"] for call in messages[-3]["tool_calls"]] == [
        "call_0",
        "call_1",
    ]
    first_assistant_call = messages[-3]["tool_calls"][0]
    assert (
        first_assistant_call["id"],
        first_assistant_call["function"]["name"],
        first_assistant_call["function"]["arguments"],
    ) == ("call_0", "read_file", args_a)
    assert messages[-2] == {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": "alpha\n",
    }
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "beta\n",
    }
