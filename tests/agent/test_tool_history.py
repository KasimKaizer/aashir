import json
from pathlib import Path

import httpx2
import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from aashir.agent import (
    AgentRequest,
    AnswerDelta,
    ToolCallStarted,
    ToolResult,
    TurnCompleted,
    run_agent,
)
from aashir.tools import AGENT_TOOLS
from tests.helpers import sse_chunk


def test_run_agent_preserves_tool_history_and_emits_lifecycle_events_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a real file tool and a two-request provider exchange.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "note.txt").write_text("café\n", encoding="utf-8")
    tool_delta = {
        "role": "assistant",
        "tool_calls": [
            {
                "index": 0,
                "id": "call_tool",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"file_path": "note.txt"}),
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
        ],
    }
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        body = (
            sse_chunk(tool_delta, finish_reason="tool_calls")
            if len(requests) == 1
            else sse_chunk({"content": "Done"}, finish_reason="stop")
        )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body + b"data: [DONE]\n\n",
        )

    client = OpenAI(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    )

    # When: the agent completes a tool round trip.
    initial_messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "use a tool"}
    ]
    events = list(
        run_agent(
            client,
            AgentRequest(
                model="test-model",
                messages=initial_messages,
                tools=AGENT_TOOLS,
            ),
        )
    )

    # Then: each call has a matching result before the turn completes.
    tool_events = [
        event for event in events if isinstance(event, (ToolCallStarted, ToolResult))
    ]
    assert [(type(event), event.call_id) for event in tool_events] == [
        (ToolCallStarted, "call_tool"),
        (ToolResult, "call_tool"),
        (ToolCallStarted, "call_unknown"),
        (ToolResult, "call_unknown"),
    ]
    assert isinstance(events[-1], TurnCompleted)

    started = [event for event in events if isinstance(event, ToolCallStarted)]
    assert [(event.call_id, event.name) for event in started] == [
        ("call_tool", "read_file"),
        ("call_unknown", "unknown_tool"),
    ]
    assert json.loads(started[0].arguments) == {"file_path": "note.txt"}
    assert json.loads(started[1].arguments) == {"unused": "value"}

    results = [event for event in events if isinstance(event, ToolResult)]
    assert [(event.call_id, event.name, event.output) for event in results] == [
        ("call_tool", "read_file", "café\n"),
        ("call_unknown", "unknown_tool", "error: unknown tool called"),
    ]

    # Then: both assistant calls and their results are preserved for the next request.
    assert len(requests) == 2
    messages = json.loads(requests[1].content)["messages"]
    assistant_message = messages[-3]
    tool_results = messages[-2:]
    assert assistant_message["role"] == "assistant"
    assert [call["id"] for call in assistant_message["tool_calls"]] == [
        "call_tool",
        "call_unknown",
    ]
    assert tool_results == [
        {"role": "tool", "tool_call_id": "call_tool", "content": "café\n"},
        {
            "role": "tool",
            "tool_call_id": "call_unknown",
            "content": "error: unknown tool called",
        },
    ]
    assert initial_messages == [{"role": "user", "content": "use a tool"}]


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
                {"index": 0, "id": "call_0", "type": "function"},
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
        {"tool_calls": [
            {
                "index": 0,
                "function": {"name": "read_file", "arguments": args_a[:split_a]},
            }
        ]},
        {"tool_calls": [{"index": 0, "function": {"arguments": args_a[split_a:]}}]},
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

    client = OpenAI(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    )

    # When: the agent assembles and runs the completed calls.
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
        event for event in events if isinstance(event, (ToolCallStarted, ToolResult))
    ]
    assert [(type(event), event.call_id) for event in tool_events] == [
        (ToolCallStarted, "call_0"),
        (ToolResult, "call_0"),
        (ToolCallStarted, "call_1"),
        (ToolResult, "call_1"),
    ]
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
