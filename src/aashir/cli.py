import argparse
import os
from collections.abc import Iterable
from pathlib import Path
from typing import assert_never

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from aashir.agent import (
    AgentEvent,
    AgentRequest,
    AnswerDelta,
    ReasoningDelta,
    ReasoningSummary,
    ToolCallStarted,
    ToolResult,
    TurnCompleted,
    run_agent,
)
from aashir.tools import AGENT_TOOLS

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("BASE_MODEL_NAME")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT_FILE")
REASONING_EFFORT = os.getenv("REASONING_EFFORT")


def _configure_request() -> AgentRequest:
    """Parse CLI input and build the validated request for an agent run.

    Returns:
        The agent request assembled from the command-line prompt and settings.

    Raises:
        RuntimeError: If required settings are missing.
        SystemExit: If command-line arguments are invalid.
    """
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    cli_args = p.parse_args()

    required_vars = {
        "OPENAI_API_KEY": API_KEY,
        "OPENAI_BASE_URL": BASE_URL,
        "BASE_MODEL_NAME": MODEL_NAME,
        "REASONING_EFFORT": REASONING_EFFORT,
    }
    missing: list[str] = [
        var for var, val in required_vars.items() if val in {"", None}
    ]
    if missing:
        missing_list: str = ", ".join(missing)
        raise RuntimeError(f"Missing required variables: {missing_list}")

    prompt: str = "You are a helpful assistant."
    if SYSTEM_PROMPT:
        try:
            if custom_prompt := Path(SYSTEM_PROMPT).read_text(encoding="utf-8"):
                prompt = custom_prompt
        except (UnicodeError, OSError) as e:
            print(
                f"warn: custom system prompt could not be loaded because of the following: {e}, using default prompt"
            )

    message_history: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": cli_args.p},
    ]

    return AgentRequest(
        model=MODEL_NAME,
        messages=message_history,
        tools=AGENT_TOOLS,
        reasoning_effort=REASONING_EFFORT,
    )


def _render_events(events: Iterable[AgentEvent]) -> None:
    """Print answer deltas as they arrive and ignore all other event types.

    Args:
        events: Agent events to render in arrival order.
    """
    for event in events:
        match event:
            case AnswerDelta(text=text):
                print(text, end="", flush=True)
            case (
                ReasoningSummary()
                | ReasoningDelta()
                | ToolCallStarted()
                | ToolResult()
                | TurnCompleted()
            ):
                continue
            case unreachable:
                assert_never(unreachable)


def main() -> None:
    request = _configure_request()
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    _render_events(run_agent(client, request))


if __name__ == "__main__":
    main()
