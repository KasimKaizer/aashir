import argparse
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionReasoningEffort,
)

from aashir.agent import (
    AgentEvent,
    AgentRequest,
    AnswerDelta,
    ReasoningDelta,
    ReasoningSummary,
    ToolCallReceived,
    ToolResult,
    TurnCompleted,
    run_agent,
)
from aashir.tools import AGENT_TOOLS


# TODO: Replace this temporary CLI settings container with an Agent class that owns
# conversation messages, configuration, and tools.
@dataclass(frozen=True, slots=True)
class _CLISettings:
    """Environment-backed settings loaded for one CLI invocation."""

    api_key: str | None
    base_url: str | None
    model_name: str
    system_prompt_file: str | None
    reasoning_effort: str | None


def _load_settings() -> _CLISettings:
    """Load dotenv and environment settings for the current invocation."""
    load_dotenv()
    return _CLISettings(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model_name=os.getenv("BASE_MODEL_NAME", ""),
        system_prompt_file=os.getenv("SYSTEM_PROMPT_FILE"),
        reasoning_effort=os.getenv("REASONING_EFFORT"),
    )


def _parse_reasoning_effort(value: str | None) -> ChatCompletionReasoningEffort:
    """Validate the configured reasoning effort against SDK-supported values."""
    if value is None:
        return None

    match value:
        case "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max":
            return value
        case _:
            raise RuntimeError(
                f"Invalid REASONING_EFFORT {value!r}; expected one of: "
                "none, minimal, low, medium, high, xhigh, or max."
            )


def _configure_request(settings: _CLISettings) -> AgentRequest:
    """Parse CLI input and build the validated request for an agent run.

    Returns:
        The agent request assembled from the command-line prompt and settings.

    Raises:
        RuntimeError: If required settings are missing or invalid.
        SystemExit: If command-line arguments are invalid.
    """
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    cli_args = p.parse_args()

    required_vars = {
        "OPENAI_API_KEY": settings.api_key,
        "OPENAI_BASE_URL": settings.base_url,
        "BASE_MODEL_NAME": settings.model_name,
        "REASONING_EFFORT": settings.reasoning_effort,
    }
    missing: list[str] = [
        var for var, val in required_vars.items() if val in {"", None}
    ]
    if missing:
        missing_list: str = ", ".join(missing)
        raise RuntimeError(f"Missing required variables: {missing_list}")

    reasoning_effort = _parse_reasoning_effort(settings.reasoning_effort)
    prompt: str = "You are a helpful assistant."
    if settings.system_prompt_file:
        try:
            if custom_prompt := Path(settings.system_prompt_file).read_text(
                encoding="utf-8"
            ):
                prompt = custom_prompt
        except (UnicodeError, OSError) as e:
            print(
                f"warn: custom system prompt could not be loaded because of the following: {e}, using default prompt",
                file=sys.stderr,
            )

    message_history: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": cli_args.p},
    ]

    return AgentRequest(
        model=settings.model_name,
        messages=message_history,
        tools=AGENT_TOOLS,
        reasoning_effort=reasoning_effort,
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
                | ToolCallReceived()
                | ToolResult()
                | TurnCompleted()
            ):
                continue
            case unreachable:
                assert_never(unreachable)


def main() -> None:
    settings = _load_settings()
    try:
        request = _configure_request(settings)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None

    with OpenAI(api_key=settings.api_key, base_url=settings.base_url) as client:
        _render_events(run_agent(client, request))


if __name__ == "__main__":
    main()
