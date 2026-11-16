"""Command-line entry point: parse flags, load settings, run the agent."""

import argparse
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, assert_never

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from aashir.agent import (
    AgentEvent,
    AgentRequest,
    AnswerDelta,
    ProviderError,
    ReasoningDelta,
    ReasoningSummary,
    ToolCallReceived,
    ToolResult,
    TurnCompleted,
    run_agent,
)
from aashir.settings import (
    REASONING_EFFORT_CHOICES,
    Settings,
    SettingsError,
    SettingsOverrides,
    load_settings,
)
from aashir.tools import AGENT_TOOLS

_DEFAULT_SYSTEM_PROMPT: Final = "You are a helpful assistant."


@dataclass(frozen=True, slots=True)
class _CLIArguments:
    """Arguments that control one CLI invocation."""

    prompt: str
    dotenv_path: Path | None
    settings_overrides: SettingsOverrides


def _parse_arguments() -> _CLIArguments:
    """Parse the prompt, the credential file, and every user-definable setting."""
    parser = argparse.ArgumentParser(
        epilog=(
            "Each setting flag overrides its environment variable. The API key has "
            "no flag so it stays out of shell history: set OPENAI_API_KEY in the "
            "environment or in a credential-only file selected with --env-file."
        )
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        metavar="PROMPT",
        help="text to send to the assistant",
    )
    parser.add_argument(
        "-e",
        "--env-file",
        type=Path,
        metavar="PATH",
        help="read provider credentials from this credential-only dotenv file",
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        help="base URL of the provider's OpenAI-compatible API (OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        help="name of the model to use (OPENAI_MODEL)",
    )
    parser.add_argument(
        "--system-prompt-file",
        metavar="PATH",
        help="text file containing the system prompt (SYSTEM_PROMPT_FILE)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        help=(
            "reasoning level to request; choose one your model supports "
            "(REASONING_EFFORT)"
        ),
    )
    parser.add_argument(
        "--tool-argument-byte-cap",
        type=int,
        metavar="BYTES",
        help=(
            "largest accepted tool-call argument size in UTF-8 bytes "
            "(TOOL_ARGUMENT_BYTE_CAP)"
        ),
    )
    arguments = parser.parse_args()
    return _CLIArguments(
        prompt=arguments.prompt,
        dotenv_path=arguments.env_file,
        settings_overrides=SettingsOverrides(
            base_url=arguments.base_url,
            model=arguments.model,
            system_prompt_file=(
                Path(arguments.system_prompt_file)
                if arguments.system_prompt_file
                else None
            ),
            reasoning_effort=arguments.reasoning_effort,
            tool_argument_byte_cap=arguments.tool_argument_byte_cap,
        ),
    )


def _read_system_prompt(prompt_file: Path | None) -> str:
    """Return the prompt in ``prompt_file``, or the default prompt with a warning.

    The default is used without a warning when no file was chosen. If the file
    cannot be read or is empty, the user is warned on stderr.
    """
    if prompt_file is None:
        return _DEFAULT_SYSTEM_PROMPT

    try:
        custom_prompt = prompt_file.read_text(encoding="utf-8")
    except (UnicodeError, OSError) as error:
        print(
            f"warn: custom system prompt could not be loaded because of the following: {error}, using default prompt",
            file=sys.stderr,
        )
        return _DEFAULT_SYSTEM_PROMPT

    if not custom_prompt:
        print(
            "warn: custom system prompt file is empty, using default prompt",
            file=sys.stderr,
        )
        return _DEFAULT_SYSTEM_PROMPT
    return custom_prompt


def _configure_request(settings: Settings, user_prompt: str) -> AgentRequest:
    """Build the request for an agent run from validated settings.

    Returns:
        The agent request assembled from the prompt and settings.
    """
    message_history: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": _read_system_prompt(settings.system_prompt_file)},
        {"role": "user", "content": user_prompt},
    ]

    return AgentRequest(
        model=settings.model,
        messages=message_history,
        tools=AGENT_TOOLS,
        reasoning_effort=settings.reasoning_effort,
        tool_argument_byte_cap=settings.tool_argument_byte_cap,
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
    """Run one agent turn for the prompt and settings given on the command line."""
    arguments = _parse_arguments()
    try:
        settings = load_settings(arguments.settings_overrides, arguments.dotenv_path)
    except SettingsError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None

    if arguments.dotenv_path is not None:
        print("info: read selected credential file", file=sys.stderr)
    request = _configure_request(settings, arguments.prompt)

    with OpenAI(
        api_key=settings.api_key.get_secret_value(),
        base_url=str(settings.base_url),
    ) as client:
        try:
            _render_events(run_agent(client, request))
        except ProviderError as error:
            print(f"error: {error.category}: {error}", file=sys.stderr)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
