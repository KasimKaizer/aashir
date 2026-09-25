import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from aashir.tools import AGENT_TOOLS, execute_tool_call

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("BASE_MODEL_NAME")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT_FILE")
REASONING_EFFORT = os.getenv("REASONING_EFFORT")


def main():
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

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    prompt: str = "You are a helpful assistant."
    try:
        if custom_prompt := Path(SYSTEM_PROMPT).read_text(encoding="utf-8"):
            prompt = custom_prompt
    except (UnicodeError, OSError, TypeError) as e:
        print(
            f"warn: custom system prompt could not be loaded because of the following: {e}, using default prompt"
        )

    message_histry: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": cli_args.p},
    ]

    while True:
        with client.chat.completions.stream(
            model=MODEL_NAME,
            reasoning_effort=REASONING_EFFORT,
            messages=message_histry,
            tools=AGENT_TOOLS,
        ) as stream:
            for event in stream:
                if event.type == "content.delta":
                    print(event.delta, end="", flush=True)

            completion = stream.get_final_completion()

        if not completion.choices or len(completion.choices) == 0:
            raise RuntimeError("no choices in response")

        response = completion.choices[0].message
        if not response.tool_calls:
            break

        message_histry.append(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "function": tool_call.function.model_dump(exclude_none=True),
                        "type": tool_call.type,
                    }
                    for tool_call in response.tool_calls
                ],
            }
        )
        for tool_call in response.tool_calls:
            message_histry.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": execute_tool_call(
                        tool_call.function.name, tool_call.function.arguments
                    ),
                }
            )


if __name__ == "__main__":
    main()
