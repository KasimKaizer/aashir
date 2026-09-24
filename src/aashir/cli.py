import argparse
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

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
        "SYSTEM_PROMPT_FILE": SYSTEM_PROMPT,
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
        warnings.warn(
            f"custom system prompt could not be loaded because of {e}, using default prompt"
        )

    message_histry: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": cli_args.p},
    ]

    while True:
        chat = client.chat.completions.create(
            model=MODEL_NAME,
            reasoning_effort=REASONING_EFFORT,
            messages=message_histry,
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        response = chat.choices[0].message
        print(response.content, end="")
        break


if __name__ == "__main__":
    main()
