# Aashir

The Agent

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- An API key and a chat model supported by your chosen endpoint

## Set up

```sh
uv sync
cp .env.example .env
```

Put `OPENAI_API_KEY` in `.env`. Aashir does not discover `.env` files automatically; select this credential-only file with `--env-file`. The file may contain only `OPENAI_API_KEY`. The API key has no command-line flag, so it stays out of shell history. A `OPENAI_API_KEY` in the process environment takes precedence over the file.

Every other setting is a command-line flag or an environment variable. A flag overrides its environment variable, and an empty value counts as unset.

| Flag                       | Environment variable     | Meaning                                                                                     |
| -------------------------- | ------------------------ | ------------------------------------------------------------------------------------------- |
| `--base-url`               | `OPENAI_BASE_URL`        | Required. HTTP(S) base URL of the provider's OpenAI-compatible API.                         |
| `--model`                  | `OPENAI_MODEL`           | Required. Name of the model to use.                                                         |
| `--system-prompt-file`     | `SYSTEM_PROMPT_FILE`     | Path to a text file containing your system prompt.                                          |
| `--reasoning-effort`       | `REASONING_EFFORT`       | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`. Unset sends nothing, so the provider's default applies; `none` explicitly asks for no reasoning. |
| `--tool-argument-byte-cap` | `TOOL_ARGUMENT_BYTE_CAP` | Largest accepted tool-call argument size in UTF-8 bytes. Defaults to 1048576.               |

## Use

```sh
uv run aashir -h
uv run aashir -p "Help me with this project" --env-file .env \
  --base-url https://api.openai.com/v1 --model <model-name>
```
