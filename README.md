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

Edit `.env` to set `OPENAI_API_KEY`, `BASE_MODEL_NAME`, and `REASONING_EFFORT` for your provider. Set `OPENAI_BASE_URL` to the provider's API base URL.

`SYSTEM_PROMPT_FILE` is the path to a text file containing your system prompt. You can create `prompt.txt` in the project directory, for example.

## Use

```sh
uv run aashir -h
```
