# Aashir

Aashir is an AI agent for software-engineering tasks. In code, readability takes precedence over brevity. "Smallest" means the least unnecessary complexity. Keep useful names and intermediate steps.

## Commands

- Package manager: `uv`. Use the Python version declared in `pyproject.toml`.
- CLI: `argparse`. Do not switch CLI frameworks without a concrete project decision.
- After a code change, from the repository root, run `uv run ruff check .`, `uv run pyrefly check`, and `uv run pyrefly coverage check`, plus the affected tests. The phase-exit suite is in [Testing](docs/agents/testing.md).

## Every task

- Read the linked file for a task before changing the behavior it covers.
- Answer a question before editing or running implementation commands. On feedback, say whether you agree or disagree before describing the change.
- No emojis in commits, issues, PR comments, or code.
- Ask before removing functionality or intentional code, including a compatibility break.
- Before editing, check `git status` when practical and preserve existing staged and unstaged work.
- Do not commit unless asked. Do not stage or commit `plan/` unless asked. Stage only explicit paths changed in this session.
- Never run `git reset --hard`, `git checkout .`, `git clean -fd`, `git stash`, `git add .`, `git add -A`, `git commit --no-verify`, or a force-push.
- Never expose API keys or other secrets in output, tests, or commits. Do not include `.env` contents in prompts or tracked files. Use `.env.example` for variable names.
- Do not claim a test or check passed unless it was run and it passed. Report a check that could not be run.
- This guide grants no permission to perform side effects and does not replace higher-priority tool permissions or platform restrictions.
- If the user's instructions conflict with a rule in this guide, explain the conflict and ask before overriding it. Do not use that confirmation to bypass a permission or verification gate.

## When the work needs it

- [Conversation](docs/agents/conversation.md) — explaining a non-trivial design.
- [Python](docs/agents/python.md) — editing Python: types, models, errors, comments, and dependencies.
- [Package API](docs/agents/api.md) — changing public imports or provider-visible schemas.
- [Testing](docs/agents/testing.md) — writing tests or running the phase-exit suite.
- [Git](docs/agents/git.md) — committing, reviewing, or rebasing.
- [Roadmap](docs/agents/roadmap.md) — checkpoint or `plan/` work.
- [Safety](docs/agents/safety.md) — changing tool, workspace, snapshot, publish, or integration behavior.

Durable conventions live in this guide. Capability status and checkpoint acceptance details stay in `plan/`.
