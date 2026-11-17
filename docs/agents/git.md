# Git

Commit, review, and rebase rules. Read this before committing or rebasing. Commands that must not be run, and the rules to commit only when asked and to stage explicit paths, are in [AGENTS.md](../../AGENTS.md).

- Before committing, check `git status` and the staged diff. Confirm that only the intended files are staged.
- Use this message pattern:
  - `ci:` for tests and CI changes.
  - `dev:` for development and agent-rule changes.
  - `fix(component):`, `feat(component):`, `enh(component):`, and `ref(component):` for bug fixes, features, enhancements, and refactors.
  - `docs:` for documentation.
  - `chore:` for maintenance.
- For a component scope, use an established name when one exists. Otherwise choose a short lowercase name for the affected component and keep using it.
- The description should focus on the intent, not the implementation details.
- Do not add a `Co-authored-by` trailer for an AI tool unless the user asks.
- Before committing, flag a critical issue: an unintended unhandled exception, a logical inconsistency, or a protocol or interface change without a matching implementation update. Do not inflate the review with style nits.

## Rebase conflicts

- Resolve conflicts only in files you modified.
- If a conflict is in a file you did not modify, abort the rebase and ask the user.
