# Testing

Rules for tests and checks. Read this before writing a test or running the phase-exit suite. Passing claims belong in [AGENTS.md](../../AGENTS.md): do not claim a check passed unless it was run and it passed.

- Use `pytest`, not `unittest`.
- Structure every test with explicit Given, When, and Then sections (`# Given`, `# When`, `# Then`): set up preconditions, perform one action under test, and assert its observable outcome. Split a test that needs more than one When action.
- Do not mock or patch anything unless the task or these rules call for it. Provider tests use a deterministic mocked HTTP stream at the boundary between the provider client and the external service. Do not call a real provider API, use a real API key, load real credentials from `.env` or the ambient environment, or spend paid tokens.
- Use pytest fixtures for process inputs such as the working directory or the test environment. Do not replace application behavior for convenience. A focused test may replace a side-effect boundary with a fail-fast function when the assertion is that the boundary is not reached.
- For CLI tests, supply arguments and configuration through their normal entry points. Prefer real local filesystem behavior with `tmp_path`. Use the provider client's HTTP boundary only when the test needs it.
- Keep each test focused on one behavior. It should fail when that behavior regresses, rather than restating implementation details.
- Keep tests deterministic and isolated. Avoid sleeps, wall-clock assumptions, and shared fixed paths.
- For `pytest.mark.parametrize`, pass a tuple as the first argument and a list of cases as the second.
- Prefer a direct assertion for a simple one-off call. Name the result when that improves clarity, reuse, failure diagnostics, or a multi-part assertion.
- After creating or modifying a test file, run `uv run pytest path/to/test_file.py` from the repository root and iterate until it passes. When changing behavior, run the affected tests and check observable results, protocol behavior, failure cases, and side effects.
- After a code change, other than a documentation-only change, run these from the repository root and fix every finding:

  ```sh
  uv run ruff check .
  uv run pyrefly check
  uv run pyrefly coverage check
  ```

- At roadmap phase exit, also run the full suite:

  ```sh
  uv run pytest
  uv run ruff check .
  uv run pyrefly check
  uv run pyrefly coverage check
  ```
