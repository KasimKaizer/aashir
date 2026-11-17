# Python

Rules for Python in this repository. Read this before editing Python.

## Types

- Annotate public and cross-module function parameters and return values, including explicit `-> None` and optional results such as `T | None`.
- Avoid `typing.Any` unless absolutely necessary.
- Declare type aliases with the PEP 695 `type` statement (`type AgentEvent = ...`), not a plain assignment.
- Parse untrusted data at the boundary into a typed shape. Do not carry raw dictionaries or `Any` through internal APIs.
- Treat `cast()` and `# type: ignore` as exceptional: fix the type or dependency where possible, and explain any unavoidable workaround.
- For closed enums and unions, use exhaustive `match` statements with `assert_never` where the project's Python version and type checker support it. Keep ordinary boolean and predicate checks as `if` statements.

## Files and imports

- Prefer `Path.read_text()` and `Path.write_text()` over `open()` when they fit the operation.
- Specify UTF-8 where encoding matters. Pass `newline=""` when exact line endings must be preserved.
- Prefer `with` or `async with` for a resource that supports a context manager. When it does not, clean it up on every path.
- Keep imports at the top of the file. Do not use inline imports.
- Check the installed SDK, its type definitions, or its implementation for an external API shape. Do not guess.
- Load packaged resources with `importlib.resources` rather than a path beside the source file.

## Models

Boundary input and output, including configuration that enters from outside the process, stays a `pydantic.BaseModel`. Use `TypeAdapter` when validating, serializing, or generating a schema for an existing type is simpler than adding a `BaseModel`.

In-process values use a standard-library dataclass when static typing and explicit invariants are enough. Keep the Pydantic model after boundary validation. Do not convert it to a dataclass, and do not create a second wire or domain type, for stylistic consistency.

Validate untrusted data at meaningful boundaries. Configure coercion, extra fields, and assignment validation intentionally. A Pydantic model or a JSON Schema does not by itself prove the data was validated.

Value and configuration objects are frozen unless mutation is part of their purpose. Dataclasses use `frozen=True`. Pydantic models use `ConfigDict(frozen=True)`, as `Settings` and tool-argument models do. `frozen=True` does not freeze nested lists or dictionaries. Use `slots=True` when preventing arbitrary per-instance attributes or reducing instance memory is useful.

Raised and caught errors are `Exception` subclasses, not Pydantic models. A change to an exported data type is an API change.

## Errors

- Define custom errors as `@dataclass(frozen=True, slots=True)` `Exception` subclasses with typed fields and an `@override` `__str__`, as `ProviderError` does. Do not use a bare-string `__init__`.
- Mark module-level constants `Final`.
- Do not catch an exception merely to suppress it. Let an unexpected failure propagate. Catch narrowly when the code has a defined recovery, needs to add useful error context, or must return a defined tool or protocol error. Do not use a broad catch that hides a provider or runtime failure.
- When adding context while re-raising an exception, preserve its cause so the original failure remains available.

## Comments and docstrings

- Write a `#` comment only for a non-obvious reason or caveat the code cannot express, such as a platform limit. Put a caveat that spans several lines in one comment or docstring.
- That comment states what happens and the consequence, in precise technical terms a human engineer can check. Example: "After a crash, the replacement may be lost."
- Module docstring: a one-line noun phrase, as in `"""The ``read_file`` tool."""`. Class docstring: one line. Simple helper: a one-line summary. Non-trivial function: Google-style `Args`, `Returns`, `Yields`, and `Raises` sections, imperative mood ("Convert...", "Process...").

## Changes

- Avoid a single-line helper with one call site when it only wraps one expression. Inline it. Keep a helper when its name makes the behavior easier to understand.
- Do not add a dependency or an abstraction without a concrete need.
- Do not remove or downgrade code to silence a type error from an outdated dependency. Check compatibility and update the dependency.
- Treat a dependency change as code. Update `pyproject.toml`, regenerate `uv.lock` with `uv`, and review both diffs. Before an intentional upgrade, check compatibility and the relevant release notes.
- Do not run package installation or build hooks unless the task calls for them or the user asks. `uv run` commands required by the task or the project checks are allowed.
- For an ad-hoc script, write it to a temporary file, run that file, and remove it. Do not embed a multi-line script in a shell command.
- Do not append to the README unless asked. Update existing interface and configuration documentation when a change would make it false.
- When a feature emits generated Python, change the generator and regenerate. Do not edit generated files directly.
- Keep terminal-UI key checks in the keybinding configuration, with defaults. Do not hardcode them.
