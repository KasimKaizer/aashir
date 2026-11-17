# Roadmap

Rules for checkpoint work. Read this before implementing a roadmap task.

`plan/README.md` indexes the local roadmap and checkpoint specifications. Git ignores `plan/`. Do not stage or commit it unless asked.

- Read the relevant checkpoint and its prerequisites before coding. Work on one checkpoint at a time, in checkpoint order. An optional checkpoint may be deferred when the roadmap says it does not block the milestone.
- Treat a checkpoint acceptance scenario as work to verify. It is not evidence that the feature or test already exists. Turn the scenario into tests and focused checks.
- Follow the checkpoint's completion criteria. A phase is complete only after its demo and the required phase-exit checks pass. Report manual checks separately from automated tests when a real terminal, platform, or editor is needed.
- Do not implement later roadmap work in the current checkpoint. If the plan is absent or inaccessible, ask for the intended scope.
- Durable conventions live in [AGENTS.md](../../AGENTS.md) and the other files under `docs/agents/`. Capability status, checkpoint acceptance details, and other changing roadmap information stay in `plan/`. Update this guide when a durable convention changes, not when implementation status changes.
