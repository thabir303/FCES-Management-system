---
name: build-task
description: Open and close a PROJECT_PLAN.md §11 build task — check dependencies passed their acceptance criteria, implement, run scoped tests, build the paper if main.tex was touched, commit exactly once, then update .claude/STATE.md. Use when starting or finishing any lettered task (A1, C5, D3, G4…).
---

# Working a §11 task

§11 is the dependency-ordered build table. **Do not start a task until its dependencies have passed
their acceptance criteria** — not merely been written.

## Open

1. Read the task's row in §11: its id, dependencies, and *Done when* criterion.
2. Verify each dependency actually passed. Existence is not passing — a module can exist with its
   criterion unmet. Check the criterion, not the file.
3. Read the §6 contract the task points into. Module docstrings cite their section; keep that.
4. Check `.claude/STATE.md` for anything blocking this task.

## Implement

Follow the §6 contract. Where you must deviate, say so in the docstring **and** amend
PROJECT_PLAN.md in the same commit, so plan and code never drift.

**Acceptance criteria test the implementation, never the outcome.** A criterion may assert a value
is produced, is internally consistent, or falls in a range only a bug could violate. It may never
require a measured result to take a particular value. Where a number appears in a criterion it is a
bug detector: record the measured figure as an observation regardless of where it lands.

## Close

1. **Scoped tests first**, then the full suite:
   ```bash
   .venv/bin/pytest research/tests/test_<module>.py -q
   .venv/bin/pytest research/tests -q
   ```
   The suite must pass from the repository root **and** from `research/` — paths are anchored and
   `test_paths.py` guards it.
2. If `main.tex` was touched, `make paper` must build clean. Paper changes go in their **own**
   commit, never swept in with code.
3. **Exactly one commit per §11 task.** A run's git SHA is worth little if the whole codebase is one
   SHA. Subject names the task id: `C5: llm.py — cache, ledger, hard cap`. The body says what was
   decided and why, not what changed.
4. Update `.claude/STATE.md`: current task, next three, anything newly blocked, and the *Last
   verified* line with today's date and the `make test` / `make paper` results.

## When the criterion cannot be met

Stop and report. Do not weaken the criterion, and do not mark a task done with a caveat. If the
criterion turns out to be impossible as written — this has happened, with B5 — say so, propose the
correction, and let the supervisor rule. `.claude/docs/session-knowledge.md` records the precedents.
