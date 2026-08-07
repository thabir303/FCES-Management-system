---
name: record-run
description: Produce a number that will reach the paper — confirm a clean git tree, run the config-driven runner, confirm results/runs/<run_id>/ was written, regenerate tables, confirm the run_id reached the generated table, then commit. Use whenever a measurement is intended for main.tex or a report.
---

# Recording a reportable number

Every figure in the paper is `\input{}` from `results/tables/`, generated from
`results/runs/<run_id>/`. This is the only path a number may take.

## 1. Clean the tree first

```bash
git status --porcelain
```

Must be empty. `capture_env()` records `git_dirty`, and `make_tables.py` **refuses** a run made
against a dirty tree. Commit the code, then run — not the other way round. A number that cannot be
traced to a commit does not go in the paper.

## 2. Run the runner

```bash
.venv/bin/python research/scripts/run_<x>.py --config research/configs/<x>.yaml
```

Runners are config-driven and take no tuning flags. If a parameter needs changing, change the
config and commit it — a flag that alters a measured outcome is a way to tune without leaving a
trace.

## 3. Confirm the run artefacts

```bash
ls results/runs/<run_id>/     # params.yaml metrics.json env.json [predictions.parquet]
```

There is **no per-run ledger**. LLM spend goes to the single global `results/ledger.jsonl`, every
row carrying `run_id`.

## 4. Regenerate the tables

```bash
make tables
grep -l "<run_id>" results/tables/*.tex
```

The `run_id` must appear in the generated table — as the header comment and in the caption. If it
does not, the table was built from an older run and the paper would cite a number that no longer
matches the code. **Never hand-edit `results/tables/*`**; a hook blocks it. Change the builder in
`make_tables.py` and regenerate.

## 5. Commit

The run directory and the regenerated tables go in together, with the measured figures in the commit
body so the number is greppable from `git log`.

## The rule that matters most

**A disappointing measurement is a finding, and is reported as it stands.** Do not tune a threshold,
refit on the evaluation split, narrow a sweep, or drop a condition because a number came out lower
than hoped. Precedents where a poor number was the result: exact matching at F1 0.000 on Abt-Buy,
the single-key blocking formulation at PC 0.171, Corpus B pair completeness far below its floor at
every severity. Each stands in the record. If a result invites you to adjust the method, stop and
raise it with the supervisor instead.
