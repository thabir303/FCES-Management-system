# STATE

Session handoff. Volatile — update at the end of every §11 task. Hard cap 60 lines.

## Current task

No §11 task in progress. The 50-pair distractor sample is judged — by Claude, per the
supervisor's explicit delegation (the model under test must never adjudicate its own negative
set). `annotation/labels/distractor_judgements.jsonl` written, one reason per pair naming the
signal: **42.0% contaminated (Wilson 95% CI 29.4%–55.8%), n=50; unsure 4/50 = 8%**, under the
15% stop threshold. This is v3's *first* measured rate — not comparable to v2's 35% (different
rule, different sample; earlier eyeballing was explicitly declined for the same reason, see
session-knowledge.md). Pool (570) used downstream unfiltered; nothing dropped on this rate.

C5 (`llm.py`) and the `embed.py` cache-dir fix both closed in prior sessions.

Done in Phase C: C1–C5. `metrics.py` has `prf1` only — C8 still owes `macro_weighted_f1`,
`confusion`, `operating_point.py`.

## Blocked or waiting on the supervisor

- **Paper edit pending, supervisor's.** `main.tex` §Methodology promises hand verification of
  a bounded set — false under amendment 7. Both inputs now settled: sample size (50) and rate
  (42.0%, CI 29.4%–55.8%). Corpus B precision to be reported as a lower bound alongside this
  rate — never corrected by it arithmetically.
- **`make data`/`make annotate` are both broken**, not unbuilt: call
  `research/scripts/build_taxonomy.py` and `annotation/annotate.py`, neither exists. Latter is
  RQ3's critical path (label-noise estimate, `mean_seconds_per_item`).
- **Open:** is separator-blindness at key level intended? `normalise_key` collapses
  `1.5kW`/`1,5 kW`, immunising exact-match to one of seven error classes by construction.
- **Unresolved:** the agreed page budget (three tables, two figures, `T8_cost` to prose)
  was never written into PROJECT_PLAN.md; §14 records no such amendment.

## Next three tasks in §11 order

1. **C8** — `metrics.py` + `operating_point.py`. Unblocked, needs only C3.
2. **C7** — `classify.py`, three conditions (§10.1). Needs C1, C5, B3.
3. **C6** — Embedding + Cascade matchers. **Unblocked** — rate measured (amendment 7);
   negative set carries a known 42.0% impurity into Corpus B precision, unfiltered.

## Gotchas that are not plan amendments

- Quota arithmetic, not measured: ~710 tokens/adjudication against 200k tokens/day is about
  **280/day**, ~11/minute. Verify against `x-ratelimit-*` at runtime; `limit-requests` is per
  *day*, `limit-tokens` per *minute* — RPM and TPD are in no header.
- TeX is installed but `/Library/TeX/texbin` is not on PATH; `make paper` prepends it.
- `make experiments` names runners that don't exist yet — build order, not a bug, for that
  target only. Does **not** excuse `make data` or `make annotate`.
- Blocking figures in `results/runs/` are **severity-free** — re-run now C4 exists, don't
  compare the two sets silently.
- Corpus B pair completeness at `t=8` is far below the 0.98 floor, non-monotonic — both
  findings, neither refitted.
- Full background, dead ends, rejected alternatives: `.claude/docs/session-knowledge.md`.

## Last verified

**2026-08-08** — `make test`: 295 passed, identical from the repository root, `research/`
and `/tmp`, plus `annotation/test_judge_distractors.py`: 13 passed. `make paper` untouched
this session. `annotation/judge_distractors.py --summary` reproduces 42.0% (CI 29.4%–55.8%)
against the committed judgements file.
