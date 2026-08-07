# STATE

Session handoff. Volatile — update at the end of every §11 task. Hard cap 60 lines.

## Current task

**C5 — `llm.py`: cache, ledger, hard cap.** Not started; `research/src/fcesreg/llm.py` does not
exist. Depends on A3, which passed.

Done in Phase C: C1 `embed.py`, C2 `blocking.py`, C3 `dedup.py` (Exact + Tfidf), C4 `degrade.py`.
`metrics.py` exists with `prf1` only — C8 still owes `macro_weighted_f1`, `confusion` and
`operating_point.py`.

## Blocked or waiting on the supervisor

- **200-pair distractor verification.** `annotation/judge_distractors.py` is built and smoke-tested;
  `annotation/labels/distractor_judgements.jsonl` does not exist yet. The judgements are the
  supervisor's to make (`make judge-distractors`). **C6 and `run_transfer.py` consume this set and
  must not run until it is complete.**
- **`make data` is broken**, not merely unbuilt: it calls `research/scripts/build_taxonomy.py`,
  which does not exist, while the README presents it as step two of reproduction.
- **`make annotate` is broken**: it calls `annotation/annotate.py`, which does not exist. This is on
  the critical path for RQ3 — the exercise produces both the label-noise estimate and
  `mean_seconds_per_item`.
- **Open decision:** is separator-blindness at key level intended? `normalise_key` collapses
  `1.5kW`/`1,5 kW` to one string, so the exact-match baseline is immune by construction to one of
  the seven injected error classes. Raised twice, never ruled on.
- **Conflict, unresolved:** the agreed page budget (three tables, two figures, `T8_cost` to prose)
  was never written into PROJECT_PLAN.md. §10 still specifies `T4_abtbuy.tex`,
  `T6_classification.tex` and `T4_cf_sweep.tex` in full, and §14 records no such amendment.

## Next three tasks in §11 order

1. **C5** — `llm.py`. Criterion: a $0.20 pilot runs; re-running the identical set costs exactly
   $0.00 with every row logging `cache_hit=true`; ledger rows land in the one global
   `results/ledger.jsonl` carrying `run_id`.
2. **C6** — Embedding + Cascade matchers. *Blocked on the verified distractor set.*
3. **C7** — `classify.py`, three conditions (§10.1), or **C8** — `metrics.py` + `operating_point.py`,
   which is unblocked and needs only C3.

## Gotchas that are not plan amendments

- TeX is installed but `/Library/TeX/texbin` is not on PATH; `make paper` prepends it.
- `make experiments` names runners that do not exist yet. For that target only, a missing script is
  the build order rather than a bug — this does **not** excuse `make data` or `make annotate`.
- Blocking figures currently in `results/runs/` are **severity-free**, measured on undegraded
  records. Re-run blocking across the severity range now that C4 exists, and do not compare the two
  sets silently.
- Corpus B pair completeness at `t=8` is far below the 0.98 floor at every severity, and
  non-monotonic. Both are findings; neither is refitted.
- Full background, dead ends and rejected alternatives: `.claude/docs/session-knowledge.md`.

## Last verified

**2026-08-07** — `make test`: 261 passed, 2 warnings. `make paper`: clean build, 6 pages.
Working tree clean at this commit.
