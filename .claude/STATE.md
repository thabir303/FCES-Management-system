# STATE

Session handoff. Volatile — update at the end of every §11 task. Hard cap 60 lines.

## Current task

No §11 task in progress. Two out-of-band fixes this session, each its own commit:
`embed.py`'s `DEFAULT_CACHE_DIR` was cwd-relative (same defect class as C5's fix below); the
distractor tool now draws a 50-pair random sample with a Wilson-interval contamination rate
(was: hand-verify a 200-pair bounded pool), reported alongside precision rather than used to
filter the negative set (amendment 7). `degrade.py`'s `max_pairs` comment, which stated the old
verification rationale, is corrected to a mining-cost cap.

C5 — `llm.py` — **closed** last session (G1, Groq `openai/gpt-oss-120b` free tier). Live
pilot passes; ledger evidence committed.

Done in Phase C: C1 `embed.py`, C2 `blocking.py`, C3 `dedup.py` (Exact + Tfidf), C4
`degrade.py`, C5 `llm.py`. `metrics.py` has `prf1` only — C8 still owes
`macro_weighted_f1`, `confusion` and `operating_point.py`.

## Blocked or waiting on the supervisor

- **Distractor sample not yet run.** `annotation/labels/distractor_judgements.jsonl` doesn't
  exist. 50 judged pairs now, not 200 — ~12 min. **C6, `run_transfer.py` still wait** — a
  measured rate now, not a clean set (amendment 7), so the requirement changed, not the block.
- **Paper edit pending, supervisor's.** `main.tex` §Methodology promises hand verification of
  a bounded set — false under amendment 7. Supervisor will edit once settled; size now is (50).
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
3. **C6** — Embedding + Cascade matchers. *Blocked on the 50-pair contamination sample (amendment 7).*

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

**2026-08-08** — `make test`: 295 passed, 2 warnings, identical from the repository root,
`research/` and `/tmp`, plus `annotation/test_judge_distractors.py`: 13 passed. `make paper`
untouched this session (last clean build 6 pages). `make llm-pilot`: all 8 checks pass live.
