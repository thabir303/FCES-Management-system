# STATE

Session handoff. Volatile — update at the end of every §11 task. Hard cap 60 lines.

## Current task

**C8 closed** — `metrics.py` gains `macro_weighted_f1`, `confusion`, `threshold_sweep`;
`operating_point.py` built (curve, `automated_share_at_precision`, `residual_effort`),
criterion met. Two bugs found and fixed in their own commits: `select_threshold` split
tie groups and promised precision it did not deliver (worst on `ExactMatcher`, which
scores only 1.0/0.0); pandas nulls are truthy, so degradation crashed on Corpus A and
`merge_fields` silently planted `"nan"` in titles.

Amendment 8 landed (paper + plan, separate commits): **Corpus B carries no precision or
F1**, measured contamination 42.0% (CI 29.4%–55.8%). Corpus A carries precision, F1, RQ3.

## Blocked or waiting on the supervisor

- **STOP before starting the sweep. `select_threshold` can pick a threshold supported by
  one pair.** Its `tp > 0` guard let dev claim ≥0.95 at severity 0.75 on a 1-pair
  threshold that delivers **precision 0.000 on test**; at 0.3, 14 pairs → 0.800. It sets
  the cascade's `upper`, so a sweep started now burns quota on a degenerate bound. The
  apparent non-monotone cliff was this artefact: under any support floor the target is
  unreachable **between severity 0.1 and 0.25**, not 0.5. Candidate rules and what each
  costs are in `session-knowledge.md` §3 — each changes a reported number, so the choice
  is the supervisor's. Protocol A confirmed; cost stands at ~7.4 days for 5 severities,
  to be recosted at 3 once the rule is settled.
- **40-item annotation tool not built.** `annotation/annotate.py` absent, `make annotate`
  broken. Author judges it (label noise must not go to a model RQ2 measures); produces
  `mean_seconds_per_item` + label-noise rate in one pass. `make data` is broken too —
  `research/scripts/build_taxonomy.py` does not exist.
- **Open:** is separator-blindness at key level intended? `normalise_key` collapses
  `1.5kW`/`1,5 kW`, immunising exact-match to one of seven error classes by construction.
- **Unresolved:** the agreed page budget (3 tables, 2 figures, `T8_cost` to prose) is in no amendment.

## Next three tasks in §11 order

1. **C6** — Embedding + Cascade matchers. Unblocked (rate measured, amendment 7).
2. **C7** — `classify.py`, three conditions (§10.1). Needs C1, C5, B3.
3. **`run_dedup.py`** — both sweeps. *Blocked on the threshold-protocol decision above.*

## Gotchas that are not plan amendments

- Quota: 200k tokens/day binds before 1000 requests/day → **~530 adjudications/day** at the
  measured 377 tokens each. `limit-requests` is per *day*, `limit-tokens` per *minute*.
- TeX is installed but `/Library/TeX/texbin` is not on PATH; `make paper` prepends it.
- `make experiments` names runners that don't exist yet — build order, not a bug, for that
  target only. Does **not** excuse `make data` or `make annotate` being broken.
- Blocking figures in `results/runs/` are **severity-free** — re-run now C4 exists, don't
  compare the two sets silently.
- Corpus B pair completeness at `t=8` is far below the 0.98 floor and non-monotonic —
  both findings, neither refitted. `F1_severity.pdf` has **no Corpus A panel** as
  specified; Corpus A dedup is `T4_abtbuy`.
- Full background, dead ends, rejected alternatives: `.claude/docs/session-knowledge.md`.

## Last verified

**2026-08-11** — `make test`: 347 passed, identical from the repository root, `research/`
and `/tmp`. `make paper`: clean build, 7 pages (amendment 8 text). `judge_distractors.py
--summary` reproduces 42.0% (CI 29.4%–55.8%) against the committed judgements file.
