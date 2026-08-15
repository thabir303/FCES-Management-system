# STATE

Session handoff. Volatile — update at the end of every §11 task. Hard cap 60 lines.

## Current task

**C8 and C6 both closed.** `metrics.py`, `operating_point.py`, `EmbeddingMatcher`,
`CascadeMatcher`, `adjudicate.py`, `run_dedup.py`, `annotate.py`. **The Wilson floor is
adopted**: a threshold meets a precision target when the lower bound of a *one-sided 95%*
Wilson interval reaches it, in both `select_threshold` and `automated_share_at_precision`.
Consequences, tested not special-cased: 0.95 needs ≥52 accepted items, 0.99 needs 268,
and target 1.0 is unreachable at any n.

Amendment 8 landed (paper + plan, separate commits): **Corpus B carries no precision or
F1**, measured contamination 42.0% (CI 29.4%–55.8%). Corpus A carries precision, F1, RQ3.

## Blocked or waiting on the supervisor

- **Corpus A cascade sweep: RE-RUN THIS DAILY.** `.venv/bin/python
  research/scripts/run_dedup.py --config research/configs/dedup.yaml --corpus abtbuy
  --cascade`. Exit **code 2** = `DailyQuotaExhausted` = success; re-run tomorrow.
  Cached calls replay free — **never clear `.cache/llm`**. 305 of 2,325 done
  (2026-08-15, 199k/200k tokens); at a measured 638 tok/adj, **~6.4 days left**.
  Severity 0.0 is complete: band 127/1916 (6.6%), R 0.553 P 0.983 F1 0.708, against
  TF-IDF alone at R 0.155 P 0.970. No run record is written on a quota stop, so
  `results/runs/` stays empty until the sweep finishes — that is deliberate.
- **40-item annotation exercise is the author's to run.** `annotation/annotate.py` is
  built and `make annotate` works; nobody has judged the 40 items yet. Judged by a
  person, never a model — RQ2 measures a model on this task. Produces
  `mean_seconds_per_item` **and** the label-noise rate in one pass. `make data` is still
  broken: `research/scripts/build_taxonomy.py` does not exist.
- **Open:** separator-blindness at key level — `normalise_key` collapses `1.5kW`/`1,5 kW`,
  immunising exact-match to one error class by construction. Page budget: in no amendment.

## Next three tasks in §11 order

1. **`run_transfer.py`** — recall + pair completeness only (amendment 8).
2. **C7** — `classify.py`, three conditions (§10.1). Needs C1, C5, B3.
3. **Plan amendment 9** — §10 still omits the Corpus A sweep and still puts the cascade
   in the cf sweep; the paper now says the cascade is Corpus A only.

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

**2026-08-15** — `make test`: 391 passed + 2 skipped (research), 26 (annotation), from
the repository root, `research/` and `/tmp`. `make paper`: clean, 7 pages.
`judge_distractors.py --summary` still reproduces 42.0% (CI 29.4%–55.8%) after
`wilson_interval` moved into `fcesreg.metrics`.
