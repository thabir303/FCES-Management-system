# STATE

## RUN THIS FIRST, EVERY DAY

    .venv/bin/python research/scripts/run_dedup.py --config research/configs/dedup.yaml --corpus abtbuy --cascade

Exit **code 2** = daily quota spent = success. Re-run tomorrow. **Never clear
`.cache/llm`** — it is what makes replay free. A missed day cannot be recovered;
the quota does not roll over.

**Missed so far: 1** (2026-08-16). 605 of ~1,600 adjudications banked.
Report missed days and the projected finish in the first line of each session.

Session handoff. Volatile — update at the end of every §11 task. Hard cap 60 lines.


## Current task

**C6, C8 closed.** Wilson floor adopted (one-sided 95%) in `select_threshold` and
`automated_share_at_precision`: 0.95 needs ≥52 accepted items, 0.99 needs 268, 1.0 is
unreachable at any n. Amendment 8: **Corpus B carries no precision or F1** (42.0%
contamination, CI 29.4%–55.8%); Corpus A carries precision, F1, RQ3.

**Cascade results so far** — sev 0.0: band 6.6%, R 0.553 P 0.983. sev 0.15: band 14.7%,
R 0.417, **P combined 0.851 — below the 0.95 floor**. The threshold constrains only what
it auto-accepts; nothing constrains the adjudicator, so both precisions are now reported
per severity. Where upper is undefined nothing is auto-accepted and the combined figure
is entirely the adjudicator's.

## Blocked or waiting on the supervisor

- **Handling time needs the author**: `annotation/annotate.py --timing-only 15`, ~8 min.
  Label noise is done (13.2%, CI 5.8%–27.3%, n=38 decided, judged by Claude). A model
  must not time its own reading — that measures LLM latency, and RQ3 derives residual
  hours from it. `make data` is still broken: `build_taxonomy.py` does not exist.
- **Open:** separator-blindness in `normalise_key`; page budget in no amendment.

## Next three tasks in §11 order

1. **C7 `classify.py` + `run_classify.py`** — RQ2 has NO implementation. Highest
   priority. Classical pair (TF-IDF+SVM, embed+logreg) costs no quota; LLM condition
   **n=500** (ruled), ~2.0 days, waits behind the sweep.
2. **`run_transfer.py`** — recall + pair completeness only (amendment 8).
3. **Amendment 9** — §10 omits the Corpus A sweep and still puts the cascade in the cf
   sweep; the paper now says Corpus A only. Plus the naive-floor write-up.

## Gotchas that are not plan amendments

- Quota: 200k/day binds before 1000 requests/day → **~310 adjudications/day** at 638
  tok each; RQ2 classification is 784 tok/record. Cascade sev 0.5 is subsampled to
  **m=800** stratified by label (ruled) — full band would cost 6.1 days against 2.6.
- TeX at `/Library/TeX/texbin`, not on PATH; `make paper` prepends it. `make
  experiments` names unbuilt runners (build order); `make data` is a real break.
  Blocking figures in `results/runs/` are **severity-free** — don't compare silently.
- Corpus B pair completeness at `t=8` far below the 0.98 floor, non-monotonic — both
  findings, neither refitted. `F1_severity.pdf` has **no Corpus A panel** as specified.
- Full background, dead ends, rejected alternatives: `.claude/docs/session-knowledge.md`.

## Last verified

**2026-08-15** — `make test`: 391 passed + 2 skipped (research), 26 (annotation), from
the repository root, `research/` and `/tmp`. `make paper`: clean, 7 pages.
`judge_distractors.py --summary` still reproduces 42.0% (CI 29.4%–55.8%) after
`wilson_interval` moved into `fcesreg.metrics`.
