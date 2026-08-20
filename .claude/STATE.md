# STATE

## RUN THIS FIRST, EVERY DAY

    .venv/bin/python research/scripts/run_dedup.py --config research/configs/dedup.yaml --corpus abtbuy --cascade

Exit **code 2** = quota spent = success. **Never clear `.cache/llm`** — it makes replay
free. Verify by **live (non-cache) ledger rows today**; CPU runners do not substitute.
**Missed: 2** (2026-08-16, 2026-08-19). **904 of 1,209 banked** (127+282+800). Report
missed days and the projected finish in the first line of each session.
Volatile handoff — update after every §11 task. Hard cap 60 lines.

## Current task
**Waiting on three supervisor rulings** (measurement done, nothing applied): the
Methodology cuts; whether `build_cf` is fixed; what the cap sweep says about the transfer.

**C7.** Division macro F1 0.759 tfidf / 0.709 embed; class **0.560** / 0.410 over 74 genuine
classes, **25.8% routed to review**. TF-IDF wins 69 of 74. 4431 ("Wire products") collision
propagated: P 0.359 R 0.301, 28 of 78 scored correct on IT kit.

**G9.** sev 0.0 floor 0.95 → **0.934 automated, 84 of 206 duplicates lost, ceiling R 0.592**
vs cascade 0.553; floor 0.99 → 0.644, 16 lost, ceiling 0.922. Leverage moves with severity.

**G10.** Transfer carried entirely by pair completeness: A 0.985 → B **0.411**, **not yet
attributable** (Corpus B saturates `max_block_size=500`). Recall transfer **unmeasurable**.

## Blocked or waiting on the supervisor

- **`build_cf` is broken and confounded — measured, not asserted.** Negatives resolve
  **0/943** in the degraded frame (the Corpus B sweep crashes), and sit **0.000** from
  source at every severity against positives at 0.665 (sev 0.25) / 0.801 (sev 0.5). Scores
  invert: at sev 0.25 positives mean **0.169**, negatives **0.673**. No valid recall figure
  until fixed.
- **Page budget: cut ~200 lines.** Methodology is 463 of 839 (55%); eight cuts total −71,
  rest from Introduction (117), Related Work (76), Corpus A/B prose. **Open:**
  separator-blindness in `normalise_key`.

## Next tasks

1. **System, stopping at the ruled line** — schema incl. six review-queue logging fields,
   asset CRUD + list/detail, bulk import wizard, review queue. QR, floor plan, reminders,
   role UI, audit browser are 14 Sep: not before the report ships.
2. **RQ2 language-model condition** — class n=350 (ruled); division n **still uncosted**,
   needs `RagFewShotLLMClassifier` built to measure rather than estimate.
3. **`run_costs.py` does not exist** — T8_cost has no runner, record or builder.

## Gotchas that are not plan amendments

- **`make_tables.py` builds only T1_* and T3**; T4/T6/T8/T9 lack builders, and it refuses
  any run made against a dirty tree — an uncommitted `main.tex` marks every run.
- **`run_dedup` exits 2 on quota before `write_run`**, so no run record exists yet; T4
  arrives on the first day the whole sweep replays from cache.
- Quota: 200k/day → **~300 adjudications/day** at 638 tok; RQ2 is 784 tok/record at k=12,
  ~1,421 with all 74 codes. TeX at `/Library/TeX/texbin`; `make paper` prepends it.
  **`make data` is a real break** — no `build_taxonomy.py`.
- Corpus A pair completeness collapses 0.985 → 0.248 → 0.049 with **zero** blocks dropped:
  key failure, not capping. Naive floor **0.000 on Corpus A** at every severity.

## Last verified
**2026-08-20** — `make paper` clean, 8 pages; `make test` 478 + 5 skipped, 29 annotation.
