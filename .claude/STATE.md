# STATE

## RUN THIS FIRST, EVERY DAY

    .venv/bin/python research/scripts/run_dedup.py --config research/configs/dedup.yaml --corpus abtbuy --cascade

Exit **code 2** = quota spent = success. **Never clear `.cache/llm`** — it makes replay
free. Verify by **live (non-cache) ledger rows today**; CPU runners do not substitute.
**Missed: 2** (2026-08-16, 2026-08-19). **1,204 of 1,209 banked, 5 remain** — finishes on
tomorrow's run. Report missed days and the projected finish in the first line of each session.
Volatile handoff — update after every §11 task. Hard cap 60 lines.

## Current task
**Waiting on two supervisor rulings**: the revised Methodology cut ledger (target now
~230, not 200); what the cap sweep says about transfer attribution (running, no output yet).

**`build_cf` fixed and confirmed 2026-08-20.** Negatives now remap onto degraded copies
`(i::a, j::b)`. Resolution 983/983 at every severity (was 0/943); noise parity within
0.002–0.015 of positives (was inverted: negatives flat 0.000 vs positives 0.665–0.801).
Regression test `test_degrade.py::TestBuildCfNoiseParity` pins both properties.

**C7.** Division macro F1 0.759 tfidf / 0.709 embed; class **0.560** / 0.410 over 74 genuine
classes, **25.8% routed to review**. TF-IDF wins 69 of 74. 4431 ("Wire products") collision
propagated: P 0.359 R 0.301, 28 of 78 scored correct on IT kit.

**G9.** sev 0.0 floor 0.95 → **0.934 automated, 84 of 206 duplicates lost, ceiling R 0.592**
vs cascade 0.553; floor 0.99 → 0.644, 16 lost, ceiling 0.922. Leverage moves with severity.

**G10.** Transfer carried entirely by pair completeness: A 0.985 → B **0.411**, **not yet
attributable** (Corpus B saturates `max_block_size=500`). Recall transfer **unmeasurable**.

## Blocked or waiting on the supervisor

- **Page budget target is ~230** (bibliography was omitted from the first count). 8
  approved cuts −71; supervisor took 6 of 9 outside candidates for −98 more (−169 total);
  revised ledger owed from Corpus B prose (346–403), Study Design (305–323), Partitioning
  (614–628). **Open:** separator-blindness in `normalise_key`.

## Next tasks

1. **T6_classification** builder written (`df3907d`); generation pending a genuinely
   clean `run_classify` re-run (the prior "clean" record was mistaken — in progress).
2. **T9_transfer** builder written, unwired until the cap sweep rules on 0.985→0.411.
3. **`run_costs.py` / T8_cost** — build after the cascade completes (tomorrow).
4. System build to the ruled line; RQ2 division-level LLM condition, still uncosted.

## Gotchas that are not plan amendments

- **`make_tables.py` refuses the whole build if the latest `run_blocking` record is dirty**
  (`run_blocking-...-15338713`, pre-existing) — call a builder function directly, not `main()`.
- **`run_dedup` exits 2 on quota before `write_run`**; T4 needs a day where all 1,209 pairs
  are cached and only the live remainder completes inside quota — a cleared cache costs it.
- Quota: 200k/day → **~300 adjudications/day** at 638 tok; RQ2 is 784 tok/record at k=12,
  ~1,421 with all 74 codes. TeX at `/Library/TeX/texbin`; `make paper` prepends it.
  **`make data` is a real break** — no `build_taxonomy.py`.
- Corpus A pair completeness collapses 0.985 → 0.248 → 0.049 with **zero** blocks dropped:
  key failure, not capping. Naive floor **0.000 on Corpus A** at every severity.

## Last verified
**2026-08-20** — `make paper` clean, 8 pages; `research/tests` 486 + 7 skipped.
