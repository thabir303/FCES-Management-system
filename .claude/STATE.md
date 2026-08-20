# STATE

## RUN THIS FIRST, EVERY DAY

    .venv/bin/python research/scripts/run_dedup.py --config research/configs/dedup.yaml --corpus abtbuy --cascade

Exit **code 2** = quota spent = success. **Never clear `.cache/llm`** — it makes replay
free. Verify by **live (non-cache) ledger rows today**; CPU runners do not substitute.
**Missed: 2** (2026-08-16, 2026-08-19). **1,204 of 1,209 banked, 5 remain** — finishes on
tomorrow's run. Report missed days and the projected finish in the first line of each session.
Volatile handoff — update after every §11 task. Hard cap 60 lines.

## Current task
**Waiting on supervisor rulings**: revised cut ledger to ~230/~245 (owed, not yet sent);
what the transfer section should say now the cap sweep is attributed; whether the system
build continues past D3/E5.

**Cap sweep done and attributed** (`run_blocking_cap-20260820T093302-a3c575e`). Corpus A
flat **0.985** across every cap. Corpus B: 0.411 (500) → 0.885 (2000) → 0.979 (10000) →
**0.982 (none)** — converges within 0.003 of Corpus A. Short titles constant **850 (1.6%)**
at every cap, confirming that mechanism is cap-independent. **The −0.575 gap at the shipped
500-cap config is a guard-scale artifact, not domain shift.** `T9_transfer` still unwired
pending the ruling on what External Validation should say with this now in hand.

**System built to the ruled line** (`c652b14`): schema (D1, verified upgrade/downgrade
clean against real Postgres), auth+roles (D2), asset CRUD+search (D3), pipeline boundary
+import wizard+review queue (E1–E5). 22 tests, all against a live DB. Three real bugs found
verifying against Postgres rather than mocking: enum columns typed `String` against native
Postgres enums; `GENERATED ALWAYS` columns needed `Computed()` or the ORM tried to write
them; `Settings.storage_root` was a bare relative default, silently refitting the cached
classifier depending on cwd — fixed, anchored to repo root.

**`build_cf` fixed 2026-08-20** — negatives remap onto `(i::a, j::b)`; noise parity
confirmed, test pins it. **T6_classification generated** from a verified-clean run.

## Blocked or waiting on the supervisor
- **Page budget target ~230–245** (bibliography + External Validation growth uncosted).
  8 approved cuts −71; 6 of 9 outside candidates approved for −98 more. Revised ledger owed.
- **Open:** separator-blindness in `normalise_key`.
## Next tasks
1. `run_costs.py` / T8_cost — build after tomorrow's cascade completion.
2. RQ2 division-level LLM condition — still uncosted, needs the RAG prompt built.
3. Wire `T9_transfer` once the supervisor rules on the External Validation wording.

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
- `system/`: run `docker compose up -d db` before `alembic`/tests; seed order is
  `seed_users.py` then `seed_categories.py` (categories FK'd by assets.cpv_code).

## Last verified
**2026-08-20** — `make test`: research 486+7 skipped, system/api 22, annotation 29, all
clean. `make paper` clean, 8 pages.
