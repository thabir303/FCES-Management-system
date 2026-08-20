# STATE

## RUN THIS FIRST, EVERY DAY

    .venv/bin/python research/scripts/run_dedup.py --config research/configs/dedup.yaml --corpus abtbuy --cascade

Exit **code 2** = quota spent = success. **Never clear `.cache/llm`**. Verify by **live
ledger rows today**; CPU runners do not substitute. **Quota day is UTC**, not local — local
midnight passing does NOT reset it. **Missed: 2** (08-16, 08-19). **1,204/1,209, 5 remain**;
08-21's local-day run replayed 0 live calls (same UTC day as the 20th), resumes at UTC
midnight. Report missed days + projected finish first line of each session. Hard cap 60 lines.

## Current task
Cap sweep attributed AND severity-0.25 domain-shift cell measured. Waiting: revised cut
ledger to ~245; `T9_transfer` wording now both are in hand; system build past D3/E5?

**Sev 0.0** (`run_blocking_cap-20260820T093302-a3c575e`): −0.575 gap at shipped cap 500 is
a guard-scale artifact, but rescaling costs volume — B candidates 36,614 (500) → 1.75M →
23.6M → 44.3M (none), **1,209× at PC 0.982**. Withdrawn: "converges within 0.003 of A" — A's
0.985 is genuinely-different retailer text, B's sev-0.0 positives are byte-identical.

**Sev 0.25, both corpora** (`...-3223ff6-d081d1cc`, clean): only condition where B's
positives actually differ. Rescaled cap 10,000: **PC(A)=0.248, PC(B)=0.441 — B higher**,
reversed from sev 0.0. Candidates(B)=90.9M, guard barely binds. median_grams/title 32 vs 33
— length doesn't explain it; mechanism open. `T9_transfer` unwired pending the ruling.

**System to the ruled line** (`c652b14`): schema/auth/asset CRUD/import wizard/review
queue (D1–D3, E1–E5), 22 tests vs live Postgres. 3 bugs: enum cols typed `String` against
native enums, `GENERATED ALWAYS` needed `Computed()`, `storage_root` cwd-relative (silent
cache miss) — fixed, anchored to repo root. `build_cf` fixed 08-20, noise parity pinned.
`T6_classification` generated clean.

## Blocked or waiting on the supervisor
- **Page budget ~230–245.** 8 approved cuts −71; 6/9 outside candidates −98 more. Ledger
  owed. **Open:** separator-blindness in `normalise_key`.

## Next tasks
1. **RQ2 — NEITHER condition has run**: class n=350 (all 74 codes, no shortlist) AND
   division n=1000, nested. ~3 quota-days, first quota spent every day. Projected done
   ~08-24 against 08-31 deadline if no day is missed.
2. `run_costs.py`/T8_cost — code doesn't wait on quota; only the final run does.
3. Path sweep: assert every configured root is absolute (storage_root was the 3rd
   cwd-relative-default bug — after `paths.py`, `embed.py`'s cache dir).
4. Wire `T9_transfer` once ruled on the severity-0.25 reversal's wording.
5. Results/Discussion/Conclusion still empty headings — skeleton owed.

## Gotchas that are not plan amendments
- `make_tables.py` refuses the whole build if the latest `run_blocking` record is dirty
  (pre-existing, `-15338713`) — call a builder function directly, not `main()`.
- `run_dedup` exits 2 on quota before `write_run`; a cleared cache costs T4 a day.
- Quota: 200k/day → **~300 adjudications/day** at 638 tok; RQ2 is 784 tok/record at k=12,
  ~1,421 with all 74 codes. TeX at `/Library/TeX/texbin`; `make paper` prepends it.
- Corpus A PC collapses 0.985 → 0.248 → 0.049, **zero** blocks dropped: key failure, not
  capping. Naive floor **0.000 on Corpus A** at every severity.
- `system/`: `docker compose up -d db` before `alembic`/tests; seed `seed_users.py` then
  `seed_categories.py` (categories FK'd by assets.cpv_code).

## Last verified
**2026-08-20** — `make test`: research 486+7 skipped, system/api 22, annotation 29, all
clean. `make paper` clean, 8 pages.
