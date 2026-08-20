# STATE

## RUN THIS FIRST, EVERY DAY

    .venv/bin/python research/scripts/run_dedup.py --config research/configs/dedup.yaml --corpus abtbuy --cascade

Exit **code 2** = quota spent = success — BUT check live-call count in the output first.
**The quota day resets at 06:00 Dhaka time (UTC midnight), not local midnight.** A run
before 06:00 replays cache, banks nothing, still exits 2 and prints "success" — the runner
now prints `*** ZERO LIVE CALLS ***` when this happens, do not ignore it. This is very
likely what both missed days actually were. **Never clear `.cache/llm`**. **Missed: 2**
(08-16, 08-19). **1,204/1,209, 5 remain**; 08-21's pre-06:00 run replayed 0 live, confirmed
by the new warning. Report missed days + projected finish first line of each session. 60 lines.

## Current task
Cap sweep + damage check landed. Owed in one message: cut ledger ~245, Results skeleton,
4 Related Work citations. Waiting: system build past D3/E5?

**Sev 0.0** (`run_blocking_cap-20260820T093302-a3c575e`): −0.575 gap at shipped cap 500 is
a guard-scale artifact, but rescaling costs volume — B candidates 36,614 (500) → 1.75M →
23.6M → 44.3M (none), **1,209× at PC 0.982**. Withdrawn: "converges within 0.003 of A" — A's
0.985 is genuinely-different retailer text, B's sev-0.0 positives are byte-identical.

**Sev 0.25, both corpora** (`...-3223ff6-d081d1cc`, clean): PC(A)=0.248, PC(B)=0.441 — B
higher, reversed from sev 0.0. Candidates A=28,102, B=90.9M (~2,482×). **Damage check**
(`run_degradation_damage-...-98086b49`): B is damaged MORE (mean dist 0.652 vs 0.516) yet
still retains higher PC — outcome 3, the strong one, reversal has an owner. `T9_transfer`
still unwired, ready to word once the supervisor rules.

**System to the ruled line** (`c652b14`): schema/auth/asset CRUD/import wizard/review
queue (D1–D3, E1–E5), 22 tests vs live Postgres. 3 bugs: enum cols typed `String` against
native enums, `GENERATED ALWAYS` needed `Computed()`, `storage_root` cwd-relative (silent
cache miss) — fixed, anchored to repo root. `build_cf` fixed 08-20, noise parity pinned.
`T6_classification` generated clean.

## Blocked or waiting on the supervisor
- **Page budget ~230–245.** 8 cuts −71; 6/9 outside candidates −98 more. Open: separator-
  blindness in `normalise_key`.

## Next tasks
1. **RQ2 — NEITHER condition has run**: class n=350 (all 74 codes, no shortlist) AND
   division n=1000, nested. ~3 quota-days, first quota spent every day. Projected ~08-24
   against 08-31 if no day is missed.
2. `run_costs.py`/T8_cost — code doesn't wait on quota; only the final run does.
3. Wire `T9_transfer`, cut ledger, Results skeleton, 4 Related Work citations — one message.
4. Path sweep done (`247645a`): 4th+5th bugs, `seed_categories.py`'s own `_repo_root()`
   and `.env`'s `STORAGE_ROOT=./storage` beating the anchored default.

## Gotchas that are not plan amendments
- `make_tables.py` refuses the build if the latest `run_blocking` record is dirty
  (pre-existing, `-15338713`) — call a builder function directly, not `main()`.
- `run_dedup` exits 2 on quota before `write_run`; a cleared cache costs T4 a day. Quota
  200k/day → ~300 adjudications/day at 638 tok; RQ2 784 tok/record at k=12, ~1,421 at 74.
- Corpus A PC collapses 0.985→0.248→0.049, zero blocks dropped: key failure, not capping.
  TeX at `/Library/TeX/texbin`, `make paper` prepends it.
- `system/`: `docker compose up -d db` before tests; seed `seed_users.py` then `seed_categories.py`.

## Last verified
**2026-08-20** — system/api 30 (was 22) vs live Postgres, clean. `make paper` clean, 8
pages, untouched pending the Results skeleton ruling.
