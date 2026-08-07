# STATE

Session handoff. Volatile — update at the end of every §11 task. Hard cap 60 lines.

## Current task

**C5 — `llm.py`: cache, ledger, quota governance.** Built and tested; **criterion not yet
closed**. Provider moved to `openai/gpt-oss-120b` on Groq's free tier (amendment G1), so
`llm.py` targets the OpenAI-compatible endpoint over `httpx`, every call is synchronous, and
rate limits replace money as the binding constraint. 34 tests cover cache, ledger, cap, pacing,
429 backoff and quota stops against a stub transport.

**To close C5:** put `GROQ_API_KEY` in `.env`, then `make llm-pilot`. It runs 20 live calls,
replays them, and prints the criterion checks. I cannot run it — no key, and spending the
quota is the supervisor's call.

Done in Phase C: C1 `embed.py`, C2 `blocking.py`, C3 `dedup.py` (Exact + Tfidf), C4 `degrade.py`,
C5 `llm.py`. `metrics.py` has `prf1` only — C8 still owes `macro_weighted_f1`, `confusion` and
`operating_point.py`.

## Blocked or waiting on the supervisor

- **Two paper edits proposed, neither applied.** (1) §Evaluation Metrics says cost is reported
  "for any method invoking a paid model" — on a free tier no method invokes one, and the figure
  is now notional. (2) Reproducibility needs the open-weights sentence. Both change what the
  paper *claims*, so rule 2 says propose and wait. `main.tex` is untouched.
- **200-pair distractor verification.** `annotation/labels/distractor_judgements.jsonl` does not
  exist. **C6 and `run_transfer.py` must not run until it is complete.**
- **`make data` is broken**, not merely unbuilt: it calls `research/scripts/build_taxonomy.py`,
  which does not exist, while the README presents it as step two of reproduction.
- **`make annotate` is broken**: it calls `annotation/annotate.py`, which does not exist. On the
  critical path for RQ3 — it produces both the label-noise estimate and `mean_seconds_per_item`.
- **Open decision:** is separator-blindness at key level intended? `normalise_key` collapses
  `1.5kW`/`1,5 kW`, so the exact-match baseline is immune by construction to one of the seven
  injected error classes. Raised twice, never ruled on.
- **Conflict, unresolved:** the agreed page budget (three tables, two figures, `T8_cost` to
  prose) was never written into PROJECT_PLAN.md. §10 still specifies `T4_abtbuy.tex`,
  `T6_classification.tex` and `T4_cf_sweep.tex` in full, and §14 records no such amendment.

## Next three tasks in §11 order

1. **C8** — `metrics.py` + `operating_point.py`. Unblocked, needs only C3.
2. **C7** — `classify.py`, three conditions (§10.1). Needs C1, C5, B3.
3. **C6** — Embedding + Cascade matchers. *Blocked on the verified distractor set.*

## Gotchas that are not plan amendments

- **`embed.DEFAULT_CACHE_DIR` is `Path(".cache/embeddings")` — cwd-relative.** Same defect class
  `paths.py` exists to prevent: a run from `research/` silently misses every entry a run from the
  root wrote. `llm.py` anchors its cache against `repo_root()`. Not fixed here because it is C1's
  module — reported, not silently touched. `test_paths.py`'s grep guard does not catch `.cache/`.
- Quota arithmetic from the supervisor's stated free-tier limits, not measured: ~710 tokens per
  adjudication against 200k tokens/day is about **280 adjudications a day**, ~11 a minute. Verify
  against `x-ratelimit-*` headers at runtime; note `limit-requests` is per *day* and
  `limit-tokens` per *minute*, so RPM and TPD are not in any header.
- TeX is installed but `/Library/TeX/texbin` is not on PATH; `make paper` prepends it.
- `make experiments` names runners that do not exist yet. For that target only, a missing script
  is the build order rather than a bug — this does **not** excuse `make data` or `make annotate`.
- Blocking figures in `results/runs/` are **severity-free**, measured on undegraded records.
  Re-run across the severity range now that C4 exists; do not compare the two sets silently.
- Corpus B pair completeness at `t=8` is far below the 0.98 floor at every severity, and
  non-monotonic. Both are findings; neither is refitted.
- Full background, dead ends and rejected alternatives: `.claude/docs/session-knowledge.md`.

## Last verified

**2026-08-08** — `make test`: 295 passed, 2 warnings, identical from the repository root, from
`research/` and from `/tmp`. `main.tex` untouched, so `make paper` not re-run (last clean build
6 pages, 2026-08-07).
