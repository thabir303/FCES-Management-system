# STATE

## Where things stand (2026-08-21, final session)

**`main.tex` is written and builds clean: 10 pages, 0 `[TBC]`, two pdflatex passes, zero
undefined refs** (`b89cbf3` + table commits through `de1ee43`). Cuts applied, four Related
Work citations added and verified against publisher records (not recalled), one real
citation error fixed (Ditto: 2020→2021, VLDB pub year not arXiv year). Results/Discussion/
Conclusion written from committed run records, every number traced to a run_id. Five tables
(`T1_corpus` merged, `T4_abtbuy`, `T6_classification`, `T8_cost`, `T9_transfer`), two figures
**not built — no time left**, flagged to supervisor.

**RQ2 division-level LLM: n=41 of 300 requested**, not what was ordered. A token-window
pacing fix (below) made per-minute pacing overly conservative on retries, collapsing
throughput to ~1 call/25min; recovered by reading the 41 already-cached completions
directly. macro F1: rag_fewshot_llm 0.805, tfidf_svm 0.680, embedding_logreg 0.674 (same
41-record subset). Class-level LLM condition **not attempted** (ruled: 135/day too few to
decide anything). Needs a follow-up session at low traffic to reach n=300.

**RUN THIS FIRST, EVERY DAY** (still applies): `run_dedup.py --config dedup.yaml --corpus
abtbuy --cascade`. Exit 2 = success, but check live-call count — quota resets at 06:00 Dhaka
(UTC midnight), not local. Cascade itself is now **complete, 1,209/1,209**
(`run_dedup-20260821T051431-ea950b6`), so this only matters again once new quota-consuming
work resumes (RQ2 completion, RQ2 class level).

## Real bugs found and fixed this session
- `llm._estimate_tokens` didn't count `json_schema` size — undercounted pacing for a
  schema with real content (enums), causing genuine 429s. Fixed.
- `llm._call` never recorded failed-attempt tokens in the per-minute window (only a 200's
  `usage` fed it). Fixed by recording the estimate on 429/400-retry paths — **but this is
  implicated in the RQ2 slowdown above**: per-retry recording may overcorrect when one
  record retries several times. Look here first before the next quota-heavy run.
- `table_costs`/`table_classification` wrote raw `%` into LaTeX — silently ate the row as a
  comment. Undetected until this session, since neither table had been `\input{}` before.
- Conclusion edit accidentally deleted `\begin{thebibliography}{99}` (old_string consumed
  it, new_string didn't reproduce it) — caught by the build, not by review.
- Reasoning-model output starvation: `openai/gpt-oss-120b` spends tokens on hidden
  chain-of-thought before the answer; `max_tokens=150` covered the JSON alone, not
  reasoning+JSON, so it surfaced as `json_validate_failed`, not as a token-budget symptom.
  `dedup.yaml`'s cascade already documented this exact class (`max_tokens: 1024`) — should
  have been checked before building at 150.

## Gotchas that are not plan amendments
- `make_tables.py` refuses the build if the latest of ANY wired run type is dirty, even from
  an untracked run directory alone. Commit after every single run in a batch, not at the end.
- `run_dedup --cascade` dirties `results/ledger.jsonl` even on 100%-cache-replay (ledger
  rows append on cache hits too) — dirties the tree for whatever runs next until committed.
- Quota 200k/day. Cascade ~657 tok/call (~300/day) does NOT transfer to RQ2 division's
  ~1,400 tok/call (verbose Corpus B examples + reasoning tokens) — budget accordingly.
- `system/`: unexamined this session — D1–D3/E1–E5 stand as of `c652b14`, not touched.

## Next tasks, in order
1. Finish RQ2 to n=300 at low-traffic time, watching the pacing fix for a slowdown repeat.
2. Two figures (recall vs. severity, operating-point curve) — not built, no plotting script
   exists yet, matplotlib not installed.
3. "Naive precision floor" sub-finding needs its own measurement; omitted from Discussion.
4. Backend (QR/barcode, floor plans, reminders, audit log) — sacrificed this session per the
   ruled priority (quota > paper > backend).
5. Re-verify system/ still clean; untouched this session but unchecked for a full session.

## Last verified
**2026-08-21** — research: 487 passed + 28 (test_paths) + 12 skipped. `fcesreg`/`system`
boundary intact (`grep -r "import fcesreg" system/` → `pipeline.py` only). `make paper`:
10 pages, clean, two passes, zero undefined refs.
