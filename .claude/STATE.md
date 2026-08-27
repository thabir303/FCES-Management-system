# STATE

## Where things stand (2026-08-27, mid-session)

**RQ3 now reports the record-level automated share, not just the pair-level one**
(`8896d09`, `afd3b33`, `02fd76e`). The pipeline's operating point was fitted and reported
over candidate *pair* decisions; RQ3 asks what share of a register — records — migrates
unattended, and a record entering several candidate pairs is not resolved until every one
of them is. `operating_point.record_level_share()` computes this from the same fitted
bounds `band_operating_point` already produces, so the two shares are directly
comparable. At severity 0 on Corpus A: **0.95 floor — 83.0% of records automated (1,192
of 1,437, 245 to review) against a pair-level 93.4%; 0.99 floor — 39.3% (565 of 1,437,
872 to review) against a pair-level 64.4%.** The gap is largest exactly where the floor
is safest, and at 0.99 nothing is auto-accepted at all — every automated decision is a
confident rejection. Threaded through Abstract, RQ3's question wording, Results,
Discussion (a new paragraph with the reframing and the zero-acceptance observation),
Conclusion (leads with the record-level number, states the floor binds only the accepted
portion) and Threats (denominator clause: comparison-eligible records, not a real
register's full holdings). Run record: `run_operating_point-20260826T211147-8896d09-945f8c5b`.

**Four corrections made** (`d62d695`): HSE wording no longer implies duplicate/misclassified
records contributed to the national £22.9bn ill-health figure — reworded to the defensible
claim (wrong servicing schedule/risk assessment attached to a misidentified item). The
language-model condition is now framed as exploratory up front (division level only, class
level not attempted and priced, no winner named while intervals overlap). "Hand-reviewed"
removed from the CPV label-noise description — states plainly that an AI coding assistant
performed the review, not a human annotator, no second reviewer. A commit-id `[TBC]` added
beside the repository URL — the paper's one remaining placeholder, not yet resolved (see
below). Repository audited: no API keys, no credentials, no committed `.env` (only
`.env.example`, placeholders throughout), no personal data beyond three fictional
`@fces.internal` seed addresses.

**`T_degradation` moved to the appendix** (`73f91c5`): illustration (seven before/after
example strings), not a measurement any argument in the paper depends on, unlike every
other table. `main.tex` keeps the `p_s = s` statement, the stress-parameter framing, and a
cross-reference to "the appendix's degradation-examples table" by name (the two documents
don't share `\ref{}`). Same generated `T_degradation.tex`, now `\input` only by
`appendix_system.tex`. One real defect this surfaced and fixed: the table's "Field merging"
row cites `\cite{mudgal2018}`, which the appendix had no bibliography to resolve —
added a one-entry `\thebibliography` there, text identical to `main.tex`'s own `bibitem`.

**All eight exhibits the supervisor asked for are built** (`d7760f9` for F2–F7 + A1,
`ecd0cb0` for the cost-scale table), every one from a committed run record, vector PDF,
distinguished by marker/linestyle not colour. `research/scripts/make_figures.py` is new
infrastructure (matplotlib added to `research/pyproject.toml`; `results/figures/`
gitignored and regenerated like `results/tables/`). Figure 1 (review share vs. duplicates
lost) is the only one placed in `main.tex` so far — it fit into existing float slack when
tried (page count unchanged at 11); Figure 2 was tried and reverted when it pushed the
count to 12; Figures 3–7 were never attempted in the main paper, per the stated mechanism
("stop the moment an addition would exceed the limit"). **Two more page tests are queued
for this session** (remove Figure 1 and re-measure; if that's not enough, drop three named
non-essential references) — not yet run as of this state note.

**`appendix_system.tex` is built and verified** (`9c60ec0`, `73f91c5`): a standalone
article-class document, exempt from the paper's page limit. Contains the
requirement-to-test table (one row per client-brief requirement from `PROJECT_PLAN.md`'s
own mapping, populated from the 62 tests that actually run, not the brief's wording, plus
a tenth row for path-safety infrastructure tests), an architecture diagram, real
API-level evidence captured against live Postgres (`capture_appendix_evidence.py`, a new
script under `system/api/src/fcesapi/scripts/` — seeds one asset, uploads a two-row CSV
where one row is an exact textual duplicate, captures the resulting flagged-duplicate row
with its real 0.8301 confidence and competing candidate, the six-field review decision,
the resulting asset, and an audit row, then deletes everything it created and verifies the
DB is unchanged), the explicit no-UI note, and the exhibits that don't fit the main paper
plus the moved `T_degradation` table. 11 pages, no page-limit constraint on it. Evidence
JSON committed at `results/appendix_evidence.json`.

**Page count: still 11, not 10**, after every cut authorized so far (T1_corpus dropped
earlier this session, bibliography URL preambles trimmed, Related Work tightened,
Conclusion/Discussion compression, `jisc`/`kitcatalogue` dropped, `T_degradation` moved).
The overflow is now small and precisely characterised: pages 9–10 are completely full,
both columns, zero slack; page 11 carries only the Conclusion and all 17 references, with
more than half of both columns still blank. **Two more page tests are the immediate next
task** (see Next tasks) — not yet run.

**RQ2 is at n=293 of 298 drawn (up from 167), still `partial_run: true`.** Today's UTC
quota window (rolled over 05:50, exhausted again ~06:35) advanced it from n=167 to n=293
— only 5 records short. `run_rag_classify-20260827T055047-9c60ec0-f8b41ee3`. The
noise-shrinkage trend across n=41/167/293 is a real finding, not just a trend: TF-IDF's
drift from its own known full-partition value falls 9.9 → 4.0 → 0.7 points; the language
model's accuracy margin falls 14.6 → 6.6 → 4.4 points, moving together. At n=293 the
language model's lower bound (0.771) sits *below* TF-IDF's point estimate (0.775) — the
sharper way to say the comparison does not separate the conditions.

**RQ2's remaining 5 records need a MANUAL run after 00:00 UTC / 06:00 Dhaka.** A
background watcher was armed this session to wait for the UTC rollover and resume
automatically, but **it is not durable** — no script, no log, no pid file survive in the
repository, so it exists only inside a session that ends when the machine sleeps. **Do
not rely on it or report RQ2 as scheduled.** Run this by hand once the quota window is
open (git tree must be clean first):

```
.venv/bin/python research/scripts/run_rag_classify.py --config research/configs/rag_classify.yaml
```

Should complete the last 5 records within the first few calls (128.7 calls/day measured
today). Once the record reads `partial_run: false` and `n_test_sample_completed: 298`:
rewrite the RQ2 paragraph budget-neutral at that n (accuracy carrying the Wilson intervals,
macro F1 alongside without one, stating the drawn sample as 298 of 300 requested rather
than rounding), add the noise-shrinkage sentence above, update the cost figures to that
run's measured mean tokens/call and days-to-n300 (must not carry the n=167 or n=293
values, nor the older 1,623/2.43), then tag the commit and resolve the paper's last
`[TBC]` beside the repository URL.

## Real bugs found and fixed this session
- The retry-pacing bug flagged as unverified in the previous state note: orphaned
  phantom token-window entries on 429 retries never expired, compounding into an observed
  ~1-call/25-min collapse. Fixed by removing the phantom-entry-on-429 behaviour (kept for
  400 `json_validate_failed`, where real generation occurred). Verified by simulation and
  a live probe.
- `run_label_noise.py` divided the disagreement count by the full n (including "unsure")
  despite its own docstring claiming unsure counts as agreement neither way. Fixed to
  exclude unsure from the denominator, matching `annotate.py`'s own summary logic — moves
  the original n=40 figure from 12.5% to 13.2% from the fix alone, separate from the
  further move to 17.4% from the 160 added records.
- The Abstract oversold the pair-level automated share as if it were the register-level
  share RQ3 asks about — the record-level correction above.
- The RQ2 paragraph mixed an accuracy-drift number with a macro-F1-margin number into one
  invalid ratio. Fixed to argue entirely from accuracy, with macro F1 as supporting detail.
- `table_cost_scale`'s caption embedded run_ids with literal, unescaped underscores —
  "Missing $ inserted" against plain pdflatex. Fixed by routing through `_esc()`, the
  helper every other table already uses for this.
- `appendix_system.tex` had no bibliography to resolve `T_degradation`'s `\cite{mudgal2018}`
  once that table moved there — added a one-entry `\thebibliography`.
- The HSE wording implied a causal link between duplicate/misclassified records and a
  national injury-cost statistic that this paper's data does not speak to. Reworded.
- "Hand-reviewed" mischaracterised the CPV label-noise review as human when it was an AI
  coding assistant — corrected to state this plainly.
- This file (`STATE.md`) was stale by five days and claimed the institutional email was
  the paper's remaining `[TBC]` — that line was deleted days ago; the real remaining
  placeholder is the commit identifier. Caught only because the supervisor asked
  explicitly; a stale handoff is how a fixed problem gets re-fixed.

## Gotchas that are not plan amendments
- `RagFewShotLLMClassifier.predict_cached_only` bypasses `complete_many` entirely — use
  this, not a fresh `predict()` call, to recompute statistics on an already-partial run
  without attempting every remaining live call.
- `annotate.py`'s `load_sample()` cannot be re-run at a larger `n` to get a superset of an
  existing sample — verified empirically only 2/40 IDs recur at n=200/seed=0. Extending a
  label-noise sample needs a dedicated script holding the original draw fixed and pulling
  only the per-division shortfall (see `annotation/extend_label_noise_sample.py`).
- `make seed`'s guard checks `system/api/scripts/seed_categories.py`, but the real path is
  `system/api/src/fcesapi/scripts/seed_categories.py` — the condition is always false, so
  `make seed` always prints "SKIPPED." Run the seed scripts directly by their real path.
- Any background watcher/monitor armed inside an agent session is session-local, not
  durable — it does not survive a sleep, a crash, or the session ending. Never report
  time-delayed work as "scheduled" on that basis; report it as pending a manual step, with
  the exact command to run.
- Removing content does not reliably reduce `main.tex`'s page count by a proportional
  amount — floats interact non-linearly (removing `T1_corpus` once made overflow *worse*
  by entry count; `[t]` vs `[tb]` was isolated and shown byte-identical, ruling out float
  spec as the mechanism). Always rebuild and measure; never assume a cut's effect.

## Next tasks, in order
1. **Page test A**: remove Figure 1 from `main.tex`, rebuild, report the count. If it
   closes at 10, move Figure 1 into `appendix_system.tex` beside the other exhibits.
2. **Page test B**, only if A is not enough: drop three references (not `koepcke2010`,
   `leipzigdbs`, `contractsfinder`, `mudgal2018`, or `iso55000`) and their citing clauses.
3. If neither closes it: stop, report the count, and wait for the supervisor's call —
   do not touch Results, Discussion, threats to validity, or the RQ paragraphs.
4. Fix commit `7df3edc`'s message ("Implement code changes to enhance functionality and
   improve performance" — says nothing) if it is safely amendable (tip of history, not
   pushed); otherwise leave it and say so.
5. Run RQ2's last 5 records manually after 00:00 UTC / 06:00 Dhaka (command above).
6. Once RQ2 reads `partial_run: false`: rewrite the paragraph, update cost figures, tag
   the commit, resolve the last `[TBC]`.
7. Final verification (page count, TBC=0, undefined refs, overfull gate, `make test` all
   three suites, table `run_id`s, appendix builds standalone) — read the rendered pages,
   not the log. Regenerate `main_overleaf.tex` **last**, after the final rebuild.
8. "Naive precision floor" sub-finding, the label-sheet scope question, and a real
   frontend for `/a/{public_id}` remain open from before this session and are unchanged.

## Last verified
**2026-08-27** — research: 524 passed + 14 skipped (up from 515+12: six new tests for
`record_level_share`). `system/api`: 62 passed. `annotation`: 29 passed. Zero failures
across all three suites. `fcesreg`/`system` boundary intact. `main.tex`: 11 pages (not 10,
page tests pending), 1 `[TBC]` (the commit identifier — not the email, which was resolved
before this session), two `pdflatex` passes, zero undefined refs, overfull gate at
5.887pt. `appendix_system.tex`: 11 pages, zero undefined refs, builds standalone.
