# STATE

## Where things stand (2026-08-28, session end)

**The paper is complete and ships at eleven pages.** Supervisor ruling, 2026-08-28: both
authorized page tests were run and reported honestly when neither closed the gap (removing
Figure 1 changed nothing — it was absorbing existing float slack either way; dropping three
non-essential references left the paper at 11, closer than any earlier attempt but not 10).
Everything left to cut is a measured result or a citation, feeding Results/Interpretation/
Difficulty/Quality of Approach (double-weighted), against a page limit under Clarity
(five per cent, single-weighted). **Ruled: ship at eleven. Do not revisit this.**

**RQ2 is finished.** `run_rag_classify-20260828T050059-2c53d19-f8b41ee3`: n=298 of 298
drawn, `partial_run: false`. Rewritten in `main.tex` with two statements kept apart: on the
full 3,962-record test partition the classical-over-embedding result is established at
interval strength (TF-IDF [0.769, 0.795] vs. embedding [0.737, 0.764], disjoint); on the
298-record subsample no pairwise difference among the three conditions is established (all
intervals overlap; the language model's own lower bound, 0.771, sits below TF-IDF's point
estimate, 0.775). Carries the asymmetric-drift caution (TF-IDF drifts −0.7 macro points from
its full-partition value, the embedding classifier drifts +3.2 in the *opposite* direction —
citing only the smaller drift would have been selecting the evidence that suits the
argument) and the noise-shrinkage sentence (LM margin over TF-IDF: 14.6 → 6.6 → 4.4 points
across n=41/167/298, moving with TF-IDF's own drift: 9.9 → 4.0 → 0.7 — not extrapolated past
n=298). Cost figures updated everywhere: 1,559 tokens/call, 128.3 calls/day, 2.34 days to
n=300. `research/scripts/make_figures.py`'s F5 regenerated from the final run (n=298, not
partial n=167); `appendix_system.tex`'s F5 caption updated to match.

**The paper's last `[TBC]` is resolved.** Commit `d97a02f` (RQ2 rewrite + cost figures) is
tagged `paper-2026-08-28` — that tag and its short SHA are named beside the repository URL
in Reproducibility. `grep -c 'TBC' main.tex` reads 0.

**A real bug was caught and fixed in the same pass**: the TBC-resolution edit used
`\texttt{}` around the tag/SHA, which needs a Type1 Courier metric (`pcrr7t`) this TeX
install cannot rasterize — the exact defect class `ca47d4a` fixed once already this project
(session of 2026-08-22). `-interaction=nonstopmode` (used for every page-count check this
whole session) recovers from the missing font by substituting `nullfont` and continuing, so
every plain `pdflatex` build "succeeded" while the actual rendered PDF read "tagged (commit
) for the analysis" — every character of the tag and SHA silently dropped. **Caught only
because this was finally checked with `make paper` (`-halt-on-error`, the real acceptance
gate CLAUDE.md names) instead of a plain `pdflatex` call.** Fixed the same way as before:
plain text, no monospace. **Lesson for next time: use `make paper`, not raw `pdflatex`, for
any check that needs to catch a silent content failure — page counts and TBC counts were
unaffected by this gap (nonstopmode still completes those), but text content is not safe
under nonstopmode alone.**

**All eight exhibits are built** (`research/scripts/make_figures.py`, `matplotlib` in
`research/pyproject.toml`, `results/figures/` gitignored like `results/tables/`). Figure 1
(review share vs. duplicates lost) is in `main.tex`, in existing float slack, costing
nothing either way. Figures 2–7 and the architecture diagram (A1) are in
`appendix_system.tex`, each captioned with its source run_id.

**`appendix_system.tex` is complete**: the requirement-to-test table (62 real tests, one row
per client-brief requirement plus one for path-safety infrastructure), the architecture
diagram, real API-level evidence against live Postgres (`capture_appendix_evidence.py`,
cleans up after itself, verified DB unchanged), the explicit no-UI note, `T_degradation`
(moved from the main paper — illustration, not a result any argument depends on), and every
exhibit that didn't fit the main paper plus the extrapolated cost table. 11 pages, no
page-limit constraint on it (handbook exempts appendices).

**`main_overleaf.tex` regenerated last**, after every other rebuild: zero `\input{`, zero
`[TBC]`, `F1_review_vs_lost.pdf` referenced by bare filename (the `overleaf` Makefile
target now rewrites `results/figures/*` paths the same way it already inlined
`results/tables/*`). **Verified by actually compiling it**: copied `main_overleaf.tex` +
`F1_review_vs_lost.pdf` flat into an isolated temp directory and built with
`pdflatex -halt-on-error` — clean, 11 pages, zero undefined references. That plus the
appendix PDF is the submission bundle.

**Commit `7df3edc`** ("Implement code changes to enhance functionality and improve
performance," 189 lines appended to `results/ledger.jsonl`, an auto-committed RQ2 progress
snapshot) **could not be amended**: already on `origin/main`, 4 commits behind `HEAD` at the
time — fails both the "not pushed" and "trivially the tip" conditions for a safe rewrite.
Left as is, flagged rather than force-pushed over.

## Real bugs found and fixed this session
- The retry-pacing bug (orphaned phantom token-window entries on 429 retries compounding
  into ~1-call/25-min collapse) — fixed and verified by simulation and a live probe.
- `run_label_noise.py` divided disagreement count by the full n including "unsure," against
  its own docstring's claim that unsure counts as agreement neither way — fixed to exclude
  it, matching `annotate.py`'s own summary logic.
- The Abstract oversold the pair-level automated share as the register-level share RQ3
  actually asks about — record-level correction throughout (Abstract, RQ3, Results,
  Discussion, Conclusion, Threats).
- The RQ2 paragraph mixed an accuracy-drift number with a macro-F1-margin number into one
  invalid ratio — fixed to argue from accuracy throughout, macro F1 as supporting detail.
- `table_cost_scale`'s caption embedded run_ids with unescaped underscores — "Missing $
  inserted." Routed through the existing `_esc()` helper.
- `appendix_system.tex` had no bibliography to resolve `T_degradation`'s
  `\cite{mudgal2018}` once that table moved there — added a one-entry `\thebibliography`.
- The HSE wording implied a causal link between duplicate/misclassified records and a
  national injury-cost statistic this paper's data does not speak to — reworded.
- "Hand-reviewed" mischaracterised the CPV label-noise review as human when it was an AI
  coding assistant — corrected to state this plainly.
- An earlier draft of the Related Work reference cut paired the "assumes a labelled
  training set" critique with `narayan2022`, whose entire point is avoiding task-specific
  training data — a real logical contradiction, caught before committing; kept `li2020ditto`
  (whose cross-encoder approach genuinely needs labelled training data) instead.
- `\texttt{}` around the commit tag silently dropped under `-interaction=nonstopmode`'s
  font-substitution recovery — see above. The same defect class as `ca47d4a`.
- `STATE.md` was five days stale and claimed the institutional email was the paper's
  remaining `[TBC]` — that line was deleted days ago; caught only because asked explicitly.

## Gotchas that are not plan amendments
- `RagFewShotLLMClassifier.predict_cached_only` bypasses `complete_many` entirely — use
  this, not a fresh `predict()` call, to recompute statistics on an already-partial run.
- `annotate.py`'s `load_sample()` cannot be re-run at a larger `n` to get a superset —
  extending a label-noise sample needs a dedicated script holding the original draw fixed.
- `make seed`'s guard checks the wrong path (`system/api/scripts/` instead of
  `system/api/src/fcesapi/scripts/`) — always prints "SKIPPED." Run seed scripts directly.
- A background watcher/monitor armed inside an agent session is session-local, not durable
  — never report time-delayed work as "scheduled" on that basis alone.
- Removing content does not reliably reduce `main.tex`'s page count proportionally — floats
  interact non-linearly. Always rebuild and measure; never assume a cut's effect.
- **`-interaction=nonstopmode` silently substitutes a missing font and continues; only
  `-halt-on-error` (what `make paper` actually runs) turns a font failure into a build
  failure.** A page count, TBC count, or overfull-gate check is safe under nonstopmode; a
  claim that specific *text content* rendered correctly is not — verify content claims
  against a `make paper` build, or by reading the actual rendered page image/pdftotext
  output, never against a nonstopmode log alone.
- `\texttt{}` needs a Type1 Courier metric (`pcrr7t`) this TeX install cannot rasterize.
  Nothing in `main.tex` may use it; every existing identifier is plain text instead. This
  is now the second time this exact defect has been introduced and fixed — check for
  `\texttt{` in any future diff touching `main.tex` before considering it done.

## Next tasks, in order
Nothing blocking remains from this session's ordered work. Open items carried from before
this session, unchanged:
1. "Naive precision floor" sub-finding — has its one sentence in Methodology; no further
   measurement planned unless requested.
2. The label-sheet scope question (multi-asset PDF beyond the named single endpoint) —
   built and tested, flagged as a scope reopening, not revisited.
3. A real frontend for `/a/{public_id}` — the API-side resolver is a stand-in.
4. Submission packaging: assemble `main_overleaf.tex` + `F1_review_vs_lost.pdf` +
   `appendix_system.pdf` (or its own Overleaf project) for the 5pm Monday 31 August 2026
   deadline. All three build clean as of this session.

## Last verified
**2026-08-28** — research: 524 passed + 14 skipped. `system/api`: 62 passed. `annotation`:
29 passed. Zero failures across all three suites. `fcesreg`/`system` boundary intact.
`main.tex`: **11 pages** (ruled final, not a target to keep chasing), **0** `[TBC]`, two
`pdflatex -halt-on-error` passes via `make paper`, zero undefined refs, overfull gate at
5.887pt (well under the 10pt failure threshold). `appendix_system.tex`: 11 pages, zero
undefined refs, builds standalone with `-halt-on-error`. `main_overleaf.tex`: regenerated
after the final rebuild, zero `\input{`, zero `[TBC]`, figure by bare filename, verified by
an actual standalone compile in an isolated directory. Tag `paper-2026-08-28` on commit
`d97a02f`.
