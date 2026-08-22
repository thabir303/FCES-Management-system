# STATE

## Where things stand (2026-08-22, session end)

**`main.tex` has a title and author for the first time** (`a41f72f`): "Automating Legacy
Equipment Register Migration: Duplicate Detection and Taxonomy Assignment at a Precision
Floor", Md Ashik Reza, Faculty of Computing, Engineering and Science, University of South
Wales. The institutional email is the paper's one remaining `[TBC]` — left for the author to
insert, not guessed.

**RQ2's n=41 comparison now carries Wilson 95% intervals and the TF-IDF drift argument**
(`a41f72f`), closing a real correctness gap: the paper argues at length that point estimates
without intervals are unreliable, then was reporting exactly that for its own three-way
comparison. All three intervals overlap (`rag_fewshot_llm` 0.829 [0.687, 0.915],
`embedding_logreg` 0.756 [0.607, 0.862], `tfidf_svm` 0.683 [0.530, 0.804]); `tfidf_svm`'s
known full-partition value (0.782/0.759) sits 9.9/7.9 points above its own subsample value,
against which the language model's 12.5-point apparent margin is only ~1.6x — not
distinguishable from sampling noise at this n. The completed-41 subset was checked, not
assumed, to be a random prefix rather than a biased one (position-vs-date correlation
−0.085, largest label-share drift 11 points, both persisted to the run record's
`completed_subset_check`, `run_rag_classify-20260821T144758-c9f8686-f8b41ee3`). The run
record also now carries the measured per-call cost (1,623 tokens/call, 123.2 calls/day,
2.43 days of quota to reach n=300), via a new `--cache-only` recompute mode
(`RagFewShotLLMClassifier.predict_cached_only`) that re-derives statistics from the LLM
cache with zero further network calls — the right tool for "add stats to a partial run
without risking the pacing bug again."

**Bibliography verified against real publisher records, all 18 (now 19) entries**
(`9dc262c`): Papadakis et al. was 2020, DBLP/Crossref give 2021 as the citable year — fixed.
Everything else (Fellegi, Christen, Rahm/Do, Mudgal, Köpcke, Li/Ditto, Narayan — the one
flagged since the first draft, confirmed correct via the paper's own PVLDB reference block —
Chen/FrugalGPT, Reimers, Siciliani, ISO 55000, Lin/Gao/Koronios, Woodall et al.) verified
correct as written. Two licence gaps closed: added a bibitem for the Leipzig DB Group's own
benchmark webpage (Abt-Buy's stated dual-citation requirement — paper + site — was only
half satisfied before), and added OGLv3.0's standard attribution sentence plus a link to the
licence text itself for Contracts Finder (previously named the licence without the
attribution statement or a link to it).

**Page count: 11, not 10.** The bibliography fixes above needed roughly 2 lines more than
remained once the title/author block and the RQ2 trim (also this session, see below) had
already used up every line of slack — page 10 was full to its last line in both columns
before these fixes. Wording was tightened as far as seemed reasonable without touching
previously-cut material elsewhere (the same discipline the RQ2 trim used), and it still
doesn't fit. **Left flagged rather than resolved unilaterally** — the two options are a
targeted cut somewhere already-costed, or accepting 11 pages for a set of mandatory
correctness/legal-attribution fixes. Not decided this session.

**Two figures: still not built.** No slack existed to add even one (page 10 was completely
full before the bibliography fixes pushed to 11) — shipped as five tables, zero figures, as
the ruling allowed.

**Backend: five feature areas built today** (`afb1468`), API-only, against live Postgres —
attachments (upload/list/download/delete, MIME-validated per kind, one-primary-per-kind
enforced by the DB's own partial unique index), floor plans + pins (image upload with real
dimensions via Pillow; pins are `Location` rows with percentage coordinates; deleting a pin
leaves no orphaned asset reference, verified by test against the existing `ON DELETE SET
NULL` FK), QR codes + a single printable label (`label.svg`, QR + Code128 in one file — the
scope fence's named single endpoint) + a multi-asset PDF label sheet (broader than that
fence, built anyway per this session's explicit instruction — **flagged as a scope
reopening, not resolved**), service reminders (`/service/due`, and a notification scheduler
proven idempotent by running it twice and asserting zero duplicate rows, not just asserted),
and an audit log query endpoint (admin-only, filtered, paginated). No new Alembic migration
was needed — every table these five areas touch already existed in `0001_initial`; verified
the existing migration still round-trips (`upgrade head → downgrade base → upgrade head`)
clean, then reseeded categories/users. 32 new tests, all against real Postgres with real
JWTs (no auth faking, matching the existing pattern exactly).

## Real bugs found and fixed this session
- RQ2's three-way comparison had no confidence interval anywhere — a correctness defect in
  the paper, not a code bug, but the standing "an incorrect number is worse than a missing
  one" rule applies to an unsupported comparison just the same. Fixed with Wilson intervals
  computed from the same run, added to both the run record and the prose.
- Papadakis et al.'s bibliography year (2020 → 2021, DBLP/Crossref both give 2021 as the
  citable print-issue year for what indexed online in 2020).
- Two licence-obligation gaps: Abt-Buy's required dual citation (paper + Leipzig DB Group
  benchmark website) was only half done; Contracts Finder's OGLv3.0 attribution was missing
  its required statement and a link to the licence text.
- The paper's title and author block were empty placeholders, uncaught through the entire
  prior session despite the paper otherwise being feature-complete — found only because this
  session was explicitly asked to check.

## Gotchas that are not plan amendments
- `RagFewShotLLMClassifier.predict_cached_only` bypasses `complete_many` entirely (reads
  `client._read_cache` directly per request) — use this, not a fresh `predict()` call, to
  recompute statistics on an already-partial run. A fresh `predict()` call replays the cached
  rows for free but then attempts every remaining live call, which is what caused the
  original slowdown; nothing about that pacing bug has been fixed, only avoided.
- The retry-pacing fix from the prior session (recording estimated tokens on 429/400 retry
  paths) is still unverified — it may be what caused RQ2's slowdown to ~1 call/25min by
  compounding perceived usage across retries of the same record. **Check this before any
  future quota-heavy run**, starting with a small probe rather than a full n=300 attempt.
- `make seed`'s guard checks `system/api/scripts/seed_categories.py`, but the real path is
  `system/api/src/fcesapi/scripts/seed_categories.py` — the condition is always false, so
  `make seed` always prints "SKIPPED" regardless of whether the scripts exist. Pre-existing,
  not touched this session (out of the ordered scope); run the seed scripts directly by their
  real path if `make bootstrap`'s seed step silently did nothing.
- `system/api`'s `base_url` setting (defaults to `http://localhost:3000`) was sitting unused
  before this session — it is now the QR code's source of truth for the asset's persistent
  URL. `system/web` is still empty (`.gitkeep` only), so nothing actually serves that host;
  this session mounted `/a/{public_id}` directly on the API itself as the interim resolver.

## Next tasks, in order
1. Decide the page-count question above (11 pages: accept, or name a specific cut).
2. Finish RQ2 to n=300 at low-traffic time — check the retry-pacing fix first (see Gotchas).
3. Two figures (recall vs. severity, operating-point curve) — matplotlib still not
   installed, no plotting script exists yet.
4. "Naive precision floor" sub-finding still needs its own measurement; still absent from
   Discussion for lack of a run record backing the specific accepted-pair counts.
5. Decide the label-sheet scope question above (multi-asset PDF beyond the named single
   endpoint) — built and tested, but flagged rather than resolved against the standing fence.
6. A real frontend for `/a/{public_id}` — the API-side resolver this session added is a
   stand-in, not a replacement for whatever `system/web` eventually serves.

## Last verified
**2026-08-22** — research: 515 passed + 12 skipped. `system/api`: 62 passed (30 existing +
32 new). `annotation`: 29 passed. Zero failures across all three suites. `fcesreg`/`system`
boundary intact (`grep -r "import fcesreg" system/` → `pipeline.py` only). Alembic
`upgrade head → downgrade base → upgrade head` verified clean, categories/users reseeded
after. `main.tex`: 11 pages (not 10, see above), 1 `[TBC]` (the author's email), two
`pdflatex` passes, zero undefined refs.
