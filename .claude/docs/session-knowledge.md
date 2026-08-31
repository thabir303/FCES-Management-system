# Session knowledge

Things this project knows that the repository does not record. Read on demand; not imported.

**Provenance tags are mandatory and load-bearing.**

- `[VERIFIED]` — re-checked against a file, `git log`, or command output while writing this
  document. The check is named.
- `[RECALLED]` — remembered from the originating conversation, not re-checked.
- `[UNCERTAIN]` — half-remembered, or possibly predating a context compaction. Treat as a lead,
  not a fact. **Verify before acting.**

A confidently wrong line here is worse than a missing one, because a future session will act on it.

---

## 1. Who the work is for, and how it is reviewed

- `[RECALLED]` The user is an MSc Data Science student; the reviewer in this conversation acts as
  supervisor. Work arrives, is reviewed in detail, and comes back with numbered decisions. The
  review standard is high and specific — they check claims against the repo themselves and have
  caught both my errors and their own.
- `[RECALLED]` **The project does not need to be publishable. It needs to answer RQ1–RQ3 well
  enough to be marked, inside a budget of a few dollars and the time remaining.** This was stated
  explicitly and it governs every scope decision. Methods that would strengthen a venue submission
  but are not needed to answer the three questions were cut.
- `[RECALLED]` The governing test for keeping a method: *removing it would leave one of RQ1–RQ3
  without an answer, or would remove the comparison that gives the answer meaning.* Everything else
  goes, and goes immediately rather than being quietly abandoned.
- `[VERIFIED]` The LLM budget is `CAP_USD = 6.00` with an estimate of ~$1.66 via the Batches API
  (PROJECT_PLAN.md §6.11). Checked by reading §6.11.

### Standing rules the supervisor imposed on paper edits

`[RECALLED]` Four rules, given when the paper was handed over:

1. Correcting a statement of fact that a measurement contradicts **is** mine to do — make the
   change and say what changed and which run contradicted it.
2. Changing what the paper **claims, promises or argues** is not mine. Propose and wait. The test:
   does the sentence state something measurable that turned out false, or something the study
   intends to do?
3. **Never adjust a claim to accommodate a disappointing result.** A disappointing result is
   reported as it stands. If I find myself softening a sentence so a number reads better, stop and
   raise it.
4. Never remove or weaken a threat to validity, and never remove a `[TBC]` by writing in a number
   I have not measured.

`[RECALLED]` Mechanical paper constraints: single file, bibliography inline as `thebibliography`,
no `.bib`, no `bibtex`, two `pdflatex` passes. No comments and no instructions to a reader in the
file; the only placeholders are bare `[TBC]` markers.

---

## 2. Decisions, with the rejected alternative

### Blocking: per-gram q-gram indexing with an overlap threshold

- `[VERIFIED]` Adopted: `n=3`, `min_overlap=8`, chosen on the Corpus A dev partition by *highest
  reduction ratio subject to pair completeness ≥ 0.98*. Checked `research/configs/blocking.yaml`.
- `[RECALLED]` **Rejected: the single-key formulation the plan originally specified** ("first `k`
  sorted char n-grams" as one composite key). It retains only 17% of true pairs and degrades
  monotonically as `k` grows — 0.254 at `k=4` down to 0.028 at `k=32` — because agreement on the
  `k` alphabetically-earliest n-grams is an exact-match key over a derived string. **Kept in the
  sweep as a reported negative result**, worth ~3 lines of Discussion showing empirically why the
  standard formulation is standard. Do not delete the single-key code path.
- `[RECALLED]` Rejected: raising `k`. It makes the key *more* specific and the result worse. This
  was measured before concluding, not assumed.
- `[RECALLED]` Plain q-gram indexing at `min_overlap=1` gives PC 1.000 but RR 0.588 on Corpus A and
  21,433,548 candidates on Corpus B — too little pruning to carry the pipeline. The overlap
  threshold is what made it usable.

### Distractor mining: three rules, each replacing the last

- `[RECALLED]` **v1, keyed on `record_id`: 48% contaminated**, measured by hand audit of 40 pairs.
- `[RECALLED]` **v2, keyed on the procurement reference (`tender_id`) with stage suffixes
  stripped, plus exclusion of same-buyer-identical-title: 35% contaminated**, hand audit of 40
  fresh pairs (seed 7). Judgements in `annotation/labels/distractor_audit_v2.judged.json`
  `[VERIFIED]` — file exists, checked with `ls`.
- `[VERIFIED]` **v3 adds extraction of references embedded in titles** (`UKRI-3547` etc.). Mined
  set 588 → 570. Checked by running `audit_distractors.py`.
- `[RECALLED]` **Rejected: `ocid` as the process identifier.** Tested explicitly — all 40 audited
  pairs had *distinct* OCIDs, because Contracts Finder mints one per notice rather than per
  contracting process. An award notice and its tender notice differ in `id` and in `ocid` while
  describing one procurement.
- `[RECALLED]` **Rejected: matching on a shared alphanumeric run of ≥5 characters.** I proposed it;
  the supervisor refused it. Inside nineteen-digit references it fires by coincidence as well as by
  relation, so it would drop genuine distractors and bias the set, and its cost on that side is
  unmeasured.
- `[RECALLED]` Root cause, and it is the finding: **Contracts Finder carries no reliable process
  identifier.** The published reference often names the *issuing organisation* rather than the
  procurement — two notices from one body share only a trailing org code — and `buyer_id` failed to
  resolve to one body in 9 of the 14 contaminated pairs. No stack of heuristics closes this; each
  rule buys less than the last while being harder to justify.
- `[RECALLED]` Therefore: bound the set at 200 and verify every pair by hand. The trajectory
  **48% → 35% → (verification) → 0% by construction** earns ~3 sentences in Discussion.
  **Superseded (amendment 7, below) — do not act on this bullet.** Full verification of a
  small bounded pool never certified the unbounded pool actually used downstream; kept here
  as the record of the reasoning that led to the sampling redesign, not as current practice.
- `[RECALLED]` I declined to publish a v3 contamination rate from my own eyeballing of 40 fresh
  pairs. The draw was from a different population so it is not comparable with 35%, and several
  calls would have been guesses without reference and date evidence to hand. The supervisor's
  200-pair verification settles it — **superseded by amendment 7**: a 50-pair random sample
  with a Wilson interval settles it now instead.
- `[VERIFIED]` **Amendment 7 — contamination is measured, not verified away.** The supervisor
  ruled that Contracts Finder's contamination is a property of the corpus (one procurement,
  several notices), not a defect the mining rule could be corrected out of — so exhaustively
  verifying a small bounded pool was never actually certifying the full pool used downstream,
  only itself. `annotation/judge_distractors.py` rewritten: draws 50 pairs uniformly at random
  from the mined pool (no longer bounded for verification, only for mining cost), reports the
  rate with a Wilson score 95% CI, and **the pool is used unfiltered by C6 and the transfer
  runner** — no pair is dropped on a judgement's strength. Corpus B precision is reported as a
  lower bound with the rate and interval stated alongside; the rate is never used to correct
  the precision figure arithmetically. `degrade.py`'s `max_pairs` comment, which stated the
  old verification rationale, corrected to a mining-cost cap. The 48%→35% mining-rule story is
  untouched — a measurement of the rule's own correction, not of what full verification left.
  Discussion cost, stated plainly: Corpus B's in-domain duplicate-detection result now carries
  a known, quantified impurity in its negative set; Corpus A is unaffected (published labels).
  Paper edit (the "verified by hand" / "bounded... complete manual verification" sentence in
  §Methodology) is the supervisor's to make, not applied here.
- `[VERIFIED]` **Amendment 7, the sample itself: 42.0% contaminated (Wilson 95% CI
  29.4%–55.8%), n=50, unsure 4/50 = 8%.** Judged by Claude, not the Groq adjudicator, on the
  supervisor's explicit instruction: the model under test must never sit on both sides of the
  comparison it is part of. Judged seed=0, the tool's default, against the live 570-pair v3
  pool — `annotation/labels/distractor_judgements.jsonl` records a reason per pair naming the
  signal (shared reference-number roots across a suffix, explicit "AWARD NOTICE"/"previous
  notice published under..." self-citation, verbatim title+description reuse, lot/phase/group
  language distinguishing genuinely separate line items). Reproduced via `--summary` against
  the committed file. **Not comparable to v2's 35%**: different rule (v3 adds title-reference
  extraction), different sample, and a v3 rate was explicitly declined earlier from an
  unstructured 40-pair eyeball (the "several calls would have been guesses" bullet above) —
  this is the first rate v3 has actually earned. Four pairs called `unsure`: two turned on a
  shared reference suffix that also appears on an unrelated pair elsewhere in the same sample
  (case 34, "-72814"), disqualifying it as a clean signal for that pair specifically; one on
  a title-specific framework name with no corroborating structural evidence (case 49); one on
  two subcontract-package names inside one large scheme that could be the same package
  renamed or two genuinely separate ones (case 47).

### CPV division set: eight divisions

- `[VERIFIED]` Adopted `{30,31,32,33,38,42,43,44}`; 39 (furniture) and 48 (software) dropped.
  Checked `research/configs/blocking.yaml` and `T1_division_choice.tex`.
- `[RECALLED]` Reasoning: division 48 alone is 15,939 records = **35.4% of the candidate corpus**,
  so a third of an "equipment register" would be software licences. Support survives the cut —
  81 classes ≥50 examples at 88.3% coverage, versus 122 at 90.7%.
- `[RECALLED]` Ingest deliberately **retains all ten divisions** so `run_profile.py` can measure
  both sets from one parquet. The restriction is applied downstream.

### Terminology: the four-digit level is a *class*, not a group

- `[VERIFIED]` `cpv.py` exposes `division()` and `cpv_class()`; `LEVELS == ("division","class")`.
  Checked by reading the file.
- `[RECALLED]` The supervisor corrected this. Official CPV is division (2) / group (3) / class (4)
  / category (5). Calling the four-digit level a "group" is wrong in a paper about that vocabulary.

### Splits

- `[RECALLED]` **The plan's original B5 criterion was impossible.** It demanded zero `record_id`
  overlap on both corpora, but Abt-Buy's supplied splits are pair-level over a fixed record pool
  and 1,359 records appear on both sides. Re-splitting would break comparability, which §4.4
  forbids. Resolved by splitting the guarantee: **record-level for Contracts Finder, pair-level for
  Abt-Buy.** The paper decided it.
- `[RECALLED]` `splits.json` was frozen over the **ten-division** ingest while analysis runs on
  eight. This is fine — filtering after a temporal split preserves the temporal guarantee — and the
  paper states which corpus the split was defined over.

### Scope cuts made for budget

- `[RECALLED]` **Zero-shot classification removed.** Three approaches, not four. It answered what
  the in-context examples contribute, a narrower question than RQ2 asks; moved to further work.
  Halves classification call volume.
- `[RECALLED]` **Band subsampling and bootstrap intervals removed.** I proposed three options for
  fitting the cascade into budget (fewer seeds / fewer severities / subsample the band with a
  bootstrap CI). **All three were rejected** — the budget was solved by cutting scope rather than
  cutting precision.
- `[RECALLED]` Instead: the three matchers with no marginal cost keep the full severity ×
  repetition factorial; the **cascade runs at 3 severity levels × 1 repetition, adjudicating every
  pair in its band**. Resolution traded for exactness.
- `[RECALLED]` Earlier cuts (Amendment 4/5): natural-duplicate experiment, Amazon-Google and
  Walmart-Amazon, dashboard, user admin UI, audit viewer, label sheet builder, Playwright, trigram
  search, soft delete + status enum + delete endpoint, `value_gbp`, `purchase_date`,
  `POST /auth/logout`.

---

## 3. Dead ends and bugs — the highest-value section

### Bugs found by measurement, not by tests

- `[VERIFIED]` **`np.save` appends `.npy`.** In `embed.py` the write-then-rename cache is only
  atomic if the written path and the renamed path are the same string; `np.save(path, arr)` silently
  writes to `path + ".npy"`, so a partial run would read back as real. Fixed by opening the handle
  explicitly (`tmp.open("wb")` then `np.save(f, vector)`). Checked lines 78–84 of `embed.py`, where
  the comment records the reasoning.
- `[VERIFIED]` **`capture_env()` ran after the work, not before.** Commit `bceb670` "A3 fix:
  capture the environment before writing run artefacts" touches `runs.py` and `run_profile.py`.
  Checked with `git show --stat bceb670`.
- `[RECALLED]` **The `git_dirty` gate refused its own first real run** and turned out to be right.
  The supervisor called this the more valuable of two bugs: "a gate that passes when it should fail
  teaches nothing".
- `[VERIFIED]` **Paths resolved against the working directory.** `DEFAULT_LEXICON_PATH` and six
  other constants were bare relative paths. From the repo root the suite passed; from `research/`
  it gave **14 failures**. Diagnosed by running `cd research && pytest`. Fixed by `paths.py`, which
  walks upward from `__file__` for a marker file. **Rejected: counting parents** (`parents[3]`) —
  shorter but silently wrong the moment a module moves. Verified by running the suite from three
  directories.

- `[VERIFIED]` **`select_threshold` could split a tie group and promise a precision it did not
  deliver.** It swept precision per *item*; a threshold admits every item scoring at or above it,
  so stopping part-way through a run of equal scores returns a threshold whose real precision is
  below target. `scores [0.9, 0.9, 0.8]`, `labels [1, 0, 1]`, target 0.95 → returned 0.9,
  delivered 0.5. **Not a corner case**: `ExactMatcher` emits only 1.0 and 0.0, so on it every pair
  ties with most others and the per-item sweep is wrong almost everywhere. It sets the cascade's
  band and RQ3's operating point, so the failure was a silently overstated guarantee on the
  headline number. Fixed by `metrics.threshold_sweep` (C8), which collapses each tie group to one
  attainable point; both modules now share it. Found while writing C8, not by an existing test.
- `[VERIFIED]` **A pandas null is truthy, so `if not value:` does not skip it.** 441 of 2,173
  Abt-Buy records (20%) carry a null description stored as float `nan`. Two failures from the one
  root: `degrade_record` raised `TypeError` in `re.sub` (so `degrade_frame` and
  `make_duplicate_pairs` could not run on Corpus A **at all**), and `merge_fields` silently wrote
  the literal `"nan"` into the title. The silent one is worse — it does not stop the run, and both
  copies of a degraded pair receive the same spurious token, making duplicates *easier* to match
  and flattering every Corpus A figure. Fixed with `_present_text`, matching `schema.text_of`,
  which reads the same fields through `fillna("")`. Corpus B has no null descriptions, which is
  why this survived: nothing exercised the degradation model on Corpus A until the Corpus A sweep
  was ordered.

- `[VERIFIED]` **`select_threshold`'s `tp > 0` guard admits degenerate operating points, and
  they do not generalise.** Measured on Corpus A dev with Tfidf, the thresholds it returns at
  severity 0.25, 0.3 and 0.75 admit **2, 14 and 1 pairs** respectively (recall 0.002, 0.017,
  0.001). Precision estimated from one pair is not a precision estimate, and the consequence
  on test is concrete, not theoretical:

  | severity | dev picks `upper` claiming ≥0.95 | delivered on test |
  |---|---|---|
  | 0.30 | 0.2184 | auto-accepts 10/1916 at precision **0.800** |
  | 0.75 | 0.1362 | auto-accepts **0**/1916 at precision **0.000** |

  This is live, not cosmetic: `select_threshold` sets the cascade's `upper`, so a degenerate
  pick auto-accepts on a threshold supported by a single pair. **The apparent
  non-monotonicity of the "cliff" was entirely this artefact.** Under a support floor the
  picture is monotone and the cliff moves:

  | rule | reachable at 0.95 (Tfidf, dev) |
  |---|---|
  | `tp > 0` (current) | 0.0, 0.05, 0.1, **0.25, 0.3, 0.75** (last three degenerate) |
  | `tp >= 20` | 0.0, 0.05, 0.1 — **unreachable from 0.25 up** |
  | Wilson lower bound ≥ 0.95 | **0.0 only**, and recall there falls 0.259 → 0.146 |

  So the target becomes unreachable **between severity 0.1 and 0.25**, not at 0.5 as the
  first pass suggested. **Raised, not fixed** — every candidate rule changes a number the
  paper reports (Wilson lowers the severity-0 automated share; `tp >= 20` carries an
  arbitrary constant), so the choice is the supervisor's. `tp >= 20` and Wilson agree that
  nothing above 0.25 is reachable, which is the part that does not depend on the rule.

### Normalisation subtleties that cost time

- `[RECALLED]` **Mojibake repair must run before NFKC**, contrary to the order §6.2 originally
  listed. NFKC maps `™` → `TM`, which destroys the byte sequence the repair recognises: `â„¢`
  becomes `â„TM` and no longer round-trips. §4.1 documents 331 mojibake rows in the 2025 subset, so
  the stated order would have silently failed on known data.
- `[RECALLED]` cp1252 leaves five slots undefined (0x81, 0x8D, 0x8F, 0x90, 0x9D). A strict encode
  refuses text containing them, and `”` misread as cp1252 lands on U+009D — the commonest damaged
  character in procurement text. A latin-1 fallback was needed.
- `[RECALLED]` Apostrophes are **deleted**, other punctuation becomes a space. `pump/valve` must
  give two tokens; `buyer's` must give one.
- `[VERIFIED]` **The two tiers behave differently on unit notation, and an earlier version of this
  file got it wrong.** Measured just now by calling both functions on all four variants:

  | input | `normalise_text` | `normalise_key` |
  |---|---|---|
  | `1.5kW` | `1 5kw` | `15kw` |
  | `1,5kW` | `1 5kw` | `15kw` |
  | `1.5 kW` | `1 5 kw` | `15kw` |
  | `1,5 kW` | `1 5 kw` | `15kw` |

  `normalise_key` collapses all four to `15kw`, so **the exact-match baseline is genuinely blind to
  the separator**. `normalise_text` is blind to `.` versus `,` (both become a space) but **not** to
  whether a space precedes the unit — `1 5kw` versus `1 5 kw` is a token-count difference. Since
  `text_of` feeds TF-IDF and the embedder, those two tiers see a distinction the exact tier does
  not. Do not state that all variants normalise identically; that is true only of the key path.
- `[RECALLED]` **Still open:** whether separator-blindness at key level is intended. `vary_units`
  injects exactly this variation, so the exact-match baseline is by construction immune to one of
  the seven error classes. Raised twice, never ruled on. This is the one genuine open decision left
  from the original four uncertainties.

### Test-writing mistakes I made repeatedly

- `[RECALLED]` I wrote assertions on **single random draws** that can legitimately reproduce the
  input. `vary_units("230V", rng(5), 1.0)` returned `"230V"` because the identity is one of the
  reachable forms. Fixed by asserting variation *across* draws.
- `[RECALLED]` I asserted **arbitrary absolute thresholds** (`s[0] > 0.7` on a TF-IDF cosine over a
  three-document corpus, where IDF weights are extreme). The meaningful claim was the ordering.
- `[RECALLED]` My test fixtures repeatedly lacked columns the code had just started requiring
  (`buyer_id`, `tender_ref`). Twice the right fix was to make the *code* tolerant and warn, not to
  patch the fixture silently.
- `[RECALLED]` `grep -F` against a hard-wrapped file misses phrases that span a line break. This
  produced a **false** verification failure on the paper fingerprint. Collapse whitespace before
  matching, or use single-line phrases.

### Blocking design trap

- `[RECALLED]` Generating n-grams over the **concatenated** title produces grams straddling word
  boundaries, whose presence depends on word order — reintroducing exactly the sensitivity that
  sorting the grams exists to remove. n-grams are generated **per token**.

### The U-shape is a finding, not a defect

- `[RECALLED]` Corpus B pair completeness is **non-monotonic in severity**: 0.758 → 0.285 → 0.216 →
  0.259 → 0.384 across severities 0.1–0.9. Cause: `merge_fields` moves the description into the
  title and `char_noise` inserts characters, so titles get *longer* at high severity and share more
  grams again. **Severity is not a monotone difficulty axis for threshold-based q-gram blocking**,
  because one of the seven classes adds content while the other six remove or corrupt it.
- `[RECALLED]` The supervisor ruled: **do not make field merging severity-invariant to smooth the
  curve.** That would trade a real finding for a tidier figure. It must be explained in Results and
  carried into Discussion, since a reader assuming monotonicity will misread every degradation curve.

---

## 4. Corrections the supervisor gave me

- `[RECALLED]` **Acceptance criteria must test the implementation, never the outcome.** C6's
  original `band_fraction < 0.15` and C3's `F1 ≥ 0.55` made measured results into build gates,
  creating pressure to tune until they pass. A band fraction of 0.30 is a finding about the method.
- `[RECALLED]` **The profile exists to test what the Methodology assumed.** The paper claimed
  "class imbalance is severe because a small number of divisions account for most equipment"; the
  measurement gave 8.2:1 at division level (moderate) and 1951:1 at class level (severe). The
  measurement wins and the paper is corrected. *This should not have needed a round trip through
  the supervisor* — correcting a contradicted statement of fact is mine to do.
- `[RECALLED]` **Do not average an asymmetry away.** `buyer_id` exists on Corpus B and not on
  Corpus A; which schemes are available where is part of what the blocking result says.
- `[RECALLED]` **Leaving a figure unmeasured is better than estimating it.** Reporting Corpus B
  pair completeness as `None` rather than inferring it was explicitly endorsed.
- `[RECALLED]` **Stopping on a failed verification check is always right, even when inconvenient.**
  "A verification rule that gets waived the first time it is inconvenient is not a verification
  rule." The one round trip costs nothing against the failure mode.
- `[RECALLED]` **Commit per §11 task.** One commit for all of Phase A and B undermined §12.6: a run's
  git SHA is worth little if the whole codebase is one SHA.
- `[RECALLED]` **Keep the paper in its own commits.** A commit that swept the plan reductions and
  the paper file swap together left the two of us reading different states.

---

## 5. Half-finished work, and what "finished" means

- `[VERIFIED]` **C5 (`llm.py`) is not started.** `research/src/fcesreg/llm.py` does not exist
  (checked with `ls`). Finished = disk cache keyed on `sha256(model+system+prompt+schema)`, one
  global `results/ledger.jsonl` carrying `run_id`, hard `cap_usd` guard, `complete_batch` via the
  Message Batches API, and the C5 criterion: a $0.20 pilot runs, re-running the identical set costs
  **exactly $0.00** with every row logging `cache_hit=true`.
- `[VERIFIED]` **The 50-pair distractor sample is unstarted** (redesigned, amendment 7 —
  was 200-pair full verification, see §2). `annotation/labels/distractor_judgements.jsonl`
  does not exist; only the v2 audit judgements do. `annotation/judge_distractors.py` rewritten
  and unit-tested (`annotation/test_judge_distractors.py`, the sampling math and Wilson
  interval — not smoke-tested only, as the old tool was). Finished = 50 pairs judged by the
  supervisor (~12 min), contamination rate + CI reported. **The pool itself is used by C6 and
  the transfer runner unfiltered** — finishing this no longer means producing a retained set,
  only a rate. C6 and `run_transfer.py` are still blocked on it.
- `[VERIFIED]` **`annotation/annotate.py` does not exist** (checked with `ls`), but the Makefile's
  `annotate` target invokes it and §6.15/§13.3 specify it. The 300-item timed annotation exercise
  is unstarted, and it produces *two* results RQ3 needs: the label-noise estimate and
  `mean_seconds_per_item`.
- `[VERIFIED]` **`system/` is essentially unbuilt** — Phase D onward. Only
  `system/api/pyproject.toml` exists.
- `[RECALLED]` **The in-domain blocking configuration is specified but not run.** Decision: run
  both the transferred configuration (External Validation) and a Corpus-B-selected one (in-domain
  RQ1 answer), report both, **never combine them**.
- `[RECALLED]` **The degradation-model-versus-audit comparison** (`audit_real_errors.py`, G2) is
  not built. Where the audit shows an error class at a rate the model does not reproduce, that goes
  in the run record **even where it does not change the parameters**, because External Validation
  compares exactly those two distributions.

---

## 6. Where PROJECT_PLAN.md or main.tex may now be wrong

- `[VERIFIED]` **`make data` and `make annotate` are broken today, not merely unbuilt.** `make data`
  calls `research/scripts/build_taxonomy.py`, which does not exist, while the README presents
  `make data` as step two of reproduction; `make annotate` calls `annotation/annotate.py`, which
  does not exist and is on the critical path for RQ3. CLAUDE.md's line that "a missing script there
  is the build order, not a bug" is **fair for `make experiments` only** — the supervisor confirmed
  they over-generalised it. These two belong under *Blocked*, not under *not yet built*.
- **§6.8 versus the code: no conflict — closed.** The supervisor checked. §6.8 already carries
  `mode: str = "per_gram"` in the signature, annotates per-gram as "what the study adopts", marks
  `single_key` as "retained only for the sweep", and its operating-point paragraph independently
  states per-gram, `n=3`, `min_overlap=8`.
- **CONFIRMED CONFLICT — the page-budget decision was never written into PROJECT_PLAN.md.** The
  supervisor verified: §10's runner table still lists `T4_abtbuy.tex` (line 1415) and
  `T6_classification.tex` (line 1417) with full specs, line 1460 still names `T4_cf_sweep.tex`, and
  §14's amendment log records no page-budget decision. The agreed budget (three tables, two figures,
  `T8_cost` collapsed to prose) exists only in conversation. **§10 is stale.** Not resolved here.
- `[RECALLED]` The paper reserves a `[TBC]` for the Corpus B deduplication sample size, and the
  plan says that number must come from a run rather than a config. **No run produces it yet.**
- `[VERIFIED]` **OPEN CONFLICT — the paper's cascade specification has no defined behaviour when
  no threshold meets the precision target, and on degraded Corpus A none does.** §Methodology says
  "Both thresholds are selected on the development partition to satisfy the precision target of
  RQ3 while minimising the size of the adjudicated band." Measured on Corpus A dev, best precision
  at any usable operating point (`tp>=20`): Tfidf 1.000 clean → **0.533** at severity 0.5 → 0.462
  at 1.0; embeddings 1.000 → **0.352** → 0.190. **Embeddings are worse, so the base matcher does
  not rescue it.** With no threshold at 0.95 the upper bound is undefined and the two readings
  diverge by a factor of 38 in cost, measured on the real test split at severities (0.0, 0.5, 1.0):
  - **A — re-fit on dev at each severity** (the faithful reading; band = everything the base
    matcher cannot confidently place): 103 + 1916 + 1895 = **3,914 adjudications**, 1.48M tokens,
    **~7.4 days** of quota, tokens binding.
  - **B — fit once on clean dev, carry unchanged**: 103 + 0 + 0 = **103 adjudications**, ~0.2 days
    — but zero at severity 0.5 and 1.0 means every degraded pair falls below the clean *lower*
    threshold and is auto-rejected. That is not a cheaper measurement of the cascade, it is the
    cascade never firing at the severities it exists for.

  Token cost is **measured, not assumed**: 377 tokens/adjudication (281 in + 96 out) over 8 live
  calls on real Abt-Buy test pairs, tagged `condition="cost_probe"` in the ledger so `run_costs.py`
  excludes them as it excludes `c5_pilot`. Output hit the `max_tokens=96` cap on every call, so
  that half is set by configuration, not by the model. Quota: 200k tokens/day binds before 1000
  requests/day, giving ~530 adjudications/day. **Raised, not resolved** — the supervisor's arithmetic
  ("low hundreds to low thousands, which the daily quota absorbs") holds only under reading B.

### Page budget as agreed

`[RECALLED]` Reserve prose first: ~¾ page Results, 1 page Discussion, 0.4 page Conclusion, leaving
~2 pages of exhibits. **Three tables and two figures.** Full: `T1_corpus_b` (with the division
choice folded in as extra rows), `T3_blocking`, `T6_classification`. Prose: corpus A sizes, the
discard tally, leaf sparsity, **and `T8_cost`**. Figures `F1_severity` and `F2_operating_point`
both stay — neither survives as prose. If it still overruns, Reproducibility drops to two lines.

---

## 7. Environment facts that cost time to discover

- `[VERIFIED]` **TeX is installed but not on PATH.** BasicTeX lives at `/Library/TeX/texbin`, which
  is absent from the default PATH, so `pdflatex` appears missing when it is not. `make paper`
  prepends it. Checked by running `make paper` → "clean build, 6 pages".
- `[RECALLED]` `IEEEtran.cls` is present in the TeX Live 2026 basic distribution; no `tlmgr install`
  was needed.
- `[RECALLED]` Postgres runs on **5433**, not 5432, to avoid clashing with a local install.
  `pgcrypto` is available in the image.
- `[RECALLED]` `torch.backends.mps.is_available()` is `True` on this machine. §12.7 forbids using
  it — CPU only.
- `[RECALLED]` The Contracts Finder bundle sizes in §4.1 were originally wrong (stated 14–17 MB;
  actual 54–59 MB compressed). Corrected in the plan.
- `[RECALLED]` The supervisor's own `pdflatex` reports 9 pages because they build against a stub
  class; **our 6-page figure is the correct one**.

---

## 8. Machine-local memory directory

`[VERIFIED]` the project's `~/.claude/projects/<encoded-repo-path>/memory/` directory exists
but is **empty** — nothing to fold in. Checked with `ls -la`.
