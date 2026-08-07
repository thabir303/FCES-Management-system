# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@.claude/STATE.md

Background this file does not carry — decisions and their rejected alternatives, dead ends, bugs and
how they were diagnosed — is in `.claude/docs/session-knowledge.md`. Read it on demand; it is not
imported. Path-scoped rules load automatically from `.claude/rules/` when a matching file is touched.

## What this is

An MSc project with two deliverables sharing one repository, plus the paper that reports on them:

- **`research/`** — `fcesreg`, the register migration pipeline: duplicate detection, CPV taxonomy
  assignment, and the automated-share-at-fixed-precision analysis answering RQ1–RQ3.
- **`system/`** — the FCES asset management system (FastAPI + Next.js + Postgres) the pipeline is
  deployed inside.
- **`main.tex`** — the paper, and the artefact that is actually submitted.

## Authority order — read before changing anything

1. **`main.tex` is the paper and the authority.** Anything it claims, the code must deliver;
   anything the code produces that the paper does not claim is not part of this project.
2. **`PROJECT_PLAN.md` is the build specification** (~1700 lines). §11 is the dependency-ordered
   build table; §3–§10 are the reference material each task points into. Module docstrings cite
   plan sections (`§6.8`) and build tasks (`C4`) — those citations are load-bearing, keep them.
3. **Where the plan and the paper disagree, stop and raise it rather than choosing a side.**
   §14 is an amendment log of exactly such conflicts and how they were resolved.

Standing rules from the plan that govern every change:

- **Never invent a number.** Unmeasured quantities stay unmeasured. Every `[TBC]` in the paper is
  filled from `results/tables/`, never by hand.
- **Acceptance criteria test the implementation, never the outcome.** A criterion may assert a
  value is produced, internally consistent, or inside a range only a bug could violate. A
  disappointing measurement is a finding about the method, not a build failure. Do not tune a
  method to make a number look better.
- **If a ground fact in §4 does not hold once the real files are opened, stop and report** rather
  than adapting silently. The paper's Methodology is written against those facts.

## Commands

```bash
make bootstrap      # venv (pyenv 3.12.8), editable installs of both packages, docker db, migrate, seed
make data           # ingest both corpora, build the CPV taxonomy, freeze the splits
make annotate       # timed annotation exercise — must run before `make experiments`
make experiments    # every runner, in dependency order
make tables         # regenerate results/tables/ from results/runs/
make test
make smoke          # 100 records, 1 severity, 1 seed — <3 min, $0.00 API spend
make paper          # two pdflatex passes; TeX lives at /Library/TeX/texbin, not on PATH
make dev            # uvicorn api on :8000 + next dev
make clean-db       # docker compose down -v && rm -rf .pgdata
```

Single test / single module:

```bash
.venv/bin/pytest research/tests/test_blocking.py -q
.venv/bin/pytest research/tests/test_degrade.py::test_same_seed_is_byte_identical -q
```

Runners are config-driven and never take tuning flags:

```bash
.venv/bin/python research/scripts/run_blocking.py --config research/configs/blocking.yaml
```

`make annotate` precedes `make experiments` because it produces both the label-noise estimate and
the `mean_seconds_per_item` figure the operating-point analysis consumes.

`make paper` must build clean before any commit touching `main.tex`.

## Architecture

Path-specific rules — the `fcesreg` prohibitions, provenance, the paper, the system scope fence —
load automatically from `.claude/rules/` and are not repeated here.

### The boundary that makes the deployment claim true

`fcesreg` is pure Python over pandas DataFrames and numpy arrays. It **must never import from
`system/`**, touch a database, or reach the network outside `ingest_*` and `llm`. The API adapts
between HTTP/DB and DataFrames in exactly one file, `system/api/src/fcesapi/services/pipeline.py`
— `grep -r "import fcesreg" system/` should return that file alone.

### The canonical Record shape

Both corpora and the spreadsheet importer map into `schema.RECORD_COLUMNS`; everything downstream
of ingest sees only that shape. Critically, `schema.NULL_IN_BOTH_CORPORA` —
`manufacturer`, `model`, `serial_number` — are **wholly null in both research corpora**. Nothing in
the research path may key, block, group or score on them; doing so silently yields one enormous
block, no blocks, or an empty negative set. Where a brand proxy is needed, use
`blocking.block_by_leading_token`, which the paper states and evaluates as an approximation.

Corpus letters follow the paper: **Corpus A is Abt-Buy** (duplicate ground truth), **Corpus B is
Contracts Finder** (category ground truth).

### Frozen splits

`data/processed/splits.json` is written once and never regenerated; `splits.freeze()` refuses to
overwrite it. The guarantee differs by corpus and that difference is deliberate: Contracts Finder
is **record-level** (temporal split at `2025-01-01`, no `record_id` on both sides), Abt-Buy is
**pair-level** (supplied splits used exactly as given for comparability with the literature; 1,359
records legitimately appear on both sides).

### Corpora are never committed

`data/raw/`, `data/processed/`, `results/tables/*` and `storage/` are gitignored. The repository
carries acquisition scripts and record identifiers instead; README has the download instructions.
Ingest verifies the 32-column Contracts Finder header and **fails loudly with the exact diff** on
any difference — never coerce, never rename silently, never fall back to positional columns.

Tests that assert against real corpora use `conftest.requires(...)` to skip with a reason naming
the missing artefact. On a clean clone those skips are expected, not failures.

### Dependency split

`rapidfuzz` belongs to `system/api` only, where it guesses import column headers. **No fuzzy-string
matcher is part of the duplicate-detection tier** — the paper does not claim one. Do not add it to
`research/pyproject.toml`.

## Conventions

- `snake_case` Python, `camelCase` TypeScript, `snake_case` SQL. API JSON keys stay `snake_case` —
  no translation at the boundary.
- FastAPI errors are `{"detail": {"code": "asset_not_found", "message": "..."}}`. 401 unauthenticated,
  403 wrong role, 404 missing, 409 conflict, 422 validation. Never leak a stack trace.
- All settings come from environment through a Pydantic `Settings` class. **Never `os.getenv`
  inline.**
- Postgres runs on **5433**, not 5432.
- Scope fences (§1.1, §13) — these were removed deliberately, do not reintroduce: no dashboard, no
  user-administration UI, no audit viewer, no label-sheet builder, no Playwright or browser E2E, no
  trigram search, no delete endpoint or soft delete or status enum, no hazard taxonomy, no
  eight-digit CPV leaf classification, no `--corpus natural` hand-labelling.
