# FCES Asset Management System — Implementation Specification

**Audience:** an AI coding agent. This document is the single source of truth for what to build.
It contains file paths, function signatures, database DDL, API contracts and acceptance criteria.
The open decisions formerly listed in §13 have been settled; §13 now records the answers.

**How to use this document:** work through §11 (Build Order) top to bottom. Each task names the
files it creates and the acceptance criterion that proves it is done. Sections §3–§10 are the
reference material each task points into. Nothing outside §11 needs to be done in order.

**Standing rules.**

1. `main.tex` is the paper. Anything the paper claims, the code delivers;
   anything the code produces that the paper does not claim is not part of this project. If this
   document and the paper disagree, **stop and raise it** — do not choose a side.
2. **Never invent a number.** If a quantity is unmeasured, say so and leave it unmeasured. Every
   `[TBC]` in the paper is filled from `results/tables/`, never by hand (§12.6).
3. **Acceptance criteria test the implementation, never the outcome.** A criterion may assert that
   a value is produced, is internally consistent, or is within a range that only a bug could
   violate. It may never require a measured result to take a particular value. A disappointing
   measurement is a finding about the method, not a build failure.
4. If a ground fact in §4 does not hold once the real files are opened, **stop and report** rather
   than adapting silently. Those facts are what the paper's Methodology is written against.

---

## 1. What we are building

Two deliverables that share one codebase.

### 1.1 The system (client requirement, verbatim)

> Creation of an asset management system for FCES. FCES has a large volume of equipment being
> moved into Calon, currently recorded in a static spreadsheet. A more manageable live system is
> required. The ability to manage equipment via a live asset system that can be accessed via
> QR/barcode, can be updated live and provides an easy-to-use system. We would like a variety of
> data recorded per item: photos, data, PDFs. Extras like floor plans showing location of machine,
> link to health and safety / risk assessments for that equipment. Reminders when it is due for
> service. Levelled access, potentially read-only for limited staff.

Requirement → implementation mapping:

| Requirement | Where it is built |
|---|---|
| Live register, not a spreadsheet | §5 schema + §7 API + §8 UI |
| Accessed via QR **and** barcode | `assets.public_id` (UUID → QR) and `assets.asset_tag` (→ Code128). §7.4, §8 `/a/[publicId]` |
| Updated live | §7 write endpoints; optimistic UI on the detail page |
| Photos, data, PDFs per item | `attachments` table, `kind` enum. §5.6 |
| Floor plans showing machine location | `floorplans` + `locations.x_pct/y_pct`. §5.3, §8 `/floorplans/[id]` |
| Link to H&S / risk assessments | `attachments.kind = 'risk_assessment'`, surfaced first on the asset page and the QR landing page. §8.3 |
| Service-due reminders | `assets.next_due_at` (generated column) + **a daily scheduled job writing `notifications` rows** + `/service` view + `GET /service/due`. §5.5, §5.8, §7.6 |
| Levelled access (read-only tier) | `users.role` enum `admin` / `technician` / `readonly`. §7.2 |
| Audit log of changes | `audit_log` table, written by the service layer. §5.6 |
| Bulk import of the legacy spreadsheet | §9, the two queues (auto / review) |

#### System scope fence

The system's obligation is the brief above plus exactly what the paper's *Integration into the
Delivered System* section claims: item records with attached photographs and documents, a QR label
per item resolving to a persistent item URL, pin placement on an uploaded floor plan image, service
interval tracking with scheduled reminders, tiered access across administrator / technician /
read-only, an audit log, and the bulk import wizard with its two queues. **Nothing else is in
scope.** Specifically out of scope, and not to be built:

- a dashboard route
- a user administration interface — users are seeded by script (`scripts/seed_users.py`)
- an audit *browsing* interface — the paper claims the log, not a viewer
- a label sheet builder beyond a single printable label endpoint
- Playwright end-to-end coverage
- a trigram fuzzy search path — one working search mechanism is enough at this scale
- soft delete, the four-value status enum, and any delete endpoint
- `value_gbp` and `purchase_date`
- `POST /auth/logout`, which does nothing server side

**Protection order** if time runs short. Cut from the bottom: (1) the bulk import wizard,
(2) the review queue, (3) the `fcesreg` / `system/` boundary, (4) the experiment runners, (5) the
provenance machinery in §12.6.

### 1.2 The research pipeline

A Python package `fcesreg` that measures how much of a spreadsheet→register migration can be
automated: duplicate detection, taxonomy classification, and the automated-share-at-fixed-precision
analysis. It is not a side project — **the system's bulk import wizard imports `fcesreg` directly.**
One deduplication implementation, one classifier implementation, used by both.

### 1.3 The contract between them

```
research/src/fcesreg/   ← the algorithms (pure, no web deps, no DB deps)
        ▲          ▲
        │          │
system/api/         research/scripts/run_*.py
(imports fcesreg)   (imports fcesreg)
```

`fcesreg` must never import from `system/`. It takes DataFrames and returns DataFrames. The API
layer adapts between HTTP/DB and those DataFrames. This is what makes "the pipeline is deployed
inside the delivered system" a true statement rather than a claim.

---

## 2. Stack and bootstrap

| Layer | Choice | Version |
|---|---|---|
| Frontend | Next.js App Router + TypeScript + Tailwind | Next 15+, Node 22 LTS |
| Backend | FastAPI + Pydantic v2 + SQLAlchemy 2.x + Alembic + rapidfuzz | Python 3.12 |
| DB | PostgreSQL | 16, via Docker |
| Research | numpy, pandas, pyarrow, scikit-learn, sentence-transformers, anthropic | Python 3.12 |
| Object storage | local filesystem `storage/` behind an interface | — |

**`rapidfuzz` is a system dependency, not a research one.** Its only use is guessing column-header
mappings in the import wizard (§9.1). **No fuzzy-string matcher (Jaro-Winkler or otherwise) is part
of the duplicate-detection tier** — the paper does not claim one, and character n-gram TF-IDF
already covers the lexical tier on the only short field these corpora have.

**Python version note — verified.** `pyenv install 3.12.8` succeeds on this machine, and
`torch 2.13.0` + `sentence-transformers 5.6.1` install and import cleanly under it. Use 3.12.8. The
fallback to 3.14 is no longer needed. `torch.backends.mps.is_available()` is `True` on this hardware
— **do not use it** (§12.7).

```bash
pyenv install 3.12.8
pyenv local 3.12.8
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e "./research[dev]"
pip install -e "./system/api[dev]"

docker compose up -d db          # postgres:16, port 5433, volume ./.pgdata
cd system/api && alembic upgrade head
cd system/web && npm install
```

`docker-compose.yml` exposes Postgres on **5433** (not 5432) to avoid clashing with any local
install. `DATABASE_URL=postgresql+psycopg://fces:fces@localhost:5433/fces`.

---

## 3. Repository layout

```
fces-asset-register/
├── docker-compose.yml
├── Makefile                          # see §12.5
├── .env.example
├── README.md
│
├── data/
│   ├── raw/                          # gitignored — downloaded bundles
│   │   ├── 2022/ 2023/ 2024/ 2025/    # Contracts Finder CSV bundles
│   │   └── abtbuy/
│   ├── interim/                      # gitignored
│   └── processed/                    # gitignored
│       ├── corpus_b_contractsfinder.parquet          # Contracts Finder, canonical Record schema
│       ├── corpus_a_abtbuy.parquet
│       ├── cpv_taxonomy.parquet
│       └── splits.json               # frozen dev/test assignment — never regenerate
│
├── research/
│   ├── pyproject.toml
│   ├── src/fcesreg/
│   │   ├── __init__.py
│   │   ├── schema.py                 # canonical Record model  §6.1
│   │   ├── normalise.py              # §6.2
│   │   ├── ingest_contractsfinder.py # §6.3
│   │   ├── ingest_abtbuy.py          # §6.4
│   │   ├── cpv.py                    # §6.5
│   │   ├── degrade.py                # §6.6
│   │   ├── embed.py                  # §6.7
│   │   ├── blocking.py               # §6.8
│   │   ├── dedup.py                  # §6.9
│   │   ├── classify.py               # §6.10
│   │   ├── llm.py                    # §6.11
│   │   ├── metrics.py                # §6.12
│   │   ├── operating_point.py        # §6.13
│   │   ├── runs.py                   # §6.14
│   │   ├── timing.py                 # §6.15 — per-item monotonic timer
│   │   └── splits.py                 # §6.16 — frozen dev/test assignment
│   ├── configs/*.yaml                # one per experiment, committed
│   ├── scripts/run_*.py              # §10
│   └── tests/
│
├── system/
│   ├── api/
│   │   ├── pyproject.toml
│   │   ├── alembic/versions/
│   │   └── src/fcesapi/
│   │       ├── main.py
│   │       ├── config.py  db.py  deps.py  security.py  storage.py
│   │       ├── models/                # SQLAlchemy ORM
│   │       ├── schemas/               # Pydantic request/response
│   │       ├── routers/               # assets, attachments, imports, floorplans,
│   │       │                          #   service, categories, labels, audit, auth, users
│   │       └── services/
│   │           ├── pipeline.py        # the ONLY place that imports fcesreg
│   │           ├── importer.py        # §9
│   │           ├── labels.py          # QR + Code128 §7.4
│   │           └── audit.py
│   └── web/                           # Next.js App Router, routes in §8
│
├── results/
│   ├── ledger.jsonl                  # ONE global LLM ledger, carries run_id — §6.11
│   ├── runs/<run_id>/                # params.yaml metrics.json predictions.parquet env.json
│   └── tables/                       # generated only — never hand-edit
│
├── annotation/
│   ├── protocol.md
│   ├── annotate.py                   # terminal tool, per-item monotonic timer — §6.15
│   ├── samples/                      # generated task files
│   └── labels/                       # human output, incl. per-item seconds_taken
└── storage/                          # gitignored — uploaded files
```

---

## 4. Ground facts about the data

These were measured from the actual files on disk. **Build against these numbers, not assumptions.**

### 4.1 Contracts Finder

- Bundle download URL: `https://data.open-contracting.org/en/publication/128/download?name=<YEAR>.csv.tar.gz`
- Every bundle lives under `data/raw/<YEAR>/`, which `.gitignore` covers. **No corpus is committed**: the paper states that third-party corpora are distributed as identifiers and reconstruction scripts, not redistributed data, and the README carries the acquisition instructions.
- **Measured on download (B1), superseding the earlier size estimates.** The stated 14–17 MB
  figures were wrong; the 2022–2024 archives are 3–4× larger. All four bundles carry 13 CSVs and an
  identical 32-column `main.csv` header, differing in column *order* only.

  | Year | `.tar.gz` | extracted | notices | `date` populated | `tender_datePublished` | in scope (10 div) |
  |---|---|---|---|---|---|---|
  | 2022 | 54 MB | 257 MB | 68,576 | 100% | 31.3% | 13,863 |
  | 2023 | 59 MB | 280 MB | 76,674 | 100% | 27.5% | 15,010 |
  | 2024 | 57 MB | 273 MB | 74,019 | 100% | 25.7% | 14,332 |
  | 2025 | (local) | 173 MB | **46,183** | 100% | 15.5% | **9,099** |

  The 2025 notice count and in-scope count match the original measurement exactly.
- `main.csv` is the one that matters; `tender_additionalClassifications.csv` supplies CPV ancestors.
- **`id` is unique within every bundle and across all four** (0 intra-year and 0 cross-year
  duplicates), so ingest needs no cross-bundle deduplication and `DiscardReport` keeps six fields.
- All CPV codes are exactly 8 digits with `scheme == "CPV"`; there is no check-digit suffix to strip.
- **`tender_datePublished` is populated for only 15% of rows. Use `main.csv:date` instead** — it is
  100% populated and spreads across all 12 months (351–1,413/month in the in-scope subset).
- Descriptions contain embedded newlines inside quoted fields. `wc -l` overcounts.
  **Always parse with a real CSV reader and `csv.field_size_limit(10**9)`.**
- Encoding damage is present and is experimentally useful, not noise: 331 records with
  mojibake/replacement characters and 256 with control characters in the 2025 in-scope subset.

### 4.2 CPV division filter and its consequence

The candidate division set is `{30, 31, 32, 33, 38, 39, 42, 43, 44, 48}`. In the 2025 bundle alone
that yields **9,099 records**; across four year bundles it yields **45,051 after the §4.3 discard
rules** — the earlier "~30,000" estimate was low.

**Decision (B3, measured).** The set adopted is the **eight divisions
`{30, 31, 32, 33, 38, 42, 43, 44}`**, dropping 39 and 48.

| Division set | records | dev | test | classes ≥50 | class coverage |
|---|---|---|---|---|---|
| 10 divisions (with 39, 48) | 45,051 | 37,250 | 7,801 | 122 | 90.7% |
| **8 divisions (adopted)** | **26,413** | **22,451** | **3,962** | **81** | **88.3%** |

Support is ample without them: 81 classes clear the 50-example floor at 88.3% coverage, against 122
at 90.7%. The cost of dropping them is 2.4 points of coverage; the cost of keeping them is that
**division 48 alone is 15,939 records, 35% of the corpus** — a third of the material would be
software licences, which distorts the label distribution the classifier learns away from anything a
faculty equipment register contains. Division 39 adds a further 2,699 furniture records.

**Sentence for the paper:** *The retained divisions are those whose contents a faculty of computing,
engineering and science would place on an equipment register and label for servicing — computing and
office machinery, electrical and communications equipment, medical and laboratory instruments,
industrial machinery, and installation and construction plant; furniture and software packages are
excluded because neither is a serviceable physical item bearing an asset tag, and software alone
would otherwise supply a third of the corpus.*

Both counts stay in `T1_corpus_b.tex`; the discarded set is reported, not deleted.

**The set was not settled by assertion.** Division 39 is furniture, which
belongs to estates rather than to an equipment register, and division 48 is software, which has no
physical item to label or service. Both inflate the corpus with records the delivered system would
never hold. Therefore:

- `run_profile.py` measures corpus size and per-class support at class level **both with and
  without 39 and 48**, and reports both counts (`T1_corpus_b.tex`).
- `configs/profile.yaml` carries both sets so the comparison stays reproducible.
- **Ingest retains all ten divisions** into `corpus_b_contractsfinder.parquet`; the eight-division restriction is
  applied downstream by filtering on `cpv_code[:2]`. One parquet therefore supports both analyses
  and no re-ingest is needed to reproduce the comparison.

**Critical: 8-digit leaf-level classification is not viable and must not be built.** Measured on
the 2025 in-scope subset: 1,032 distinct leaf codes, only 58 with ≥20 examples, only 17 with ≥50.
Restricting to divisions 30/31/38/42 is worse — 385 leaves, 171 of them singletons.

Measured again on the full four-year corpus (B3), the picture does not improve with scale:

| Division set | records | distinct leaves | ≥20 examples | ≥50 examples | singletons |
|---|---|---|---|---|---|
| 10 divisions | 45,051 | 1,983 | 328 | 128 | 474 |
| **8 divisions (adopted)** | 26,413 | 1,593 | 234 | **87** | 385 |

87 leaves clear the 50-example floor out of 1,593, and 385 codes occur exactly once. Four times the
data leaves the leaf level as sparse as before.

**Therefore the two classification levels are `division` (2-digit) and `class` (4-digit)**, with
classes restricted to those having ≥50 training examples (2025 alone: 82 classes with ≥20, covering
87.8% of records). `cpv.py` must expose exactly these two levels and nothing else.

**The leaf-level sparsity counts are a reported result, not a justification note.** The paper states
in RQ2 that the eight-digit level is not evaluated and that the measured sparsity appears in
Section V. `run_profile.py` must therefore emit, into `T1_leaf_sparsity.tex`, the number of distinct leaf
codes, the number with ≥20 examples, the number with ≥50, and the singleton count — measured on the
final corpus, not quoted from the 2025 figures above.

### 4.3 Discard rules (count each, they are a reported result)

Drop a record if any holds:
- `len(normalise(title) + " " + normalise(description)) < 60`
- `normalise(description) == normalise(title)`
- `normalise(description)` is in the boilerplate blocklist:
  `"as per tender"`, `"contract award notice"`, `"transparency only"`,
  `"notice of awarded contract following a mini competition"`,
  `"call off from fcdo services ref xly120 121 21cc"`

**Blocklist entries are stored already normalised.** They are compared against
`normalise_text(description)`, which turns `:` and `/` into spaces, so an entry carrying raw
punctuation would never match anything. A test asserts `normalise_text(entry) == entry` for every
entry.

**Rules are applied in order, and each dropped row is attributed to the first rule it fails.** That
is what makes the five counts sum to the total dropped, which is B2's acceptance criterion. It also
means a count here is not the same as the independent frequency of that condition: on the 2025
in-scope subset, `desc == title` holds for **745** rows measured independently, but 208 of those are
also under 60 characters and are attributed to the short rule, leaving **537** in the
`dropped_desc_equals_title` column. (The earlier figure of 716 for the independent count was
measured under slightly different normalisation; 745 is the value the released code reproduces.)

Measured across all four bundles (B2):

| Year | in | out | out of scope | short | desc = title | boilerplate |
|---|---|---|---|---|---|---|
| 2022 | 68,576 | 11,980 | 54,713 | 1,046 | 836 | 1 |
| 2023 | 76,674 | 12,868 | 61,664 | 1,203 | 917 | 22 |
| 2024 | 74,019 | 12,402 | 59,687 | 981 | 822 | 127 |
| 2025 | 46,183 | 7,801 | 37,084 | 611 | 537 | 150 |
| **total** | **265,452** | **45,051** | 213,148 | 3,841 | 3,112 | 300 |

### 4.4 Abt-Buy benchmark

Verified reachable, HTTP 200:
```
https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/exp_data/{tableA,tableB,train,valid,test}.csv
```
Shapes: `tableA` 1,081 rows (`id,name,description,price`), `tableB` 1,092,
`train` 5,743 pairs at **10.7% positive**, `valid` and `test` 1,916 each at 10.8%.
All verified on download (B4). The splits are pre-defined — **use them as given, never re-split.**
Mirror if Wisconsin is down: `https://dbs.uni-leipzig.de/file/Abt-Buy.zip`.

**The supplied splits are pair-level, not record-level.** They partition labelled *pairs* drawn from
a fixed pool of 2,173 records, so records recur across sides by design: **1,359 records appear in
both the dev and test partitions**, while **zero pairs** do. This is a property of the benchmark, not
a defect in the ingest, and it constrains what B5 can assert — see B5.

`price` is dropped at ingest: it has no counterpart in `RECORD_COLUMNS` and nothing uses it.
`buyer_id` is null on this corpus, which is what makes the `buyer` blocking scheme inapplicable here
(§6.8).

### 4.5 Natural duplicate structure (descriptive statistic only)

The 2025 in-scope subset contains **154 `(buyer_id, normalised_title)` groups with more than one
notice**, covering 424 notices, plus 277 titles repeated across different buyers.

These counts are reported by `run_profile.py` as a descriptive statistic about the corpus, and that
is their only use. **There is no natural-duplicate experiment.** Deciding whether such a group is a
genuine duplicate or a legitimate annual repeat procurement would need roughly 200 further human
labels on genuinely ambiguous cases, and the paper's External Validation section does not describe
that experiment. No mining rule, no hand-labelling, no `--corpus natural`.

---

## 5. Database schema

Single Alembic migration `0001_initial.py`. Postgres 16. Requires `pgcrypto` for `gen_random_uuid()`.

### 5.1 Enums

```sql
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE user_role       AS ENUM ('admin','technician','readonly');
CREATE TYPE attachment_kind AS ENUM ('photo','pdf','document','link','risk_assessment','manual','certificate');
CREATE TYPE import_status   AS ENUM ('pending','processing','ready_for_review','committed','failed');
CREATE TYPE import_route    AS ENUM ('auto','review');
CREATE TYPE dedup_call      AS ENUM ('new','duplicate','uncertain');
CREATE TYPE review_action   AS ENUM ('accept','override','reject','skip');
```

`asset_status` is **removed** (scope fence, §1.1): it existed to carry the `disposed` soft-delete
state and to feed the dropped dashboard, and neither the client brief nor the paper claims an asset
lifecycle state. Deletion and retention are out of scope and are stated as a limitation in the
report.

### 5.2 Users

```sql
CREATE TABLE users (
  id            BIGSERIAL PRIMARY KEY,
  email         TEXT UNIQUE NOT NULL,
  name          TEXT NOT NULL,
  password_hash TEXT NOT NULL,                       -- argon2id
  role          user_role NOT NULL DEFAULT 'readonly',
  is_active     BOOLEAN NOT NULL DEFAULT TRUE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 5.3 Floor plans and locations

```sql
CREATE TABLE floorplans (
  id          BIGSERIAL PRIMARY KEY,
  building    TEXT NOT NULL,
  floor       TEXT NOT NULL,
  name        TEXT,
  image_path  TEXT NOT NULL,                          -- storage/floorplans/<uuid>.png
  image_w     INT NOT NULL,
  image_h     INT NOT NULL,
  uploaded_by BIGINT REFERENCES users(id),
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (building, floor, name)
);

CREATE TABLE locations (
  id           BIGSERIAL PRIMARY KEY,
  building     TEXT,
  floor        TEXT,
  room         TEXT,
  label        TEXT,
  floorplan_id BIGINT REFERENCES floorplans(id) ON DELETE SET NULL,
  x_pct        NUMERIC(5,2) CHECK (x_pct BETWEEN 0 AND 100),   -- % of image width
  y_pct        NUMERIC(5,2) CHECK (y_pct BETWEEN 0 AND 100)    -- % of image height
);
```

Percentage coordinates, **not pixels** — the plan image can be re-exported at any resolution and
pins stay correct.

### 5.4 Categories (CPV taxonomy)

```sql
CREATE TABLE categories (
  cpv_code                      TEXT PRIMARY KEY,      -- 2, 4 or 8 digit
  cpv_description               TEXT NOT NULL,
  level                         SMALLINT NOT NULL CHECK (level IN (2,4,8)),
  parent_code                   TEXT REFERENCES categories(cpv_code),
  hazard_class                  TEXT,                  -- nullable; illustrative designation only
  default_service_interval_days INT
);
```

Seeded from `data/processed/cpv_taxonomy.parquet` by `scripts/seed_categories.py`.

**`hazard_class` is a designation, not a vocabulary.** Do not design a hazard taxonomy — the paper
requires only that *some* divisions are designated as carrying hazard or servicing rules, so that
the per-class breakdown in RQ2 has something to report. `seed_categories.py` therefore designates
exactly three divisions:

| Division | `hazard_class` | `default_service_interval_days` |
|---|---|---|
| 33 — medical equipment | `regulated` | 365 |
| 38 — laboratory / optical / precision instruments | `calibration_required` | 365 |
| 42 — industrial machinery | `mechanical` | 180 |

These three are the classes the paper's per-class breakdown covers. **The designation is
illustrative and must be marked as such in both the seed script (a comment at the constant, plus a
log line on run) and the report** — it stands in for a faculty health-and-safety schedule that does
not yet exist, and no claim is made that these values are the faculty's real policy. Every other
category keeps `hazard_class = NULL`.

### 5.5 Assets

```sql
CREATE TABLE assets (
  id                    BIGSERIAL PRIMARY KEY,
  public_id             UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),  -- QR target
  asset_tag             TEXT UNIQUE NOT NULL,          -- 'FCES-000123', Code128 barcode payload
  name                  TEXT NOT NULL,
  description           TEXT,
  manufacturer          TEXT,
  model                 TEXT,
  serial_number         TEXT,
  cpv_code              TEXT REFERENCES categories(cpv_code),
  location_id           BIGINT REFERENCES locations(id) ON DELETE SET NULL,
  owning_department     TEXT,
  service_interval_days INT,
  last_serviced_at      DATE,
  next_due_at           DATE GENERATED ALWAYS AS (
                          CASE WHEN service_interval_days IS NOT NULL
                                AND last_serviced_at IS NOT NULL
                               THEN last_serviced_at + service_interval_days
                          END) STORED,
  source_row_id         BIGINT,                        -- import_rows.id if imported
  created_by            BIGINT REFERENCES users(id),
  created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
  search_tsv            TSVECTOR GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(name,'')), 'A') ||
    setweight(to_tsvector('english', coalesce(manufacturer,'') || ' ' || coalesce(model,'')), 'B') ||
    setweight(to_tsvector('english', coalesce(description,'')), 'C')) STORED
);

CREATE INDEX assets_search_idx ON assets USING GIN (search_tsv);
CREATE INDEX assets_due_idx    ON assets (next_due_at);
CREATE INDEX assets_cpv_idx    ON assets (cpv_code);
```

**One search mechanism only.** `search_tsv` + the GIN index is it. No `pg_trgm`, no trigram index,
no fuzzy fallback path (scope fence, §1.1). At a few thousand assets, full-text search over a
weighted `tsvector` is sufficient, and a second search path is a second thing to get wrong.

`asset_tag` is generated by a Postgres sequence formatted as `FCES-%06d`. It is the human-readable
identifier printed as a Code128 barcode; `public_id` is the opaque UUID encoded in the QR code.
**Both are on every label** — the client asked for QR *and* barcode.

### 5.6 Attachments, service history, audit

```sql
CREATE TABLE attachments (
  id           BIGSERIAL PRIMARY KEY,
  asset_id     BIGINT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
  kind         attachment_kind NOT NULL,
  title        TEXT,
  filename     TEXT,
  mime         TEXT,
  size_bytes   BIGINT,
  storage_path TEXT,                                  -- null when kind='link'
  url          TEXT,                                  -- null unless kind IN ('link','risk_assessment')
  is_primary   BOOLEAN NOT NULL DEFAULT FALSE,        -- primary photo / primary risk assessment
  uploaded_by  BIGINT REFERENCES users(id),
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (storage_path IS NOT NULL OR url IS NOT NULL)
);
CREATE INDEX attachments_asset_idx ON attachments (asset_id, kind);
CREATE UNIQUE INDEX attachments_one_primary_idx
  ON attachments (asset_id, kind) WHERE is_primary;

CREATE TABLE service_events (
  id           BIGSERIAL PRIMARY KEY,
  asset_id     BIGINT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
  performed_at DATE NOT NULL,
  performed_by BIGINT REFERENCES users(id),
  provider     TEXT,
  notes        TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE audit_log (
  id          BIGSERIAL PRIMARY KEY,
  actor_id    BIGINT REFERENCES users(id),
  entity_type TEXT NOT NULL,
  entity_id   BIGINT NOT NULL,
  action      TEXT NOT NULL,                          -- create | update | delete | import_commit
  before      JSONB,
  after       JSONB,
  at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX audit_entity_idx ON audit_log (entity_type, entity_id, at DESC);
```

Inserting a `service_events` row must update `assets.last_serviced_at` to `MAX(performed_at)` —
do this in the service layer, not a trigger, so it is testable.

### 5.8 Notifications — what makes "scheduled reminders" true

```sql
CREATE TABLE notifications (
  id         BIGSERIAL PRIMARY KEY,
  asset_id   BIGINT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
  kind       TEXT NOT NULL CHECK (kind IN ('due_soon','overdue')),
  due_at     DATE NOT NULL,
  generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  acknowledged_at TIMESTAMPTZ,
  acknowledged_by BIGINT REFERENCES users(id),
  UNIQUE (asset_id, kind, due_at)
);
CREATE INDEX notifications_open_idx ON notifications (kind, due_at)
  WHERE acknowledged_at IS NULL;
```

A view a user has to visit is not a reminder, and the client asked for reminders explicitly. The
paper's word is *scheduled*, so a scheduled job satisfies it honestly:

- **A daily job** computes due-soon and overdue items and inserts `notifications` rows. The unique
  constraint makes it idempotent — running twice in one day produces no duplicates.
- Rows surface **on the service view and on login**. Nothing else.
- **No email delivery, no external service, no preferences UI.** Those are out of scope and stay
  out; the claim is a scheduled reminder, not a notification platform.
- The scheduler is a plain job (APScheduler in-process, or a container-level cron), configured once
  and covered by a test that advances the clock rather than waiting.

### 5.7 Import tables

```sql
CREATE TABLE import_batches (
  id               BIGSERIAL PRIMARY KEY,
  filename         TEXT NOT NULL,
  uploaded_by      BIGINT REFERENCES users(id),
  row_count        INT,
  auto_count       INT,
  review_count     INT,
  precision_target NUMERIC(4,3) NOT NULL,             -- P* for this batch
  column_mapping   JSONB NOT NULL,                    -- {"Equipment Name":"name", ...}
  pipeline_run_id  TEXT,                              -- ties batch to results/runs/<id>
  status           import_status NOT NULL DEFAULT 'pending',
  error            TEXT,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE import_rows (
  id                       BIGSERIAL PRIMARY KEY,
  batch_id                 BIGINT NOT NULL REFERENCES import_batches(id) ON DELETE CASCADE,
  row_index                INT NOT NULL,
  raw                      JSONB NOT NULL,            -- original spreadsheet row
  normalised               JSONB,                     -- mapped to Record schema
  dedup_decision           dedup_call,
  dedup_score              NUMERIC(5,4),
  dedup_candidate_asset_id BIGINT REFERENCES assets(id),
  class_cpv_code           TEXT,
  class_score              NUMERIC(5,4),
  class_alternatives       JSONB,   -- [{"code":"38430000","desc":"...","score":0.21}, ...]
  evidence                 JSONB,   -- {"matched_ngrams":[...], "nearest":[{...}], "method":"..."}
  route                    import_route NOT NULL,
  resolved_by              BIGINT REFERENCES users(id),
  resolved_at              TIMESTAMPTZ,
  final_asset_id           BIGINT REFERENCES assets(id),
  UNIQUE (batch_id, row_index)
);
CREATE INDEX import_rows_review_idx ON import_rows (batch_id, route) WHERE resolved_at IS NULL;

CREATE TABLE review_decisions (
  id             BIGSERIAL PRIMARY KEY,
  import_row_id  BIGINT NOT NULL REFERENCES import_rows(id) ON DELETE CASCADE,
  actor_id       BIGINT REFERENCES users(id),
  decision_type  TEXT NOT NULL CHECK (decision_type IN ('dedup','classification')),
  action         review_action NOT NULL,
  chosen_value   TEXT,
  seconds_taken  INT NOT NULL CHECK (seconds_taken >= 0),
  at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

`review_decisions.seconds_taken` is written automatically by the review UI.

**This table is NOT the source of the residual-manual-effort figure.** The headline RQ3 result must
not depend on a full-stack React review queue, which is the most fragile component in the project.
The paper specifies the correct source: handling time is measured by timing manual curation during
the annotation exercise (§6.15, §13.3). Once the review queue works, its observed handling time is
reported as **corroboration** of the annotation figure — a second, independent measurement — and
never as the number that feeds `residual_effort`.

---

## 6. `fcesreg` module contracts

Every module is pure Python operating on pandas DataFrames or numpy arrays. **No imports from
`system/`. No database access. No network access except in `ingest_*` and `llm`.**

### 6.1 `schema.py`

```python
RECORD_COLUMNS = [
    "record_id", "title", "description", "manufacturer", "model",
    "serial_number", "buyer_id", "cpv_code", "release_date", "source",
]

class Record(BaseModel):
    record_id: str
    title: str
    description: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None
    buyer_id: str | None = None
    cpv_code: str | None = None
    release_date: date | None = None
    source: Literal["contractsfinder", "abtbuy", "upload"]

def validate_frame(df: pd.DataFrame) -> None:
    """Raise if df is missing RECORD_COLUMNS or has wrong dtypes."""

def text_of(df: pd.DataFrame) -> pd.Series:
    """Concatenated field used for embedding/TF-IDF: title + ' ' + description."""
```

Both corpora and the spreadsheet importer map into this one shape. Anything downstream of ingest
only ever sees `RECORD_COLUMNS`.

**`manufacturer`, `model` and `serial_number` are null across both research corpora.** Contracts
Finder supplies `record_id`, `title`, `description`, `buyer_id`, `cpv_code` and `release_date`;
Abt-Buy supplies name, description and price. The three fields survive in `Record` only because
`source="upload"` — a real FCES spreadsheet — can populate them, and the system's `assets` table
carries them. Therefore:

- **Nothing in the research path may key, block, group or score on these three columns.** A blocking
  scheme or distractor rule that reads them is keying on an always-null column and will silently
  produce one enormous block, no blocks at all, or an empty negative set.
- **Do not manufacture a `manufacturer` column by parsing titles with hand-written rules.** Where a
  brand proxy is needed, use the leading-token key (§6.8), which is stated as an approximation in
  the paper and evaluated as one.
- `validate_frame` must warn (not raise) when a frame carries these columns wholly null, so the
  condition is visible in logs rather than assumed.

### 6.2 `normalise.py`

```python
def fix_mojibake(s: str) -> str:            # 'â€™' → '’', strip U+FFFD
def strip_control(s: str) -> str:           # drop unicodedata category 'Cc' except \n\r\t
def normalise_text(s: str | None) -> str:   # NFKC → fix_mojibake → strip_control →
                                            # casefold → collapse whitespace → strip punctuation
def normalise_key(s: str | None) -> str:    # normalise_text + remove all non-alphanumerics
def normalise_frame(df: pd.DataFrame) -> pd.DataFrame:   # adds *_norm columns, non-destructive
```

`normalise_key` is what the exact-match dedup baseline compares.

**Two implementation notes, both deviations from the line above, both deliberate.**

1. **Mojibake repair runs before NFKC, not after.** NFKC maps `™` to `TM` and `…` to `...`, which
   destroys the byte sequences the repair recognises — `â„¢` would become `â„TM` and no longer round
   trip. Since §4.1 records 331 mojibake-carrying rows in the 2025 subset alone, NFKC-first would
   silently fail to repair a documented part of the corpus. The repair also falls back to latin-1
   for the five slots cp1252 leaves undefined (0x81, 0x8D, 0x8F, 0x90, 0x9D), because `”` misread as
   cp1252 lands on U+009D and a strict encode would refuse it.
2. **Apostrophes are deleted; all other punctuation becomes a space.** `pump/valve` must yield two
   tokens, but `buyer's` must yield one. Note the consequence that decimal separators do not survive
   — `1.5kW` and `1,5 kW` both normalise to `1 5kw`. That is what "strip punctuation" means, and the
   loss is symmetric across both members of a pair, so it cannot manufacture a false match.

### 6.3 `ingest_contractsfinder.py`

```python
FIELD_MAP = {
    "record_id":    "id",
    "title":        "tender_title",
    "description":  "tender_description",
    "buyer_id":     "buyer_id",
    "cpv_code":     "tender_classification_id",
    "release_date": "date",                 # NOT tender_datePublished — see §4.1
}

def load_bundle(year_dir: Path) -> pd.DataFrame:
    """Read <year_dir>/main.csv. csv.field_size_limit(10**9). Never use wc -l."""

def to_records(df: pd.DataFrame, year: int) -> pd.DataFrame

def apply_filters(df, divisions: set[str], min_chars: int = 60,
                  blocklist: set[str] | None = None) -> tuple[pd.DataFrame, DiscardReport]

@dataclass
class DiscardReport:
    total_in: int
    dropped_out_of_scope: int
    dropped_short: int
    dropped_desc_equals_title: int
    dropped_boilerplate: int
    total_out: int
```

CLI: `python -m fcesreg.ingest_contractsfinder --raw data/raw --years 2022 2023 2024 2025 --out data/processed/corpus_b_contractsfinder.parquet`

Emits `DiscardReport` as JSON alongside the parquet. Those counts are a reported result.

### 6.4 `ingest_abtbuy.py`

```python
BASE = "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/exp_data"

def download(dest: Path) -> None
def load(dest: Path) -> tuple[pd.DataFrame, pd.DataFrame]  # (records, pairs)
```

`records` maps `name→title`, `description→description`, `id→record_id` prefixed `A:`/`B:`.
`pairs` has `left_id, right_id, label, split` where `split ∈ {train, valid, test}` — **taken from
the supplied files, never regenerated.**

### 6.5 `cpv.py`

```python
def build_taxonomy(bundle_dirs: list[Path]) -> pd.DataFrame
    """Harvest distinct (code, description) from main.csv:tender_classification_{id,description}
    and tender_additionalClassifications.csv:{id,description}. Derive level from code shape,
    parent_code by truncation. Offline; no external CPV list required."""

def division(code: str) -> str    # code[:2]
def cpv_class(code: str) -> str   # code[:4] — the CPV *class* level

def supported_labels(train: pd.DataFrame, level: Literal["division","class"],
                     min_examples: int = 50) -> tuple[set[str], float]
    """Returns (labels, coverage_fraction). Callers must report coverage."""
```

**`level` accepts only `"division"` and `"class"`. Leaf level is not implemented** — see §4.2.

### 6.6 `degrade.py`

```python
@dataclass
class DegradationConfig:
    severity: float          # 0.0 .. 1.0, single knob
    p_abbreviate: float = 1.0    # per-class multipliers applied to severity
    p_charnoise:  float = 1.0
    p_case:       float = 1.0
    p_whitespace: float = 1.0
    p_merge:      float = 1.0
    p_omit:       float = 1.0
    p_units:      float = 1.0

# one function per error class, each (text, rng) -> text
def abbreviate(s, rng, lexicon)      # domain lexicon, data/lexicon/abbreviations.yaml
def char_noise(s, rng)               # insert / delete / substitute / transpose
def vary_case(s, rng)
def perturb_whitespace(s, rng)
def vary_units(s, rng)               # 230V ↔ 230 V ↔ 230v ; 1.5kW ↔ 1,5 kW

def merge_fields(rec: dict, rng) -> dict   # Mudgal-style: move an attribute into free text
def omit_field(rec: dict, rng) -> dict

def degrade_record(rec: dict, cfg: DegradationConfig, rng: np.random.Generator) -> dict

def make_duplicate_pairs(records: pd.DataFrame, cfg, seed: int
                         ) -> tuple[pd.DataFrame, pd.DataFrame]
    """Two independent degraded copies of each source record → a positive pair.
    Returns (degraded_records, pairs[left_id,right_id,label])."""

def make_distractors(records: pd.DataFrame, cfg, seed: int, corpus: Literal["cf","abtbuy"],
                     sim_threshold: float = 0.75) -> pd.DataFrame
    """Near-duplicate NEGATIVES mined from fields that actually exist.

    corpus="cf":     pairs sharing a CPV class, with title cosine >= sim_threshold,
                     and DISTINCT record_id.
    corpus="abtbuy": pairs sharing a leading token (first substantive token of the
                     normalised title), and distinct record_id.

    Both sides are then degraded under cfg, like the positives, so the negatives are
    not trivially separable by cleanliness. Returns pairs[left_id,right_id,label=0].
    Report n_distractors and the mined-pair rate — both go in T3/T4 captions.
    """
```

**The original specification — "same manufacturer+model family, different serial" — cannot be
built**: all three fields are null in both corpora (§6.1). The two rules above are what the paper
describes, and they are mined from `cpv_code` + title similarity on Contracts Finder and from the
leading token on Abt-Buy. These negatives matter: without them a detector that cannot separate
similar-but-distinct records reports high recall for the wrong reason, and the whole RQ1 result is
flattered.

Every function takes an explicit `rng`. **No module-level random state.** The same seed must
reproduce byte-identical output — there is a test for this.

### 6.7 `embed.py`

```python
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

def embed(texts: Sequence[str], model_id: str = DEFAULT_MODEL,
          cache_dir: Path = Path(".cache/embeddings"), batch_size: int = 64) -> np.ndarray
    """L2-normalised float32 (n, d). Disk cache keyed by sha256(model_id + '\\x00' + text).
    CPU only — do not call .to('cuda') or .to('mps')."""
```

The cache is what keeps the full experiment sweep under an hour on re-runs. Implement it before
anything calls `embed`.

### 6.8 `blocking.py`

Three keying schemes, and exactly three. Each keys on a column that is actually populated —
`block_by_manufacturer` is **deleted**, because `manufacturer` is null everywhere (§6.1) and keying
on it yields either one enormous block or none at all.

```python
SCHEMES = ("sorted_ngrams", "leading_token", "buyer")

def block_by_sorted_ngrams(df, n: int = 3, k: int = 4,
                           mode: str = "per_gram") -> dict[str, list[str]]
    """Character n-grams of the normalised title, per token. Applies to BOTH corpora.
    mode="per_gram" is q-gram indexing and is what the study adopts.
    mode="single_key" is the original formulation, retained only for the sweep."""

def ngram_overlap_candidates(df, n: int = 3, min_overlap: int = 8,
                             max_block_size: int = 500, chunk_rows: int = 2000
                             ) -> tuple[pd.DataFrame, BlockingReport]
    """Per-gram indexing requiring >= min_overlap shared n-grams. Overlap counts come
    from a sparse incidence matrix times its transpose, in row chunks."""

def block_by_leading_token(df, stopwords: frozenset[str] = LEADING_STOPWORDS
                           ) -> dict[str, list[str]]
    """Key = first substantive token of the normalised title, skipping leading stopwords
    and pure quantities ('the', 'supply', 'provision', 'of', '2x', ...). A brand proxy
    where the title carries one — an approximation, and reported as one. BOTH corpora.
    Records whose title yields no substantive token are placed in no block, and that
    count is reported."""

def block_by_buyer(df) -> dict[str, list[str]]
    """Key = buyer_id. CONTRACTS FINDER ONLY — buyer_id is null on Abt-Buy.
    Raises SchemeUnavailable if the column is wholly null."""

class SchemeUnavailable(ValueError): ...

def applicable_schemes(df) -> list[str]
    """Which of SCHEMES this frame can support, decided by column nullity, not by
    corpus name. run_blocking.py reports this per corpus."""

def candidate_pairs(df, schemes: list[str], max_block_size: int = 500) -> pd.DataFrame
    """Union of blocks. Blocks larger than max_block_size are dropped and counted
    (report the count — it is a source of lost recall)."""

def evaluate_blocking(candidates: pd.DataFrame, truth: pd.DataFrame) -> dict
    """{'pair_completeness': float, 'reduction_ratio': float,
        'n_candidates': int, 'n_possible': int, 'blocks_dropped': int,
        'n_unblocked_records': int}"""
```

**The `t=8` threshold is selected on Corpus A and carried to Corpus B unchanged**, which is what
the transfer design requires. Corpus B pair completeness is unmeasurable until C4 supplies labels,
and procurement titles have a different length distribution from product names, so what transfers is
currently unknown. Measure it the moment C4 lands. **If it falls below the floor that is the
transfer finding, not a tuning problem:** report it as measured, and additionally report what a
Corpus-B-selected threshold would have recovered as a secondary figure showing the size of the gap.
Do not refit and carry on.

**The operating point, settled on the Corpus A dev partition (G3).** Per-gram q-gram indexing with
`n=3` and `min_overlap=8`, giving **pair completeness 0.988 at reduction ratio 0.9796** — 48,194
candidates on Corpus A and 71,248 on Corpus B. The rule is *the highest reduction ratio holding pair
completeness at or above a floor of 0.98*; the floor is stated in the paper rather than tuned per
result, and the 1.2% of true pairs forgone is unrecoverable downstream and reported as such.

The threshold sweep is itself a reported result — the pair-completeness against reduction-ratio
curve is more informative than any single point. The original single-key formulation is retained in
the sweep as a **negative result**: it degrades monotonically from PC 0.254 at `k=4` to 0.028 at
`k=32`, because agreement on the `k` alphabetically-earliest n-grams is an exact-match key over a
derived string. Three lines of the paper showing empirically why the standard formulation is
standard.

`run_blocking.py` reports, **per corpus**, which schemes apply and how each performs individually
and in union. The paper states that scheme availability differs between the two corpora and that
the difference is reported alongside performance, so the availability finding is a result, not a
caveat.

### 6.9 `dedup.py`

All matchers share one interface:

```python
class Matcher(Protocol):
    name: str
    def score_pairs(self, pairs: pd.DataFrame, records: pd.DataFrame) -> np.ndarray: ...

class ExactMatcher:        # normalise_key equality → score 1.0 or 0.0
class TfidfMatcher:        # char_wb ngram_range=(2,4), sublinear_tf, cosine
class EmbeddingMatcher:    # embed() then cosine
class CascadeMatcher:
    def __init__(self, base: Matcher, lower: float, upper: float,
                 adjudicator: "LLMAdjudicator", max_adjudications: int = 5000): ...
    # score >= upper → 1.0 ; score <= lower → 0.0 ; otherwise ask the LLM.
    # Records band_fraction on self.stats — that number is a reported result.

def select_thresholds(scores, labels, precision_target: float) -> float
    """Lowest threshold achieving >= precision_target on the DEV split. Never fit on test."""
```

`CascadeMatcher.stats` must expose `{'n_pairs', 'n_adjudicated', 'band_fraction'}`.

### 6.10 `classify.py`

```python
class Classifier(Protocol):
    name: str
    def fit(self, train: pd.DataFrame, level: str) -> None: ...
    def predict(self, records: pd.DataFrame) -> ClassificationResult: ...

@dataclass
class ClassificationResult:
    codes: list[str]
    scores: np.ndarray                      # confidence of the top choice
    alternatives: list[list[tuple[str, float]]]   # top-5 competitors — the review UI needs these

class TfidfSvmClassifier         # char_wb (2,5) + LinearSVC + CalibratedClassifierCV
class EmbeddingLogRegClassifier  # embed() + LogisticRegression
class RagFewShotLLMClassifier    # shortlist + k nearest labelled dev examples, then ask

def shortlist_codes(record_text: str, taxonomy: pd.DataFrame, k: int = 12) -> list[tuple[str,str]]
    """Embedding similarity against 'code + description'. NEVER send the full taxonomy
    to the model — the shortlist is both a cost control and the retrieval condition."""
```

`alternatives` is not optional. The import review queue renders it and the paper reports it.

**Unsupported labels are routed to review, never discarded.** Class-level evaluation is restricted
to labels meeting `min_examples`, and on the adopted division set that leaves **11.7% of dev records
in classes too sparse to learn** (`T1_label_support.tex`). Those records are not dropped from the
migration pipeline:

- `run_classify.py` reports macro/weighted F1 over the supported label set **and** reports the
  uncovered share alongside it, so class-level accuracy is read against the share of the register it
  applies to rather than against the whole. A record whose true label is unsupported counts as
  routed-to-review, not as an error and not as absent.
- `services/importer.py` routes such a record to `route='review'` regardless of `class_score`
  (§9.2 step 6). An unlearnable category is exactly where human judgement belongs.
- The coverage figure reaches the paper through `results/tables/`, not as prose.

This is why the restriction is a *reported* number rather than an internal threshold: it is the
share of the migration the classifier declines to automate, and RQ3's automated share has to account
for it.

### 6.11 `llm.py`

```python
MODEL = "claude-haiku-4-5"          # $1.00/M input, $5.00/M output
CAP_USD = 6.00

class BudgetExceeded(RuntimeError): ...

class LLMClient:
    def __init__(self, model=MODEL, cache_dir=Path(".cache/llm"),
                 ledger_path=Path("results/ledger.jsonl"),   # ONE global ledger
                 run_id: str | None = None, cap_usd=CAP_USD): ...

    def complete(self, system: str, prompt: str, max_tokens: int = 64,
                 json_schema: dict | None = None) -> LLMResponse:
        """1. key = sha256(model + system + prompt + str(json_schema))
           2. cache hit → return immediately, cost 0, log cache_hit=True
           3. miss → call API, append ledger row, raise BudgetExceeded if cap crossed"""

    def complete_batch(self, requests: list[BatchRequest]) -> list[LLMResponse]:
        """Message Batches API — 50% cheaper. Use this for every offline evaluation run.
        Poll until processing_status == 'ended'. Key results by custom_id, NEVER by
        position — batch results arrive out of order."""
```

**One global ledger at `results/ledger.jsonl`**, not one per run. Every row carries `run_id`, and
`run_costs.py` groups by it. This is deliberate: spend aggregates across runs (the `cap_usd` guard
is meaningless if each run starts a fresh ledger) and cache hits span them (a run that costs $0.00
because an earlier run paid for the same prompts is only interpretable against a shared ledger).
`results/runs/<run_id>/` therefore contains `params.yaml`, `metrics.json`, `predictions.parquet` and
`env.json`, and **no ledger file**.

Ledger row, one JSON object per line:
```json
{"ts":"...","run_id":"...","condition":"rq2_zeroshot","prompt_sha256":"...",
 "model":"claude-haiku-4-5","input_tokens":712,"output_tokens":9,
 "usd":0.000757,"latency_ms":812,"cache_hit":false}
```

Appends must be atomic (open `"a"`, one `write()` of a single line ending `\n`) — the sweep runs
several conditions and a torn line loses a cost row.

Three rules that keep this inside budget:

1. **Use `complete_batch` for all accuracy runs.** 50% discount. Measure latency separately on a
   100-call synchronous subsample — the paper reports per-record latency, and batch calls cannot
   supply it.
2. **Do not implement prompt caching.** Haiku 4.5's minimum cacheable prefix is 4,096 tokens;
   these prompts are 700–1,200. `cache_control` would silently do nothing. The SHA-256 disk cache
   is the mechanism that matters.
3. **Build the cache and ledger before the first paid call**, not after.

Estimated total spend for the full experiment set: **~$1.66 via the Batches API** (~$3.30 at
standard rates). The `cap_usd=6.00` guard is headroom, not a target.

### 6.12 `metrics.py`

```python
def prf1(y_true, y_pred) -> dict                    # precision, recall, f1, tp, fp, fn
def macro_weighted_f1(y_true, y_pred, labels) -> dict
def confusion(y_true, y_pred, labels) -> pd.DataFrame
```

### 6.13 `operating_point.py`

```python
DEFAULT_TARGET = 0.95

def precision_automation_curve(scores, labels) -> pd.DataFrame
    """columns: threshold, precision, recall, automated_share. The FULL curve is a
    reported result (F2_operating_point.pdf), not just the point on it."""

def automated_share_at_precision(scores, labels, target: float = DEFAULT_TARGET
                                 ) -> tuple[float, float]
    """(threshold, automated_share) at the lowest threshold meeting `target`.
    Returns (nan, 0.0) if no threshold achieves the target — that is a finding,
    not an error, and the caller reports it as such."""

def residual_effort(n_records: int, automated_share: float,
                    mean_seconds_per_item: float) -> dict
    """{'baseline_hours': ..., 'residual_hours': ..., 'hours_saved': ...}

    mean_seconds_per_item is a PARAMETER. This function never reads a database, never
    imports from system/, and has no default for it. Its source is the timed annotation
    exercise (§6.15) — see §5.7 for why not review_decisions."""
```

This produces the headline result. **The precision floor is `DEFAULT_TARGET = 0.95`, headlined, with
0.99 reported alongside it, and the full precision–automation curve reported in full** (§13.1). The
curve is the result; the two points on it are the summary.

### 6.14 `runs.py`

```python
def new_run_id(script: str, config_path: Path) -> str
    """<script>-<UTC yyyymmddTHHMMSS>-<git short sha>-<sha256(config)[:8]>"""

def capture_env() -> dict
    """git sha, git dirty flag, python version, package versions, model ids,
    seeds, hostname, platform, timestamp"""

def write_run(run_id, params: dict, metrics: dict,
              predictions: pd.DataFrame | None = None) -> Path
    """Writes results/runs/<run_id>/{params.yaml,metrics.json,predictions.parquet,env.json}"""
```

If the git tree is dirty, `capture_env()["git_dirty"] = True` and `make_tables.py` **refuses** to
build tables from that run. This is the mechanism that stops an untracked local edit silently
changing a published number.

### 6.15 `timing.py` — per-item handling time

The handling-time figure that RQ3 converts into residual hours is measured here, during the
annotation exercise, **not** by the review queue UI (§5.7, Amendment 3).

```python
@dataclass(frozen=True)
class ItemTiming:
    item_id: str
    seconds: float           # time.monotonic() delta — never wall clock
    started_at: str          # ISO 8601 UTC, for the record only
    abandoned: bool          # item shown then skipped without a judgement

@contextmanager
def time_item(item_id: str, sink: list[ItemTiming],
              idle_cutoff_s: float = IDLE_CUTOFF_S) -> Iterator[Callable[[], None]]:
    """Starts a monotonic timer on enter, appends an ItemTiming to `sink` on exit.
    The yielded callable marks the item abandoned. An item exceeding `idle_cutoff_s`
    (default 120) is flagged `abandoned` automatically rather than silently inflating
    the mean. The timing is recorded even if the body raises, so an interrupted
    session does not lose the items already judged."""

def summarise(timings: Sequence[ItemTiming]) -> dict
    """{'n', 'n_abandoned', 'mean_seconds', 'median_seconds', 'p90_seconds',
        'total_seconds'}. Abandoned items are excluded from the mean and the
        exclusion count is reported. Mean over fewer than 30 retained items
        raises — the paper states a sample size and it has to be real."""
```

### 6.16 `splits.py` — the frozen dev/test assignment

```python
SPLITS_PATH = Path("data/processed/splits.json")
CF_CUTOFF   = date(2025, 1, 1)     # dev = 2022-2024 bundles, test = 2025 bundle

@dataclass(frozen=True)
class Splits:
    cf_dev: set[str];  cf_test: set[str]
    abtbuy_dev_pairs: set[tuple[str,str]];  abtbuy_test_pairs: set[tuple[str,str]]
    cutoff: date
    def cf(self, df, part) -> pd.DataFrame      # part in {"dev","test"}
    def abtbuy(self, pairs, part) -> pd.DataFrame

class SplitOverlap(AssertionError): ...

def freeze(corpus_b_contractsfinder, abtbuy_pairs, path=SPLITS_PATH, cutoff=CF_CUTOFF,
           overwrite: bool = False) -> Splits
    """Refuses to overwrite an existing file. Splits are frozen."""

def load(path=SPLITS_PATH) -> Splits
    """Re-checks the invariants on every load, not only at freeze time."""
```

The invariants and why they differ per corpus are set out at B5. `research/scripts/freeze_splits.py`
is the CLI; `make data` runs it last.

`annotation/annotate.py` is a **terminal script** — that is sufficient and is the point of this
amendment. It presents one item at a time, wraps each judgement in `time_item`, and writes
`annotation/labels/<task>-<annotator>.jsonl` with one row per item carrying the label, the timing
and the item id. `run_label_noise.py` reads the labels; `run_operating_point.py` reads
`summarise(...)["mean_seconds"]` and passes it to `residual_effort` as an argument.

---

## 7. Backend API

FastAPI. JSON everywhere except file upload (multipart) and label sheets (HTML/PDF).
Base path `/api/v1`. Auth is a JWT bearer token.

### 7.1 Auth

| Method | Path | Body / Query | Notes |
|---|---|---|---|
| POST | `/auth/login` | `{email, password}` | → `{access_token, token_type, user}` |
| GET | `/auth/me` | — | current user |

No `POST /auth/logout`: it would do nothing server side. The client discards the token. Users are
seeded by `scripts/seed_users.py` (one admin, one technician, one readonly, passwords from env);
there is no `/users` API and no user administration UI (scope fence, §1.1).

### 7.2 Role enforcement

Implement as a dependency: `require_role("technician")` allows technician **and** admin.

| Role | Permitted |
|---|---|
| `readonly` | every `GET` |
| `technician` | all `GET`; create/update assets, attachments, service events; run and resolve imports |
| `admin` | everything, plus editing `categories` |

`readonly` receiving a write request gets **403**, not 401.

### 7.3 Assets

| Method | Path | Notes |
|---|---|---|
| GET | `/assets` | `q, cpv_code, location_id, due_before, page, page_size, sort`. Full-text via `search_tsv` — **the only search path**. |
| POST | `/assets` | technician+ |
| GET | `/assets/{id}` | includes attachments, location, category, service history |
| PATCH | `/assets/{id}` | technician+; writes `audit_log` |
| GET | `/assets/by-public-id/{public_id}` | QR resolution |
| GET | `/assets/by-tag/{asset_tag}` | barcode resolution |

No `DELETE /assets/{id}`. Deletion and retention are out of scope and stated as a limitation
(§1.1, §13.7).

### 7.4 Labels — QR and barcode

| Method | Path | Notes |
|---|---|---|
| GET | `/labels/asset/{id}.svg` | single printable label: QR of `{BASE_URL}/a/{public_id}` + Code128 of `asset_tag` + name + tag text |

Server-side SVG generation — `qrcode` (with `qrcode.image.svg`) and `python-barcode` for Code128.
No client-side JS canvas. **One printable label endpoint, no sheet builder** — the paper claims a QR
label per item resolving to a persistent item URL, and this endpoint is that claim.

### 7.5 Attachments

| Method | Path | Notes |
|---|---|---|
| POST | `/assets/{id}/attachments` | multipart: `file`, `kind`, `title`, `is_primary`; or JSON `{kind:'link'|'risk_assessment', url, title}` |
| GET | `/assets/{id}/attachments` | |
| GET | `/attachments/{id}/content` | streams the file; 404 for `kind='link'` |
| DELETE | `/attachments/{id}` | technician+ |

Accept `image/jpeg|png|webp`, `application/pdf`, `.docx`, `.xlsx`. Max 25 MB. Store under
`storage/assets/{asset_id}/{uuid}{ext}` — never trust the client filename for the path.
Setting `is_primary=true` clears the flag on siblings of the same `kind` in one transaction.

### 7.6 Service

| Method | Path | Notes |
|---|---|---|
| POST | `/assets/{id}/service-events` | recomputes `assets.last_serviced_at` |
| GET | `/assets/{id}/service-events` | |
| GET | `/service/due` | `?window_days=30` → `{overdue: [...], due_soon: [...]}` |

### 7.7 Floor plans and locations

| Method | Path | Notes |
|---|---|---|
| POST | `/floorplans` | multipart image; store `image_w/image_h` |
| GET | `/floorplans` · `/floorplans/{id}` | detail includes every pinned asset |
| POST | `/locations` · PATCH `/locations/{id}` | sets `x_pct`, `y_pct` |
| GET | `/floorplans/{id}/assets` | assets pinned on this plan |

### 7.8 Imports — see §9 for the flow

| Method | Path | Notes |
|---|---|---|
| POST | `/imports` | multipart file → `{batch_id, detected_columns, sample_rows}`, status `pending` |
| POST | `/imports/{id}/mapping` | `{column_mapping, precision_target}` → kicks off processing |
| GET | `/imports/{id}` | status + counts |
| GET | `/imports/{id}/rows` | `?route=review&resolved=false&page=` |
| POST | `/imports/{id}/rows/{row_id}/resolve` | `{action, chosen_cpv_code?, chosen_asset_id?, seconds_taken}` |
| POST | `/imports/{id}/commit` | writes remaining accepted rows to `assets` |

### 7.9 Categories

| Method | Path | Notes |
|---|---|---|
| GET | `/categories` | `?level=2|4&q=` |
| PATCH | `/categories/{code}` | admin — sets `hazard_class`, `default_service_interval_days` |

No `/audit` endpoint and no `/users` endpoints. The audit **log** is claimed by the paper and is
written on every mutation (§5.6, D9); an audit **viewer** is not claimed and is not built. Users are
seeded by script.

---

## 8. Frontend routes (Next.js App Router)

| Route | Purpose |
|---|---|
| `/login` | |
| `/` | redirects to `/assets` — **there is no dashboard** (scope fence, §1.1) |
| `/assets` | searchable table; filters for category, location, due |
| `/assets/new` · `/assets/[id]/edit` | forms; category picker is a searchable CPV tree |
| `/assets/[id]` | **detail page — see §8.3** |
| `/a/[publicId]` | QR landing; resolves and redirects to `/assets/[id]`; unauthenticated visitors go to `/login?next=...` (§13.2) |
| `/import` | upload |
| `/import/[batchId]` | column mapping → progress → summary |
| `/import/[batchId]/review` | **review queue — see §9.3** |
| `/floorplans` · `/floorplans/[id]` | plan list; pin placement by click |
| `/service` | overdue / due-soon tabs |

No `/labels` route (the label is `GET /labels/asset/{id}.svg`, linked from the detail page), no
`/admin/users`, no `/audit`.

### 8.3 Asset detail page layout

Order matters — the client called out risk assessments explicitly, so they are above the fold:

1. Header: name, `asset_tag`, QR thumbnail (click → `GET /labels/asset/{id}.svg`)
2. **Safety strip** — primary risk assessment link, `hazard_class` from the category. If no risk
   assessment is attached, show an explicit amber "No risk assessment linked" warning rather than
   an empty area.
3. Photo gallery (primary first)
4. Core fields: manufacturer, model, serial, category, department
5. Location: room text + floor-plan thumbnail with this asset's pin highlighted
6. Service: interval, last serviced, next due (red if overdue), history table, "Log service" button
7. Documents: PDFs, manuals, certificates, links

---

## 9. The bulk import wizard

This is the centrepiece. It is where the research pipeline and the delivered system meet, and it
is the last thing to be cut if time runs short.

### 9.1 Flow

```
1. Upload CSV/XLSX                → POST /imports              status=pending
2. Column mapping UI              → POST /imports/{id}/mapping status=processing
3. Background processing (§9.2)                                status=ready_for_review
4. Summary: N auto, M to review
5. Review queue (§9.3)            → POST .../rows/{id}/resolve
6. Commit                         → POST /imports/{id}/commit  status=committed
```

**Step 2 is not optional.** Real spreadsheets do not have the column names the system expects. The
UI shows detected headers on the left, target fields on the right, and pre-fills guesses using
fuzzy matching (`rapidfuzz.process.extractOne`) against `RECORD_COLUMNS` plus common aliases
(`"Equipment Name"→name`, `"Make"→manufacturer`, `"Serial No"→serial_number`, `"Dept"→owning_department`).

### 9.2 Processing (`services/importer.py`)

Runs in a background task. For each row:

1. Map through `column_mapping` → `Record` shape → store in `import_rows.normalised`
2. `normalise_frame`
3. **Dedup against existing assets:** `blocking.candidate_pairs` against the live `assets` table,
   then the configured `Matcher`. Highest-scoring candidate → `dedup_score`,
   `dedup_candidate_asset_id`. Decision from thresholds selected at `precision_target`:
   `score ≥ upper → 'duplicate'`, `score ≤ lower → 'new'`, else `'uncertain'`
4. **Classify:** the fitted `Classifier` → `class_cpv_code`, `class_score`, `class_alternatives`
   (top 5)
5. **Evidence:** matched n-grams for TF-IDF, nearest labelled neighbours for embedding, band
   position for cascade → `evidence` JSONB
6. **Route:** `route='auto'` only if all three hold — `dedup_decision='new'`, `class_score ≥` the
   classifier threshold at `precision_target`, **and the predicted code is in the supported label
   set**. Anything else → `route='review'`. A record whose best category is a class the classifier
   was never able to learn goes to a human even if the score looks confident, because that score is
   measured over a label set the record's true category is not in (§6.10).

Update `auto_count` / `review_count`; set status `ready_for_review`.

Load the fitted models once at API startup from `models/` (joblib) — do not refit per import.

### 9.3 Review queue UI

One row at a time, keyboard-driven. Must show all three of:

- **The candidate decision** — "Likely duplicate of FCES-000412 (score 0.87)" or
  "Category: 38430000 Measuring instruments (0.62)"
- **The competing alternatives** — the other top-5 codes with scores, each one-click selectable;
  for duplicates, the other candidate assets above the lower threshold
- **The similarity evidence** — highlighted overlapping n-grams between the incoming row and the
  candidate; the nearest labelled examples that drove the classification

Actions: `Accept` (A), `Override` (O — opens picker), `Reject` (R — do not import), `Skip` (S).

**Timing:** start a monotonic timer on render, stop on action, post `seconds_taken` with the
resolve call. Do not use wall-clock deltas that a page refresh can corrupt.

This field does **not** feed `operating_point.residual_effort` — that number comes from the timed
annotation exercise (§6.15). Once the queue has been used, its handling time is reported as
independent corroboration of the annotation figure. The consequence is that RQ3 does not block on
this UI: if the review queue is late or broken, the headline result still stands.

### 9.4 Commit

Insert an `assets` row for every `route='auto'` row and every reviewed row whose action was
`accept`/`override`. Set `assets.source_row_id`. Write one `audit_log` entry per created asset
with `action='import_commit'`. Wrap in a single transaction; on failure set `status='failed'` and
record the error.

---

## 10. Experiment runners

Each script: `python research/scripts/run_X.py --config research/configs/X.yaml`.
Each writes `results/runs/<run_id>/`. `make tables` regenerates everything in `results/tables/`
from those directories.

| Script | Produces | Output artefact |
|---|---|---|
| `run_profile.py` | corpus counts per year/division/level, class distribution, discard tallies, split sizes, **leaf-level sparsity counts (§4.2)**, **division set with and without 39/48 (§4.2)**, same-buyer/same-title group counts (§4.5), **Abt-Buy split sizes and positive rates** | `T1_corpus_a.tex`, `T1_corpus_b.tex`, `T1_discard.tex`, `T1_leaf_sparsity.tex` |
| `audit_real_errors.py` | observed rate of each error class in real Contracts Finder text (§4.1) | feeds T2 |
| `run_degrade_check.py` | injected vs observed rates per class | `T2_degradation.tex` |
| `run_blocking.py` | pair completeness, reduction ratio per scheme × severity, **per corpus, plus which schemes apply to which corpus (§6.8)** | `T3_blocking.tex` |
| `run_dedup.py --corpus abtbuy` | P/R/F1 × 4 matchers on the given test split | `T4_abtbuy.tex` |
| `run_dedup.py --corpus cf --sweep` | P/R/F1 for the three zero-marginal-cost matchers × 5 severities × 3 seeds; **the cascade at 3 severities spanning the same range × 1 seed, adjudicating every pair in its band** | `F1_severity.pdf` |
| `run_classify.py` | macro/weighted F1 × **3** classifiers × 2 levels. The two classical approaches on the **full** test partition; the language model condition on a **stratified sample** of it, with the classical figures reported on that same sample alongside their full ones | `T6_classification.tex` |
| `run_classify.py --per-class` | confusion matrix for hazard-carrying divisions 33, 38, 42 | `T7_perclass.tex` |
| `run_label_noise.py` | disagreement rate vs published CPV + 95% CI + intra-annotator κ; **mean handling time from the timed annotation (§6.15)** | inline figure |
| `run_transfer.py` | **the External Validation comparison.** Thresholds selected on the Corpus A dev partition, carried across **unchanged**, evaluated on degraded Corpus B. Reports both figures and their difference as one paired comparison | `T9_transfer.tex` |
| `run_costs.py` | ms/record, USD/1000 records, measured cascade band fraction (reads the global ledger, grouped by `run_id`) | `T8_cost.tex` |
| `run_operating_point.py` | full precision–automation curve; automated share at 0.95 and 0.99; residual hours | `F2_operating_point.pdf` |

There is no `run_dedup.py --corpus natural` and no `T5_natural.tex` (§4.5).

**Dependency order:** profile → blocking → (abtbuy, severity sweep) → degrade check → classify →
label noise (which the timed annotation feeds) → costs → **operating point**.

`run_operating_point.py` consumes exactly three things: the severity sweep, the classification
results, and `mean_seconds_per_item` passed as an argument. **It does not depend on E4** — the
review queue is not on its critical path (§5.7, §9.3). Because the annotation exercise produces the
handling-time figure, it must be run before the operating-point analysis, which is why it sits early
in Phase G rather than late.

### 10.1 Scope reductions and the budget they buy

This project answers three research questions inside a few dollars and the time remaining. A method
earns its place only if removing it would leave one of RQ1–RQ3 without an answer, or remove the
comparison that gives the answer meaning. The following were removed on that test, and are removed
rather than deferred.

| Removed | Why it fails the test |
|---|---|
| The fourth classification condition (zero-shot) | It answered what the in-context examples contribute, which is a narrower question than RQ2 asks. The comparison that matters to someone migrating a register — does a language model given the labels you already have beat a classifier trained on those same labels — survives with three conditions, at half the call volume. Moved to further work |
| Band subsampling, bootstrap intervals, cascade accuracy by band position | These bought a cheaper cascade by making its accuracy an estimate. Cutting scope was preferred to cutting precision |

**The cascade trades resolution for exactness.** The three matchers with no marginal cost per
decision keep the full severity × repetition factorial and produce the degradation curve. The
cascade runs at **3 severity levels spanning that same range, at a single repetition, adjudicating
every pair in its band**. Three exact points readable against the other methods' curves are worth
more than fifteen estimated ones.

**Corpus B deduplication corpus size.** Sized so the cascade fits *comfortably* inside the budget
rather than exactly filling it. Corpus B's role in RQ1 is in-domain material for the transfer
comparison, not exhaustive coverage.

**The size is a measured result, not a config decision.** The paper reserves a `[TBC]` for it in the
Corpus B subsection, so it must reach the document the way every other number does: `run_dedup.py
--corpus cf --sweep` emits the sampled record count into its run, `make_tables` writes it into
`T4_cf_sweep.tex`, and the paper `\input{}`s it. Recording it only in
`configs/dedup_cf_sweep.yaml` would leave a number in the paper that no run produced.

**Partitioning rule enforced in code:** Contracts Finder splits by `release_date` (train = earlier
months, test = later months), **never at random** — near-identical repeat notices from the same
buyer would otherwise straddle the split. Abt-Buy uses its supplied splits. Write `splits.json`
once and load it everywhere; a test asserts no `record_id` appears in both sides.

**Compute budget:** the heaviest run is `run_dedup.py --corpus cf --sweep`. Expect 45–75 min on
CPU with the embedding cache warm. If it exceeds 2 hours, reduce seeds from 3 to 2 before reducing
severity levels — for the three free matchers only; the cascade's condition count is fixed by §10.1
and is a budget decision, not a compute one.

---

## 11. Build order

Dependency-ordered. Each task lists what it creates and how to know it is done. **Do not start a
task until its dependencies pass their acceptance criteria.**

**Every criterion below tests the implementation, never the outcome** (standing rule 3). Where a
number appears in a criterion it is a bug detector — a value only a broken implementation could
produce — and the measured figure is recorded as an observation regardless of where it falls.

### Phase A — Foundations

| # | Task | Creates | Done when |
|---|---|---|---|
| A1 | Repo skeleton, `docker-compose.yml`, `Makefile`, `.env.example`, both `pyproject.toml`s | §3 tree | `make bootstrap` completes on a clean clone |
| A2 | `schema.py`, `normalise.py` + tests | `research/src/fcesreg/{schema,normalise}.py` | `pytest research/tests/test_normalise.py` passes; mojibake and control-char cases covered; `validate_frame` warns on wholly-null `manufacturer`/`model`/`serial_number` |
| A3 | `runs.py` | `runs.py` | `new_run_id()` is stable for the same config; `capture_env()["git_dirty"]` reflects reality |
| A4 | `timing.py` + tests | `timing.py` | `time_item` measures monotonic elapsed time; `summarise` excludes abandoned items and refuses a mean over <30 retained items |

### Phase B — Data

| # | Task | Depends | Done when |
|---|---|---|---|
| B1 | Download 2022–2024 bundles; symlink existing 2025 | A1 | four dirs under `data/raw/` each containing `main.csv` |
| B2 | `ingest_contractsfinder.py` | A2, B1 | `corpus_b_contractsfinder.parquet` exists; `DiscardReport` JSON written with all six counts, and `total_in - dropped_* == total_out`. **Column names verified across all four years — a mismatch fails loudly and is reported, never coerced (§13.6)** |
| B3 | `cpv.py` + `cpv_taxonomy.parquet` | B2 | `supported_labels` returns a `(labels, coverage)` pair for both levels and both candidate division sets; coverage is a fraction in [0,1] and the labels all meet `min_examples`. **The counts are recorded, not gated** |
| B4 | `ingest_abtbuy.py` | A2 | `corpus_a_abtbuy.parquet`; test split is exactly 1,916 pairs at ~10.7% positive; no pair references an id absent from `tableA`/`tableB` |
| B5 | Freeze `splits.json` (temporal for CF, given for Abt-Buy) | B2, B4 | test asserts **zero `record_id` overlap** between CF dev and test, **and zero *pair* overlap** between Abt-Buy dev and test — see the note below |

**B5's guarantee differs by corpus, because the plan as first written asked for something
impossible.** The original criterion demanded zero `record_id` overlap between dev and test for both
corpora. On Abt-Buy that cannot hold: the supplied splits are defined over pairs from a fixed record
pool and 1,359 records appear on both sides (§4.4). Obtaining a record-level guarantee would require
re-splitting, which §4.4 forbids and which `main.tex` explicitly rules out ("the published train,
validation and test splits are used unchanged, so that results remain comparable with the
literature"). The two requirements contradicted each other; the paper decides it. So:

- **Contracts Finder — record-level.** We control the split, so no `record_id` may appear on both
  sides. Enforced in `splits._check`, and a test additionally asserts that no publication *month*
  straddles the boundary.
- **Abt-Buy — pair-level.** No labelled pair may appear on both sides. Record recurrence is
  documented as a property of the benchmark rather than asserted away.

Measured (B5): CF dev 37,250 / test 7,801 (17.3% held out, cutoff `2025-01-01`, so the 2022–2024
bundles are dev and the 2025 bundle is test); Abt-Buy dev 7,659 pairs / test 1,916 pairs.
`freeze()` refuses to overwrite an existing `splits.json` — the assignment is frozen, and silently
rewriting it would invalidate every result already computed against it.

### Phase C — Algorithms

| # | Task | Depends | Done when |
|---|---|---|---|
| C1 | `embed.py` with disk cache | A2 | second call on identical input is >50× faster and hits zero network |
| C2 | `blocking.py` | B5 | all three §6.8 schemes implemented, no `block_by_manufacturer`; `applicable_schemes` returns `buyer` for CF and not for Abt-Buy; `evaluate_blocking` recovers known pair-completeness and reduction-ratio values on a hand-built fixture, and on the real corpora returns metrics in [0,1] with `n_candidates ≤ n_possible`. **Measured values are recorded, not gated** |
| C3 | `dedup.py` — Exact + Tfidf | C2 | both matchers run end to end on the Abt-Buy test split and emit P/R/F1 with `tp+fp+fn` consistent with the pair count. **Bug detector only: F1 below 0.40 indicates a pair-construction fault — stop and fix.** The actual F1 is recorded as an observation, not a pass mark |
| C4 | `degrade.py` + tests | A2 | same seed ⇒ byte-identical output; each of the **7** error classes has its own test (abbreviation, character noise, casing, whitespace, units, field omission, and Mudgal's field merge — matching the seven knobs in §6.6 and the six classes plus merge the paper lists); `make_distractors` returns a non-empty label-0 set for both corpora under their respective mining rules and touches none of the three null columns |
| C5 | `llm.py` — cache, ledger, hard cap | A3 | a $0.20 pilot runs; re-running the identical set costs **exactly $0.00** and every row logs `cache_hit=true`; ledger rows land in the single `results/ledger.jsonl` carrying `run_id` |
| C6 | `dedup.py` — Embedding + Cascade | C1, C3, C5 | `CascadeMatcher.stats` is populated with all three keys; the counts are mutually consistent (`n_adjudicated ≤ n_pairs`, `band_fraction == n_adjudicated / n_pairs`); and **every pair sent to the adjudicator has a base score strictly inside `(lower, upper)`, with no pair outside the band adjudicated**. The measured band fraction is recorded as a finding — a fraction of 0.30 is a fact about the method, not a build failure |
| C7 | `classify.py` — all three (§10.1) | C1, C5, B3 | every classifier returns non-empty `alternatives`; the language model condition never receives the full taxonomy, only the shortlist |
| C8 | `metrics.py`, `operating_point.py` | C3 | `automated_share_at_precision` recovers a known answer on a synthetic fixture |

### Phase D — System backend

| # | Task | Depends | Done when |
|---|---|---|---|
| D1 | Alembic `0001_initial` (all of §5) | A1 | `alembic upgrade head` then `downgrade base` runs clean |
| D2 | Auth + role dependency + `seed_users.py` | D1 | `readonly` POST → 403; `technician` POST → 201; three seeded users exist. No `/users` API |
| D3 | Assets CRUD + search | D2 | `GET /assets?q=microscope` uses the GIN index (verify with EXPLAIN); no trigram path exists |
| D4 | `seed_categories.py` | D1, B3 | `categories` populated at levels 2 and 4; divisions 33/38/42 carry the illustrative `hazard_class` and interval, everything else NULL; the script logs that the designation is illustrative |
| D5 | Attachments + storage | D3 | photo, PDF and `risk_assessment` link all upload; `is_primary` is exclusive per kind |
| D6 | Label service — QR + Code128 | D3 | `GET /labels/asset/{id}.svg` renders; scanning the QR opens `/a/{public_id}`; the barcode reads back `asset_tag` |
| D7 | Service events + due view | D3 | `next_due_at` recomputes on service log; `/service/due` splits overdue from due-soon |
| D7b | **Notification scheduler** (§5.8) | D7 | the daily job writes `notifications` rows for due-soon and overdue items; running it twice in one day inserts no duplicates; rows surface on `/service` and on login |
| D8 | Floor plans + pins | D3 | pin placed at 25%/75% renders in the same relative spot after the image is re-uploaded at a different resolution |
| D9 | Audit log writes | D3 | every write records before/after JSON. Log only — no viewer |

### Phase E — The wizard

| # | Task | Depends | Done when |
|---|---|---|---|
| E1 | `services/pipeline.py` — model loading, the only `fcesreg` import site | C6, C7, D3 | models load once at startup; `grep -r "import fcesreg" system/ ` returns exactly this one file |
| E2 | Upload + column detection + mapping UI | D3 | a spreadsheet with headers that match nothing still maps successfully by hand |
| E3 | `services/importer.py` background processing | E1, E2 | 50-row file → rows split auto/review with scores, alternatives and evidence populated |
| E4 | Review queue UI with timer | E3 | all three of candidate / alternatives / evidence visible; `seconds_taken` written and non-zero. **Not on the critical path for RQ3** (§5.7) |
| E5 | Commit | E4 | accepted rows appear in `assets` with `source_row_id` set; one audit row each |

### Phase F — Frontend completion

F1 asset list + filters · F2 asset detail (§8.3 order, safety strip above the fold) · F3 floor plan
pin UI · F4 service view.

No dashboard, no label sheet builder, no admin or audit screens (scope fence, §1.1).

### Phase G — Experiments

G1 `run_profile` (incl. the 39/48 comparison and leaf sparsity) · G2 `audit_real_errors` +
`run_degrade_check` · G3 `run_blocking` · **G4 annotation exercise (§13.3) — 300 items, timed** ·
G5 `run_label_noise` (labels + handling time) · G6 `run_dedup` (abtbuy, sweep) ·
G7 `run_classify` (+per-class) · G8 `run_costs` · **G9 `run_operating_point`** · **G10 `run_transfer`** · G11 `make tables`.

The annotation exercise moves **early**: it produces both the label-noise estimate and the
`mean_seconds_per_item` figure that G9 consumes, so everything downstream waits on it. G9 depends on
G6, G7 and that number — **not on Phase E**.

---

## 12. Conventions

### 12.1 Naming
`snake_case` Python, `camelCase` TypeScript, `snake_case` SQL. API JSON keys are `snake_case` —
do not translate at the boundary.

### 12.2 Errors
FastAPI returns `{"detail": {"code": "asset_not_found", "message": "..."}}`. Never leak a stack
trace. 401 unauthenticated, 403 authenticated-but-wrong-role, 404 missing, 409 conflict,
422 validation.

### 12.3 Config
All settings from environment through a Pydantic `Settings` class. Never `os.getenv` inline.
Required: `DATABASE_URL`, `JWT_SECRET`, `ANTHROPIC_API_KEY`, `STORAGE_ROOT`, `BASE_URL`.

### 12.4 Tests
`pytest` for both packages. Minimum: normalisation edge cases, degradation determinism, blocking
metrics on a fixture, distractor mining on a fixture, threshold selection, timing summarisation,
role enforcement per endpoint, the full import flow on a 10-row fixture driven through the API.

**No Playwright, no browser end-to-end coverage** (scope fence, §1.1). The import flow is tested at
the API layer, which is where the logic lives; the UI is verified by hand.

### 12.5 Makefile targets
```
make bootstrap     # venv, deps, docker up, migrate, seed users + categories
make data          # ingest both corpora, build taxonomy, freeze splits
make annotate      # the timed annotation exercise — run before `make experiments`
make experiments   # every runner in dependency order
make tables        # regenerate results/tables/ from results/runs/
make dev           # api + web with reload
make test
make smoke         # 100 records, 1 severity, 1 seed — must finish <3 min, $0.00 API spend
```

### 12.6 Result provenance and the paper build
1. Run ID = `<script>-<UTC ts>-<git sha>-<config sha>`.
2. A dirty git tree marks the run; `make_tables.py` refuses it.
3. Every generated table caption carries its source `run_id`.
4. **`make paper` must build clean before any commit touching `main.tex`.** Two `pdflatex` passes,
   no `bibtex` — the bibliography is inline as `thebibliography`. TeX lives at `/Library/TeX/texbin`,
   which is not on the default PATH. A paper that is correct in content and refuses to build is
   discovered at the worst possible moment, so this is a precondition in the same way a clean git
   tree is a precondition for `make tables`.
5. **No number is ever typed into a paper or report by hand** — everything is `\input{}` from
   `results/tables/`. Re-run `make tables` as the final act before any submission.
5. **Commit at the granularity of the §11 build-order table — one commit per task.** The run id
   embeds a git SHA so that a result can be traced to the code that produced it, and that is worth
   nothing if the whole codebase is a single commit. A SHA has to identify a meaningful state.
6. **No corpus is committed.** `data/raw/` and `data/processed/` are gitignored, and the README
   carries acquisition instructions instead. The paper states that corpora derived from third-party
   sources are distributed as record identifiers and reconstruction scripts rather than as
   redistributed data, in accordance with each licence; a repository containing the CSVs would
   contradict that claim. Build artefacts (`*.egg-info/`) are likewise untracked.

### 12.7 CPU only
No CUDA, no MPS. Do not call `.to(device)`. `sentence-transformers` defaults to CPU — leave it.
Embedding batch size 64. `bge-small-en-v1.5` or `all-MiniLM-L6-v2`, nothing larger.
No transformer fine-tuning anywhere.

---

## 13. Decisions — settled

These were the open questions. All are now answered. Build to the answer.

1. **The precision floor `P*` — `0.95`.** `DEFAULT_TARGET = 0.95` and it is the headline. Report the
   **full precision–automation curve**, with the automated share at both 0.95 and 0.99 called out on
   it. The curve is the result; the two points are the summary.
2. **`/a/{public_id}` requires authentication** and redirects unauthenticated visitors to
   `/login?next=...`. Anonymous read of a limited field set is **not built**; it is written up as a
   limitation in the report — a QR label on a machine in a corridor is scannable by anyone, and the
   trade-off between that and open access is discussed, not implemented.
3. **CPV label-noise annotation — 300 items, labelled by the supervisor/author, timed per
   judgement.** Blind judgement against a 12-code shortlist, stratified per `annotation/protocol.md`.
   `annotation/annotate.py` (§6.15) times every judgement with a monotonic timer. **This exercise
   produces two results: the label-noise rate and the `mean_seconds_per_item` figure that RQ3
   consumes — so it is built and run before the experiments that depend on it** (Phase G4).
4. **Natural duplicate labels — removed.** No `--corpus natural`, no hand-labelling of the 154
   groups. The counts survive as a descriptive statistic only (§4.5).
5. **`hazard_class` — designation, not vocabulary.** Divisions 33, 38 and 42 are designated with a
   default service interval per the §5.4 table. **No hazard taxonomy is designed.** The designation
   is illustrative and is marked as such in the seed script and in the report; the paper needs only
   that *some* divisions carry hazard or servicing rules so the per-class breakdown has something to
   report.
6. **Column schema across year bundles — fail loudly.** If 2022–2024 differ from 2025 in column
   names or shape, `load_bundle` raises with the exact diff and **reports it to the supervisor**.
   Never coerce, never rename silently, never fall back to positional columns. If reconciling the
   older bundles turns out to be significant work, proceed with the years that parse and switch the
   temporal split to **within-year months**, recording that as a stated limitation in the paper.
7. **Retention and deletion — out of scope.** No delete endpoint, no soft delete, no status enum.
   What happens to an asset that leaves the register, and to its attached files, is stated as a
   limitation rather than implemented.

---

## 14. Amendment log

Changes applied to this document after the first draft, in the order the supervisor raised them.

| # | Change | Sections touched |
|---|---|---|
| 1 | `manufacturer` / `model` / `serial_number` are null in both corpora. Blocking re-specified as sorted char n-grams + leading token + `buyer_id`; `block_by_manufacturer` deleted. `make_distractors` re-specified per corpus (CPV class + title similarity on CF; leading token on Abt-Buy) | §6.1, §6.6, §6.8, §10, C2, C4 |
| 2 | Acceptance criteria test the implementation, never the outcome. C6 rewritten around `stats` consistency and band membership; C3's floor kept only as a bug detector; the same reading applied to B2, B3, C2 | standing rules, §11 |
| 3 | Handling time comes from the timed annotation exercise, not `review_decisions`. New `timing.py` (§6.15) and `annotation/annotate.py`; `residual_effort` takes the figure as an argument; G9 no longer depends on E4 | §5.7, §6.13, §6.15, §9.3, §10, §11 |
| 4 | Removed: `--corpus natural` + T5 + §4.5 mining rule + §13.4; Amazon-Google and Walmart-Amazon; any Jaro-Winkler matcher | §4.5, §10, §11, §13 |
| 5 | System scope fence: removed dashboard, user admin UI, audit viewer, label sheet builder, Playwright, trigram search, soft delete + status enum + delete endpoint, `value_gbp`, `purchase_date`, `POST /auth/logout` | §1.1, §5.1, §5.5, §7, §8, §11, §12.4 |
| 6 | CPV division set must be justified, not asserted: measure with and without 39 and 48, report both, one argued sentence in the paper | §4.2, §10, B3 |
| — | One global `results/ledger.jsonl` carrying `run_id`, not one per run | §3, §6.11 |
| — | Leaf-level sparsity counts are a reported result and must reach `T1_leaf_sparsity.tex` | §4.2, §10 |
| B | Phase B measurements folded back in: corrected bundle sizes and per-year counts, the division-set decision (8 divisions, 39 and 48 dropped) with its justifying sentence, four-year leaf sparsity, the discard tally, the corrected `desc == title` figure, and `splits.py` | §3, §4.1–4.5, §6.16, §11 |
| B | B5's acceptance criterion was impossible as written — record-level overlap cannot be zero on Abt-Buy's pair-level splits. Split into a record-level guarantee for CF and a pair-level one for Abt-Buy, per `main.tex` | §4.4, §11 |
| R1 | **The four-digit level is a CPV *class*, not a group.** Official CPV is division (2) / group (3) / class (4) / category (5). The study evaluates 2 and 4, so those are division and class. `group()` → `cpv_class()`, `Level` → `"division" \| "class"`, and the note claiming divergence from official nomenclature is deleted because after the rename there is none | §4.2, §6.5, §10, tests |
| R1 | **Corpus letters were transposed and stale.** The paper's Corpus A is Abt-Buy (duplicates) and Corpus B is Contracts Finder (categories). Artefacts renamed `corpus_a_abtbuy{,_pairs}.parquet` and `corpus_b_contractsfinder.parquet` | §3, §10, Makefile, tests |
| R1 | Corpus removed from git history; `data/raw/2025/` replaces the root `2025/`; `*.egg-info/` untracked; commit granularity fixed at one per build-order task | §3, §4.1, §12.6 |
| R1 | Artefact-dependent tests skip with a reason naming the missing file, so a clean clone reads "waiting on `make data`" rather than showing failures | `research/tests/conftest.py`, §12.4 |
| R1 | Corpus descriptive statistics reach the paper through `results/tables/` like every other number, not as prose in this document | §10, `run_profile.py`, `make_tables.py` |
| R1 | `main.tex` is the paper. `main.md` deleted rather than maintained as a second copy that would drift | README, standing rule 1 |

**Two observations returned to the supervisor.** Neither Amazon-Google/Walmart-Amazon nor a
Jaro-Winkler matcher was present in the draft to remove — the plan already claimed Abt-Buy alone and
already listed exactly four matchers. `rapidfuzz` was listed as a research dependency but is used
only for column-header guessing in the import wizard, so it moved to `system/api` (§2).
