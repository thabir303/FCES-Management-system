# FCES Asset Register

Two deliverables sharing one codebase:

- **`research/`** — `fcesreg`, the migration pipeline. Duplicate detection, CPV taxonomy
  assignment, and the automated-share-at-fixed-precision analysis that answers RQ1–RQ3.
- **`system/`** — the asset management system delivered to the Faculty of Computing, Engineering
  and Science: item records with photographs and documents, QR labels, floor-plan pins, service
  reminders, tiered access, an audit log, and the bulk import wizard.

`fcesreg` never imports from `system/`. The API adapts between HTTP/DB and DataFrames in exactly
one file, `system/api/src/fcesapi/services/pipeline.py`. That boundary is what makes "the pipeline
is deployed inside the delivered system" a true statement rather than a claim.

`PROJECT_PLAN.md` is the build specification. **`main.tex` is the paper** — it is the authority on
what the code must produce, and it is what compiles to the submitted artefact. Where the plan and
the paper disagree, stop and raise it rather than choosing a side.

## Bootstrap

```bash
pyenv install 3.12.8 && pyenv local 3.12.8
cp .env.example .env        # fill in JWT_SECRET, GROQ_API_KEY, seed passwords
make bootstrap
```

Postgres runs in Docker on port **5433**, not 5432, to avoid clashing with a local install.

## Obtaining the corpora

**No corpus is committed to this repository.** Both are third-party data redistributed under their
own licences, so the repository carries the acquisition scripts and the record identifiers needed to
reconstruct them, not the data itself. `data/raw/` and `data/processed/` are gitignored.

### Corpus A — Abt-Buy (duplicate ground truth)

Fetched automatically by `make data`. Five files from
`https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/exp_data/`
(`tableA`, `tableB`, `train`, `valid`, `test`), landing in `data/raw/abtbuy/`. Mirror if Wisconsin
is unreachable: `https://dbs.uni-leipzig.de/file/Abt-Buy.zip`. The supplied train/valid/test splits
are used exactly as given and are never regenerated.

### Corpus B — UK Contracts Finder (category ground truth)

Four annual OCDS bundles, released under the Open Government Licence. Download each from:

```
https://data.open-contracting.org/en/publication/128/download?name=<YEAR>.csv.tar.gz
```

for `<YEAR>` in **2022, 2023, 2024, 2025** (roughly 54–59 MB compressed, 170–280 MB extracted
each). Each archive unpacks to a nested `<YEAR>/` directory containing 13 CSVs; flatten it so the
layout is:

```
data/raw/
├── 2022/main.csv, tender_additionalClassifications.csv, ...
├── 2023/…
├── 2024/…
├── 2025/…
└── abtbuy/tableA.csv, tableB.csv, train.csv, valid.csv, test.csv
```

`main.csv` supplies the notices; `tender_additionalClassifications.csv` supplies the CPV ancestors
from which the taxonomy is reconstructed. Ingest verifies the 32-column header against the schema it
was built for and **fails loudly** on any difference rather than coercing.

## Order of work

```bash
make bootstrap      # venv, deps, database, migrations, seed data
make data           # ingest both corpora, build the CPV taxonomy, freeze the splits
make annotate       # the timed annotation exercise — must precede experiments
make experiments    # every runner, in dependency order
make tables         # regenerate results/tables/ from results/runs/
```

`make annotate` comes before `make experiments` because it produces both the label-noise estimate
and the mean handling time that the operating-point analysis consumes.

## Tests

```bash
make test
```

Tests that assert against the real corpora skip with a reason naming the missing artefact until
`make data` has run. On a clean clone this is expected and is not a failure.

## Reproducibility

No number reaches the paper by hand. Every figure is `\input{}` from `results/tables/`, generated
from `results/runs/<run_id>/`. A dirty git tree marks the run and `make_tables.py` refuses it.
Commits are made per build-order task so that a run's git SHA identifies a meaningful state of the
code.
