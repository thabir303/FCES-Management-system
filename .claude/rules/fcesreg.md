---
paths:
  - "research/src/fcesreg/**"
---

# Rules for the `fcesreg` package

**The boundary.** `fcesreg` must never import from `system/`. It takes DataFrames and returns
DataFrames; the API adapts between HTTP/DB and those DataFrames in exactly one file. This boundary
is what makes "the pipeline is deployed inside the delivered system" a true statement rather than a
claim. No database access, and no network access outside `ingest_*` and `llm`.

**`manufacturer`, `model` and `serial_number` are null across both research corpora.** They survive
in `Record` only because `source="upload"` can populate them. Nothing in the research path may key,
block, group or score on them — see `schema.NULL_IN_BOTH_CORPORA`. A scheme that reads them
silently produces one enormous block, no blocks at all, or an empty negative set. Do not
manufacture a `manufacturer` column by parsing titles; where a brand proxy is needed, use
`blocking.block_by_leading_token`, which the paper states as an approximation and evaluates as one.

**No path may resolve against the working directory.** Use `paths.data_path()`,
`paths.results_path()`, `paths.annotation_path()`. A bare `Path("data/…")` is correct from the repo
root and wrong from anywhere else; this exact defect gave 14 failures from `research/` while the
root suite passed. `test_paths.py` greps for the pattern — a new module reintroducing it fails.

**CPU only.** No CUDA, no MPS, no `.to(device)`. `torch.backends.mps.is_available()` is `True` on
this machine and must still not be used (§12.7). Embedding batch size 64, `bge-small-en-v1.5` or
`all-MiniLM-L6-v2`, nothing larger. No transformer fine-tuning anywhere.

**Determinism.** Every function taking randomness takes an explicit `rng`. No module-level random
state. The same seed must reproduce byte-identical output, from any working directory.

**Two taxonomy levels only**, `"division"` (2-digit) and `"class"` (4-digit). Leaf level is not
implemented and must not be — §4.2 measured why. Note the four-digit level is a CPV *class*; the
official *group* is three digits and is not evaluated here.

**Unmeasured is not estimated.** A metric with no ground truth available returns `None` with a note,
never an inferred value. Pair completeness on a corpus without labelled pairs is the live example.
