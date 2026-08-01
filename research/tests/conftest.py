"""Shared fixtures and the artefact guard.

Some tests assert against the real corpora rather than a fixture, because the properties
they check — the supplied Abt-Buy split sizes, the taxonomy's parent closure, the absence
of record overlap in the frozen splits — are properties of the actual data and are worth
nothing when checked against something synthetic.

Those artefacts are gitignored, so a clean clone does not have them until ``make data``
has run. The guard below skips such tests with a reason naming the missing file and the
command that produces it, so a fresh checkout reports "waiting on make data" rather than
twenty-two failures that look like broken code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROCESSED = Path("data/processed")

CORPUS_A = PROCESSED / "corpus_a_abtbuy.parquet"
CORPUS_A_PAIRS = PROCESSED / "corpus_a_abtbuy_pairs.parquet"
CORPUS_B = PROCESSED / "corpus_b_contractsfinder.parquet"
TAXONOMY = PROCESSED / "cpv_taxonomy.parquet"
SPLITS = PROCESSED / "splits.json"
ABTBUY_RAW = Path("data/raw/abtbuy")


def requires(*paths: Path) -> pytest.MarkDecorator:
    """Skip unless every named artefact exists, saying which one is missing."""
    missing = [str(p) for p in paths if not p.exists()]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"needs {', '.join(missing)} — run `make data` first",
    )
