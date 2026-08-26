"""One-time extension of the CPV label-noise sample from n=40 to n=200 (supervisor
instruction, 2026-08-26): "keep the existing 40 and add to them... the sample must be a
strict superset drawn under the same rule."

``annotate.py.load_sample`` cannot simply be re-run at ``--n 200`` to get a superset:
proportional allocation recomputes every division's quota from scratch, which shifts every
``rng.choice`` call -- verified only 2 of the original 40 record_ids recur if you just widen
n and keep seed=0. This script instead holds the original 40 fixed and draws the shortfall
per division (k_200 - k_40, all non-negative -- checked below) from the remaining pool,
under a new seed so the draw is independent of and does not overlap the first.

Writes the 160 new (unjudged) records to stdout as JSON lines for manual review, in the same
shape ``annotate.py.render`` expects. Judging happens separately -- this script only draws
the sample.

    python annotation/extend_label_noise_sample.py > /tmp/label_noise_extension.jsonl
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from fcesreg.paths import annotation_path, data_path
from fcesreg.splits import load as load_splits

DIVISIONS = ["30", "31", "32", "33", "38", "42", "43", "44"]
TARGET_N = 200
EXTENSION_SEED = 1  # distinct from the original draw's seed=0


def _pool() -> pd.DataFrame:
    corpus = pd.read_parquet(data_path("processed", "corpus_b_contractsfinder.parquet"))
    corpus = corpus[corpus["cpv_code"].str[:2].isin(set(DIVISIONS))]
    test_ids = load_splits().cf_test
    corpus = corpus[corpus["record_id"].isin(test_ids)].reset_index(drop=True)
    return corpus.assign(_division=corpus["cpv_code"].str[:2])


def _allocate(shares: pd.Series, n: int) -> pd.Series:
    exact = shares / shares.sum() * n
    take = np.floor(exact).astype(int)
    remainder = (exact - take).sort_values(ascending=False, kind="stable")
    for division in remainder.index[: n - int(take.sum())]:
        take[division] += 1
    return take


def main() -> int:
    existing_path = annotation_path("labels", "cpv_label_noise.jsonl")
    existing = [
        json.loads(line)
        for line in existing_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    existing_ids = {row["record_id"] for row in existing}
    if len(existing_ids) != 40:
        raise SystemExit(f"expected 40 existing judged records, found {len(existing_ids)}")

    corpus = _pool()
    shares = corpus["_division"].value_counts().sort_index()
    k200 = _allocate(shares, TARGET_N)

    existing_by_division = (
        corpus[corpus["record_id"].isin(existing_ids)]["_division"].value_counts()
    )

    rng = np.random.default_rng(EXTENSION_SEED)
    picked = []
    for division, target in k200.items():
        have = int(existing_by_division.get(division, 0))
        need = int(target) - have
        if need <= 0:
            continue
        block = corpus[
            (corpus["_division"] == division) & (~corpus["record_id"].isin(existing_ids))
        ]
        idx = rng.choice(len(block), size=min(need, len(block)), replace=False)
        picked.append(block.iloc[np.sort(idx)])

    extension = pd.concat(picked).reset_index(drop=True)
    assert len(extension) == TARGET_N - len(existing_ids), (
        f"expected {TARGET_N - len(existing_ids)} new records, drew {len(extension)}"
    )
    assert not set(extension["record_id"]) & existing_ids, "extension overlaps the original 40"

    taxonomy = pd.read_parquet(data_path("processed", "cpv_taxonomy.parquet"))
    lookup = dict(zip(taxonomy["cpv_code"], taxonomy["cpv_description"], strict=True))
    extension = extension.assign(
        _division_desc=extension["cpv_code"].str[:2].map(lookup),
        _class_desc=extension["cpv_code"].str[:4].map(lookup),
    )

    for _, row in extension.iterrows():
        print(json.dumps({
            "record_id": row["record_id"],
            "title": row["title"],
            "description": row["description"],
            "cpv_code": row["cpv_code"],
            "division_desc": row["_division_desc"],
            "class_desc": row["_class_desc"],
        }))
    print(f"drew {len(extension)} new records, 0 overlap with the original 40", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
