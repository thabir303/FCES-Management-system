"""Frozen dev/test partitions (§10, B5).

Every corpus is partitioned **before any tuning**. The development partition is used for
threshold selection, prompt design and model fitting; the held-out partition is used only
for the final reported results.

The assignment is written once to ``data/processed/splits.json`` and loaded everywhere.
It is never regenerated: a split that drifts as the code changes makes every number before
the drift incomparable with every number after it.

The two corpora are partitioned differently, and for different reasons.

Contracts Finder
    Partitioned by publication period, never at random. Near-identical repeat notices from
    the same buyer would otherwise straddle the split and flatter every result. The
    guarantee is **record-level**: no ``record_id`` appears on both sides.

Abt-Buy
    The supplied train/valid/test files are used exactly as given, so results stay
    comparable with the published literature. Those splits are defined over *pairs* drawn
    from a fixed record pool, so a record can and does appear on both sides — 1,359 of
    them do. The guarantee here is therefore **pair-level**: no labelled pair appears on
    both sides. Re-splitting to obtain a record-level guarantee would break comparability
    with every published figure on this benchmark, which §4.4 forbids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

__all__ = [
    "SPLITS_PATH",
    "CF_CUTOFF",
    "SplitOverlap",
    "Splits",
    "freeze",
    "load",
]

SPLITS_PATH = Path("data/processed/splits.json")

#: Dev is every notice published before this date, test is everything from it onwards.
#: This puts the 2022–2024 bundles in dev and the 2025 bundle in test, which is both a
#: clean temporal boundary and the way a register actually accumulates.
CF_CUTOFF = date(2025, 1, 1)


class SplitOverlap(AssertionError):
    """Dev and test share something they must not."""


@dataclass(frozen=True)
class Splits:
    cf_dev: set[str]
    cf_test: set[str]
    abtbuy_dev_pairs: set[tuple[str, str]]
    abtbuy_test_pairs: set[tuple[str, str]]
    cutoff: date

    def cf(self, df: pd.DataFrame, part: str) -> pd.DataFrame:
        ids = self.cf_dev if part == "dev" else self.cf_test
        return df[df["record_id"].isin(ids)].reset_index(drop=True)

    def abtbuy(self, pairs: pd.DataFrame, part: str) -> pd.DataFrame:
        wanted = self.abtbuy_dev_pairs if part == "dev" else self.abtbuy_test_pairs
        keys = list(zip(pairs["left_id"], pairs["right_id"], strict=True))
        mask = [k in wanted for k in keys]
        return pairs[mask].reset_index(drop=True)


def _check(splits: Splits) -> None:
    """The invariants B5 exists to protect."""
    shared_records = splits.cf_dev & splits.cf_test
    if shared_records:
        raise SplitOverlap(
            f"{len(shared_records)} Contracts Finder record_ids appear in both dev "
            f"and test, e.g. {sorted(shared_records)[:3]}"
        )
    shared_pairs = splits.abtbuy_dev_pairs & splits.abtbuy_test_pairs
    if shared_pairs:
        raise SplitOverlap(
            f"{len(shared_pairs)} Abt-Buy pairs appear in both dev and test"
        )
    if not splits.cf_dev or not splits.cf_test:
        raise SplitOverlap("a Contracts Finder partition is empty")


def freeze(
    corpus_b: pd.DataFrame,
    abtbuy_pairs: pd.DataFrame,
    path: Path = SPLITS_PATH,
    cutoff: date = CF_CUTOFF,
    overwrite: bool = False,
) -> Splits:
    """Compute and write the partitions. Refuses to overwrite an existing file.

    The refusal is the point: ``splits.json`` is frozen, and silently rewriting it would
    invalidate every result already produced against the old assignment.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists and splits are frozen. Pass overwrite=True only if "
            "you intend to invalidate every result computed against the old assignment."
        )

    dates = pd.to_datetime(corpus_b["release_date"]).dt.date
    is_test = dates >= cutoff
    splits = Splits(
        cf_dev=set(corpus_b.loc[~is_test, "record_id"]),
        cf_test=set(corpus_b.loc[is_test, "record_id"]),
        abtbuy_dev_pairs={
            (a, b)
            for a, b in zip(
                abtbuy_pairs.loc[abtbuy_pairs["split"] != "test", "left_id"],
                abtbuy_pairs.loc[abtbuy_pairs["split"] != "test", "right_id"],
                strict=True,
            )
        },
        abtbuy_test_pairs={
            (a, b)
            for a, b in zip(
                abtbuy_pairs.loc[abtbuy_pairs["split"] == "test", "left_id"],
                abtbuy_pairs.loc[abtbuy_pairs["split"] == "test", "right_id"],
                strict=True,
            )
        },
        cutoff=cutoff,
    )
    _check(splits)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "contractsfinder": {
                    "strategy": "temporal",
                    "guarantee": "record-level: no record_id on both sides",
                    "cutoff": cutoff.isoformat(),
                    "dev": sorted(splits.cf_dev),
                    "test": sorted(splits.cf_test),
                },
                "abtbuy": {
                    "strategy": "supplied",
                    "guarantee": (
                        "pair-level: no labelled pair on both sides. Records DO recur "
                        "across sides by the benchmark's own design; re-splitting would "
                        "break comparability with published results (§4.4)."
                    ),
                    "dev_pairs": sorted(splits.abtbuy_dev_pairs),
                    "test_pairs": sorted(splits.abtbuy_test_pairs),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return splits


def load(path: Path = SPLITS_PATH) -> Splits:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = Splits(
        cf_dev=set(data["contractsfinder"]["dev"]),
        cf_test=set(data["contractsfinder"]["test"]),
        abtbuy_dev_pairs={tuple(p) for p in data["abtbuy"]["dev_pairs"]},
        abtbuy_test_pairs={tuple(p) for p in data["abtbuy"]["test_pairs"]},
        cutoff=date.fromisoformat(data["contractsfinder"]["cutoff"]),
    )
    _check(splits)
    return splits
