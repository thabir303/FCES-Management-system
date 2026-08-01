"""Freeze the dev/test assignment (B5). Run once; never regenerate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from fcesreg.splits import SPLITS_PATH, freeze


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus-b",
        type=Path,
        default=Path("data/processed/corpus_b_contractsfinder.parquet"),
        help="Corpus B — Contracts Finder, category ground truth",
    )
    p.add_argument(
        "--corpus-a-pairs",
        type=Path,
        default=Path("data/processed/corpus_a_abtbuy_pairs.parquet"),
        help="Corpus A — Abt-Buy, duplicate ground truth",
    )
    p.add_argument("--out", type=Path, default=SPLITS_PATH)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="invalidates every result computed against the existing assignment",
    )
    args = p.parse_args(argv)

    corpus_b = pd.read_parquet(args.corpus_b)
    pairs = pd.read_parquet(args.corpus_a_pairs)

    try:
        splits = freeze(corpus_b, pairs, path=args.out, overwrite=args.overwrite)
    except FileExistsError as e:
        print(f"\n{e}\n", file=sys.stderr)
        return 1

    n_dev, n_test = len(splits.cf_dev), len(splits.cf_test)
    total = n_dev + n_test
    print(
        f"contractsfinder  dev {n_dev:>6,} | test {n_test:>6,} "
        f"({100 * n_test / total:.1f}% held out, cutoff {splits.cutoff})",
        file=sys.stderr,
    )
    print(
        f"abtbuy           dev {len(splits.abtbuy_dev_pairs):>6,} pairs | "
        f"test {len(splits.abtbuy_test_pairs):>6,} pairs",
        file=sys.stderr,
    )
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
