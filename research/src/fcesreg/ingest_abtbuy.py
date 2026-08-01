"""Abt-Buy ingest (§6.4).

The text-heavy benchmark from the DeepMatcher suite. It is used because at least one of
its attributes is long free text, which is the property distinguishing equipment
descriptions from purely structured records, and because published results exist for it
so the methods here can be positioned against the literature.

**The splits are supplied and are used exactly as given — never regenerated.** Re-splitting
would break comparability with every published figure on this benchmark.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from fcesreg.schema import RECORD_COLUMNS, validate_frame

__all__ = ["BASE", "MIRROR", "FILES", "SPLITS", "download", "load", "build"]

BASE = "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/exp_data"
MIRROR = "https://dbs.uni-leipzig.de/file/Abt-Buy.zip"

FILES = ("tableA.csv", "tableB.csv", "train.csv", "valid.csv", "test.csv")
SPLITS = ("train", "valid", "test")


def download(dest: Path, force: bool = False) -> None:
    """Fetch the five supplied files into ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = dest / name
        if target.exists() and not force:
            continue
        url = f"{BASE}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                target.write_bytes(r.read())
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(
                f"could not fetch {url}: {e}. If Wisconsin is down, retrieve the "
                f"benchmark from the mirror at {MIRROR} and unpack it into {dest}."
            ) from e


def load(dest: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(records, pairs)``.

    ``records`` is in RECORD_COLUMNS shape with ``record_id`` prefixed ``A:``/``B:`` so
    the two tables share one identifier space. ``pairs`` carries
    ``left_id, right_id, label, split``, with ``split`` taken from which supplied file the
    row came from.

    The ``price`` column is dropped: it has no counterpart in the Record schema and
    nothing in the pipeline uses it.
    """
    dest = Path(dest)

    frames = []
    for prefix, filename in (("A", "tableA.csv"), ("B", "tableB.csv")):
        t = pd.read_csv(dest / filename, dtype=str, keep_default_na=False)
        out = pd.DataFrame(
            {
                "record_id": prefix + ":" + t["id"].astype(str),
                "title": t["name"],
                "description": t["description"].replace("", None),
                "manufacturer": None,
                "model": None,
                "serial_number": None,
                "buyer_id": None,  # Abt-Buy has no publishing authority — see §6.8
                "cpv_code": None,
                "release_date": None,
                "source": "abtbuy",
            }
        )
        out["table"] = prefix
        frames.append(out)
    records = pd.concat(frames, ignore_index=True)[[*RECORD_COLUMNS, "table"]]

    pair_frames = []
    for split in SPLITS:
        p = pd.read_csv(dest / f"{split}.csv")
        pair_frames.append(
            pd.DataFrame(
                {
                    "left_id": "A:" + p["ltable_id"].astype(str),
                    "right_id": "B:" + p["rtable_id"].astype(str),
                    "label": p["label"].astype(int),
                    "split": split,
                }
            )
        )
    pairs = pd.concat(pair_frames, ignore_index=True)

    known = set(records["record_id"])
    dangling = (~pairs["left_id"].isin(known)) | (~pairs["right_id"].isin(known))
    if dangling.any():
        raise ValueError(
            f"{int(dangling.sum())} pairs reference ids absent from tableA/tableB — "
            "the download is incomplete or mismatched"
        )

    return records, pairs


def build(dest: Path, out_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    download(dest)
    records, pairs = load(dest)
    validate_frame(records)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records.to_parquet(out_path, index=False)
    pairs.to_parquet(out_path.with_name(out_path.stem + "_pairs.parquet"), index=False)

    for split in SPLITS:
        s = pairs[pairs["split"] == split]
        print(
            f"  {split}: {len(s):>5,} pairs, {100 * s['label'].mean():.1f}% positive",
            file=sys.stderr,
        )
    return records, pairs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest the Abt-Buy benchmark.")
    p.add_argument("--dest", type=Path, default=Path("data/raw/abtbuy"))
    p.add_argument("--out", type=Path, default=Path("data/processed/corpus_a_abtbuy.parquet"))
    args = p.parse_args(argv)

    records, pairs = build(args.dest, args.out)
    print(
        f"wrote {len(records):,} records and {len(pairs):,} pairs to {args.out.parent}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
