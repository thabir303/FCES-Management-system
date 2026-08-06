"""Hand-audit a sample of mined distractors before the generator is trusted (C4).

Distractors are the part of the degradation model with no ground truth of its own. The
Corpus B rule mines pairs sharing a CPV class with high title similarity and distinct
record identifiers, and calls them non-matches. That is only sound if the two records
really are distinct items rather than the same procurement published twice — a repeat
notice, an amendment, or a framework call-off against a parent notice. Where it is the
same procurement, the pair is labelled 0 but is really a 1, and every recall figure
computed against it is wrong in the same direction.

No automatic check can separate those cases, which is why this prints a sample for a
human and records what they found. It writes nothing to the labels directory: its output
is a note in the run record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fcesreg.paths import annotation_path, data_path
from fcesreg.degrade import DegradationConfig, make_distractors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path,
                   default=data_path("processed", "corpus_b_contractsfinder.parquet"))
    p.add_argument("--divisions", nargs="+",
                   default=["30", "31", "32", "33", "38", "42", "43", "44"])
    p.add_argument("--sample", type=int, default=40)
    p.add_argument("--sim-threshold", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=annotation_path("samples", "distractor_audit.json"))
    args = p.parse_args(argv)

    corpus = pd.read_parquet(args.corpus)
    corpus = corpus[corpus["cpv_code"].str[:2].isin(set(args.divisions))]

    pairs = make_distractors(
        corpus, DegradationConfig(0.0), seed=args.seed, corpus="cf",
        sim_threshold=args.sim_threshold,
    )
    print(f"mined {len(pairs):,} distractor pairs from {len(corpus):,} records")
    if pairs.empty:
        return 1

    rng = np.random.default_rng(args.seed)
    take = rng.choice(len(pairs), size=min(args.sample, len(pairs)), replace=False)
    sample = pairs.iloc[take]

    by_id = corpus.set_index("record_id")
    rows = []
    for left, right in zip(sample["left_id"], sample["right_id"], strict=True):
        a, b = by_id.loc[left], by_id.loc[right]
        rows.append(
            {
                "left_id": left,
                "right_id": right,
                "left_title": a["title"],
                "right_title": b["title"],
                "same_buyer": bool(a["buyer_id"] == b["buyer_id"]),
                "same_cpv": bool(a["cpv_code"] == b["cpv_code"]),
                "left_date": str(a["release_date"]),
                "right_date": str(b["release_date"]),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    same_buyer = sum(r["same_buyer"] for r in rows)
    print(f"wrote {len(rows)} pairs to {args.out}")
    print(f"  same buyer: {same_buyer}/{len(rows)} — the population most at risk of being")
    print("  the same procurement published twice rather than two distinct items\n")

    for r in rows[:15]:
        flag = "SAME-BUYER" if r["same_buyer"] else "diff-buyer"
        print(f"  [{flag}] {r['left_date'][:10]} | {r['left_title'][:78]}")
        print(f"               {r['right_date'][:10]} | {r['right_title'][:78]}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
