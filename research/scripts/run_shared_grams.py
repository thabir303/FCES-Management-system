"""Does the severity-0.25 blocking reversal come from an absolute threshold meeting two
different amounts of gram headroom? (§6.8 follow-up, supervisor ruling 2026-08-21.)

``min_overlap`` is an ABSOLUTE count of shared grams, not a proportion. Corpus B's true
pairs at severity 0 are two degraded copies of ONE source, so before any degradation they
share every gram; Corpus A's are two independently-written descriptions of the same product
by two different retailers, sharing far fewer grams to begin with. If Corpus A's true pairs
sit close to the ``min_overlap=8`` line while Corpus B's sit far above it, degradation
pushes A below the line long before B, and the reversal follows from an absolute threshold
meeting two different amounts of headroom -- no domain story required.

Measures the distribution of shared 3-gram counts over true duplicate pairs at severity 0,
for both corpora, using the exact same population (`truth`, the test partition) the cap
sweep computed pair completeness against.

Zero quota, CPU only.

    python research/scripts/run_shared_grams.py --config research/configs/shared_grams.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fcesreg.blocking import _grams, _norm_titles
from fcesreg.paths import repo_root
from fcesreg.runs import capture_env, new_run_id, write_run

SCRIPT = "run_shared_grams"

sys.path.insert(0, str(Path(__file__).resolve().parent))


def shared_gram_counts(records: pd.DataFrame, pairs: pd.DataFrame, n: int) -> np.ndarray:
    titles = dict(zip(records["record_id"], _norm_titles(records)))
    grams = {rid: _grams(title, n) for rid, title in titles.items()}
    counts = []
    for left, right in zip(pairs["left_id"], pairs["right_id"], strict=True):
        left_grams = grams.get(left)
        right_grams = grams.get(right)
        if left_grams is None or right_grams is None:
            continue
        counts.append(len(left_grams & right_grams))
    return np.array(counts)


#: Beyond the threshold itself: the diagnostic question is not just "how many pairs are
#: already below the line" (nearly identical between corpora at severity 0) but "how much
#: mass sits close enough above the line that a little erosion pushes it under" -- that
#: shape difference is what a single cutoff at min_overlap cannot show.
_CUTOFFS = (8, 10, 12, 15, 16)


def summarise(counts: np.ndarray, min_overlap: int) -> dict:
    if len(counts) == 0:
        return {"n": 0}
    cutoffs = sorted(set(_CUTOFFS) | {min_overlap, 2 * min_overlap})
    return {
        "n": int(len(counts)),
        "median": float(np.median(counts)),
        "q1": float(np.percentile(counts, 25)),
        "q3": float(np.percentile(counts, 75)),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "share_at_or_below": {c: float((counts <= c).mean()) for c in cutoffs},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    build_cfg = {
        k: str(repo_root() / cfg[k]) if k.startswith("corpus") else cfg[k]
        for k in ("corpus_a", "corpus_a_pairs", "corpus_b", "divisions")
    }
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    from run_dedup import build_abtbuy
    from run_transfer import build_cf_positives

    severity, seed, n, min_overlap = cfg["severity"], cfg["seed"], cfg["n"], cfg["min_overlap"]

    records_a, _, truth_a = build_abtbuy(build_cfg, severity, seed)
    records_b, _, truth_b = build_cf_positives(build_cfg, severity, seed)

    positives_a = truth_a[truth_a["label"] == 1] if "label" in truth_a else truth_a
    positives_b = truth_b[truth_b["label"] == 1] if "label" in truth_b else truth_b

    counts_a = shared_gram_counts(records_a, positives_a, n)
    counts_b = shared_gram_counts(records_b, positives_b, n)

    summary_a = summarise(counts_a, min_overlap)
    summary_b = summarise(counts_b, min_overlap)

    print(f"severity {severity}, seed {seed}, n-gram {n}, min_overlap {min_overlap}\n")
    for name, summary in (("corpus_a", summary_a), ("corpus_b", summary_b)):
        print(f"{name}: n={summary['n']}  median={summary.get('median')}  "
              f"q1={summary.get('q1')}  q3={summary.get('q3')}")
        for cutoff, share in summary.get("share_at_or_below", {}).items():
            print(f"    share <= {cutoff:<3} A/B  {share:.3f}")

    metrics = {
        "severity": severity, "seed": seed, "n_gram": n, "min_overlap": min_overlap,
        "corpus_a": summary_a, "corpus_b": summary_b,
    }
    out = write_run(run_id, params=cfg, metrics=metrics, env=env)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
