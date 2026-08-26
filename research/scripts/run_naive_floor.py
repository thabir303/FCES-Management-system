"""Naive point-estimate threshold selection vs. the confidence-bound rule, on Corpus A,
across the same severity/seed sweep ``run_dedup.py``'s free-matcher run already uses
(supervisor instruction, 2026-08-26).

The paper's Duplicate Detection subsection argues that a threshold chosen because a
handful of accepted dev pairs happened to score well will not hold up on held-out data,
and that :func:`fcesreg.dedup.select_threshold`'s Wilson lower-bound rule is what prevents
it. That argument had no measurement behind it: the finding was dropped from an earlier
session for lack of one, and ``select_threshold``'s own docstring illustrates the failure
mode with an invented example (severity 0.30, "14 pairs") that traces back to commit
f7fe45f as rhetorical prose, not a run record -- no severity in any committed sweep is
0.30, and the commit's own measured summary only reports severity 0.

This measures the real thing: at each (severity, seed, matcher), sweep the SAME dev scores
two ways -- ``sweep.precision >= target`` (naive, point estimate) vs.
``select_threshold``'s existing Wilson-bound rule -- and evaluate whatever each rule
selects on the SAME held-out test split. Where the naive rule accepts a threshold the
confidence rule would refuse (too little dev evidence), the divergence in test precision
is the finding.

Zero quota, CPU only.

    python research/scripts/run_naive_floor.py --config research/configs/naive_floor.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fcesreg.dedup import ExactMatcher, TfidfMatcher, EmbeddingMatcher, select_threshold
from fcesreg.metrics import prf1, threshold_sweep, wilson_lower_bound
from fcesreg.paths import repo_root
from fcesreg.runs import capture_env, new_run_id, write_run
from run_dedup import build_abtbuy

SCRIPT = "run_naive_floor"

FREE_MATCHERS = {"exact": ExactMatcher, "tfidf": TfidfMatcher, "embedding": EmbeddingMatcher}


def naive_select(scores: np.ndarray, labels: np.ndarray, precision_target: float):
    """Lowest threshold whose POINT-ESTIMATE precision reaches the target -- no evidence
    floor. Mirrors ``select_threshold`` exactly except for the qualifying condition, so the
    two are comparable on the same sweep. Returns ``(threshold, n_accepted)``, ``(inf,
    None)`` if nothing qualifies."""
    sweep = threshold_sweep(scores, labels)
    ok = np.flatnonzero(sweep.precision >= precision_target)
    if ok.size == 0:
        return float("inf"), None
    idx = ok[-1]
    return float(sweep.threshold[idx]), int(sweep.n_selected[idx])


def confident_select_with_n(scores: np.ndarray, labels: np.ndarray, precision_target: float):
    """``select_threshold``, but also reporting the accepted-pair count backing it (the
    production function returns only the threshold)."""
    sweep = threshold_sweep(scores, labels)
    confident = np.array([
        wilson_lower_bound(int(tp), int(n))
        for tp, n in zip(sweep.tp, sweep.n_selected, strict=True)
    ])
    ok = np.flatnonzero(confident >= precision_target)
    if ok.size == 0:
        return float("inf"), None
    idx = ok[-1]
    return float(sweep.threshold[idx]), int(sweep.n_selected[idx])


def test_precision_at(matcher, test: pd.DataFrame, records: pd.DataFrame, threshold: float):
    if np.isinf(threshold):
        return None, 0
    scores = matcher.score_pairs(test, records)
    predicted = (scores >= threshold).astype(int)
    got = prf1(test["label"].to_numpy(), predicted)
    return got["precision"], int(predicted.sum())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["corpus_a"] = str(repo_root() / cfg["corpus_a"])
    cfg["corpus_a_pairs"] = str(repo_root() / cfg["corpus_a_pairs"])
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    rows = []
    for severity in cfg["severities"]:
        for seed in cfg["seeds"]:
            records, dev, test = build_abtbuy(cfg, severity, seed)
            for name, factory in FREE_MATCHERS.items():
                matcher = factory()
                dev_scores = matcher.score_pairs(dev, records)
                dev_labels = dev["label"].to_numpy()

                naive_t, naive_n_dev = naive_select(dev_scores, dev_labels, cfg["precision_target"])
                conf_t, conf_n_dev = confident_select_with_n(dev_scores, dev_labels, cfg["precision_target"])

                naive_test_p, naive_n_test = test_precision_at(matcher, test, records, naive_t)
                conf_test_p, conf_n_test = test_precision_at(matcher, test, records, conf_t)

                row = {
                    "matcher": name, "severity": severity, "seed": seed,
                    "naive_threshold": None if np.isinf(naive_t) else naive_t,
                    "naive_n_accepted_dev": naive_n_dev,
                    "naive_test_precision": naive_test_p,
                    "naive_n_accepted_test": naive_n_test,
                    "confidence_threshold": None if np.isinf(conf_t) else conf_t,
                    "confidence_n_accepted_dev": conf_n_dev,
                    "confidence_test_precision": conf_test_p,
                    "confidence_n_accepted_test": conf_n_test,
                    # A real divergence: the naive rule accepted a threshold evidenced by
                    # too little dev data to survive the Wilson bound.
                    "naive_only": (naive_n_dev is not None) and (conf_n_dev is None or naive_t != conf_t),
                }
                rows.append(row)
                naive_str = (
                    f"naive t={naive_t:.4f} n_dev={naive_n_dev} test_p={naive_test_p}"
                    if not np.isinf(naive_t) else "naive: no threshold reaches point estimate"
                )
                conf_str = (
                    f"confident t={conf_t:.4f} n_dev={conf_n_dev} test_p={conf_test_p}"
                    if not np.isinf(conf_t) else "confident: no threshold clears the Wilson bound"
                )
                print(f"  {name:<10} sev {severity:<5} seed {seed}  {naive_str}  |  {conf_str}")

    metrics = {"precision_target": cfg["precision_target"], "rows": rows}
    out = write_run(run_id, params=cfg, metrics=metrics, env=env)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
