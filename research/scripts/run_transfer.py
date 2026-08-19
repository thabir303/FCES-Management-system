"""External Validation: what transfers from the benchmark corpus to the procurement one
(§10, G10, amendment 8).

The paper's central honesty problem. Corpus A carries human labels but a commercial domain;
Corpus B carries a domain close to institutional equipment but synthetic duplicate labels.
Neither is a real asset register, and **the difference between the two figures is the best
available estimate of how far the reported performance would transfer**. That difference is
reported whatever its size, and it is the first item in the threats to validity.

**Nothing is refitted on Corpus B.** Thresholds are selected on the Corpus A *development*
partition and carried across unchanged; the blocking configuration likewise. A number
recovered by refitting would answer a different question — how well the method can be made
to work on Corpus B — and the question here is what survives the move.

**Recall and pair completeness only** (amendment 8). Corpus B's negative set carries 42.0%
measured contamination, which corrupts precision directly while leaving these two untouched:
both are computed over duplicates whose membership is known by construction. Precision and
F1 are not computed here at all, so there is nothing to withhold later.

One paired comparison per severity, never two independent results.

**Corpus B is built with positives only here.** Neither reported measure involves the
negative set, and ``run_dedup.build_cf``'s mined negatives are drawn from *undegraded*
records while its positives are two independently degraded copies — so at any severity
above zero the two classes differ systematically in noise level independently of whether
they are duplicates. That affects precision, which amendment 8 withholds on this corpus
anyway; it must not be allowed to leak into a recall figure that does not need it.

    python research/scripts/run_transfer.py --config research/configs/transfer.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fcesreg.blocking import candidate_pairs, evaluate_blocking
from fcesreg.dedup import select_threshold
from fcesreg.degrade import DegradationConfig, make_duplicate_pairs
from fcesreg.paths import repo_root
from fcesreg.runs import capture_env, new_run_id, write_run
from fcesreg.splits import load as load_splits

from run_dedup import FREE_MATCHERS, build_abtbuy

SCRIPT = "run_transfer"


def build_cf_positives(cfg: dict, severity: float, seed: int):
    """Corpus B records and its **positive** pairs, at one severity.

    Deliberately not ``run_dedup.build_cf``. That builder adds a mined negative set, and
    neither measure reported here touches it: recall is computed over duplicates whose
    membership is known by construction, and pair completeness likewise. Pulling the
    negatives in would import a confound this runner does not need — see the note in the
    module docstring.
    """
    corpus = pd.read_parquet(cfg["corpus_b"])
    corpus = corpus[corpus["cpv_code"].str[:2].isin(set(cfg["divisions"]))]
    splits = load_splits()
    config = DegradationConfig(severity)

    frames, pair_sets = [], []
    for ids in (splits.cf_dev, splits.cf_test):
        block = corpus[corpus["record_id"].isin(ids)]
        degraded, positives = make_duplicate_pairs(block, config, seed=seed)
        frames.append(degraded)
        pair_sets.append(positives)
    records = pd.concat(frames, ignore_index=True).drop_duplicates("record_id")
    return records, pair_sets[0], pair_sets[1]


def recall_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float | None:
    """Recall of ``scores >= threshold``.

    ``None`` where the threshold is undefined — nothing is accepted, and a recall of 0.0
    would read as a measurement of the matcher rather than of the selection failing.
    """
    if not np.isfinite(threshold) or not labels.sum():
        return None
    return float(labels[scores >= threshold].sum() / labels.sum())


def blocking_completeness(records: pd.DataFrame, truth: pd.DataFrame, cfg: dict) -> dict:
    """Pair completeness under the Corpus A configuration, applied unchanged.

    The block-size cap is reported alongside, because completeness lost to *dropping
    oversized blocks* and completeness lost to *the key not grouping duplicates together*
    are different failures with different fixes, and the ratio between the two corpora is
    uninterpretable without knowing which one moved.
    """
    block = cfg["blocking"]
    pairs, reports = candidate_pairs(
        records,
        [block["scheme"]],
        max_block_size=block["max_block_size"],
        scheme_kwargs={
            block["scheme"]: {
                "n": block["n"], "mode": block["mode"],
                "min_overlap": block["min_overlap"],
            }
        },
    )
    got = evaluate_blocking(pairs, truth, n_records=len(records))
    for report in reports:
        got |= {
            "blocks_dropped": report.blocks_dropped,
            "records_in_dropped_blocks": report.records_in_dropped_blocks,
            "n_unblocked_records": report.n_unblocked_records,
            "largest_block": report.largest_block,
            "n_blocks": report.n_blocks,
        }
    return got


def fit_on_corpus_a(cfg: dict) -> dict[str, float]:
    """One threshold per matcher, from the Corpus A dev partition, fitted **once**.

    Fitted at severity zero and carried unchanged to every severity on both corpora. This
    is what "tuned on Corpus A and then evaluated, without further tuning, on degraded
    records from Corpus B" means, and it is not the same as re-fitting at each severity:
    re-fitting would ask how well the method can be made to work on each condition, where
    the question here is what survives being moved without adjustment.
    """
    dedup_cfg = _dedup_cfg(cfg)
    records, dev, _ = build_abtbuy(dedup_cfg, 0.0, cfg["seed"])
    labels = dev["label"].to_numpy()
    return {
        name: select_threshold(
            factory().score_pairs(dev, records), labels, cfg["precision_target"]
        )
        for name, factory in FREE_MATCHERS.items()
    }


def _dedup_cfg(cfg: dict) -> dict:
    return {
        "corpus_a": cfg["corpus_a"], "corpus_a_pairs": cfg["corpus_a_pairs"],
        "corpus_b": cfg["corpus_b"], "divisions": cfg["divisions"],
    }


def paired(cfg: dict, severity: float, thresholds: dict[str, float]) -> list[dict]:
    """One severity, both corpora, under thresholds already fixed on Corpus A dev."""
    dedup_cfg = _dedup_cfg(cfg)
    a_records, _, a_test = build_abtbuy(dedup_cfg, severity, cfg["seed"])
    b_records, _, b_test = build_cf_positives(dedup_cfg, severity, cfg["seed"])

    a_block = blocking_completeness(a_records, a_test, cfg)
    b_block = blocking_completeness(b_records, b_test, cfg)

    rows = []
    for name, factory in FREE_MATCHERS.items():
        matcher = factory()
        threshold = thresholds[name]
        a_recall = recall_at(
            matcher.score_pairs(a_test, a_records), a_test["label"].to_numpy(), threshold
        )
        b_recall = recall_at(
            matcher.score_pairs(b_test, b_records), b_test["label"].to_numpy(), threshold
        )
        rows.append(
            {
                "matcher": name,
                "severity": severity,
                "threshold_from_corpus_a_dev": (
                    None if not np.isfinite(threshold) else float(threshold)
                ),
                # Corpus B's positives are two degraded copies of one record, so at
                # severity zero they are IDENTICAL and every matcher scores 1.0 on all of
                # them. That row is an artefact of the construction, not a result, and is
                # flagged rather than quietly reported as perfect recall.
                "corpus_b_positives_are_identical": severity == 0.0,
                "recall_corpus_a": a_recall,
                "recall_corpus_b": b_recall,
                # The paired quantity, and the point of the runner. Undefined where either
                # side is, rather than substituting a zero for a measurement not made.
                "recall_transfer_gap": (
                    None if a_recall is None or b_recall is None else b_recall - a_recall
                ),
                "pair_completeness_corpus_a": a_block["pair_completeness"],
                "pair_completeness_corpus_b": b_block["pair_completeness"],
                "pair_completeness_gap": (
                    None
                    if a_block["pair_completeness"] is None
                    or b_block["pair_completeness"] is None
                    else b_block["pair_completeness"] - a_block["pair_completeness"]
                ),
                "reduction_ratio_corpus_a": a_block["reduction_ratio"],
                "reduction_ratio_corpus_b": b_block["reduction_ratio"],
                "blocks_dropped_corpus_a": a_block.get("blocks_dropped"),
                "blocks_dropped_corpus_b": b_block.get("blocks_dropped"),
                "records_in_dropped_blocks_corpus_a": a_block.get(
                    "records_in_dropped_blocks"
                ),
                "records_in_dropped_blocks_corpus_b": b_block.get(
                    "records_in_dropped_blocks"
                ),
                "largest_block_corpus_a": a_block.get("largest_block"),
                "largest_block_corpus_b": b_block.get("largest_block"),
            }
        )
    return rows


def show(row: dict) -> None:
    def fmt(x):
        return "  n/a " if x is None else f"{x:>+6.3f}" if x < 0 else f"{x:>6.3f}"

    print(
        f"  {row['matcher']:<10} sev {row['severity']:<5} "
        f"R  A {fmt(row['recall_corpus_a'])} -> B {fmt(row['recall_corpus_b'])}  "
        f"gap {fmt(row['recall_transfer_gap'])}   "
        f"PC A {fmt(row['pair_completeness_corpus_a'])} -> "
        f"B {fmt(row['pair_completeness_corpus_b'])}  "
        f"gap {fmt(row['pair_completeness_gap'])}"
        + ("" if row["threshold_from_corpus_a_dev"] is not None
           else "   (no confident threshold on A)")
        + ("   [B positives identical at sev 0 -- artefact]"
           if row["corpus_b_positives_are_identical"] else "")
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key in ("corpus_a", "corpus_a_pairs", "corpus_b"):
        cfg[key] = str(repo_root() / cfg[key])
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    thresholds = fit_on_corpus_a(cfg)
    print("thresholds fitted ONCE on Corpus A dev at severity 0, carried unchanged:")
    for name, t in thresholds.items():
        print(f"    {name:<10} {'undefined' if not np.isfinite(t) else f'{t:.4f}'}")
    print()
    rows: list[dict] = []
    for severity in cfg["severities"]:
        batch = paired(cfg, severity, thresholds)
        rows.extend(batch)
        for row in batch:
            show(row)
        first = batch[0]
        print(
            f"      blocking: A dropped {first['blocks_dropped_corpus_a']} blocks "
            f"({first['records_in_dropped_blocks_corpus_a']} records, largest "
            f"{first['largest_block_corpus_a']})   B dropped "
            f"{first['blocks_dropped_corpus_b']} blocks "
            f"({first['records_in_dropped_blocks_corpus_b']} records, largest "
            f"{first['largest_block_corpus_b']})"
        )

    metrics = {
        "withheld": (
            "precision and f1 are not computed here at all: Corpus B's negative set "
            "carries 42.0% measured contamination (amendment 8), which corrupts precision "
            "while leaving recall and pair completeness untouched"
        ),
        "refitting": "none; every parameter comes from the Corpus A dev partition",
        "paired": rows,
    }
    out = write_run(
        run_id, params=cfg, metrics=metrics, predictions=pd.DataFrame(rows), env=env
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
