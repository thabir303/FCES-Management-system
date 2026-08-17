"""RQ3: the automated share at a fixed precision floor, and the review volume left (§6.13, G9).

The headline result. Not "how accurate is the pipeline" but "how much of a register migrates
without a human, given that every error it makes has to be found again by hand".

**Residual effort is review volume, and total effort is a formula** (ruled 2026-08-17). This
runner takes no handling-time argument, because no handling time is measured anywhere in the
project: timing an author's own reading measures that author, and timing a model's measures
endpoint latency. Neither is curation effort. Nothing is lost by the volume framing — the
reduction is a ratio, so it is the same number whatever a handling time turns out to be.

**The two tasks are reported apart and never summed.** Duplicate detection automates a
decision about a *pair*; categorisation automates a decision about a *record*. An automated
share of pairs and an automated share of records are different quantities over different
denominators, and adding them or averaging them would produce a number with no referent.

**Duplicate detection automates at both ends.** A pair is resolved without a human when it
is confidently a duplicate *or* confidently not one; only the band between the two bounds
is reviewed. Scoring only the accepts as automated — the single-threshold model — counted
every obvious non-duplicate as outstanding human work and put the automated share at 1.7%
where it is in fact above 99%. The two-bound rule is :func:`operating_point.band_operating_point`.

**An automated share is never reported without the duplicates it cost.** Clearing the field
discards duplicates along with the noise and they never reach a reviewer. The two errors are
not symmetric: a wrong acceptance is a false merge a reader can see and undo, and the
precision floor is fitted to bound it, whereas a wrong rejection leaves a duplicate in the
register indistinguishable from a genuine second item, with nothing bounding it and nobody
looking. So every operating point carries ``n_duplicates_lost`` and ``recall_ceiling``
beside its share, in counts as well as rates, because the choice a faculty is making is how
many undetected duplicates to accept in exchange for how much review to avoid.

**Thresholds are selected on dev and reported on test.** The share a threshold promises
where it was chosen is not the share it delivers, and the gap between them is itself worth
seeing — it is reported rather than hidden.

    python research/scripts/run_operating_point.py --config research/configs/operating_point.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fcesreg.operating_point import (
    EFFORT_FORMULA,
    band_operating_point,
    residual_effort,
)
from fcesreg.paths import repo_root, results_path
from fcesreg.runs import capture_env, new_run_id, write_run

from run_dedup import FREE_MATCHERS, build_abtbuy

SCRIPT = "run_operating_point"


def latest_run(prefix: str) -> Path:
    runs = sorted(results_path("runs").glob(f"{prefix}-*"))
    if not runs:
        raise FileNotFoundError(
            f"no {prefix} run in results/runs; RQ3's categorisation half consumes it, so "
            f"run {prefix}.py first rather than reporting the duplicate half alone"
        )
    return runs[-1]


def delivered(scores: np.ndarray, labels: np.ndarray, lower: float, upper: float) -> dict:
    """What dev-selected bounds actually do on test, under the two-bound rule.

    A pair is automated at either end — auto-accepted above ``upper``, auto-rejected below
    ``lower`` — and only the band between them reaches a human. Counting only the accepts
    would score every obvious non-duplicate as un-automated work, which is not what a
    reviewer would ever be shown.
    """
    accepted = scores >= upper
    rejected = scores < lower
    band = ~(accepted | rejected)
    n_positive = int(labels.sum())
    n_lost = int(labels[rejected].sum())
    return {
        "lower": None if not np.isfinite(lower) else float(lower),
        "upper": None if not np.isfinite(upper) else float(upper),
        "automated_share": float(1.0 - band.mean()),
        "band_fraction": float(band.mean()),
        "n_positive": n_positive,
        "n_auto_accepted": int(accepted.sum()),
        "n_auto_rejected": int(rejected.sum()),
        "n_band": int(band.sum()),
        "precision_auto_accepted": (
            float(labels[accepted].mean()) if accepted.any() else None
        ),
        "purity_auto_rejected": (
            float(1.0 - labels[rejected].mean()) if rejected.any() else None
        ),
        "recall_auto_accepted": (
            float(labels[accepted].sum() / n_positive) if n_positive else None
        ),
        # The half the automated share hides: duplicates discarded without ever reaching a
        # reviewer. Nothing bounds this the way the precision floor bounds a false merge.
        "n_duplicates_lost": n_lost,
        "duplicates_lost_rate": (n_lost / n_positive) if n_positive else None,
        "recall_ceiling": (
            float((labels[accepted].sum() + labels[band].sum()) / n_positive)
            if n_positive
            else None
        ),
    }


def dedup_points(cfg: dict, dedup_cfg: dict) -> list[dict]:
    """Automated share per matcher × severity × target, on PAIRS, two-bound rule."""
    rows = []
    for severity in cfg["severities"]:
        records, dev, test = build_abtbuy(dedup_cfg, severity, cfg["seed"])
        dev_labels, test_labels = dev["label"].to_numpy(), test["label"].to_numpy()
        for name, factory in FREE_MATCHERS.items():
            matcher = factory()
            dev_scores = matcher.score_pairs(dev, records)
            test_scores = matcher.score_pairs(test, records)
            for target in cfg["targets"]:
                # Both bounds fitted on dev, then applied unchanged to test.
                fitted = band_operating_point(dev_scores, dev_labels, target)
                got = delivered(test_scores, test_labels, fitted["lower"], fitted["upper"])
                rows.append(
                    {
                        "matcher": name,
                        "severity": severity,
                        "target": target,
                        "promised_share_on_dev": fitted["automated_share"],
                        "bounds_crossed_on_dev": fitted["bounds_crossed"],
                        "n_test_pairs": int(len(test_labels)),
                        **got,
                    }
                )
                precision = got["precision_auto_accepted"]
                print(
                    f"  {name:<10} sev {severity:<5} P>={target}  "
                    f"automated {got['automated_share']:.3f}  "
                    f"band {got['n_band']:>5}  "
                    + (
                        f"accept P {precision:.3f} on {got['n_auto_accepted']:>3}"
                        if precision is not None
                        else "nothing auto-accepted    "
                    )
                    # Never printed without its other half.
                    + f"   LOST {got['n_duplicates_lost']:>3} of {got['n_positive']} "
                    f"duplicates"
                    + (
                        f" ({got['duplicates_lost_rate']:.1%}), ceiling R "
                        f"{got['recall_ceiling']:.3f}"
                        if got["recall_ceiling"] is not None
                        else ""
                    )
                )
    return rows


#: Precision floors the curve is traced over. Stops short of 1.0, which the one-sided 95%
#: Wilson floor makes unreachable at any sample size — plotting it would draw a point that
#: cannot exist rather than showing a trade-off.
CURVE_TARGETS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]


def dedup_curve(cfg: dict, dedup_cfg: dict) -> pd.DataFrame:
    """The trade-off curve F2 plots: automated share against the precision floor demanded.

    Traced over *targets*, not over raw thresholds, because under the two-bound rule an
    operating point is a pair of bounds and both move with the floor. A single-threshold
    curve would plot a quantity the pipeline does not use.

    Fitted on dev and evaluated on test at every point, so the curve is the one a faculty
    would actually get rather than the one selection saw.
    """
    rows = []
    for severity in cfg["severities"]:
        records, dev, test = build_abtbuy(dedup_cfg, severity, cfg["seed"])
        dev_labels, test_labels = dev["label"].to_numpy(), test["label"].to_numpy()
        for name, factory in FREE_MATCHERS.items():
            matcher = factory()
            dev_scores = matcher.score_pairs(dev, records)
            test_scores = matcher.score_pairs(test, records)
            for target in CURVE_TARGETS:
                fitted = band_operating_point(dev_scores, dev_labels, target)
                got = delivered(test_scores, test_labels, fitted["lower"], fitted["upper"])
                rows.append(
                    {"matcher": name, "severity": severity, "target": target, **got}
                )
    return pd.DataFrame(rows)


def classify_points(cfg: dict) -> dict:
    """Automated share for categorisation, on records.

    Read from the classification run rather than refitted, so the two runners cannot drift
    into reporting different numbers for the same thing.
    """
    run = latest_run("run_classify") if cfg["classify_run"] == "latest" else (
        results_path("runs") / cfg["classify_run"]
    )
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))

    out: dict = {"source_run": run.name, "levels": {}}
    for level, result in metrics["levels"].items():
        best = max(
            result["conditions"].items(), key=lambda kv: kv[1]["macro_f1"]
        )
        out["levels"][level] = {
            "condition": best[0],
            "accuracy": best[1]["accuracy"],
            "macro_f1": best[1]["macro_f1"],
            "n_scored": result["n_test"],
            # Already a measured automated share, and the one that matters: the share the
            # classifier declines outright, before any confidence threshold is applied.
            "routed_to_review": result["test_routed_to_review"],
            "unspecified_at_level": result["test_unspecified_at_level"],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dedup_cfg = yaml.safe_load(
        (repo_root() / cfg["dedup_config"]).read_text(encoding="utf-8")
    )
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    print("duplicate detection — automated share of PAIRS, Corpus A:")
    points = dedup_points(cfg, dedup_cfg)

    curve = dedup_curve(cfg, dedup_cfg)

    print("\ncategorisation — automated share of RECORDS, Corpus B:")
    classification = classify_points(cfg)
    for level, got in classification["levels"].items():
        print(
            f"  {level:<9} {got['condition']:<18} acc {got['accuracy']:.3f}  "
            f"{got['routed_to_review']:.1%} routed to review "
            f"({got['unspecified_at_level']:.1%} of it published above this level)"
        )

    n = cfg["illustrative_register_size"]
    print(f"\nresidual review volume, illustrated on a {n:,}-record register:")
    volumes = {}
    for row in points:
        if row["severity"] != cfg["severities"][0] or row["target"] != cfg["targets"][0]:
            continue
        got = residual_effort(n, row["automated_share"])
        # The trade, never one side of it: review avoided against duplicates merged in
        # unreviewed. Scaled to the illustrative register so both halves share a unit.
        got["duplicates_lost_per_register"] = round(
            n * row["n_duplicates_lost"] / row["n_test_pairs"]
        )
        got["duplicates_lost_rate"] = row["duplicates_lost_rate"]
        volumes[row["matcher"]] = got
        print(
            f"  {row['matcher']:<10} {got['n_automated']:>5} automated, "
            f"{got['n_review']:>5} to review  ({got['volume_reduction']:.1%} reduction)"
            f"  — and ~{got['duplicates_lost_per_register']:>3} duplicates merged in "
            f"unreviewed"
        )
    print(f"\n  {EFFORT_FORMULA}")
    print("  Handling time is not measured here; the reduction is a ratio and needs none.")

    metrics = {
        "unit_note": (
            "dedup automates a decision about a PAIR and categorisation one about a "
            "RECORD; the two shares have different denominators and are never summed"
        ),
        "dedup": points,
        "classification": classification,
        "residual_volume": volumes,
        "effort_formula": EFFORT_FORMULA,
        "handling_time": (
            "not measured; residual effort is reported as review volume (ruled 2026-08-17)"
        ),
    }
    out = write_run(run_id, params=cfg, metrics=metrics, predictions=curve, env=env)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
