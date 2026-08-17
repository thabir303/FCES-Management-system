"""CPV classification across three conditions and two taxonomy levels (§10, C7).

Answers RQ2: does a language model given the labels you already have beat a classifier
trained on those same labels?

**The two classical conditions cost no quota and run in minutes**; the language model
condition costs roughly two days of a free-tier allowance and competes with the cascade
sweep for it. They are therefore separable here — ``--conditions classical`` finishes RQ2's
first numbers without touching the endpoint.

**Unsupported labels are routed to review, not discarded** (§6.10). Class-level evaluation
is restricted to labels meeting ``min_examples``; a record whose true label falls outside
that set is not an error and not absent, it is work the classifier declines to automate,
and its share is reported beside the score. A macro F1 over labels covering half the corpus
is not the same result as one covering nearly all of it.

    python research/scripts/run_classify.py --config research/configs/classify.yaml
    python research/scripts/run_classify.py --config research/configs/classify.yaml --per-class
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fcesreg.classify import EmbeddingLogRegClassifier, TfidfSvmClassifier
from fcesreg.cpv import label_series, supported_labels
from fcesreg.metrics import confusion, macro_weighted_f1
from fcesreg.paths import repo_root
from fcesreg.runs import capture_env, new_run_id, write_run
from fcesreg.splits import load as load_splits

SCRIPT = "run_classify"

CLASSICAL = {"tfidf_svm": TfidfSvmClassifier, "embedding_logreg": EmbeddingLogRegClassifier}


def load_partitions(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus = pd.read_parquet(cfg["corpus"])
    corpus = corpus[corpus["cpv_code"].str[:2].isin(set(cfg["divisions"]))]
    splits = load_splits()
    dev = corpus[corpus["record_id"].isin(splits.cf_dev)].reset_index(drop=True)
    test = corpus[corpus["record_id"].isin(splits.cf_test)].reset_index(drop=True)
    return dev, test


def restrict(frame: pd.DataFrame, level: str, labels: set[str]) -> pd.DataFrame:
    return frame[label_series(frame, level).isin(labels)].reset_index(drop=True)


def evaluate_level(cfg: dict, dev: pd.DataFrame, test: pd.DataFrame, level: str) -> dict:
    labels, coverage = supported_labels(dev, level, cfg["min_examples"])
    dev_ok, test_ok = restrict(dev, level, labels), restrict(test, level, labels)

    ordered = sorted(labels)
    out: dict = {
        "level": level,
        "n_supported_labels": len(labels),
        "train_coverage": coverage,
        # The share the classifier declines to automate: not an error, not absent.
        "test_routed_to_review": 1.0 - len(test_ok) / len(test) if len(test) else 0.0,
        "n_train": len(dev_ok),
        "n_test": len(test_ok),
        "conditions": {},
    }

    truth = label_series(test_ok, level).to_numpy()
    for name, factory in CLASSICAL.items():
        model = factory()
        model.fit(dev_ok, level)
        predicted = model.predict(test_ok)
        scored = macro_weighted_f1(truth, np.asarray(predicted.codes), ordered)
        out["conditions"][name] = {
            k: v for k, v in scored.items() if k != "per_class"
        }
        out["conditions"][name]["per_class"] = scored["per_class"]
        out["conditions"][name]["_predictions"] = predicted.codes
        print(
            f"  {level:<9} {name:<18} macro {scored['macro_f1']:.3f}  "
            f"weighted {scored['weighted_f1']:.3f}  acc {scored['accuracy']:.3f}  "
            f"(n={len(test_ok)}, {len(labels)} labels, "
            f"{out['test_routed_to_review']:.1%} routed to review)"
        )
    out["_truth"] = truth
    out["_test"] = test_ok
    return out


def collision_check(cfg: dict, level_result: dict) -> dict:
    """Did the classifier reproduce the label-noise collision and get scored correct for it?

    A named check, not something to notice in passing. The 40-item hand sample found
    ``44316400`` — "hardware" in the builders' ironmongery sense — carrying both audio-visual
    equipment and Navy IT equipment. A character n-gram model trained on those labels may
    route the same records there and be **scored correct**, because the published label
    agrees with it. That is label noise propagating into a measured result, which is a
    stronger claim than the noise rate merely bounding it.

    Reported either way. A clean negative says the collision did not propagate, which is
    also informative and costs a sentence.
    """
    settings = cfg["collision_check"]
    if level_result["level"] != settings["level"]:
        return {}

    code = settings["code"]
    truth = level_result["_truth"]
    test = level_result["_test"]
    out = {
        "code": code,
        "description": settings["label"],
        "n_true": int((truth == code).sum()),
        "conditions": {},
    }
    for name, result in level_result["conditions"].items():
        predicted = np.asarray(result["_predictions"])
        agreed = (predicted == code) & (truth == code)
        out["conditions"][name] = {
            "n_predicted": int((predicted == code).sum()),
            "n_scored_correct": int(agreed.sum()),
            # Titles so a reader can see whether these are ironmongery or AV/IT, which is
            # the whole question and cannot be settled by a count.
            "examples": test.loc[agreed, "title"].head(8).tolist(),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--per-class", action="store_true",
                   help="confusion matrices and the collision check")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    dev, test = load_partitions(cfg)
    print(f"dev {len(dev):,} records, test {len(test):,}\n")

    metrics: dict = {"levels": {}, "conditions_run": sorted(CLASSICAL)}
    for level in cfg["levels"]:
        metrics["levels"][level] = evaluate_level(cfg, dev, test, level)

    if args.per_class:
        for level, result in metrics["levels"].items():
            found = collision_check(cfg, result)
            if not found:
                continue
            metrics["collision_check"] = found
            print(f"\ncollision check: {found['code']} — {found['description']}")
            print(f"  published under it in test: {found['n_true']}")
            for name, got in found["conditions"].items():
                print(f"  {name:<18} predicted {got['n_predicted']:>4}, "
                      f"scored correct {got['n_scored_correct']:>4}")
                for title in got["examples"][:4]:
                    print(f"      {title[:70]}")
                if got["n_scored_correct"] == 0:
                    print("      (clean negative: the collision did not propagate)")

            ordered = sorted(
                set(result["_truth"]) | {c for r in result["conditions"].values()
                                         for c in r["_predictions"]}
            )
            for name, got in result["conditions"].items():
                matrix = confusion(result["_truth"], np.asarray(got["_predictions"]), ordered)
                got["confusion"] = matrix.to_dict()

    # Arrays and frames are working state, not results; strip before serialising.
    for result in metrics["levels"].values():
        result.pop("_truth", None)
        result.pop("_test", None)
        for got in result["conditions"].values():
            got.pop("_predictions", None)

    print(f"\nwrote {write_run(run_id, params=cfg, metrics=metrics, env=env)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
