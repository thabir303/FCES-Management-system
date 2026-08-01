"""Corpus descriptive statistics for both corpora (G1).

Produces every count the paper's Corpus A and Corpus B sections need, including the ones
that were previously prose in the plan: the Abt-Buy split sizes and positive rates, the
Contracts Finder discard tally, the leaf-level sparsity that justifies not evaluating the
eight-digit level, and the division-set comparison with and without 39 and 48.

No number here is ever typed into the paper. This writes a run; ``make_tables.py`` turns
runs into ``results/tables/*.tex``; the paper ``\\input{}``s those (§12.6).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from fcesreg.cpv import LEVELS, label_series, leaf_sparsity, supported_labels
from fcesreg.normalise import normalise_key
from fcesreg.runs import new_run_id, write_run
from fcesreg.splits import load as load_splits

SCRIPT = "run_profile"


def profile_corpus_a(records: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Abt-Buy: duplicate ground truth."""
    by_split = {}
    for split, g in pairs.groupby("split"):
        by_split[split] = {
            "n_pairs": int(len(g)),
            "n_positive": int(g["label"].sum()),
            "positive_rate": float(g["label"].mean()),
        }
    return {
        "n_records": int(len(records)),
        "n_records_table_a": int((records["table"] == "A").sum()),
        "n_records_table_b": int((records["table"] == "B").sum()),
        "n_pairs": int(len(pairs)),
        "positive_rate_overall": float(pairs["label"].mean()),
        "splits": by_split,
        "split_provenance": "supplied with the benchmark; never regenerated",
    }


def profile_corpus_b(
    corpus: pd.DataFrame,
    discard: dict,
    division_sets: dict[str, list[str]],
    adopted: str,
    min_examples: int,
) -> dict:
    splits = load_splits()
    out: dict = {
        "n_records_all_candidate_divisions": int(len(corpus)),
        "discard": discard,
        "adopted_division_set": adopted,
        "division_sets": {},
        "per_year": {},
    }

    for year, g in corpus.groupby("bundle_year"):
        out["per_year"][str(int(year))] = int(len(g))

    for name, divisions in division_sets.items():
        sub = corpus[corpus["cpv_code"].str[:2].isin(set(divisions))]
        dev = splits.cf(sub, "dev")
        test = splits.cf(sub, "test")

        levels: dict = {}
        for level in LEVELS:
            labels, coverage = supported_labels(dev, level, min_examples=min_examples)
            counts = label_series(sub, level).value_counts()
            levels[level] = {
                "n_labels_observed": int(len(counts)),
                "n_labels_supported": int(len(labels)),
                "coverage_of_dev": coverage,
                "min_examples": min_examples,
            }

        out["division_sets"][name] = {
            "divisions": sorted(divisions),
            "n_records": int(len(sub)),
            "n_dev": int(len(dev)),
            "n_test": int(len(test)),
            "levels": levels,
            "leaf_sparsity": leaf_sparsity(sub),
            "division_distribution": {
                str(k): int(v)
                for k, v in label_series(sub, "division").value_counts().items()
            },
        }

    out["split"] = {
        "strategy": "temporal",
        "cutoff": splits.cutoff.isoformat(),
        "n_dev": len(splits.cf_dev),
        "n_test": len(splits.cf_test),
    }
    return out


def natural_duplicate_stats(corpus: pd.DataFrame) -> dict:
    """§4.5 — descriptive only. There is no natural-duplicate experiment."""
    work = pd.DataFrame(
        {
            "buyer_id": corpus["buyer_id"],
            "title_key": corpus["title"].map(normalise_key),
        }
    ).dropna(subset=["buyer_id"])

    by_buyer_title = work.groupby(["buyer_id", "title_key"]).size()
    repeated = by_buyer_title[by_buyer_title > 1]

    buyers_per_title = work.groupby("title_key")["buyer_id"].nunique()
    return {
        "n_buyer_title_groups_with_repeats": int(len(repeated)),
        "n_notices_in_those_groups": int(repeated.sum()),
        "n_titles_shared_across_buyers": int((buyers_per_title > 1).sum()),
        "note": "descriptive statistic only — no natural-duplicate experiment (§4.5)",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("research/configs/profile.yaml"))
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_id = new_run_id(SCRIPT, args.config)

    records_a = pd.read_parquet(cfg["corpus_a"])
    pairs_a = pd.read_parquet(cfg["corpus_a_pairs"])
    corpus_b = pd.read_parquet(cfg["corpus_b"])
    discard = json.loads(Path(cfg["discard_report"]).read_text(encoding="utf-8"))
    taxonomy = pd.read_parquet(cfg["taxonomy"])

    metrics = {
        "corpus_a_abtbuy": profile_corpus_a(records_a, pairs_a),
        "corpus_b_contractsfinder": profile_corpus_b(
            corpus_b,
            discard,
            cfg["division_sets"],
            cfg["adopted"],
            cfg["min_examples"],
        ),
        "taxonomy": {
            "n_divisions": int((taxonomy["level"] == 2).sum()),
            "n_classes": int((taxonomy["level"] == 4).sum()),
            "n_leaves": int((taxonomy["level"] == 8).sum()),
        },
    }
    if cfg.get("natural_duplicate_stats"):
        metrics["natural_duplicates"] = natural_duplicate_stats(corpus_b)

    out = write_run(run_id, params=cfg, metrics=metrics)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
