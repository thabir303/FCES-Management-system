"""Shortlist recall at k — the ceiling on the retrieval-augmented condition (§10.1, C7).

The retrieval-augmented language model condition is shown a shortlist of candidate codes
retrieved **by embedding similarity**, and embeddings have just been measured as the weaker
representation on exactly this data (macro F1 0.351 against 0.508 at class level). If the
shortlist frequently omits the true code the model cannot be right, and the condition's
measured accuracy would be a measurement of the retriever wearing the model's name.

**This runner spends no quota.** Retrieval needs no language model call and the embeddings
are already cached, so the ceiling can be established before committing two days of a free
tier allowance to the condition itself. Recall at k is reported with the results either
way: it is one number and it tells a reader what the model was actually working with.

Three retrievers are compared on identical records and identical pools:

* ``embedding`` — what the paper specifies.
* ``tfidf`` — character n-grams, the representation that wins both classification levels.
* ``tfidf_svm`` — the trained classifier's own top k. **Not a candidate for the condition**
  and reported as a reference only: shortlisting with the classifier RQ2 is asking the
  model to beat would make the comparison circular, since the model could then only
  re-rank the baseline's output and never disagree with it outright.

    python research/scripts/run_shortlist.py --config research/configs/classify.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fcesreg.classify import (
    EmbeddingShortlister,
    TfidfShortlister,
    TfidfSvmClassifier,
    recall_at_k,
)
from fcesreg.cpv import label_series, supported_labels
from fcesreg.paths import repo_root
from fcesreg.runs import capture_env, new_run_id, write_run
from fcesreg.schema import text_of

from run_classify import load_partitions, restrict

SCRIPT = "run_shortlist"

RETRIEVERS = {"embedding": EmbeddingShortlister, "tfidf": TfidfShortlister}

#: Code length by level. The taxonomy carries 2-, 4- and 8-digit rows in one frame; a pool
#: mixing lengths would let a retriever "hit" at the wrong granularity.
WIDTH = {"division": 2, "class": 4}


def pools(taxonomy: pd.DataFrame, level: str, cfg: dict, supported: set[str]) -> dict:
    """The three candidate sets a shortlist could be drawn from, widest first.

    Which one the condition should use is a real choice, not a detail. The widest is what
    "rather than the full vocabulary" means literally; the narrowest is the set the
    classical conditions can predict, and shortlisting from it makes the comparison
    like-for-like at the cost of handing the language model the restriction as a gift.
    """
    at_level = taxonomy[taxonomy["cpv_code"].str.len() == WIDTH[level]]
    divisions = set(cfg["divisions"])
    return {
        "taxonomy": at_level,
        "in_division": at_level[at_level["cpv_code"].str[:2].isin(divisions)],
        "supported": at_level[at_level["cpv_code"].isin(supported)],
    }


def supervised_reference(
    dev: pd.DataFrame, test: pd.DataFrame, level: str, truth: np.ndarray, ks
) -> dict:
    """The trained classifier's own ranking, as a reference bound (not a proposal)."""
    model = TfidfSvmClassifier()
    model.fit(dev, level)
    probabilities = model.model.predict_proba(model.vectoriser.transform(text_of(test)))
    classes = np.asarray(model.model.classes_)
    position = {code: i for i, code in enumerate(classes)}
    # A truth label outside the fitted classes has no position; -1 never matches a ranking.
    return recall_at_k(
        np.argsort(-probabilities, axis=1),
        np.array([position.get(code, -1) for code in truth]),
        ks,
    )


def measure(level: str, dev: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> dict:
    taxonomy = pd.read_parquet(repo_root() / cfg["taxonomy"])
    supported, _ = supported_labels(dev, level, cfg["min_examples"])
    dev_ok = restrict(dev, level, supported)
    supported, _ = supported_labels(dev_ok, level, cfg["min_examples"])
    dev_ok, test_ok = restrict(dev_ok, level, supported), restrict(test, level, supported)
    truth = label_series(test_ok, level).to_numpy()
    texts = text_of(test_ok).tolist()
    ks = cfg["shortlist"]["ks"]

    out: dict = {"level": level, "n_test": len(test_ok), "pools": {}}
    for pool_name, pool in pools(taxonomy, level, cfg, supported).items():
        codes = pool["cpv_code"].to_numpy()
        position = {code: i for i, code in enumerate(codes)}
        true_position = np.array([position.get(code, -1) for code in truth])
        entry: dict = {"n_codes": len(codes), "retrievers": {}}
        for name, factory in RETRIEVERS.items():
            shortlister = factory()
            shortlister.fit(pool, corpus=dev_ok)
            entry["retrievers"][name] = recall_at_k(
                shortlister.rank(texts), true_position, ks
            )
        out["pools"][pool_name] = entry

    out["pools"]["supported"]["retrievers"]["tfidf_svm_reference"] = supervised_reference(
        dev_ok, test_ok, level, truth, ks
    )
    return out


def report(result: dict, ks) -> None:
    print(f"\n{result['level']}  (n={result['n_test']:,} test records)")
    for pool_name, pool in result["pools"].items():
        print(f"  pool {pool_name:<12} {pool['n_codes']:>5} codes")
        for name, got in pool["retrievers"].items():
            cells = "  ".join(f"@{k}:{got['recall'][k]:.3f}" for k in ks)
            missing = (
                "" if not got["n_not_in_pool"]
                else f"   ({got['n_not_in_pool']} true codes absent from pool)"
            )
            rank = got["mean_rank_when_found"]
            print(f"    {name:<22} {cells}   mean rank {rank:.1f}{missing}"
                  if rank else f"    {name:<22} {cells}{missing}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()
    ks = cfg["shortlist"]["ks"]

    dev, test = load_partitions(cfg)
    metrics: dict = {"levels": {}}
    for level in cfg["levels"]:
        metrics["levels"][level] = measure(level, dev, test, cfg)
        report(metrics["levels"][level], ks)

    print(f"\nwrote {write_run(run_id, params=cfg, metrics=metrics, env=env)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
