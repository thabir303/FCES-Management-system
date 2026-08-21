"""RQ2's language-model condition (§6.10, amendment 13, C7): division level, whole label
set, few-shot examples retrieved by embedding similarity.

**Class level is not run.** One day's allowance buys roughly 300 division-level calls or
roughly 135 class-level calls (74 codes vs 8 in the prompt, more tokens per call), and 135
is too few to decide anything -- this is a ruling, not an omission, quantified from the
measured cost this runner reports.

**Same population construction as the classical conditions**, imported from
``run_classify.py`` rather than reimplemented, so the three conditions are compared on
exactly the same supported label set and the same train/test split. The language-model
condition runs on a stratified sample of the test partition (stratified by division, so
every code's share of the sample matches its share of the full partition); the classical
pair are scored on that same sample *and* on the full partition, so the sample-size cost is
visible rather than hidden inside a number that looks like the full-partition figure.

    python research/scripts/run_rag_classify.py --config research/configs/rag_classify.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fcesreg.classify import EmbeddingLogRegClassifier, RagFewShotLLMClassifier, TfidfSvmClassifier
from fcesreg.cpv import label_series, supported_labels
from fcesreg.llm import LLMClient
from fcesreg.metrics import macro_weighted_f1
from fcesreg.paths import repo_root, results_path
from fcesreg.runs import capture_env, new_run_id, write_run
from run_classify import load_partitions, restrict

SCRIPT = "run_rag_classify"

CLASSICAL = {"tfidf_svm": TfidfSvmClassifier, "embedding_logreg": EmbeddingLogRegClassifier}


def stratified_sample(frame: pd.DataFrame, label_col: pd.Series, n: int, seed: int) -> pd.DataFrame:
    """``n`` records drawn stratified by ``label_col``, same construction as
    ``run_dedup.stratified_subsample`` generalised beyond a binary label."""
    rng = np.random.default_rng(seed)
    take = []
    grouped = pd.Series(range(len(frame))).groupby(label_col.to_numpy(), sort=True)
    for _, block in grouped:
        k = round(n * len(block) / len(frame))
        idx = rng.choice(block.to_numpy(), size=min(k, len(block)), replace=False)
        take.append(idx)
    positions = np.sort(np.concatenate(take)) if take else np.array([], dtype=int)
    return frame.iloc[positions].reset_index(drop=True)


def score(name: str, truth: np.ndarray, predicted: np.ndarray, ordered: list[str]) -> dict:
    scored = macro_weighted_f1(truth, predicted, ordered)
    print(f"  {name:<18} macro {scored['macro_f1']:.3f}  weighted {scored['weighted_f1']:.3f}  "
          f"acc {scored['accuracy']:.3f}  (n={len(truth)})")
    return {k: v for k, v in scored.items() if k != "per_class"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    level = cfg["level"]
    if level != "division":
        raise SystemExit(
            f"level is {level!r}; the class-level condition is a deliberate ruling not to "
            f"run this session (see the module docstring), not something this runner refuses "
            f"by accident -- do not pass --config around that ruling"
        )

    dev, test = load_partitions(cfg)
    labels, _ = supported_labels(dev, level, cfg["min_examples"])
    dev_ok = restrict(dev, level, labels)
    labels, _ = supported_labels(dev_ok, level, cfg["min_examples"])
    dev_ok = restrict(dev_ok, level, labels)
    test_ok = restrict(test, level, labels)
    ordered = sorted(labels)

    test_labels = label_series(test_ok, level)
    sample = stratified_sample(test_ok, test_labels, cfg["llm_sample_n"], cfg["llm_sample_seed"])
    print(f"dev {len(dev_ok):,} (of {len(dev):,}), test {len(test_ok):,} (of {len(test):,}), "
          f"llm sample {len(sample):,} of {cfg['llm_sample_n']} requested, "
          f"{len(ordered)} supported {level} codes\n")

    taxonomy = pd.read_parquet(repo_root() / cfg["taxonomy"])

    # llm.py deliberately reads only os.environ -- bootstrapping it is the entrypoint's job.
    load_dotenv(repo_root() / ".env")

    llm_cfg = yaml.safe_load((repo_root() / cfg["llm_config"]).read_text(encoding="utf-8"))
    client = LLMClient(
        model=llm_cfg["model"],
        base_url=llm_cfg["base_url"],
        ledger_path=results_path("ledger.jsonl"),
        run_id=run_id,
    )

    rag = RagFewShotLLMClassifier(
        client, taxonomy, condition=cfg["condition"],
        k_examples=cfg["k_examples"], max_tokens=cfg["max_tokens"],
    )
    rag.fit(dev_ok, level)
    print(f"rag_fewshot_llm: predicting {len(sample)} records, condition={cfg['condition']!r}...")
    rag_result = rag.predict(sample)

    sample_labels = label_series(sample, level).to_numpy()
    conditions_on_sample: dict = {}
    conditions_on_sample["rag_fewshot_llm"] = score(
        "rag_fewshot_llm", sample_labels, np.asarray(rag_result.codes), ordered
    )

    full_labels = label_series(test_ok, level).to_numpy()
    conditions_on_full: dict = {}
    for name, factory in CLASSICAL.items():
        model = factory()
        model.fit(dev_ok, level)

        predicted_sample = model.predict(sample)
        conditions_on_sample[name] = score(
            f"{name} (n={len(sample)})", sample_labels, np.asarray(predicted_sample.codes), ordered
        )

        predicted_full = model.predict(test_ok)
        conditions_on_full[name] = score(
            f"{name} (full)", full_labels, np.asarray(predicted_full.codes), ordered
        )

    metrics = {
        "level": level,
        "n_supported_labels": len(labels),
        "n_dev": len(dev_ok),
        "n_test_full": len(test_ok),
        "n_test_sample_requested": cfg["llm_sample_n"],
        "n_test_sample_actual": len(sample),
        "llm_sample_seed": cfg["llm_sample_seed"],
        "k_examples": cfg["k_examples"],
        "condition": cfg["condition"],
        "conditions_on_sample": conditions_on_sample,
        "conditions_on_full_partition": conditions_on_full,
    }
    out = write_run(run_id, params=cfg, metrics=metrics, env=env)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
