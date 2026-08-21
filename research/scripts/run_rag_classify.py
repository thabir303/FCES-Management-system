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
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fcesreg.classify import (
    ClassificationResult,
    EmbeddingLogRegClassifier,
    RagFewShotLLMClassifier,
    TfidfSvmClassifier,
    _parse_classification,
)
from fcesreg.costs import summarise_costs, throughput_per_day
from fcesreg.cpv import label_series, supported_labels
from fcesreg.llm import DailyQuotaExhausted, LLMClient
from fcesreg.metrics import macro_weighted_f1, wilson_interval
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


def predict_whatever_completed(
    rag: RagFewShotLLMClassifier, sample: pd.DataFrame
) -> tuple[ClassificationResult, pd.DataFrame]:
    """Same contract as ``rag.predict(sample)``, except a mid-batch ``DailyQuotaExhausted``
    is not fatal: whatever ``complete_many`` finished before the quota ran out is real,
    already-paid-for work, and reporting it at its own (smaller) sample size is honest in a
    way that either crashing outright or padding it back up to the requested n would not be.

    Mirrors ``run_dedup.py``'s handling of the same exception around the cascade -- the
    runner catches it, the library class does not, exactly the split ``adjudicate.py`` and
    ``run_dedup.py`` already establish.
    """
    try:
        return rag.predict(sample), sample
    except DailyQuotaExhausted as e:
        record_ids = sample["record_id"].tolist()
        completed_positions = []
        codes: list[str] = []
        for position, record_id in enumerate(record_ids):
            custom_id = f"{position}|{record_id}"
            response = e.completed.get(custom_id)
            if response is None:
                continue
            try:
                code, _runner_up = _parse_classification(
                    response.text, custom_id, rag._label_set
                )
            except Exception as parse_exc:
                # Already in best-effort partial recovery; a response that came back
                # malformed here is the same "not a random sample of the band" concern
                # adjudicate.py raises on -- but crashing the whole partial report over one
                # bad reply among an otherwise-real batch is the wrong failure mode now.
                # Excluded and named, not silently dropped.
                print(f"  excluding {custom_id}: {parse_exc}")
                continue
            completed_positions.append(position)
            codes.append(code)
        partial_sample = sample.iloc[completed_positions].reset_index(drop=True)
        result = ClassificationResult(
            codes=codes,
            scores=np.ones(len(codes), dtype=np.float64),
            alternatives=[[] for _ in codes],
        )
        print(
            f"  *** QUOTA EXHAUSTED MID-BATCH *** {len(codes)} of {len(sample)} requested "
            f"records actually completed before the daily allowance ran out. Reporting at "
            f"n={len(codes)}, not the requested n={len(sample)} -- do not treat this as a "
            f"failure to fix; it is what today's allowance bought."
        )
        return result, partial_sample


def score(name: str, truth: np.ndarray, predicted: np.ndarray, ordered: list[str]) -> dict:
    scored = macro_weighted_f1(truth, predicted, ordered)
    n = len(truth)
    correct = int((truth == predicted).sum())
    point, lower, upper = wilson_interval(correct, n)
    print(f"  {name:<18} macro {scored['macro_f1']:.3f}  weighted {scored['weighted_f1']:.3f}  "
          f"acc {scored['accuracy']:.3f} [{lower:.3f}, {upper:.3f}]  (n={n})")
    result = {k: v for k, v in scored.items() if k != "per_class"}
    result["accuracy_wilson_95"] = {"point": point, "lower": lower, "upper": upper}
    return result


def _subsample_check(drawn: pd.DataFrame, completed: pd.DataFrame, level: str) -> dict:
    """Whether the completed subset looks like a random draw from ``drawn`` or a biased
    prefix of it. ``predict_whatever_completed``/``predict_cached_only`` both fill
    ``completed`` in the order the sample frame lists records, which is the order
    ``stratified_sample`` leaves them in (ascending original position, not drawn order) --
    if that position correlated with something the frame is sorted by, "completed" would be
    a biased slice of "drawn" rather than a smaller random sample of it. This measures that
    directly rather than asserting it.
    """
    check: dict = {}
    drawn_labels = label_series(drawn, level).value_counts(normalize=True)
    completed_labels = label_series(completed, level).value_counts(normalize=True)
    check["label_share_max_abs_diff"] = float(
        (drawn_labels - completed_labels.reindex(drawn_labels.index).fillna(0.0)).abs().max()
    )
    if "release_date" in drawn.columns:
        positions = np.arange(len(drawn))
        dates = pd.to_datetime(drawn["release_date"]).astype("int64").to_numpy()
        check["position_vs_date_correlation"] = float(np.corrcoef(positions, dates)[0, 1])
        check["drawn_date_range"] = [
            str(pd.to_datetime(drawn["release_date"]).min().date()),
            str(pd.to_datetime(drawn["release_date"]).max().date()),
        ]
        check["completed_date_range"] = [
            str(pd.to_datetime(completed["release_date"]).min().date()),
            str(pd.to_datetime(completed["release_date"]).max().date()),
        ]
    return check


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--cache-only", action="store_true",
        help="score only requests already in the LLM cache; never calls the network or "
             "waits on pacing. For recomputing a run's statistics (e.g. after a partial "
             "run) without any further quota risk.",
    )
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
    print(f"rag_fewshot_llm: predicting {len(sample)} records, condition={cfg['condition']!r}, "
          f"cache_only={args.cache_only}...")
    if args.cache_only:
        rag_result, completed_sample = rag.predict_cached_only(sample)
        partial_reason = (
            "recomputed from cache only (--cache-only); no network call attempted, so this "
            "cannot advance n beyond whatever a prior live run already completed and cached"
        )
    else:
        rag_result, completed_sample = predict_whatever_completed(rag, sample)
        partial_reason = (
            "per-minute pacing slowed to ~1 call/25min after a token-window accounting fix "
            "made pacing more conservative on retries; recovered by reading the already-"
            "cached live completions directly rather than re-issuing requests"
        )
    partial = len(completed_sample) < len(sample)

    # The classical conditions are scored on the SAME subset the language model actually
    # answered, not the full requested sample -- a partial run must compare like for like,
    # not silently widen the classical conditions' n past what the LLM condition achieved.
    sample_labels = label_series(completed_sample, level).to_numpy()
    conditions_on_sample: dict = {}
    conditions_on_sample["rag_fewshot_llm"] = score(
        "rag_fewshot_llm", sample_labels, np.asarray(rag_result.codes), ordered
    )

    full_labels = label_series(test_ok, level).to_numpy()
    conditions_on_full: dict = {}
    for name, factory in CLASSICAL.items():
        model = factory()
        model.fit(dev_ok, level)

        predicted_sample = model.predict(completed_sample)
        conditions_on_sample[name] = score(
            f"{name} (n={len(completed_sample)})", sample_labels,
            np.asarray(predicted_sample.codes), ordered,
        )

        predicted_full = model.predict(test_ok)
        conditions_on_full[name] = score(
            f"{name} (full)", full_labels, np.asarray(predicted_full.codes), ordered
        )

    if partial:
        print(f"\n*** PARTIAL RUN: {len(completed_sample)} of {len(sample)} requested "
              f"records completed before the daily quota ran out ***")

    metrics = {
        "level": level,
        "n_supported_labels": len(labels),
        "n_dev": len(dev_ok),
        "n_test_full": len(test_ok),
        "n_test_sample_requested": cfg["llm_sample_n"],
        "n_test_sample_drawn": len(sample),
        "n_test_sample_completed": len(completed_sample),
        "partial_run": partial,
        "partial_reason": partial_reason if partial else None,
        "llm_sample_seed": cfg["llm_sample_seed"],
        "k_examples": cfg["k_examples"],
        "condition": cfg["condition"],
        "conditions_on_sample": conditions_on_sample,
        "conditions_on_full_partition": conditions_on_full,
    }
    if partial:
        metrics["completed_subset_check"] = _subsample_check(sample, completed_sample, level)

    ledger = [json.loads(line) for line in results_path("ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    summaries = summarise_costs(ledger, conditions={cfg["condition"]})
    if cfg["condition"] in summaries:
        cost = summaries[cfg["condition"]]
        throughput = throughput_per_day(
            mean_tokens=cost.mean_tokens,
            tokens_per_day=llm_cfg["limits"]["tokens_per_day"],
            requests_per_day=llm_cfg["limits"]["requests_per_day"],
        )
        metrics["measured_cost"] = {
            "n_calls": cost.n_calls,
            "mean_tokens_per_call": cost.mean_tokens,
            "calls_per_day": throughput["calls_per_day"],
            "binding_limit": throughput["binding_limit"],
            "days_to_reach_n300": 300 / throughput["calls_per_day"],
        }
    out = write_run(run_id, params=cfg, metrics=metrics, env=env)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
