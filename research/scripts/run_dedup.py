"""Duplicate detection across the degradation range, per corpus (§10, amendments 8 and 9).

Two sweeps with deliberately different measures:

* **Corpus A (Abt-Buy)** — precision, recall and F1, plus the cascade. Its labels were made
  and published by other parties, so precision means something on it. This is the corpus
  RQ3's operating point is taken from.
* **Corpus B (Contracts Finder)** — recall only, surfaced. Its negative set carries 42.0%
  measured contamination (Wilson 95% CI 29.4%–55.8%, amendment 8), which leaves recall
  untouched but corrupts precision directly: a matcher that correctly calls a contaminated
  pair a duplicate is scored as a false positive, and the penalty grows with how good the
  matcher is. Precision and F1 are still **computed and written into the run record**, so
  the decision to withhold them stays auditable, and ``make_tables`` declines to surface
  them. They are not deleted and they are not quietly reported.

**Thresholds are selected on dev at each severity independently** (protocol A), and
qualification is by the lower bound of a one-sided 95% Wilson interval rather than a point
estimate. Where nothing qualifies the bound is undefined, which for the cascade means
nothing can be auto-accepted and the band extends to every pair not confidently rejected.
That is a reported result, not a misconfiguration.

**Resumable by construction.** Every adjudication goes through the LLM cache, so re-running
after a daily quota stop replays completed work for free and continues where it left off.
A multi-day sweep is therefore just repeated invocations of this script.

    python research/scripts/run_dedup.py --config research/configs/dedup.yaml --corpus abtbuy --sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from fcesreg.adjudicate import LLMAdjudicator
from fcesreg.dedup import (
    AdjudicationBudgetExceeded,
    CascadeMatcher,
    EmbeddingMatcher,
    ExactMatcher,
    TfidfMatcher,
    select_threshold,
)
from fcesreg.degrade import DegradationConfig, degrade_frame, make_distractors
from fcesreg.llm import DailyQuotaExhausted, LLMClient
from fcesreg.metrics import prf1
from fcesreg.operating_point import reject_bound
from fcesreg.paths import repo_root, results_path
from fcesreg.runs import capture_env, new_run_id, write_run
from fcesreg.splits import load as load_splits

SCRIPT = "run_dedup"

FREE_MATCHERS = {
    "exact": ExactMatcher,
    "tfidf": TfidfMatcher,
    "embedding": EmbeddingMatcher,
}


def evaluate(matcher, pairs: pd.DataFrame, records: pd.DataFrame, threshold: float) -> dict:
    scores = matcher.score_pairs(pairs, records)
    got = prf1(pairs["label"].to_numpy(), (scores >= threshold).astype(int))
    got["threshold"] = None if np.isinf(threshold) else float(threshold)
    return got


def build_abtbuy(cfg: dict, severity: float, seed: int):
    """Degraded records plus the supplied dev/test pair splits, used exactly as given."""
    records = pd.read_parquet(cfg["corpus_a"])
    pairs = pd.read_parquet(cfg["corpus_a_pairs"])
    if severity > 0.0:
        records = degrade_frame(records, DegradationConfig(severity), seed=seed)
    splits = load_splits()
    return records, splits.abtbuy(pairs, "dev"), splits.abtbuy(pairs, "test")


def build_cf(cfg: dict, severity: float, seed: int):
    """Positives from degraded duplicate pairs; negatives from the mined distractor pool.

    The pool is used **unfiltered** (amendment 7). Judgements on it measure contamination;
    they never remove a pair.
    """
    corpus = pd.read_parquet(cfg["corpus_b"])
    corpus = corpus[corpus["cpv_code"].str[:2].isin(set(cfg["divisions"]))]
    splits = load_splits()
    config = DegradationConfig(severity)

    frames, pair_sets = [], []
    for part, ids in (("dev", splits.cf_dev), ("test", splits.cf_test)):
        block = corpus[corpus["record_id"].isin(ids)]
        degraded, positives = make_duplicate_pairs_for(block, config, seed)
        negatives = make_distractors(block, config, seed=seed, corpus="cf")
        frames.append(degraded)
        pair_sets.append(
            pd.concat([positives, negatives], ignore_index=True).assign(_part=part)
        )
    records = pd.concat(frames, ignore_index=True).drop_duplicates("record_id")
    return records, pair_sets[0], pair_sets[1]


def make_duplicate_pairs_for(block: pd.DataFrame, config: DegradationConfig, seed: int):
    from fcesreg.degrade import make_duplicate_pairs

    return make_duplicate_pairs(block, config, seed=seed)


def run_free_matchers(cfg: dict, corpus: str) -> list[dict]:
    rows = []
    for severity in cfg["severities"]:
        for seed in cfg["seeds"]:
            build = build_abtbuy if corpus == "abtbuy" else build_cf
            records, dev, test = build(cfg, severity, seed)
            for name, factory in FREE_MATCHERS.items():
                matcher = factory()
                threshold = select_threshold(
                    matcher.score_pairs(dev, records),
                    dev["label"].to_numpy(),
                    cfg["precision_target"],
                )
                got = evaluate(matcher, test, records, threshold)
                rows.append({"matcher": name, "severity": severity, "seed": seed, **got})
                print(
                    f"  {name:<10} sev {severity:<5} seed {seed}  "
                    f"R {got['recall']:.3f}  P {got['precision']:.3f}  F1 {got['f1']:.3f}"
                    + ("  (no confident threshold)" if got["threshold"] is None else "")
                )
    return rows


def stratified_subsample(pairs: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """``n`` pairs drawn stratified by label, so the positive rate is preserved exactly.

    Not the band subsampling rejected in §10.1. The *whole* band of this subsample is
    adjudicated, so the cascade's behaviour on it is exact; only the population is smaller,
    and the cost is a wider interval rather than an estimate in place of a measurement.
    """
    rng = np.random.default_rng(seed)
    take = []
    for label, block in pairs.groupby("label", sort=True):
        k = round(n * len(block) / len(pairs))
        idx = rng.choice(len(block), size=min(k, len(block)), replace=False)
        take.append(block.iloc[np.sort(idx)])
    return pd.concat(take).sort_index().reset_index(drop=True)


def split_precision(pairs: pd.DataFrame, decision: np.ndarray, scores: np.ndarray) -> dict:
    """Precision of the auto-accepted portion beside precision of the combined output.

    **The selection procedure constrains only the first.** A threshold is chosen so that
    what it accepts without adjudication meets the target; nothing constrains what the
    adjudicator then does with the band. Where the upper threshold is undefined, nothing is
    auto-accepted at all and the combined figure is entirely the adjudicator's. Reporting
    only the combined number would attribute the adjudicator's errors to a threshold that
    never saw those pairs.
    """
    labels = pairs["label"].to_numpy()
    predicted = (scores >= 0.5).astype(int)

    auto = decision == "accept"
    out = {"n_auto_accepted": int(auto.sum())}
    if auto.any():
        out["precision_auto_accepted"] = prf1(labels[auto], predicted[auto])["precision"]
    else:
        # Unmeasured is not estimated: with no auto-accepted pairs there is no such
        # precision, and 0.0 would read as a measured failure rather than an empty set.
        out["precision_auto_accepted"] = None
    return out


def run_cascade(cfg: dict, client: LLMClient) -> list[dict]:
    """Corpus A only, three severities, one repetition, every pair in the band adjudicated."""
    settings = cfg["cascade"]
    if settings["corpus"] != "abtbuy":
        raise SystemExit(
            f"cascade corpus is {settings['corpus']!r}; the paper evaluates the cascade on "
            f"the benchmark corpus only, since precision is not reportable on the other"
        )

    adjudicator = LLMAdjudicator(
        client, condition=settings["condition"], max_tokens=settings["max_tokens"]
    )
    subsamples = {s["severity"]: s for s in settings.get("subsample", [])}

    rows = []
    for severity in settings["severities"]:
        records, dev, test = build_abtbuy(cfg, severity, settings["seed"])
        drawn = subsamples.get(severity)
        if drawn:
            test = stratified_subsample(test, drawn["n"], drawn["seed"])
            print(f"  sev {severity}: subsampled to {len(test)} pairs "
                  f"({int(test['label'].sum())} positive), whole band adjudicated")
        base = FREE_MATCHERS[settings["base"]]()
        dev_scores = base.score_pairs(dev, records)
        labels = dev["label"].to_numpy()

        upper = select_threshold(dev_scores, labels, cfg["precision_target"])
        lower = reject_bound(dev_scores, labels, cfg["precision_target"])
        cascade = CascadeMatcher(
            base, lower, upper, adjudicator, settings["max_adjudications"]
        )
        print(
            f"  cascade sev {severity}: lower "
            f"{'-inf' if np.isinf(lower) else f'{lower:.4f}'}, upper "
            f"{'inf (nothing auto-accepted)' if np.isinf(upper) else f'{upper:.4f}'}"
        )
        scores = cascade.score_pairs(test, records)
        got = prf1(test["label"].to_numpy(), (scores >= 0.5).astype(int))
        got["threshold"] = None if np.isinf(upper) else float(upper)
        split = split_precision(test, cascade.last_decision, scores)
        rows.append(
            {"matcher": "cascade", "severity": severity, "seed": settings["seed"],
             "n_subsampled": len(test) if drawn else None,
             **got, **split, **cascade.stats}
        )
        auto = split["precision_auto_accepted"]
        print(
            f"    adjudicated {cascade.stats['n_adjudicated']}/{cascade.stats['n_pairs']} "
            f"({cascade.stats['band_fraction']:.1%})  R {got['recall']:.3f}  "
            f"P combined {got['precision']:.3f}  "
            f"P auto-accepted {'n/a (nothing auto-accepted)' if auto is None else f'{auto:.3f}'}"
        )
        if auto is not None and got["precision"] < cfg["precision_target"] <= auto:
            print(
                f"      NOTE: the threshold held {cfg['precision_target']} on what it "
                f"accepted, but the combined output did not. The gap is the adjudicator's."
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--corpus", choices=("abtbuy", "cf"), required=True)
    p.add_argument("--sweep", action="store_true", help="run the full severity factorial")
    p.add_argument("--cascade", action="store_true", help="run the cascade (Corpus A only)")
    args = p.parse_args(argv)

    # `llm.py` deliberately reads only os.environ — no .env mechanics inside the library.
    # Bootstrapping it is the entrypoint's job, and every runner that can reach the endpoint
    # must do this or fail on the first call with the key unset.
    load_dotenv(repo_root() / ".env")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    metrics: dict = {"corpus": args.corpus, "protocol": "A (thresholds re-fit per severity)"}
    if args.corpus == "cf":
        metrics["withheld"] = (
            "precision and f1 are computed and stored here but must not be surfaced for "
            "this corpus: its negative set carries 42.0% measured contamination "
            "(Wilson 95% CI 29.4%-55.8%, amendment 8), which corrupts precision while "
            "leaving recall untouched"
        )

    if args.sweep:
        print(f"free matchers, corpus {args.corpus}:")
        metrics["free_matchers"] = run_free_matchers(cfg, args.corpus)

    if args.cascade:
        client = LLMClient(
            model=yaml.safe_load(
                (repo_root() / "research/configs/llm.yaml").read_text()
            )["model"],
            base_url=yaml.safe_load(
                (repo_root() / "research/configs/llm.yaml").read_text()
            )["base_url"],
            ledger_path=results_path("ledger.jsonl"),
            run_id=run_id,
        )
        print("\ncascade:")
        try:
            metrics["cascade"] = run_cascade(cfg, client)
        except DailyQuotaExhausted as e:
            print(
                f"\ndaily quota exhausted after {len(e.completed)} adjudications this "
                f"invocation. Nothing is lost: every completed call is cached, so "
                f"re-running this command tomorrow replays them for free and continues.",
            )
            return 2
        except AdjudicationBudgetExceeded as e:
            print(f"\n{e}")
            return 3

    out = write_run(run_id, params=cfg, metrics=metrics, env=env)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
