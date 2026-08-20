"""Does severity 0.25 damage the two corpora equally? (§6.6/§6.8 follow-up.)

The severity-0.25 cap-sweep cell measured Corpus B's pair completeness (0.441) *higher*
than Corpus A's (0.248) at the rescaled cap -- reversed from severity 0.0. That is only a
finding about blocking if the same severity number delivers the same amount of textual
damage to both corpora. There is no reason to assume it does: the abbreviation class draws
on a domain lexicon built for equipment/product vocabulary that may hit Abt-Buy harder;
~20% of Abt-Buy records carry a null description (the corpus that produced the `"nan"`
bug), so the two corpora enter and leave the merge/omit classes differently; and
``min_overlap`` in blocking is an absolute gram count, so any resulting length difference
converts directly into a completeness difference with nothing to do with blocking at all.

**Same instrument that settled the `build_cf` noise-parity fix**: mean normalised edit
distance from source, ``1 - difflib.SequenceMatcher(None, source, degraded).ratio()``, over
``schema.text_of`` (title + description). See
``test_degrade.py::TestBuildCfNoiseParity.distance_from_source`` -- duplicated here
verbatim rather than imported from a test module, because a runner does not import tests.

**Per-class breakdown uses `DegradationConfig`'s own isolation mechanism** (its docstring:
"a single class can be isolated for the degradation check without rebuilding the model") --
one multiplier at 1.0, the rest at 0.0, same severity and seed as the total-damage run, so
each class's contribution is measured under the exact conditions the total run used.

Zero quota, CPU only.

    python research/scripts/run_degradation_damage.py --config research/configs/degradation_damage.yaml
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fcesreg.degrade import ERROR_CLASSES, DegradationConfig, degrade_frame
from fcesreg.paths import repo_root
from fcesreg.runs import capture_env, new_run_id, write_run
from fcesreg.schema import text_of
from fcesreg.splits import load as load_splits

SCRIPT = "run_degradation_damage"


def distance_from_source(degraded: pd.DataFrame, source_text: dict[str, str]) -> np.ndarray:
    """Mean normalised edit distance from source, per record. Identical definition to
    ``test_degrade.py::TestBuildCfNoiseParity.distance_from_source``."""
    out = []
    for rid, got in zip(degraded["record_id"], text_of(degraded), strict=True):
        src = source_text.get(rid.split("::")[0])
        if src is None:
            continue
        out.append(1.0 - SequenceMatcher(None, src, got).ratio())
    return np.array(out)


def isolated_config(severity: float, only: str | None) -> DegradationConfig:
    """``only=None`` is every class at full strength (the condition the cap sweep used).
    Otherwise every multiplier is 0 except ``only``, which is 1 -- isolating that class's
    damage at the same severity."""
    if only is None:
        return DegradationConfig(severity)
    multipliers = {f"p_{c}": (1.0 if c == only else 0.0) for c in ERROR_CLASSES}
    return DegradationConfig(severity, **multipliers)


def corpus_a_population(cfg: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    records = pd.read_parquet(cfg["corpus_a"])
    source_text = dict(zip(records["record_id"], text_of(records)))
    return records, source_text


def corpus_b_population(cfg: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    """Same population `build_cf_positives` blocks over: both split partitions, filtered
    to the evaluated divisions."""
    splits = load_splits()
    corpus = pd.read_parquet(cfg["corpus_b"])
    corpus = corpus[corpus["cpv_code"].str[:2].isin(set(cfg["divisions"]))]
    ids = splits.cf_dev | splits.cf_test
    records = corpus[corpus["record_id"].isin(ids)]
    source_text = dict(zip(records["record_id"], text_of(records)))
    return records, source_text


def measure(corpus: str, records: pd.DataFrame, source_text: dict, severity: float, seed: int) -> dict:
    result: dict = {"corpus": corpus, "n_records": len(records)}

    total_cfg = isolated_config(severity, only=None)
    degraded = degrade_frame(records, total_cfg, seed=seed)
    dist = distance_from_source(degraded, source_text)
    result["total"] = {"mean": float(dist.mean()), "n": int(len(dist))}

    result["by_class"] = {}
    for error_class in ERROR_CLASSES:
        cls_cfg = isolated_config(severity, only=error_class)
        degraded = degrade_frame(records, cls_cfg, seed=seed)
        dist = distance_from_source(degraded, source_text)
        result["by_class"][error_class] = {"mean": float(dist.mean()), "n": int(len(dist))}

    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = {
        k: str(repo_root() / v) if k.startswith("corpus") else v for k, v in cfg.items()
    }
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    severity, seed = cfg["severity"], cfg["seed"]

    records_a, source_a = corpus_a_population(cfg)
    records_b, source_b = corpus_b_population(cfg)

    result_a = measure("corpus_a", records_a, source_a, severity, seed)
    result_b = measure("corpus_b", records_b, source_b, severity, seed)

    print(f"severity {severity}, seed {seed}\n")
    for result in (result_a, result_b):
        print(f"{result['corpus']}: n={result['n_records']}  "
              f"total mean distance {result['total']['mean']:.4f}")
        for cls, d in result["by_class"].items():
            print(f"    {cls:<12} {d['mean']:.4f}")
        print()

    gap = result_b["total"]["mean"] - result_a["total"]["mean"]
    print(f"total mean distance: A {result_a['total']['mean']:.4f}  "
          f"B {result_b['total']['mean']:.4f}  gap(B-A) {gap:+.4f}")

    metrics = {
        "severity": severity,
        "seed": seed,
        "corpus_a": result_a,
        "corpus_b": result_b,
        "total_gap_b_minus_a": gap,
    }
    out = write_run(run_id, params=cfg, metrics=metrics, env=env)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
