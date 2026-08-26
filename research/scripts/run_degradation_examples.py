"""One real before/after example per degradation error class, and the probability each
class fires at the severities the paper actually sweeps (§6.6, T-degradation).

**Examples are found, not invented.** For each class, every multiplier except that class's
own is set to 0 and severity is set to 1.0, so ``DegradationConfig.rate`` (``min(1, severity
* multiplier)``) is exactly 1.0 for that one class and exactly 0.0 for the other six --
whatever changes in the output is that class's effect alone, undiluted. Records are scanned
in a fixed order (sorted by ``record_id``) and the first one the class actually changes is
kept; a class that never changes anything for the whole corpus would surface as an empty
example rather than a silently-skipped row.

**The reported p(severity) values are a separate, honest computation**: the default
multiplier (1.0, unchanged) at the three severities ``run_dedup.py``/``run_transfer.py``
actually sweep. The severity-1.0 probe above exists only to produce a legible example text;
it is not what the real experiments ran at.

Zero quota, CPU only.

    python research/scripts/run_degradation_examples.py --config research/configs/degradation_examples.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from fcesreg.degrade import ERROR_CLASSES, DegradationConfig, degrade_frame
from fcesreg.paths import repo_root
from fcesreg.runs import capture_env, new_run_id, write_run

SCRIPT = "run_degradation_examples"


def isolated_config(severity: float, only: str) -> DegradationConfig:
    multipliers = {f"p_{c}": (1.0 if c == only else 0.0) for c in ERROR_CLASSES}
    return DegradationConfig(severity, **multipliers)


def _title_desc(frame: pd.DataFrame) -> list[str]:
    """Title and description shown separately, not merged through ``text_of`` -- ``merge``
    and ``omit`` change which field holds what, which a combined string can hide entirely
    (this is exactly the blind spot flagged in the session that measured degradation
    damage: ``text_of`` concatenates regardless of which field moved)."""
    out = []
    for title, desc in zip(frame["title"], frame["description"], strict=True):
        t = "" if pd.isna(title) else str(title)
        d = "∅" if pd.isna(desc) else str(desc)
        out.append(f"title: {t} | description: {d}")
    return out


def find_example(records: pd.DataFrame, error_class: str, seed: int) -> dict:
    cfg = isolated_config(1.0, only=error_class)
    degraded = degrade_frame(records, cfg, seed=seed)
    before = _title_desc(records)
    after = _title_desc(degraded)
    ids = records["record_id"].tolist()

    for rid, b, a in zip(ids, before, after, strict=True):
        if a != b:
            return {"record_id": rid, "before": b, "after": a}
    return {"record_id": None, "before": None, "after": None}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    corpus = pd.read_parquet(repo_root() / cfg["corpus_b"])
    corpus = corpus[corpus["cpv_code"].str[:2].isin(set(cfg["divisions"]))]
    records = corpus.sort_values("record_id").reset_index(drop=True)

    severities = cfg["reported_severities"]
    seed = cfg["seed"]

    classes: dict[str, dict] = {}
    for error_class in ERROR_CLASSES:
        example = find_example(records, error_class, seed)
        default_cfg_by_severity = {s: DegradationConfig(s).rate(error_class) for s in severities}
        classes[error_class] = {
            "example": example,
            "p_by_severity": {str(s): p for s, p in default_cfg_by_severity.items()},
        }
        print(f"{error_class:<12} record {example['record_id']}")
        print(f"  before: {example['before']!r}")
        print(f"  after:  {example['after']!r}")
        print(f"  p at {severities}: "
              f"{[default_cfg_by_severity[s] for s in severities]}\n")

    metrics = {"n_records_scanned": len(records), "seed": seed, "severities": severities, "classes": classes}
    out = write_run(run_id, params=cfg, metrics=metrics, env=env)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
