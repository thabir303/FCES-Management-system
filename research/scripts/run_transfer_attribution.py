"""Consolidates the cap-sweep attribution into one run record for T9_transfer (§10, G10,
amendment 8 follow-up, supervisor rulings 2026-08-20/21).

**Not a measurement -- a merge of two already-measured, already-committed run records** into
the shape ``make_tables.table_transfer`` needs: corpus, severity, cap, pair completeness,
candidate pairs, blocks dropped, for every (corpus, cap) cell across both the severity-0.0
sweep and the severity-0.25 single cell. Nothing here recomputes anything; every number is
read back from ``run_blocking_cap`` records that are already committed and already clean.

The two source run_ids are named explicitly rather than resolved by "latest", because both
matter and only one can be "latest" at a time -- resolving by recency would silently drop
one of them the next time either script runs again for an unrelated reason.

    python research/scripts/run_transfer_attribution.py --config research/configs/transfer_attribution.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from fcesreg.runs import capture_env, load_run, new_run_id, write_run

SCRIPT = "run_transfer_attribution"


def rows_from(run: dict) -> list[dict]:
    severity = run["metrics"]["severity"]
    out = []
    for corpus, corpus_metrics in run["metrics"]["corpora"].items():
        for cap in corpus_metrics["caps"]:
            if not cap.get("completed", True):
                continue
            out.append({
                "corpus": corpus,
                "severity": severity,
                "cap": cap["max_block_size"],
                "pair_completeness": cap["pair_completeness"],
                "n_candidates": cap["n_candidates"],
                "blocks_dropped": cap["blocks_dropped"],
                "largest_block": cap["largest_block"],
            })
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    sev0_run = load_run(cfg["severity_0_sweep_run_id"])
    sev025_run = load_run(cfg["severity_025_cell_run_id"])

    rows = rows_from(sev0_run) + rows_from(sev025_run)
    for row in rows:
        print(f"  {row['corpus']:<9} sev {row['severity']:<5} cap "
              f"{'none' if row['cap'] is None else row['cap']:>7}  "
              f"PC {row['pair_completeness']:.3f}  cands {row['n_candidates']:>10,}  "
              f"dropped {row['blocks_dropped']:>4}")

    metrics = {
        "source_runs": {
            "severity_0_sweep": sev0_run["run_id"],
            "severity_025_cell": sev025_run["run_id"],
        },
        "rows": rows,
        "degradation_damage_run_id": cfg.get("degradation_damage_run_id"),
        "shared_grams_run_id": cfg.get("shared_grams_run_id"),
    }
    out = write_run(run_id, params=cfg, metrics=metrics, env=env)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
