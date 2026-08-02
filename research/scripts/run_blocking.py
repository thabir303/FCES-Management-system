"""Blocking evaluation, per scheme and per corpus (G3).

Recall lost in blocking cannot be recovered downstream, so this is reported separately
from matching. Two things this script is careful about:

* **Per scheme and per corpus, never averaged.** ``buyer_id`` exists on Corpus B and not
  on Corpus A. Which schemes are available where is part of what the blocking result says,
  and an average across corpora would erase it.

* **The leading-token key is reported as measured.** It is a brand proxy applied to
  procurement titles, which mostly carry no brand. If it performs poorly on Corpus B that
  is a finding about what the corpus substitution costs, not a defect to tune away.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from fcesreg.blocking import (
    SchemeUnavailable,
    applicable_schemes,
    candidate_pairs,
    evaluate_blocking,
)
from fcesreg.runs import capture_env, new_run_id, write_run

SCRIPT = "run_blocking"


def _report_dict(r) -> dict:
    return {
        "scheme": r.scheme,
        "n_blocks": r.n_blocks,
        "n_candidates": r.n_candidates,
        "blocks_dropped": r.blocks_dropped,
        "records_in_dropped_blocks": r.records_in_dropped_blocks,
        "n_unblocked_records": r.n_unblocked_records,
        "unblocked_share": r.n_unblocked_records / r.n_records if r.n_records else None,
        "largest_block": r.largest_block,
        "mean_block_size": (
            r.n_records - r.n_unblocked_records) / r.n_blocks if r.n_blocks else None,
    }


def evaluate_corpus(
    name: str,
    records: pd.DataFrame,
    truth: pd.DataFrame | None,
    max_block_size: int,
) -> dict:
    available = applicable_schemes(records)
    unavailable = [s for s in ("sorted_ngrams", "leading_token", "buyer") if s not in available]

    out: dict = {
        "corpus": name,
        "n_records": int(len(records)),
        "schemes_available": available,
        "schemes_unavailable": unavailable,
        "unavailable_reason": {
            s: "keying column absent or wholly null on this corpus" for s in unavailable
        },
        "per_scheme": {},
        "union": {},
        "truth_available": truth is not None,
    }

    for scheme in available:
        try:
            pairs, reports = candidate_pairs(records, [scheme], max_block_size)
        except SchemeUnavailable as e:
            out["per_scheme"][scheme] = {"unavailable": str(e)}
            continue

        entry = _report_dict(reports[0])
        if truth is not None:
            entry |= evaluate_blocking(pairs, truth, n_records=len(records))
        else:
            entry["pair_completeness"] = None
            entry["pair_completeness_note"] = (
                "not measured: this corpus has no labelled duplicate pairs yet"
            )
        out["per_scheme"][scheme] = entry

    pairs, reports = candidate_pairs(records, available, max_block_size)
    union = {
        "schemes": available,
        "n_candidates": int(len(pairs)),
        "blocks_dropped_total": sum(r.blocks_dropped for r in reports),
    }
    if truth is not None:
        union |= evaluate_blocking(pairs, truth, n_records=len(records))
    else:
        union["pair_completeness"] = None
    out["union"] = union
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("research/configs/blocking.yaml"))
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    records_a = pd.read_parquet(cfg["corpus_a"])
    truth_a = pd.read_parquet(cfg["corpus_a_pairs"])

    corpus_b = pd.read_parquet(cfg["corpus_b"])
    corpus_b = corpus_b[corpus_b["cpv_code"].str[:2].isin(set(cfg["divisions"]))]

    metrics = {
        "corpus_a_abtbuy": evaluate_corpus(
            "corpus_a_abtbuy", records_a, truth_a, cfg["max_block_size"]
        ),
        "corpus_b_contractsfinder": evaluate_corpus(
            "corpus_b_contractsfinder", corpus_b, None, cfg["max_block_size"]
        ),
    }

    out = write_run(run_id, params=cfg, metrics=metrics, env=env)
    print(f"wrote {out}")

    for corpus, m in metrics.items():
        print(f"\n{corpus}  ({m['n_records']:,} records)")
        print(f"  available: {', '.join(m['schemes_available'])}")
        if m["schemes_unavailable"]:
            print(f"  unavailable: {', '.join(m['schemes_unavailable'])}")
        for scheme, e in m["per_scheme"].items():
            pc = e.get("pair_completeness")
            pc_s = f"{pc:.3f}" if pc is not None else "  n/a"
            print(
                f"    {scheme:<14} PC {pc_s}  RR {e.get('reduction_ratio', float('nan')):.4f}  "
                f"cands {e['n_candidates']:>9,}  unblocked {e['unblocked_share']:.1%}  "
                f"largest {e['largest_block']:,}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
