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
    ngram_overlap_candidates,
)
from fcesreg.splits import load as load_splits
from fcesreg.paths import repo_root
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


def sweep_overlap(
    records: pd.DataFrame,
    dev_truth: pd.DataFrame,
    n: int,
    thresholds: list[int],
    max_block_size: int,
) -> list[dict]:
    """Pair completeness against reduction ratio, as a curve. Selected on dev only."""
    curve = []
    for t in thresholds:
        pairs, _ = ngram_overlap_candidates(
            records, n=n, min_overlap=t, max_block_size=max_block_size
        )
        e = evaluate_blocking(pairs, dev_truth, n_records=len(records))
        curve.append({"min_overlap": t} | e)
    return curve


def sweep_single_key(
    records: pd.DataFrame,
    dev_truth: pd.DataFrame,
    n: int,
    ks: list[int],
    max_block_size: int,
) -> list[dict]:
    """The formulation §6.8 originally specified, retained as a negative result.

    It degrades monotonically as k grows: a larger k demands agreement on more
    alphabetically-early n-grams, and agreement on any of them is already an exact match
    over a derived string.
    """
    curve = []
    for k in ks:
        pairs, _ = candidate_pairs(
            records,
            ["sorted_ngrams"],
            max_block_size,
            {"sorted_ngrams": {"n": n, "k": k, "mode": "single_key"}},
        )
        e = evaluate_blocking(pairs[0] if isinstance(pairs, tuple) else pairs,
                              dev_truth, n_records=len(records))
        curve.append({"k": k} | e)
    return curve


def select_operating_point(curve: list[dict], floor: float) -> dict:
    """Highest reduction ratio holding pair completeness at or above ``floor``.

    Returns the chosen row plus the completeness given up, which is unrecoverable
    downstream and is reported as such.
    """
    eligible = [c for c in curve if (c["pair_completeness"] or 0.0) >= floor]
    if not eligible:
        best = max(curve, key=lambda c: c["pair_completeness"] or 0.0)
        return {
            "selected": None,
            "floor": floor,
            "reason": "no threshold met the floor",
            "best_available": best,
        }
    chosen = max(eligible, key=lambda c: c["reduction_ratio"])
    return {
        "selected": chosen,
        "floor": floor,
        "completeness_forgone": 1.0 - chosen["pair_completeness"],
        "note": (
            "completeness lost in blocking is unrecoverable downstream; the floor is "
            "stated rather than tuned per result"
        ),
    }


def evaluate_corpus(
    name: str,
    records: pd.DataFrame,
    truth: pd.DataFrame | None,
    max_block_size: int,
    scheme_kwargs: dict[str, dict] | None = None,
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
            pairs, reports = candidate_pairs(
                records, [scheme], max_block_size, scheme_kwargs
            )
        except SchemeUnavailable as e:
            out["per_scheme"][scheme] = {"unavailable": str(e)}
            continue

        entry = _report_dict(reports[0])
        entry |= evaluate_blocking(pairs, truth, n_records=len(records))
        out["per_scheme"][scheme] = entry

    pairs, reports = candidate_pairs(records, available, max_block_size, scheme_kwargs)
    union = {
        "schemes": available,
        "blocks_dropped_total": sum(r.blocks_dropped for r in reports),
    }
    union |= evaluate_blocking(pairs, truth, n_records=len(records))
    out["union"] = union
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=repo_root() / "research/configs/blocking.yaml")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    records_a = pd.read_parquet(cfg["corpus_a"])
    truth_a = pd.read_parquet(cfg["corpus_a_pairs"])

    corpus_b = pd.read_parquet(cfg["corpus_b"])
    corpus_b = corpus_b[corpus_b["cpv_code"].str[:2].isin(set(cfg["divisions"]))]

    sn = cfg["sorted_ngrams"]
    splits = load_splits()
    dev_truth = splits.abtbuy(truth_a, "dev")

    # The operating point is chosen on the Corpus A dev partition and then applied
    # unchanged to both corpora. Nothing is selected against test.
    overlap_curve = sweep_overlap(
        records_a, dev_truth, sn["n"], sn["overlap_sweep"], cfg["max_block_size"]
    )
    single_key_curve = sweep_single_key(
        records_a, dev_truth, sn["n"], sn["single_key_k_sweep"], cfg["max_block_size"]
    )
    operating_point = select_operating_point(overlap_curve, cfg["min_pair_completeness"])
    chosen_t = (operating_point["selected"] or {}).get("min_overlap", sn["min_overlap"])

    kw = {"sorted_ngrams": {"n": sn["n"], "min_overlap": chosen_t, "mode": "per_gram"}}
    metrics = {
        "selection": {
            "selected_on": "corpus A dev partition",
            "operating_point": operating_point,
            "overlap_curve": overlap_curve,
            "single_key_curve": single_key_curve,
            "single_key_note": (
                "retained as a negative result: the formulation degrades monotonically "
                "as k grows and is not a usable blocking scheme"
            ),
        },
        "severity": cfg.get("severity"),
        "severity_note": (
            "these figures are measured on UNDEGRADED records; blocking is re-run across "
            "the severity range once the degradation model exists (C4), and the two sets "
            "must not be compared silently"
        ),
        "corpus_a_abtbuy": evaluate_corpus(
            "corpus_a_abtbuy", records_a, truth_a, cfg["max_block_size"], kw
        ),
        "corpus_b_contractsfinder": evaluate_corpus(
            "corpus_b_contractsfinder", corpus_b, None, cfg["max_block_size"], kw
        ),
    }

    out = write_run(run_id, params=cfg, metrics=metrics, env=env)
    print(f"wrote {out}")

    op = metrics["selection"]["operating_point"]
    sel = op.get("selected")
    if sel:
        print(f"\noperating point (dev, floor {op['floor']}): "
              f"t={sel['min_overlap']}  PC {sel['pair_completeness']:.3f}  "
              f"RR {sel['reduction_ratio']:.4f}  "
              f"forgone {op['completeness_forgone']:.3f}")
    for corpus, m in metrics.items():
        if not isinstance(m, dict) or "per_scheme" not in m:
            continue
        print(f"\n{corpus}  ({m['n_records']:,} records)")
        print(f"  available: {', '.join(m['schemes_available'])}")
        if m["schemes_unavailable"]:
            print(f"  unavailable: {', '.join(m['schemes_unavailable'])}")
        for scheme, e in m["per_scheme"].items():
            pc = e.get("pair_completeness")
            pc_s = f"{pc:.3f}" if pc is not None else "  n/a"
            rr = e.get("reduction_ratio")
            rr_s = f"{rr:.4f}" if rr is not None else "   n/a"
            print(
                f"    {scheme:<14} PC {pc_s}  RR {rr_s}  "
                f"cands {e['n_candidates']:>9,}  unblocked {e['unblocked_share']:.1%}  "
                f"largest {e['largest_block']:,}"
            )
        u = m["union"]
        upc = u.get("pair_completeness")
        print(f"    {'UNION':<14} PC {f'{upc:.3f}' if upc is not None else '  n/a'}  "
              f"RR {u['reduction_ratio']:.4f}  cands {u['n_candidates']:>9,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
