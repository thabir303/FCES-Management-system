"""What the block-size cap costs, per corpus (§6.8; attribution for the transfer figure).

The transfer comparison reports pair completeness 0.985 on Corpus A against 0.411 on
Corpus B under one blocking configuration carried across unchanged. **That gap cannot be
reported as a domain-shift result until it is attributed.** Two guard parameters interact:
grams whose posting list exceeds ``max_block_size`` are removed *before* counting, and
``min_overlap`` is then applied to what survives — so on a large, repetitive corpus the cap
deletes precisely the high-frequency grams two similar titles rely on to reach the overlap
threshold. ``largest_block`` sits exactly on the cap for Corpus B at every severity, which
is what a binding guard looks like; on Corpus A the cap is nearly inert.

**Severity 0.0 only.** Corpus B's positives are byte-identical there, and two identical
strings share every gram, so a pair that fails to block fails for a mechanical reason and
not because degradation destroyed the text they had in common.

Two mechanisms are separated, because they have different fixes:

* the **cap** — the pair's gram was deleted for being too popular;
* the **short title** — the title yields fewer than ``min_overlap`` grams, so no cap and no
  key could ever have blocked it.

The second is counted directly and is independent of the cap, which is what makes it the
control on the first.

Zero quota, CPU only.

    python research/scripts/run_blocking_cap.py --config research/configs/blocking_cap.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from fcesreg.blocking import _grams, _norm_titles, ngram_overlap_candidates
from fcesreg.paths import repo_root
from fcesreg.runs import capture_env, new_run_id, write_run

from run_dedup import build_abtbuy
from run_transfer import build_cf_positives

SCRIPT = "run_blocking_cap"

#: Stands in for "no cap". Larger than any posting list either corpus can produce, so the
#: filter is a no-op rather than a special case threaded through the library.
NO_CAP = 10**9


def short_titles(records: pd.DataFrame, n: int, min_overlap: int) -> dict:
    """Records whose title yields fewer than ``min_overlap`` distinct n-grams.

    These can never reach the overlap threshold against anything, at any cap. Counting them
    separates a guard that is set wrong from a corpus that cannot be blocked this way at
    all, and it costs one pass over the titles.
    """
    counts = [len(_grams(t, n)) for t in _norm_titles(records)]
    n_short = sum(1 for c in counts if c < min_overlap)
    return {
        "n_records": len(counts),
        "n_short_titles": n_short,
        "short_title_share": n_short / len(counts) if counts else 0.0,
        "median_grams_per_title": float(pd.Series(counts).median()) if counts else None,
    }


def completeness(pairs: pd.DataFrame, truth: pd.DataFrame) -> float | None:
    positives = truth[truth["label"] == 1] if "label" in truth else truth
    wanted = {
        tuple(sorted((a, b)))
        for a, b in zip(positives["left_id"], positives["right_id"], strict=True)
    }
    if not wanted:
        return None
    got = {
        tuple(sorted((a, b)))
        for a, b in zip(pairs["left_id"], pairs["right_id"], strict=True)
    }
    return len(wanted & got) / len(wanted)


def at_cap(records: pd.DataFrame, truth: pd.DataFrame, cap: int | None, cfg: dict) -> dict:
    """One corpus at one cap. Records where it breaks rather than fighting it."""
    started = time.monotonic()
    row: dict = {"max_block_size": cap, "cap_applied": NO_CAP if cap is None else cap}
    try:
        pairs, reports = ngram_overlap_candidates(
            records,
            n=cfg["n"],
            min_overlap=cfg["min_overlap"],
            max_block_size=NO_CAP if cap is None else cap,
        )
    except (MemoryError, ValueError) as exc:
        # A reduction ratio that collapses is a result; an uncapped run that will not fit in
        # memory is also a result, and both are reported rather than quietly skipped.
        return row | {
            "completed": False,
            "failure": f"{type(exc).__name__}: {exc}"[:200],
            "seconds": time.monotonic() - started,
        }

    report = reports[0] if isinstance(reports, list) else reports
    n = len(records)
    n_possible = n * (n - 1) // 2
    return row | {
        "completed": True,
        "pair_completeness": completeness(pairs, truth),
        "reduction_ratio": 1 - (len(pairs) / n_possible) if n_possible else None,
        "n_candidates": len(pairs),
        "blocks_dropped": report.blocks_dropped,
        "records_in_dropped_blocks": report.records_in_dropped_blocks,
        "n_blocks": report.n_blocks,
        "largest_block": report.largest_block,
        "seconds": time.monotonic() - started,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    build_cfg = {
        k: str(repo_root() / cfg[k]) if k.startswith("corpus") else cfg[k]
        for k in ("corpus_a", "corpus_a_pairs", "corpus_b", "divisions")
    }
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    metrics: dict = {"severity": cfg["severities"][0], "corpora": {}}
    severity = cfg["severities"][0]

    a_records, _, a_truth = build_abtbuy(build_cfg, severity, cfg["seed"])
    b_records, _, b_truth = build_cf_positives(build_cfg, severity, cfg["seed"])

    for name, records, truth in (
        ("corpus_a", a_records, a_truth), ("corpus_b", b_records, b_truth)
    ):
        short = short_titles(records, cfg["n"], cfg["min_overlap"])
        print(
            f"\n{name}: {short['n_records']:,} records, "
            f"{short['n_short_titles']:,} with fewer than {cfg['min_overlap']} grams "
            f"({short['short_title_share']:.1%}), median {short['median_grams_per_title']:.0f} "
            f"grams/title"
        )
        rows = []
        for cap in cfg["max_block_sizes"]:
            got = at_cap(records, truth, cap, cfg)
            rows.append(got)
            label = "none" if cap is None else f"{cap:,}"
            if not got["completed"]:
                print(f"  cap {label:>7}  FAILED after {got['seconds']:.0f}s: {got['failure']}")
                continue
            print(
                f"  cap {label:>7}  PC {got['pair_completeness']:.3f}  "
                f"RR {got['reduction_ratio']:.6f}  "
                f"dropped {got['blocks_dropped']:>6} grams  "
                f"largest {got['largest_block']:>6}  "
                f"cands {got['n_candidates']:>9,}  {got['seconds']:>5.0f}s"
            )
        metrics["corpora"][name] = {"short_titles": short, "caps": rows}

    print(f"\nwrote {write_run(run_id, params=cfg, metrics=metrics, env=env)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
