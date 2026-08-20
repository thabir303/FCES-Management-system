"""What the cascade costs: ms/record, tokens/1000, notional USD/1000 at the ledger's own
rate card, rate-limit-bound throughput, and the measured cascade band fraction (§11,
`T8_cost`).

**The code does not wait on quota.** ``fcesreg.costs.summarise_costs`` reads whatever is in
``results/ledger.jsonl`` right now, deduplicated by ``prompt_sha256`` so a resumed sweep's
replayed rows cannot inflate the figure -- see the module docstring for why ``usd`` and a
naive row count are both wrong. Only the run that reaches the paper waits, until the
cascade's run_dedup record is a finished 1,209/1,209 rather than a partial day's -- set
``run_dedup_run_id`` in the config once it is, or leave it ``null`` to read whatever the
latest clean run_dedup record currently holds.

**Free matchers are not this runner's concern.** They cost $0.00 and their wall-clock time
is already in their own run_blocking/run_classify records under ``seconds``; T8_cost reports
the one method whose cost is not already zero by construction.

Zero *additional* quota, CPU only -- it reads the ledger, it does not call the endpoint.

    python research/scripts/run_costs.py --config research/configs/costs.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from fcesreg.costs import summarise_costs, throughput_per_day
from fcesreg.paths import repo_root, results_path
from fcesreg.runs import capture_env, new_run_id, write_run
from make_tables import latest_run as _latest_run_id  # same-directory script import

SCRIPT = "run_costs"


def load_ledger(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cascade_run_dedup_record(run_id: str | None) -> dict:
    """The run_dedup --cascade record to read band_fraction / n_pairs from.

    ``load_run`` raises DirtyRun-shaped concerns are make_tables.py's job, not this one's:
    a cost figure derived from a dirty cascade run is still refused at table-build time by
    the same guard every other table goes through, so it is not duplicated here.
    """
    from fcesreg.runs import load_run

    resolved = run_id or _latest_run_id("run_dedup")
    return load_run(resolved)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    ledger = load_ledger(results_path("ledger.jsonl"))
    summaries = summarise_costs(ledger, conditions={cfg["condition"]})
    if cfg["condition"] not in summaries:
        raise SystemExit(
            f"no ledger rows for condition {cfg['condition']!r} -- nothing to summarise"
        )
    cost = summaries[cfg["condition"]]

    llm_cfg = yaml.safe_load((repo_root() / cfg["llm_config"]).read_text(encoding="utf-8"))
    limits = llm_cfg["limits"]
    throughput = throughput_per_day(
        mean_tokens=cost.mean_tokens,
        tokens_per_day=limits["tokens_per_day"],
        requests_per_day=limits["requests_per_day"],
    )

    dedup_run = cascade_run_dedup_record(cfg.get("run_dedup_run_id"))
    cascade_rows = dedup_run["metrics"]["cascade"]

    band = []
    for row in cascade_rows:
        n_pairs = row["n_pairs"]
        per_1000 = cost.per_thousand(n_pairs)
        band.append({
            "severity": row["severity"],
            "n_pairs": n_pairs,
            "n_adjudicated": row["n_adjudicated"],
            "band_fraction": row["band_fraction"],
            **per_1000,
        })

    print(f"cascade cost, from {dedup_run['run_id']} (n_pairs source) and the live ledger:")
    print(f"  n_calls (deduplicated)  {cost.n_calls}")
    print(f"  n_replayed_rows         {cost.n_replayed_rows}")
    print(f"  mean_latency_ms (live)  "
          f"{'n/a (no live rows)' if cost.mean_latency_ms is None else f'{cost.mean_latency_ms:.0f}'}")
    print(f"  mean_tokens/call        {cost.mean_tokens:.1f}")
    print(f"  usd (one clean run)     {cost.usd:.4f}")
    print(f"  throughput/day          {throughput['calls_per_day']:.0f} "
          f"(bound by {throughput['binding_limit']})")
    print()
    for row in band:
        print(f"  sev {row['severity']:<5} band {row['band_fraction']:.1%}  "
              f"({row['n_adjudicated']}/{row['n_pairs']})  "
              f"usd/1000 {row['usd_per_1000']:.4f}  tokens/1000 {row['tokens_per_1000']:.0f}")

    metrics = {
        "condition": cfg["condition"],
        "n_calls": cost.n_calls,
        "n_replayed_rows": cost.n_replayed_rows,
        "mean_latency_ms": cost.mean_latency_ms,
        "mean_tokens_per_call": cost.mean_tokens,
        "usd_one_clean_run": cost.usd,
        "throughput_per_day": throughput,
        "dedup_run_id": dedup_run["run_id"],
        "band": band,
    }
    out = write_run(run_id, params=cfg, metrics=metrics, env=env)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
