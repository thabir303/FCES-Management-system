"""Isolated worker for one (corpus, cap) measurement — spawned by `run_blocking_cap.py`.

Not a runner: it takes a config file path and writes one result to a file. Excluded from
`test_scripts.py`'s "no tuning flags" guard by its leading underscore, which marks it as
internal rather than an entrypoint a user invokes directly.

**Runs in its own process, with a hard memory ceiling, on purpose.** Blocking on Corpus B at
a high or unbounded cap can retain enough of the sparse overlap product to start swapping,
and swapping is silent — CPU drops, the process looks alive, and there is no way from outside
to tell "still working" from "already lost". `RLIMIT_AS` converts that into a deterministic
`MemoryError` raised inside this process, at a size the orchestrator chose, so a failure is a
recorded result rather than an unkillable wait. The orchestrator applies the wall-clock
timeout from outside via `subprocess.run(..., timeout=...)`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

NO_CAP = 10**9


def main() -> int:
    # No in-process memory ceiling: RLIMIT_AS cannot be lowered on this platform --
    # `setrlimit` raises "current limit exceeds maximum limit" for any finite value, a
    # known macOS limitation rather than a bug here. The memory budget is instead enforced
    # from OUTSIDE by the orchestrator, which polls this process's RSS via `ps` and kills
    # it if it crosses the configured ceiling -- the same protection, applied externally.
    task = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(sys.argv[2])

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fcesreg.blocking import _grams, _norm_titles, ngram_overlap_candidates
    from run_dedup import build_abtbuy
    from run_transfer import build_cf_positives

    started = time.monotonic()
    print(f"  [worker] corpus={task['corpus']} cap={task['cap']} pid={__import__('os').getpid()}",
          flush=True)

    cfg = task["cfg"]
    if task["corpus"] == "corpus_a":
        records, _, truth = build_abtbuy(cfg, task["severity"], task["seed"])
    else:
        records, _, truth = build_cf_positives(cfg, task["severity"], task["seed"])

    cap = task["cap"]
    cap_applied = NO_CAP if cap is None else cap

    counts = [len(_grams(t, task["n"])) for t in _norm_titles(records)]
    n_short = sum(1 for c in counts if c < task["min_overlap"])
    short = {
        "n_records": len(counts),
        "n_short_titles": n_short,
        "short_title_share": n_short / len(counts) if counts else 0.0,
        "median_grams_per_title": float(pd.Series(counts).median()) if counts else None,
    }

    pairs, reports = ngram_overlap_candidates(
        records, n=task["n"], min_overlap=task["min_overlap"], max_block_size=cap_applied
    )
    report = reports[0] if isinstance(reports, list) else reports

    positives = truth[truth["label"] == 1] if "label" in truth else truth
    wanted = {
        tuple(sorted((a, b)))
        for a, b in zip(positives["left_id"], positives["right_id"], strict=True)
    }
    got = {
        tuple(sorted((a, b)))
        for a, b in zip(pairs["left_id"], pairs["right_id"], strict=True)
    }
    pc = (len(wanted & got) / len(wanted)) if wanted else None

    n = len(records)
    n_possible = n * (n - 1) // 2
    result = {
        "completed": True,
        "max_block_size": cap,
        "cap_applied": cap_applied,
        "short_titles": short,
        "pair_completeness": pc,
        "reduction_ratio": 1 - (len(pairs) / n_possible) if n_possible else None,
        "n_candidates": len(pairs),
        "blocks_dropped": report.blocks_dropped,
        "records_in_dropped_blocks": report.records_in_dropped_blocks,
        "n_blocks": report.n_blocks,
        "largest_block": report.largest_block,
        "seconds": time.monotonic() - started,
    }
    out_path.write_text(json.dumps(result))
    print(f"  [worker] done in {result['seconds']:.0f}s", flush=True)

    # os._exit rather than a normal return. On this platform the sparse overlap product
    # (scipy/numpy over a BLAS backend) leaves the process alive well past this point --
    # measured at 900s+ after printing "done" on Corpus B's uncapped pass, long enough to
    # exceed even a generous wall-clock budget and cause the orchestrator to discard a
    # result that had already been written to disk. The result is durable at this point
    # (out_path.write_text above already returned), so a normal interpreter shutdown that
    # waits on thread pools and atexit handlers buys nothing and has cost a real
    # measurement once already.
    import os

    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
