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

**Every (corpus, cap) measurement runs in an isolated subprocess and the run record is
rewritten after each one completes.** The first version of this runner held everything in
one process and wrote a single run record at the end; an uncapped pass on Corpus B started
swapping instead of failing, and fifteen minutes of silence gave no way to tell "still
working" from "already lost" — the whole sweep was lost with it. This version cannot lose a
completed measurement to a later one's failure: each cap gets its own process, its own
`RLIMIT_AS` memory ceiling and its own wall-clock timeout applied from outside, and a cap
that exceeds its budget is recorded as a failure rather than retried or silently dropped.

Two mechanisms are separated in what is measured, because they have different fixes:

* the **cap** — the pair's gram was deleted for being too popular;
* the **short title** — the title yields fewer than ``min_overlap`` grams, so no cap and no
  key could ever have blocked it.

Zero quota, CPU only.

    python research/scripts/run_blocking_cap.py --config research/configs/blocking_cap.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

from fcesreg.paths import repo_root
from fcesreg.runs import capture_env, new_run_id, write_run

SCRIPT = "run_blocking_cap"
WORKER = Path(__file__).resolve().parent / "_blocking_cap_worker.py"


#: How often to check the worker's memory footprint. Frequent enough that a runaway
#: allocation is caught within a couple of seconds of crossing the budget, cheap enough
#: (one `ps` invocation) that it costs nothing against a run measured in minutes.
POLL_SECONDS = 2.0


def _rss_gb(pid: int) -> float | None:
    """The worker's resident set size in GB, via `ps` -- portable, no new dependency.

    `psutil` would be the obvious tool and is not a project dependency; adding one for a
    single measurement in a diagnostic runner is not worth the dependency-split rule this
    project otherwise holds (see the fcesreg rules on rapidfuzz). `ps` ships everywhere
    this project runs.
    """
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    text = out.stdout.strip()
    return int(text) / (1024**2) if text else None  # ps reports RSS in KB


def _watch(proc: subprocess.Popen, timeout_seconds: float, memory_gb: float) -> str | None:
    """Poll a running worker; kill it on a wall-clock or memory budget breach.

    Returns ``None`` on a clean exit within budget, or a short reason string otherwise.
    Enforced from OUTSIDE the worker because `RLIMIT_AS` cannot be lowered on this
    platform (see `_blocking_cap_worker.py`) -- polling and killing is the portable
    substitute for the same protection: a swap-thrashing process gets caught and stopped
    within one poll interval instead of running silently until the machine grinds to a
    halt, which is what happened the first time this ran with no protection at all.
    """
    started = time.monotonic()
    while proc.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed > timeout_seconds:
            proc.kill()
            proc.wait(timeout=10)
            return f"TIMED OUT (budget {timeout_seconds:.0f}s)"
        rss = _rss_gb(proc.pid)
        if rss is not None and rss > memory_gb:
            proc.kill()
            proc.wait(timeout=10)
            return f"KILLED at {rss:.1f}GB (budget {memory_gb}GB)"
        time.sleep(POLL_SECONDS)
    return None


def run_one_cap(corpus: str, cap: dict, cfg: dict, build_cfg: dict) -> dict:
    """Spawn the isolated worker for one (corpus, cap) pair.

    A completed run's JSON is read back from a temp file, not stdout — stdout is left free
    for the worker's own progress lines, which are printed with ``flush=True`` and inherited
    directly by this process's stdout so they appear as the worker makes them, not buffered
    until exit.
    """
    task = {
        "corpus": corpus,
        "cap": cap["value"],
        "severity": cfg["severities"][0],
        "seed": cfg["seed"],
        "n": cfg["n"],
        "min_overlap": cfg["min_overlap"],
        "memory_gb": cap["memory_gb"],
        "cfg": build_cfg,
    }
    label = "none" if cap["value"] is None else f"{cap['value']:,}"
    print(f"  cap {label:>7}  starting (budget {cap['timeout_seconds']}s, "
          f"{cap['memory_gb']}GB)...", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        task_path = Path(tmp) / "task.json"
        out_path = Path(tmp) / "result.json"
        task_path.write_text(json.dumps(task))

        started = time.monotonic()
        proc = subprocess.Popen(
            [sys.executable, "-u", str(WORKER), str(task_path), str(out_path)],
            cwd=repo_root(),
        )
        outcome = _watch(proc, cap["timeout_seconds"], cap["memory_gb"])
        elapsed = time.monotonic() - started

        if outcome is not None:
            # Belt and suspenders alongside the worker's os._exit fix: a result the worker
            # already wrote to disk before the process hung must not be thrown away just
            # because the process itself failed to terminate promptly afterward.
            if out_path.exists():
                print(f"  cap {label:>7}  {outcome} after {elapsed:.0f}s, but a result was "
                      f"already written -- using it rather than discarding it", flush=True)
            else:
                print(f"  cap {label:>7}  {outcome} after {elapsed:.0f}s", flush=True)
                return {
                    "max_block_size": cap["value"], "completed": False,
                    "failure": f"{outcome} after {elapsed:.0f}s",
                    "seconds": elapsed,
                }

        if proc.returncode != 0 or not out_path.exists():
            print(f"  cap {label:>7}  FAILED (exit {proc.returncode}) after {elapsed:.0f}s",
                  flush=True)
            return {
                "max_block_size": cap["value"], "completed": False,
                "failure": f"subprocess exited {proc.returncode} after {elapsed:.0f}s",
                "seconds": elapsed,
            }

        result = json.loads(out_path.read_text())

    short = result["short_titles"]
    print(
        f"  cap {label:>7}  PC {result['pair_completeness']:.3f}  "
        f"RR {result['reduction_ratio']:.6f}  "
        f"dropped {result['blocks_dropped']:>6} grams  "
        f"largest {result['largest_block']:>6}  "
        f"cands {result['n_candidates']:>9,}  "
        f"short-title {short['n_short_titles']:>5} ({short['short_title_share']:.1%})  "
        f"{result['seconds']:>5.0f}s",
        flush=True,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)

    sys.stdout.reconfigure(line_buffering=True)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    build_cfg = {
        k: str(repo_root() / cfg[k]) if k.startswith("corpus") else cfg[k]
        for k in ("corpus_a", "corpus_a_pairs", "corpus_b", "divisions")
    }
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    metrics: dict = {"severity": cfg["severities"][0], "corpora": {}}
    metrics["corpora"]["corpus_a"] = {"caps": []}
    metrics["corpora"]["corpus_b"] = {"caps": []}

    def persist() -> None:
        # Rewritten after EVERY cap, not once at the end -- a run record exists from the
        # first completed cap onward, and a later cap's failure cannot cost an earlier
        # cap's result. Idempotent: write_run overwrites the same run_id's files.
        write_run(run_id, params=cfg, metrics=metrics, env=env)

    for corpus in ("corpus_a", "corpus_b"):
        print(f"\n{corpus}:", flush=True)
        for cap in cfg["caps"]:
            got = run_one_cap(corpus, cap, cfg, build_cfg)
            metrics["corpora"][corpus]["caps"].append(got)
            persist()

    out = repo_root() / "results" / "runs" / run_id
    print(f"\nwrote {out} (rewritten after every cap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
