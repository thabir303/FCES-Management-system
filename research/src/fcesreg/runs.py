"""Run identity and environment capture (§6.14).

This is the machinery that stops an untracked local edit silently changing a published
number. Every experiment writes ``results/runs/<run_id>/`` and every generated table
carries the ``run_id`` it came from.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

__all__ = [
    "RESULTS_ROOT",
    "GitInfo",
    "git_info",
    "config_digest",
    "new_run_id",
    "capture_env",
    "write_run",
    "load_run",
]

RESULTS_ROOT = Path("results/runs")

_NO_GIT = "nogit"

#: Packages whose versions change a result. Recorded with every run.
_TRACKED_PACKAGES = (
    "fcesreg",
    "numpy",
    "pandas",
    "scikit-learn",
    "sentence-transformers",
    "torch",
    "anthropic",
    "pyarrow",
)


@dataclass(frozen=True)
class GitInfo:
    sha: str
    short_sha: str
    dirty: bool
    branch: str | None


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_info() -> GitInfo:
    """Current git state.

    A repository with no commits yet reports ``sha == "nogit"`` and ``dirty == True``: an
    uncommitted tree cannot be reconstructed, which is exactly the condition ``dirty`` is
    meant to signal.
    """
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return GitInfo(sha=_NO_GIT, short_sha=_NO_GIT, dirty=True, branch=None)

    status = _git("status", "--porcelain")
    return GitInfo(
        sha=sha,
        short_sha=sha[:7],
        dirty=bool(status),
        branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
    )


def config_digest(config_path: Path | str) -> str:
    """First 8 hex characters of the SHA-256 of the config file's bytes."""
    data = Path(config_path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:8]


def new_run_id(script: str, config_path: Path | str) -> str:
    """``<script>-<UTC yyyymmddTHHMMSS>-<git short sha>-<sha256(config)[:8]>``.

    The timestamp makes every invocation distinct; the config digest makes two runs of the
    same script comparable at a glance. Identical config bytes always produce an identical
    final component, whatever else changes.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{script}-{stamp}-{git_info().short_sha}-{config_digest(config_path)}"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def capture_env(
    model_ids: dict[str, str] | None = None, seeds: dict[str, int] | None = None
) -> dict[str, Any]:
    """Everything needed to reconstruct the conditions a number was produced under."""
    gi = git_info()
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": gi.sha,
        "git_short_sha": gi.short_sha,
        "git_dirty": gi.dirty,
        "git_branch": gi.branch,
        "python_version": sys.version.split()[0],
        "packages": _package_versions(),
        "model_ids": model_ids or {},
        "seeds": seeds or {},
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.machine(),
    }


def write_run(
    run_id: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    predictions: pd.DataFrame | None = None,
    root: Path = RESULTS_ROOT,
    model_ids: dict[str, str] | None = None,
    seeds: dict[str, int] | None = None,
    env: dict[str, Any] | None = None,
) -> Path:
    """Write ``<root>/<run_id>/{params.yaml,metrics.json,predictions.parquet,env.json}``.

    No ledger file: LLM spend goes to the single global ``results/ledger.jsonl``, keyed by
    ``run_id`` (§6.11). Cost aggregates across runs and cache hits span them, so a per-run
    ledger would make both uninterpretable.

    The environment is captured **before** anything is written. Capturing it afterwards
    would see the run's own freshly created output directory as an uncommitted change and
    mark every run dirty, which would make ``make_tables`` refuse work that was in fact
    reproducible. ``env`` may be supplied by the caller to record the state at the moment
    the computation started rather than the moment its results were serialised.
    """
    if env is None:
        env = capture_env(model_ids, seeds)

    out = Path(root) / run_id
    out.mkdir(parents=True, exist_ok=True)

    (out / "params.yaml").write_text(
        yaml.safe_dump(params, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    (out / "env.json").write_text(
        json.dumps(env, indent=2, sort_keys=True), encoding="utf-8"
    )
    if predictions is not None:
        predictions.to_parquet(out / "predictions.parquet", index=False)

    return out


def load_run(run_id: str, root: Path = RESULTS_ROOT) -> dict[str, Any]:
    """Read a run back. ``make_tables.py`` uses this and refuses runs with ``git_dirty``."""
    d = Path(root) / run_id
    if not d.is_dir():
        raise FileNotFoundError(f"no such run: {d}")
    return {
        "run_id": run_id,
        "params": yaml.safe_load((d / "params.yaml").read_text(encoding="utf-8")),
        "metrics": json.loads((d / "metrics.json").read_text(encoding="utf-8")),
        "env": json.loads((d / "env.json").read_text(encoding="utf-8")),
        "predictions_path": (
            d / "predictions.parquet" if (d / "predictions.parquet").exists() else None
        ),
    }
