"""Per-item timing for the annotation tools (§6.15).

**Nothing this module produces is a reported figure** (ruled 2026-08-17). RQ3 reports
residual effort as review *volume*, with total effort left as a formula a reader
substitutes their own handling time into; see :func:`operating_point.residual_effort`.
Timing an author's own reading measures that author and timing a model's measures endpoint
latency, so neither is curation effort.

What remains is instrumentation: the annotation tools time judgements as they are made so
an annotator can see how long a session took and so an abandoned item is countable rather
than invisible. The review queue records its own per-decision duration separately (§5.7),
as operational telemetry of a deployment rather than a measurement of a corpus.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "IDLE_CUTOFF_S",
    "MIN_ITEMS_FOR_MEAN",
    "ItemTiming",
    "TooFewTimings",
    "time_item",
    "summarise",
    "write_timings",
    "read_timings",
]

#: An item that takes longer than this was not being worked on — the annotator went to
#: make tea. It is flagged abandoned rather than silently inflating the mean.
IDLE_CUTOFF_S = 120.0

#: The paper states a timing sample size. A mean over fewer than this many retained items
#: is not a measurement, so ``summarise`` refuses to produce one.
MIN_ITEMS_FOR_MEAN = 30


class TooFewTimings(ValueError):
    """Fewer than ``MIN_ITEMS_FOR_MEAN`` retained items — no mean can be reported."""


@dataclass(frozen=True)
class ItemTiming:
    item_id: str
    seconds: float
    started_at: str
    abandoned: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@contextmanager
def time_item(
    item_id: str,
    sink: list[ItemTiming],
    idle_cutoff_s: float = IDLE_CUTOFF_S,
) -> Iterator[Callable[[], None]]:
    """Time one judgement and append an :class:`ItemTiming` to ``sink``.

    Uses ``time.monotonic``, never wall clock: a clock adjustment mid-session must not be
    able to produce a negative or wildly inflated duration.

    The yielded callable marks the item abandoned explicitly — call it when the annotator
    skips without judging. An item exceeding ``idle_cutoff_s`` is marked abandoned
    automatically. Abandoned items are still recorded, so the exclusion is visible and
    countable rather than invisible.

    A timing is recorded even if the body raises, so an interrupted session does not lose
    the items already judged.
    """
    started_at = datetime.now(UTC).isoformat()
    t0 = time.monotonic()
    flag = {"abandoned": False}

    def mark_abandoned() -> None:
        flag["abandoned"] = True

    try:
        yield mark_abandoned
    finally:
        elapsed = time.monotonic() - t0
        sink.append(
            ItemTiming(
                item_id=item_id,
                seconds=elapsed,
                started_at=started_at,
                abandoned=flag["abandoned"] or elapsed > idle_cutoff_s,
            )
        )


def summarise(timings: Sequence[ItemTiming]) -> dict[str, Any]:
    """Summary statistics over retained (non-abandoned) items.

    Raises :class:`TooFewTimings` when fewer than ``MIN_ITEMS_FOR_MEAN`` items survive the
    abandonment filter. The paper reports a sample size alongside the mean and it has to
    be a real one.
    """
    retained = sorted(t.seconds for t in timings if not t.abandoned)
    n_abandoned = len(timings) - len(retained)

    if len(retained) < MIN_ITEMS_FOR_MEAN:
        raise TooFewTimings(
            f"{len(retained)} retained timings ({n_abandoned} abandoned); "
            f"need at least {MIN_ITEMS_FOR_MEAN} to report a mean"
        )

    n = len(retained)
    return {
        "n": n,
        "n_abandoned": n_abandoned,
        "mean_seconds": sum(retained) / n,
        "median_seconds": _quantile(retained, 0.5),
        "p90_seconds": _quantile(retained, 0.9),
        "total_seconds": sum(retained),
    }


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile over an already-sorted list."""
    if not sorted_values:
        raise ValueError("empty")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def write_timings(path: Path, timings: Sequence[ItemTiming]) -> None:
    """Append timings as JSON Lines. Appends are atomic per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for t in timings:
            f.write(t.to_json() + "\n")


def read_timings(path: Path) -> list[ItemTiming]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(ItemTiming(**json.loads(line)))
    return rows
