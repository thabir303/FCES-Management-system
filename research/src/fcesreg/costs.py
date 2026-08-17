"""Cost and throughput aggregation over the global ledger (§10, `T8_cost`).

**The reported cost is one clean execution of the method, not what we happened to spend.**
Three quantities in the ledger look like cost and only one is:

* ``usd`` summed over the file — what a *warm cache* cost us. A fact about our development
  process, not about the method. It also falls as the cache warms, which is the wrong
  direction for a cost figure to move.
* ``usd_uncached`` summed over the file — inflated, and **silently**. A resumable sweep
  replays every previously cached call on each daily re-run and appends a row for each, so
  replay rows accumulate roughly quadratically over the run. Measured at 1.58x partway
  through a seven-day sweep and still climbing.
* ``usd_uncached`` **deduplicated by** ``prompt_sha256`` — what one clean run would cost.
  This is the one the paper reports.

``sum(usd)`` happens to equal the deduplicated total whenever every distinct prompt was
paid for exactly once, which is true only while the cache is never cleared. Clearing it and
re-running would pay twice for one prompt: counted twice by ``sum(usd)``, once by
deduplication. The agreement is a coincidence of the current state, not a definition, so
deduplication is what is implemented.

The same exposure runs through tokens and latency and is handled the same way: tokens are
deduplicated per distinct prompt, and latency is taken over live rows only, since the
elapsed time of a cache hit measures a disk read.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["EXCLUDED_CONDITIONS", "CostSummary", "summarise_costs", "throughput_per_day"]

#: Conditions that are not measurements of the method. Acceptance checks, sizing probes and
#: schema checks all issue real calls against the real endpoint, so they land in the ledger
#: like anything else and must be filtered out by name rather than by hoping nobody notices.
EXCLUDED_CONDITIONS = frozenset(
    {"c5_pilot", "cost_probe", "schema_probe", "rq2_probe"}
)


@dataclass(frozen=True)
class CostSummary:
    """One method's cost for a single execution. Every count is per distinct prompt."""

    condition: str
    n_calls: int
    input_tokens: int
    output_tokens: int
    usd: float
    n_replayed_rows: int
    mean_latency_ms: float | None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def mean_tokens(self) -> float:
        return self.total_tokens / self.n_calls if self.n_calls else 0.0

    def per_thousand(self, n_records: int) -> dict:
        """Scale to a thousand records. ``n_records`` is what the run actually processed."""
        if n_records <= 0:
            raise ValueError(f"n_records must be positive, got {n_records}")
        factor = 1000.0 / n_records
        return {
            "usd_per_1000": self.usd * factor,
            "tokens_per_1000": self.total_tokens * factor,
            "calls_per_1000": self.n_calls * factor,
        }


def summarise_costs(
    rows, conditions: set[str] | None = None, exclude: frozenset[str] = EXCLUDED_CONDITIONS
) -> dict[str, CostSummary]:
    """Per-condition cost of a single execution, deduplicated by ``prompt_sha256``.

    Invariant, and there is a test for it: replaying a completed run appends rows but must
    not change any returned figure. That is the whole point — a cost that grew every time
    the sweep resumed would be a measurement of how many days the quota took, not of the
    method.

    The first row for a prompt is kept. Rows are appended in time order, so the first is the
    live call that paid for it; a cache hit carries the same token counts but no latency
    worth having.
    """
    first_seen: dict[tuple[str, str], dict] = {}
    replayed: dict[str, int] = {}
    latencies: dict[str, list[float]] = {}

    for row in rows:
        condition = row.get("condition", "unspecified")
        if condition in exclude:
            continue
        if conditions is not None and condition not in conditions:
            continue

        key = (condition, row["prompt_sha256"])
        if key in first_seen:
            replayed[condition] = replayed.get(condition, 0) + 1
            continue
        first_seen[key] = row
        if not row.get("cache_hit"):
            # A cache hit's latency is a disk read, not the endpoint's response time.
            latencies.setdefault(condition, []).append(float(row.get("latency_ms", 0)))

    out: dict[str, CostSummary] = {}
    for (condition, _), row in first_seen.items():
        got = out.get(condition)
        timings = latencies.get(condition, [])
        out[condition] = CostSummary(
            condition=condition,
            n_calls=(got.n_calls if got else 0) + 1,
            input_tokens=(got.input_tokens if got else 0) + int(row["input_tokens"]),
            output_tokens=(got.output_tokens if got else 0) + int(row["output_tokens"]),
            usd=(got.usd if got else 0.0) + float(row["usd_uncached"]),
            n_replayed_rows=replayed.get(condition, 0),
            mean_latency_ms=(sum(timings) / len(timings)) if timings else None,
        )
    return out


def throughput_per_day(mean_tokens: float, tokens_per_day: int, requests_per_day: int) -> dict:
    """Rate-limit-bound throughput, reported as an operational figure beside cost.

    Under a free tier the binding constraint is quota rather than money, so "how many
    adjudications a day" is the number that decides whether a method is usable — a method
    costing $0.00 and taking three weeks is not free.
    """
    if mean_tokens <= 0:
        raise ValueError(f"mean_tokens must be positive, got {mean_tokens}")
    by_tokens = tokens_per_day / mean_tokens
    return {
        "calls_per_day": min(by_tokens, float(requests_per_day)),
        "binding_limit": "tokens" if by_tokens < requests_per_day else "requests",
        "calls_per_day_if_tokens_bound": by_tokens,
        "requests_per_day": float(requests_per_day),
    }
