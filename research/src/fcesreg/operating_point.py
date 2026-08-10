"""The precision–automation trade-off, and the residual effort it implies (§6.13).

This module produces the headline result. RQ3 asks what share of a legacy register migrates
without human intervention *at a fixed precision floor*, which is a different question from
how accurate the pipeline is on average: a method that is accurate on average but cannot be
run at high precision is of little use to a faculty that has to sign off a register, because
every error it introduces has to be found again by hand.

**The decision rule the whole module encodes.** At threshold ``t`` the pipeline resolves an
item automatically when its score is at or above ``t``, and routes everything below to a
human. Raising ``t`` buys precision and spends automated share. The curve traced by that
trade is the reported result (``F2_operating_point.pdf``); the automated share at 0.95 and
at 0.99 are the two points quoted from it (§13.1).

**Ties are resolved as a block, never split.** A threshold admits every item scoring at or
above it, so an operating point may never be placed *inside* a group of equal scores — the
precision it promises would not be the precision it delivers. This matters concretely
rather than theoretically: ``ExactMatcher`` emits only 1.0 and 0.0, so every one of its
pairs is tied with most of the others.

Nothing here reads a database, imports from ``system/`` or consults a corpus. It takes
arrays and a handling time and returns numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fcesreg.metrics import threshold_sweep

__all__ = [
    "DEFAULT_TARGET",
    "precision_automation_curve",
    "automated_share_at_precision",
    "residual_effort",
]

#: The precision floor RQ3 headlines (§13.1). 0.99 is reported alongside it, from the same
#: curve, rather than by re-running anything.
DEFAULT_TARGET = 0.95


def precision_automation_curve(scores, labels) -> pd.DataFrame:
    """The full trade-off curve: one row per attainable operating point.

    Columns are ``threshold``, ``precision``, ``recall`` and ``automated_share``.
    ``automated_share`` is the fraction of items resolved without human intervention at that
    threshold, and ``precision`` is measured over exactly those resolved items — not over
    the whole set — because the items routed to review are not decisions the pipeline made
    and cannot be credited or charged to it.

    The curve is the result and the two quoted points are its summary, so it is returned in
    full rather than reduced. It starts at the highest score present; a threshold above that
    resolves nothing, and the precision of an empty set is undefined rather than perfect.

    ``recall`` is ``nan`` throughout when the label set contains no positives — there is
    nothing to recall, and an inferred 0.0 would read as a measurement.
    """
    sweep = threshold_sweep(scores, labels)

    return pd.DataFrame(
        {
            "threshold": sweep.threshold,
            "precision": sweep.precision,
            "recall": (
                sweep.tp / sweep.n_positive
                if sweep.n_positive
                else np.full(sweep.tp.shape, np.nan)
            ),
            "automated_share": sweep.n_selected / sweep.n_items,
        }
    )


def automated_share_at_precision(
    scores, labels, target: float = DEFAULT_TARGET
) -> tuple[float, float]:
    """``(threshold, automated_share)`` at the *lowest* threshold holding ``target``.

    Lowest, not highest, because among the operating points that satisfy the precision floor
    the useful one is the point automating the most work. Precision is not monotone in the
    threshold, so this is a search over the curve rather than a boundary lookup.

    Returns ``(nan, 0.0)`` when no threshold reaches the target. **That is a finding, not an
    error**: it says this method cannot be operated at this precision on this data, which is
    a reportable result about the method. The caller reports it as such and does not retry
    at a lower target.
    """
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target must be in (0, 1], got {target}")

    sweep = threshold_sweep(scores, labels)

    # tp > 0 excludes a degenerate point that admits only negatives: its precision is 0 and
    # it could never meet a positive target, but the guard states the intent.
    ok = np.flatnonzero((sweep.precision >= target) & (sweep.tp > 0))
    if ok.size == 0:
        return float("nan"), 0.0

    at = ok[-1]
    return float(sweep.threshold[at]), float(sweep.n_selected[at] / sweep.n_items)


def residual_effort(
    n_records: int, automated_share: float, mean_seconds_per_item: float
) -> dict:
    """Convert an automated share into hours of human work, saved and remaining.

    ``mean_seconds_per_item`` is a **parameter with no default**. Its only legitimate source
    is the timed annotation exercise (§6.15), and defaulting it would let an assumed number
    reach the headline result without a run behind it. This function never reads a database
    and never imports from ``system/``; in particular the review queue's ``seconds_taken`` is
    not its source (§5.7, amendment 3).

    ``baseline_hours`` is the whole register done by hand — the comparison a faculty
    actually faces, since the alternative to the pipeline is not a faster pipeline.
    """
    if n_records < 0:
        raise ValueError(f"n_records must be non-negative, got {n_records}")
    if not 0.0 <= automated_share <= 1.0:
        raise ValueError(f"automated_share must be in [0, 1], got {automated_share}")
    if mean_seconds_per_item <= 0:
        raise ValueError(
            f"mean_seconds_per_item must be positive, got {mean_seconds_per_item}; "
            "it is measured by the timed annotation exercise, never assumed"
        )

    baseline_hours = n_records * mean_seconds_per_item / 3600.0
    residual_hours = baseline_hours * (1.0 - automated_share)

    return {
        "baseline_hours": baseline_hours,
        "residual_hours": residual_hours,
        "hours_saved": baseline_hours - residual_hours,
        "n_records": int(n_records),
        "automated_share": float(automated_share),
        "mean_seconds_per_item": float(mean_seconds_per_item),
    }
