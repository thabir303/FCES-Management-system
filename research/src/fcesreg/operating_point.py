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

**Residual effort is reported as review volume, and total effort as a formula.** Handling
time is not measured anywhere in this project: an author timing their own reading measures
that author, and a model timing its own measures endpoint latency. The volume framing loses
nothing that matters, because the reduction is a ratio and is therefore the same number
whatever the handling time turns out to be.

Nothing here reads a database, imports from ``system/`` or consults a corpus. It takes
arrays and returns numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fcesreg.dedup import select_threshold
from fcesreg.metrics import threshold_sweep, wilson_lower_bound

__all__ = [
    "DEFAULT_TARGET",
    "EFFORT_FORMULA",
    "precision_automation_curve",
    "automated_share_at_precision",
    "reject_bound",
    "band_operating_point",
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

    ``precision_lower`` is the one-sided 95% Wilson lower bound on that row's precision, and
    it is what ``automated_share_at_precision`` selects against. Carried in the curve so the
    figure can show the floor being applied rather than leaving the quoted points looking
    unexplained: a row whose ``precision`` clears the target while its ``precision_lower``
    does not is a point supported by too little evidence to quote.

    ``recall`` is ``nan`` throughout when the label set contains no positives — there is
    nothing to recall, and an inferred 0.0 would read as a measurement.
    """
    sweep = threshold_sweep(scores, labels)

    return pd.DataFrame(
        {
            "threshold": sweep.threshold,
            "precision": sweep.precision,
            "precision_lower": [
                wilson_lower_bound(int(tp), int(n))
                for tp, n in zip(sweep.tp, sweep.n_selected, strict=True)
            ],
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
    """``(threshold, automated_share)`` at the *lowest* threshold **confidently** holding
    ``target``.

    Lowest, not highest, because among the operating points that satisfy the precision floor
    the useful one is the point automating the most work. Precision is not monotone in the
    threshold, so this is a search over the curve rather than a boundary lookup.

    Qualification is by the lower bound of a one-sided 95% Wilson interval, the same rule
    ``dedup.select_threshold`` uses and for the same reason: a floor should mean confidence
    that precision is at least the target, not a point estimate that happens to clear it.
    An automated share resting on a handful of accepted items is not an automated share.
    Two consequences follow from the arithmetic and are properties of the rule, not bugs:

    * meeting 0.95 needs at least 52 accepted items even if every one is correct, and 0.99
      needs 268;
    * ``target=1.0`` can never be met at any sample size, because a finite run of correct
      decisions never evidences certainty.

    Returns ``(nan, 0.0)`` when no threshold qualifies. **That is a finding, not an error**:
    it says this method cannot be operated at this precision on this data. The caller
    reports it as such and does not retry at a lower target.
    """
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target must be in (0, 1], got {target}")

    sweep = threshold_sweep(scores, labels)

    confident = np.array(
        [
            wilson_lower_bound(int(tp), int(n))
            for tp, n in zip(sweep.tp, sweep.n_selected, strict=True)
        ]
    )
    ok = np.flatnonzero(confident >= target)
    if ok.size == 0:
        return float("nan"), 0.0

    at = ok[-1]
    return float(sweep.threshold[at]), float(sweep.n_selected[at] / sweep.n_items)


def reject_bound(scores, labels, target: float = DEFAULT_TARGET) -> float:
    """Highest threshold whose *rejected* set is confidently negative.

    The mirror of :func:`dedup.select_threshold`, read from the other end and held to the
    same standard: a pair is auto-rejected only where the evidence supports rejecting it,
    not merely where the point estimate looks safe. ``-inf`` means nothing can be rejected
    without adjudication, which at high severity is the honest answer.

    Lives here rather than in a runner because :func:`band_operating_point` needs it and a
    library function defined inside a script cannot be tested or reused.
    """
    sweep = threshold_sweep(scores, labels)
    n_below = sweep.n_items - sweep.n_selected
    neg_below = n_below - (sweep.n_positive - sweep.tp)
    bound = np.array(
        [wilson_lower_bound(int(k), int(n)) for k, n in zip(neg_below, n_below, strict=True)]
    )
    ok = np.flatnonzero(bound >= target)
    return float(sweep.threshold[ok[0]]) if ok.size else float("-inf")


def band_operating_point(scores, labels, target: float = DEFAULT_TARGET) -> dict:
    """Automated share under the **two-bound** rule the duplicate pipeline actually uses.

    :func:`automated_share_at_precision` models one threshold: resolve above it, review
    everything below. That is right for a task where the decision is "accept this
    suggestion", and **wrong for pairwise duplicate detection**, where the overwhelming
    majority of candidate pairs are obvious non-duplicates that no human would ever open.
    Scoring them as un-automated says the pipeline automates almost nothing when in fact it
    automates nearly everything, and would have understated the headline result by an order
    of magnitude.

    So a pair is automated at *either* end: auto-accepted at or above ``upper``
    (:func:`dedup.select_threshold`), auto-rejected below ``lower`` (:func:`reject_bound`).
    What is left is the band a human adjudicates, and ``automated_share`` is
    ``1 - band_fraction``. Both bounds are held to the same one-sided 95% Wilson floor, so
    the pipeline is confident in what it accepts *and* in what it discards.

    ``upper = inf`` means nothing can be auto-accepted and ``lower = -inf`` means nothing
    can be auto-rejected. Both are findings and both are reported as they fall.

    ``bounds_crossed`` flags the case where both rules claim the same items — see the
    comment at the clamp. The returned precision and purity are always *measured* on the
    sets actually produced, so a crossed fit cannot quietly report a floor it did not meet.
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    upper = select_threshold(scores, labels, target)
    lower = reject_bound(scores, labels, target)

    # **The bounds can cross**, and on a well-separated scorer they do: each rule certifies
    # its own set against its own denominator, so a middle region can satisfy both "accept
    # confidently" and "reject confidently" at once. `CascadeMatcher` refuses that
    # configuration outright. Here it is recorded and resolved to a single cut at `upper`,
    # never averaged away — and the precision and purity below are then *measured* on the
    # sets that result rather than inherited from a floor that was fitted on different ones.
    crossed = bool(np.isfinite(lower) and np.isfinite(upper) and lower > upper)
    if crossed:
        lower = upper

    accepted = scores >= upper
    rejected = scores < lower
    band = ~(accepted | rejected)
    n = len(scores)

    return {
        "lower": lower,
        "upper": upper,
        "bounds_crossed": crossed,
        "upper_undefined": not np.isfinite(upper),
        "lower_undefined": not np.isfinite(lower),
        "n_items": int(n),
        "n_auto_accepted": int(accepted.sum()),
        "n_auto_rejected": int(rejected.sum()),
        "n_band": int(band.sum()),
        "band_fraction": float(band.mean()) if n else 0.0,
        "automated_share": float(1.0 - band.mean()) if n else 0.0,
        # Precision over what was accepted WITHOUT adjudication — the quantity the floor
        # constrains. It says nothing about what a human then does with the band.
        "precision_auto_accepted": (
            float(labels[accepted].mean()) if accepted.any() else None
        ),
        "purity_auto_rejected": (
            float(1.0 - labels[rejected].mean()) if rejected.any() else None
        ),
    }


#: How a reader converts the measured volume into time. ``t`` is *their* seconds per item,
#: not ours: this project does not measure handling time, and a symbol a reader substitutes
#: into is honest where a number we never ran would not be.
EFFORT_FORMULA = "hours = n_review * t / 3600, for a handling time of t seconds per item"


def residual_effort(n_records: int, automated_share: float) -> dict:
    """The human work an operating point leaves behind, **as review volume**.

    Volume, not hours. Converting to hours needs a mean handling time, and this project
    does not measure one — timing an author's own reading measures how fast that author
    read, and timing a model's measures endpoint latency. Reporting either as curation
    effort would not be a limitation, it would be a false measurement.

    So the quantity RQ3 headlines is ``n_review``: how many records still reach a human.
    That is measured end to end and needs no assumption. :data:`EFFORT_FORMULA` is carried
    beside it so a faculty holding its own handling time can convert, and so a reader can
    see that the conversion was declined rather than forgotten.

    ``baseline_review`` is the whole register by hand — the comparison a faculty actually
    faces, since the alternative to the pipeline is not a faster pipeline. The reduction is
    a ratio of volumes and is therefore invariant to whatever ``t`` turns out to be, which
    is the reason the volume framing loses nothing: **any** handling time gives the same
    proportional saving.

    The review queue's ``seconds_taken`` is still recorded per decision (§5.7), and is
    still not this function's input: it is operational telemetry from a deployment, not a
    measurement of the corpus this result is about (amendment 3).
    """
    if n_records < 0:
        raise ValueError(f"n_records must be non-negative, got {n_records}")
    if not 0.0 <= automated_share <= 1.0:
        raise ValueError(f"automated_share must be in [0, 1], got {automated_share}")

    n_review = round(n_records * (1.0 - automated_share))

    return {
        "n_records": int(n_records),
        "automated_share": float(automated_share),
        "baseline_review": int(n_records),
        "n_review": int(n_review),
        "n_automated": int(n_records - n_review),
        # A ratio of volumes: the same number whatever a handling time turns out to be.
        "volume_reduction": float(automated_share),
        "effort_formula": EFFORT_FORMULA,
    }
