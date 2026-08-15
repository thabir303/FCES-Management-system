"""Evaluation metrics (§6.12).

Every function here takes the label set as an explicit argument rather than inferring it
from the data. Inferring it would make a score depend on which classes happened to appear
in a split, so the same classifier would score differently on two samples of one corpus for
reasons that have nothing to do with the classifier.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

__all__ = [
    "prf1",
    "macro_weighted_f1",
    "confusion",
    "ThresholdSweep",
    "threshold_sweep",
    "Z_ONE_SIDED_95",
    "Z_TWO_SIDED_95",
    "wilson_interval",
    "wilson_lower_bound",
]

#: 95th percentile of the standard normal — a **one-sided** 95% lower confidence bound.
#: Used where the claim is one-directional ("precision is at least P"), since there is no
#: upper-side risk to insure against.
Z_ONE_SIDED_95 = 1.6448536269514722

#: 97.5th percentile — the half-width of a **two-sided** 95% interval. Used where an
#: interval is reported, as for the distractor contamination rate.
Z_TWO_SIDED_95 = 1.959963984540054


def wilson_interval(
    successes: int, n: int, z: float = Z_TWO_SIDED_95
) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion: ``(point, lower, upper)``.

    Preferred to the naive normal (Wald) interval, which at small ``n`` and a rate near 0
    or 1 pushes a bound outside ``[0, 1]`` and misstates coverage. Both bounds are clipped.

    ``z`` decides what is being claimed and the two are not interchangeable:
    :data:`Z_TWO_SIDED_95` for a reported interval, :data:`Z_ONE_SIDED_95` for a one-sided
    bound. Passing the two-sided constant to a one-sided claim silently asserts 97.5%
    confidence while calling it 95%.
    """
    if n <= 0:
        raise ValueError("cannot estimate a rate from zero judged items")
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    spread = (z / denom) * ((p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) ** 0.5)
    return p_hat, max(0.0, centre - spread), min(1.0, centre + spread)


def wilson_lower_bound(successes: int, n: int, z: float = Z_ONE_SIDED_95) -> float:
    """Lower confidence bound alone. ``0.0`` for ``n == 0``, where nothing is evidenced.

    Defaults to the **one-sided** constant, because every caller of this function is asking
    a one-directional question: is the true rate at least this? Returning 0.0 rather than
    raising on an empty set makes it usable inside a vectorised threshold sweep, where
    "no evidence" and "fails the bar" want the same treatment.
    """
    if n <= 0:
        return 0.0
    return wilson_interval(successes, n, z)[1]


class ThresholdSweep(NamedTuple):
    """Every attainable operating point of a scorer, one entry per distinct score."""

    threshold: np.ndarray
    tp: np.ndarray
    n_selected: np.ndarray
    n_items: int
    n_positive: int

    @property
    def precision(self) -> np.ndarray:
        return self.tp / self.n_selected


def threshold_sweep(scores, labels) -> ThresholdSweep:
    """Sort by descending score and accumulate, collapsing each tie group into one row.

    **A threshold may never be placed inside a group of equal scores.** Selecting at or
    above ``t`` admits every item scoring exactly ``t``, so an operating point computed
    part-way through such a group promises a precision the threshold does not deliver.
    Collapsing to the last index of each run is what makes the promise true.

    This is not a theoretical edge: ``ExactMatcher`` emits only 1.0 and 0.0, so on that
    matcher every pair is tied with most of the others and a per-index sweep is wrong
    almost everywhere.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: {scores.shape} vs {labels.shape}")
    if scores.size == 0:
        raise ValueError("no items: there are no thresholds to sweep")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain nan or inf; a threshold over them is meaningless")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")

    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    tp = np.cumsum(y)
    boundary = np.append(np.flatnonzero(np.diff(s) != 0), s.size - 1)

    return ThresholdSweep(
        threshold=s[boundary],
        tp=tp[boundary],
        n_selected=boundary + 1,
        n_items=int(s.size),
        n_positive=int(labels.sum()),
    )


def prf1(y_true, y_pred) -> dict:
    """Precision, recall, F1 and the counts they are built from.

    The counts are returned alongside the rates so a caller can check that
    ``tp + fp + fn + tn`` accounts for every pair — a mismatch means the pair set and the
    label set disagree, which is a construction fault rather than a poor result.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_pairs": int(y_true.size),
    }


def _check_label_coverage(y_true: np.ndarray, y_pred: np.ndarray, labels: list) -> None:
    """Refuse to score labels the caller did not declare.

    Silently dropping an out-of-set label would remove real items from the macro average
    and quietly improve the score. §6.10 routes a record whose true label is unsupported to
    review — it is the *caller's* job to have done that, and this is where the omission
    surfaces instead of becoming a better number.
    """
    known = set(labels)
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        unknown = sorted({v for v in values.tolist() if v not in known})
        if unknown:
            raise ValueError(
                f"{name} contains {len(unknown)} label(s) absent from `labels`, e.g. "
                f"{unknown[:3]}. Restrict the evaluation set or widen `labels`; they are "
                f"not dropped silently."
            )


def macro_weighted_f1(y_true, y_pred, labels) -> dict:
    """Macro and support-weighted F1 over exactly the labels given.

    Macro averages per-class F1 without regard to support, so a rare class counts as much
    as a common one; weighted averages by the support each class has in ``y_true``. Both are
    reported because they answer different questions, and on a distribution as skewed as
    CPV they diverge widely — a classifier that only learns the head scores well on one and
    badly on the other.

    A declared label with no instances in ``y_true`` and none in ``y_pred`` contributes 0.0
    to the macro average. That is the pessimistic reading and it is deliberate: the
    alternative, dropping it, would make the macro score improve as the test split got
    smaller. ``labels_without_support`` is returned so the effect is visible rather than
    buried in an average.
    """
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    labels = list(labels)
    if not labels:
        raise ValueError("`labels` is empty; there is nothing to average over")
    if len(set(labels)) != len(labels):
        raise ValueError("`labels` contains duplicates, which would double-count a class")
    _check_label_coverage(y_true, y_pred, labels)

    per_class: dict = {}
    for label in labels:
        true_is = y_true == label
        pred_is = y_pred == label
        tp = int((true_is & pred_is).sum())
        fp = int((~true_is & pred_is).sum())
        fn = int((true_is & ~pred_is).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(true_is.sum()),
        }

    supports = np.array([per_class[k]["support"] for k in labels], dtype=float)
    f1s = np.array([per_class[k]["f1"] for k in labels], dtype=float)
    total = supports.sum()

    return {
        "macro_f1": float(f1s.mean()),
        # Weighted by support in y_true, so it sums over the items actually present.
        "weighted_f1": float((f1s * supports).sum() / total) if total else 0.0,
        "accuracy": float((y_true == y_pred).mean()) if y_true.size else 0.0,
        "per_class": per_class,
        "n_labels": len(labels),
        "labels_without_support": int((supports == 0).sum()),
        "n_items": int(y_true.size),
    }


def confusion(y_true, y_pred, labels) -> pd.DataFrame:
    """Confusion matrix, rows indexed by true label and columns by predicted label.

    Label order follows ``labels`` as given rather than being sorted, so a caller
    presenting the operationally significant classes (§10, T7) controls the reading order
    of its own table.
    """
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    labels = list(labels)
    if len(set(labels)) != len(labels):
        raise ValueError("`labels` contains duplicates, which would double-count a class")
    _check_label_coverage(y_true, y_pred, labels)

    position = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, predicted in zip(y_true.tolist(), y_pred.tolist(), strict=True):
        matrix[position[actual], position[predicted]] += 1

    return pd.DataFrame(matrix, index=list(labels), columns=list(labels))
