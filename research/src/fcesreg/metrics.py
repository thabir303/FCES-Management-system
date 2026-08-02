"""Evaluation metrics (§6.12)."""

from __future__ import annotations

import numpy as np

__all__ = ["prf1"]


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
