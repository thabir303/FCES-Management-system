"""Duplicate detection matchers (§6.9).

Four methods are compared in increasing order of cost. This module carries the two
cheapest; the embedding matcher and the cascade arrive at C6.

Every matcher shares one interface, ``score_pairs(pairs, records) -> np.ndarray``, so the
cascade can wrap any of them and the runners can treat them uniformly.

Thresholds are always selected on a development partition. ``select_threshold`` exists so
that no caller is tempted to pick one by looking at test scores.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from fcesreg.normalise import normalise_key
from fcesreg.schema import text_of

__all__ = [
    "Matcher",
    "ExactMatcher",
    "TfidfMatcher",
    "select_threshold",
    "score_to_prediction",
]


@runtime_checkable
class Matcher(Protocol):
    name: str

    def score_pairs(
        self, pairs: pd.DataFrame, records: pd.DataFrame
    ) -> np.ndarray: ...


def _aligned_texts(records: pd.DataFrame) -> tuple[dict[str, int], pd.Series]:
    """Map record_id to row position, and the text each row contributes."""
    index = {rid: i for i, rid in enumerate(records["record_id"])}
    return index, text_of(records)


def _pair_positions(
    pairs: pd.DataFrame, index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    missing = [
        rid
        for rid in (*pairs["left_id"], *pairs["right_id"])
        if rid not in index
    ]
    if missing:
        raise KeyError(
            f"{len(missing)} pair ids are absent from the record frame, e.g. "
            f"{sorted(set(missing))[:3]}. The pair set and the record set disagree."
        )
    left = np.fromiter((index[r] for r in pairs["left_id"]), dtype=np.int64, count=len(pairs))
    right = np.fromiter((index[r] for r in pairs["right_id"]), dtype=np.int64, count=len(pairs))
    return left, right


class ExactMatcher:
    """Normalised exact match: the floor, and a measure of how much is trivially solvable.

    Compares ``normalise_key`` of the concatenated title and description, so every
    difference of casing, spacing and punctuation is collapsed and what survives is a
    genuine difference in content. Scores are 1.0 or 0.0 and nothing in between, which is
    the point — this method has no operating curve to tune.
    """

    name = "exact"

    def score_pairs(self, pairs: pd.DataFrame, records: pd.DataFrame) -> np.ndarray:
        index, texts = _aligned_texts(records)
        keys = np.array([normalise_key(t) for t in texts], dtype=object)
        left, right = _pair_positions(pairs, index)
        return (keys[left] == keys[right]).astype(np.float64)


class TfidfMatcher:
    """Character n-gram TF-IDF with cosine similarity.

    Character features rather than word features, because the dominant error classes —
    typos, abbreviation, unit notation — operate *within* words, and a word-level model
    sees a misspelled token as a wholly unrelated one.

    The vectoriser is fitted on the record frame it is asked to score, not on a separate
    corpus, so the IDF weights describe the collection being deduplicated.
    """

    name = "tfidf"

    def __init__(
        self,
        ngram_range: tuple[int, int] = (2, 4),
        analyzer: str = "char_wb",
        sublinear_tf: bool = True,
        min_df: int = 1,
    ):
        self.vectoriser = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            lowercase=True,
        )

    def score_pairs(self, pairs: pd.DataFrame, records: pd.DataFrame) -> np.ndarray:
        index, texts = _aligned_texts(records)
        matrix = self.vectoriser.fit_transform(texts)
        # TfidfVectorizer L2-normalises rows, so the cosine is a plain dot product.
        left, right = _pair_positions(pairs, index)
        return np.asarray(matrix[left].multiply(matrix[right]).sum(axis=1)).ravel()


def select_threshold(
    scores: np.ndarray, labels: np.ndarray, precision_target: float
) -> float:
    """Lowest threshold reaching ``precision_target``. **Fit on dev, never on test.**

    Returns the threshold that maximises recall subject to precision, which is the
    operating point RQ3 asks for: a method that is accurate on average but cannot be run
    at high precision is of little use to a faculty that has to sign off a register.

    Returns ``inf`` when no threshold reaches the target — a finding, not an error. The
    caller reports it as such rather than falling back to a lower target.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)

    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]

    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)

    ok = np.flatnonzero((precision >= precision_target) & (tp > 0))
    if ok.size == 0:
        return float("inf")
    return float(s[ok[-1]])


def score_to_prediction(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Pairs at or above the threshold are called duplicates."""
    return (np.asarray(scores, dtype=float) >= threshold).astype(int)
