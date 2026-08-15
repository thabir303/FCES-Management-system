"""Duplicate detection matchers (§6.9).

Four methods are compared in increasing order of cost: normalised exact match, character
n-gram TF-IDF, sentence embeddings, and a cascade that spends a language model call only
where the cheap signal is inconclusive.

Every matcher shares one interface, ``score_pairs(pairs, records) -> np.ndarray``, so the
cascade can wrap any of them and the runners can treat them uniformly.

Thresholds are always selected on a development partition. ``select_threshold`` exists so
that no caller is tempted to pick one by looking at test scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from fcesreg.embed import DEFAULT_CACHE_DIR as EMBED_CACHE_DIR
from fcesreg.embed import DEFAULT_MODEL as EMBED_MODEL
from fcesreg.embed import embed
from fcesreg.metrics import threshold_sweep, wilson_lower_bound
from fcesreg.normalise import normalise_key
from fcesreg.schema import text_of

__all__ = [
    "Matcher",
    "Adjudicator",
    "ExactMatcher",
    "TfidfMatcher",
    "EmbeddingMatcher",
    "CascadeMatcher",
    "AdjudicationBudgetExceeded",
    "select_threshold",
    "score_to_prediction",
]


@runtime_checkable
class Matcher(Protocol):
    name: str

    def score_pairs(
        self, pairs: pd.DataFrame, records: pd.DataFrame
    ) -> np.ndarray: ...


@runtime_checkable
class Adjudicator(Protocol):
    """Decides the pairs the base matcher could not place. Returns 1.0 or 0.0 per pair.

    Kept as a protocol so the cascade's band logic is testable without a network call, and
    so the expensive tier can be swapped without touching the cascade.
    """

    def adjudicate(
        self, pairs: pd.DataFrame, records: pd.DataFrame
    ) -> np.ndarray: ...


class AdjudicationBudgetExceeded(RuntimeError):
    """The band is larger than ``max_adjudications``.

    Raised rather than silently adjudicating a prefix and deciding the remainder by some
    fallback. A partial cascade is a different method from the one the paper describes, and
    one that reported itself under the same name would make the cost figure and the accuracy
    figure describe different things.
    """

    def __init__(self, n_band: int, cap: int):
        self.n_band = n_band
        self.cap = cap
        super().__init__(
            f"{n_band} pairs fall in the adjudication band, above the cap of {cap}. "
            f"Raise max_adjudications deliberately or narrow the band; the cascade will "
            f"not adjudicate part of a band and report the result as whole."
        )


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


class EmbeddingMatcher:
    """Sentence-embedding cosine similarity.

    Expected to tolerate abbreviation and paraphrase better than character features and
    character noise worse, since a typo moves a token off its learned representation
    whereas a synonym does not. Whether that expectation holds is measured, not assumed.

    CPU only (§12.7), and the encoder is the small one — the embedding tier has to be
    affordable to be a fair comparison against TF-IDF. Vectors come back L2-normalised, so
    the cosine is a dot product.
    """

    name = "embedding"

    def __init__(
        self,
        model_id: str = EMBED_MODEL,
        cache_dir: Path = EMBED_CACHE_DIR,
        batch_size: int = 64,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.batch_size = batch_size

    def score_pairs(self, pairs: pd.DataFrame, records: pd.DataFrame) -> np.ndarray:
        index, texts = _aligned_texts(records)
        vectors = embed(
            texts.tolist(),
            model_id=self.model_id,
            cache_dir=self.cache_dir,
            batch_size=self.batch_size,
        )
        left, right = _pair_positions(pairs, index)
        return (vectors[left] * vectors[right]).sum(axis=1).astype(np.float64)


class CascadeMatcher:
    """Cheap similarity first; a language model only where it is inconclusive.

    At or above ``upper`` a pair is accepted, at or below ``lower`` it is rejected, and
    strictly between them it goes to the adjudicator. The band is where the cost lives, so
    ``band_fraction`` is a reported result rather than an implementation detail.

    **``upper`` may be infinite, and that is a result rather than a misconfiguration.**
    Thresholds are chosen on dev to meet RQ3's precision target; where no threshold on the
    base similarity reaches it, nothing can be accepted without adjudication and the band
    extends to every pair not confidently rejected. The severity at which that happens
    bounds what a pipeline of this shape can offer on records of that quality.

    Output is 1.0 or 0.0 with nothing in between: the cascade emits decisions, not a
    ranking, so it reads as points against the other matchers' curves rather than a curve
    of its own.
    """

    name = "cascade"

    def __init__(
        self,
        base: Matcher,
        lower: float,
        upper: float,
        adjudicator: Adjudicator,
        max_adjudications: int = 5000,
    ):
        if lower > upper:
            raise ValueError(
                f"lower ({lower}) is above upper ({upper}); the band would be empty and "
                f"every pair decided by whichever bound is tested first"
            )
        self.base = base
        self.lower = lower
        self.upper = upper
        self.adjudicator = adjudicator
        self.max_adjudications = max_adjudications
        self.stats: dict = {}

    def score_pairs(self, pairs: pd.DataFrame, records: pd.DataFrame) -> np.ndarray:
        base_scores = np.asarray(self.base.score_pairs(pairs, records), dtype=float)

        # Strictly inside the band. A pair exactly on either bound is decided by the cheap
        # tier, which is what makes `>= upper` and `<= lower` exhaustive of the remainder.
        in_band = (base_scores > self.lower) & (base_scores < self.upper)
        n_band = int(in_band.sum())

        if n_band > self.max_adjudications:
            raise AdjudicationBudgetExceeded(n_band, self.max_adjudications)

        out = (base_scores >= self.upper).astype(np.float64)
        if n_band:
            verdicts = np.asarray(
                self.adjudicator.adjudicate(pairs.loc[in_band].copy(), records),
                dtype=float,
            )
            if verdicts.shape != (n_band,):
                raise ValueError(
                    f"adjudicator returned {verdicts.shape} for {n_band} banded pairs"
                )
            out[in_band] = verdicts

        self.stats = {
            "n_pairs": int(len(pairs)),
            "n_adjudicated": n_band,
            "band_fraction": (n_band / len(pairs)) if len(pairs) else 0.0,
            "lower": float(self.lower),
            "upper": float(self.upper),
            "upper_undefined": bool(np.isinf(self.upper)),
        }
        return out


def select_threshold(
    scores: np.ndarray, labels: np.ndarray, precision_target: float
) -> float:
    """Lowest threshold **confidently** reaching ``precision_target``. Fit on dev, never test.

    A threshold qualifies when the *lower bound of a one-sided 95% Wilson interval* on its
    precision reaches the target — not when its point estimate does. The distinction is not
    pedantry; it is what stops the function returning a threshold supported by almost
    nothing:

    ==========  ===================  ==============================
    severity    point estimate says  the same threshold on test
    ==========  ===================  ==============================
    0.30        ≥0.95 on 14 pairs    0.800 precision on 10 accepted
    0.75        ≥0.95 on **1** pair  0.000 precision on 0 accepted
    ==========  ===================  ==============================

    A precision floor should mean confidence that precision is at least the target, so the
    evidence demanded scales with the strictness of the target — which a fixed minimum on
    accepted pairs cannot do, and which needs no invented constant.

    **One-sided, at 95%.** The claim is one-directional: there is no upper-side risk to
    insure against, and using the lower limit of a *two-sided* 95% interval would assert
    97.5% confidence while calling it 95%. ``metrics`` names both constants so the choice
    is visible at the call site.

    The precision promised is also the precision delivered, which additionally requires
    tied scores to be admitted as a block — see ``metrics.threshold_sweep``.

    Returns ``inf`` when no threshold qualifies — a finding, not an error, and after this
    change a common one. The caller reports it as such rather than falling back to a lower
    target; the cascade reads it as "nothing can be auto-accepted at this severity".
    """
    sweep = threshold_sweep(scores, labels)

    confident = np.array(
        [
            wilson_lower_bound(int(tp), int(n))
            for tp, n in zip(sweep.tp, sweep.n_selected, strict=True)
        ]
    )
    ok = np.flatnonzero(confident >= precision_target)
    if ok.size == 0:
        return float("inf")
    return float(sweep.threshold[ok[-1]])


def score_to_prediction(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Pairs at or above the threshold are called duplicates."""
    return (np.asarray(scores, dtype=float) >= threshold).astype(int)
