"""CPV code assignment, three conditions (§6.10, C7).

RQ2 asks whether a language model given the labels you already have beats a classifier
trained on those same labels. Three conditions answer it, in increasing order of cost:

* :class:`TfidfSvmClassifier` — character n-grams and a linear SVM. Character features for
  the same reason as in duplicate detection: the errors that matter operate inside words.
* :class:`EmbeddingLogRegClassifier` — sentence embeddings and a logistic head.
* :class:`RagFewShotLLMClassifier` — a shortlist of candidate codes retrieved by embedding
  similarity, plus the nearest labelled examples, then one call.

**Two taxonomy levels only**, ``division`` (2-digit) and ``class`` (4-digit). The leaf level
is not implemented and must not be: §4.2 measured the label distribution as too sparse there
for a result to mean anything.

``alternatives`` is not optional. The import review queue renders the runner-up codes and
the paper reports them, so every condition returns them even where producing them costs
something — a classifier that can only say its first choice cannot be reviewed, only
accepted or rejected.

**The classical pair costs no quota and the language model condition does**, which is why
they are separable here and sequenced apart in the runner. Nothing in this module reaches
the network except through an injected adjudicator-style client.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from fcesreg.cpv import label_series
from fcesreg.embed import embed
from fcesreg.schema import text_of

__all__ = [
    "LEVELS",
    "Classifier",
    "ClassificationResult",
    "TfidfSvmClassifier",
    "EmbeddingLogRegClassifier",
    "Shortlister",
    "EmbeddingShortlister",
    "TfidfShortlister",
    "recall_at_k",
    "shortlist_codes",
    "taxonomy_text",
]

#: The two levels evaluated. Leaf (8-digit) is deliberately absent — see the module
#: docstring and §4.2. Note the 4-digit level is a CPV *class*; the official *group* is
#: three digits and is not evaluated here.
LEVELS = ("division", "class")

_N_ALTERNATIVES = 5


@dataclass
class ClassificationResult:
    """One prediction per record, with the runners-up the review queue needs."""

    codes: list[str]
    scores: np.ndarray
    alternatives: list[list[tuple[str, float]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.codes) != len(self.scores):
            raise ValueError(
                f"{len(self.codes)} codes against {len(self.scores)} scores"
            )
        if self.alternatives and len(self.alternatives) != len(self.codes):
            raise ValueError(
                f"{len(self.alternatives)} alternative lists against {len(self.codes)} codes"
            )


@runtime_checkable
class Classifier(Protocol):
    name: str

    def fit(self, train: pd.DataFrame, level: str) -> None: ...

    def predict(self, records: pd.DataFrame) -> ClassificationResult: ...


def _top_k(probabilities: np.ndarray, classes: np.ndarray, k: int):
    """Best label and the next ``k`` competitors, per row."""
    order = np.argsort(-probabilities, axis=1)
    best = classes[order[:, 0]]
    scores = probabilities[np.arange(len(probabilities)), order[:, 0]]
    alternatives = [
        [(classes[j], float(probabilities[i, j])) for j in order[i, 1 : k + 1]]
        for i in range(len(probabilities))
    ]
    return list(best), scores, alternatives


class TfidfSvmClassifier:
    """Character n-gram TF-IDF into a calibrated linear SVM.

    ``LinearSVC`` gives a decision margin rather than a probability, and the review queue
    needs a confidence it can threshold, so it is wrapped in ``CalibratedClassifierCV``.
    Without that the ``scores`` field would be a margin masquerading as a confidence, and
    the operating-point analysis would be thresholding an uncalibrated quantity.
    """

    name = "tfidf_svm"

    def __init__(self, ngram_range: tuple[int, int] = (2, 5), min_df: int = 2, cv: int = 3):
        self.vectoriser = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, sublinear_tf=True,
            min_df=min_df, lowercase=True,
        )
        self.cv = cv
        self.model: CalibratedClassifierCV | None = None

    def fit(self, train: pd.DataFrame, level: str) -> None:
        y = label_series(train, level)
        matrix = self.vectoriser.fit_transform(text_of(train))
        # A class with fewer members than the fold count cannot be cross-validated; drop
        # the fold count rather than the class, so support stays as the caller set it.
        folds = max(2, min(self.cv, int(y.value_counts().min())))
        self.model = CalibratedClassifierCV(LinearSVC(), cv=folds)
        self.model.fit(matrix, y)

    def predict(self, records: pd.DataFrame) -> ClassificationResult:
        if self.model is None:
            raise RuntimeError("fit() before predict()")
        probabilities = self.model.predict_proba(self.vectoriser.transform(text_of(records)))
        codes, scores, alternatives = _top_k(
            probabilities, np.asarray(self.model.classes_), _N_ALTERNATIVES
        )
        return ClassificationResult(codes=codes, scores=scores, alternatives=alternatives)


class EmbeddingLogRegClassifier:
    """Sentence embeddings into a logistic head.

    The head is deliberately shallow. Fine-tuning the encoder is out of scope (§12.7, CPU
    only), and the comparison RQ2 makes is between representations, not between training
    budgets.
    """

    name = "embedding_logreg"

    def __init__(self, max_iter: int = 1000, C: float = 1.0):
        self.model = LogisticRegression(max_iter=max_iter, C=C)
        self.fitted = False

    def fit(self, train: pd.DataFrame, level: str) -> None:
        y = label_series(train, level)
        self.model.fit(embed(text_of(train).tolist()), y)
        self.fitted = True

    def predict(self, records: pd.DataFrame) -> ClassificationResult:
        if not self.fitted:
            raise RuntimeError("fit() before predict()")
        probabilities = self.model.predict_proba(embed(text_of(records).tolist()))
        codes, scores, alternatives = _top_k(
            probabilities, np.asarray(self.model.classes_), _N_ALTERNATIVES
        )
        return ClassificationResult(codes=codes, scores=scores, alternatives=alternatives)


def taxonomy_text(taxonomy: pd.DataFrame) -> list[str]:
    """``"code description"`` per row — what a retriever matches a record against."""
    return (taxonomy["cpv_code"] + " " + taxonomy["cpv_description"].fillna("")).tolist()


@runtime_checkable
class Shortlister(Protocol):
    """Ranks the taxonomy against a record. The retrieval half of the RAG condition.

    Separated from :func:`shortlist_codes` so that **shortlist recall at k is measurable
    without spending a single language model call**. If the true code is not in the
    shortlist the model cannot be right, so a measured accuracy for the condition would
    conflate the model choosing badly with the retriever never offering the right option.
    That ceiling is a reported result, not an implementation detail.
    """

    name: str

    def fit(self, taxonomy: pd.DataFrame, corpus: pd.DataFrame | None = None) -> None: ...

    def rank(self, texts: Sequence[str]) -> np.ndarray: ...


class EmbeddingShortlister:
    """Cosine similarity between record text and ``"code description"``.

    What the paper specifies. Note it is *untrained*: it never sees a label, so it can only
    match a record against how the taxonomy describes itself.
    """

    name = "embedding"

    def __init__(self) -> None:
        self.matrix: np.ndarray | None = None

    def fit(self, taxonomy: pd.DataFrame, corpus: pd.DataFrame | None = None) -> None:
        if taxonomy.empty:
            raise ValueError("taxonomy is empty; there is nothing to shortlist")
        self.matrix = embed(taxonomy_text(taxonomy))

    def rank(self, texts: Sequence[str]) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("fit() before rank()")
        return np.argsort(-(embed(list(texts)) @ self.matrix.T), axis=1)


class TfidfShortlister:
    """Character n-gram TF-IDF, same features as :class:`TfidfSvmClassifier`.

    ``corpus`` contributes only its *text* to the vectoriser's vocabulary and document
    frequencies, never its labels — the taxonomy alone is 1,209 short strings, whose
    document frequencies describe the taxonomy's own prose rather than the procurement
    vocabulary a record is written in. This retriever remains untrained in the sense that
    matters: it has no access to which code a record was published under.
    """

    name = "tfidf"

    def __init__(self, ngram_range: tuple[int, int] = (2, 5), min_df: int = 2):
        self.vectoriser = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, sublinear_tf=True,
            min_df=min_df, lowercase=True,
        )
        self.matrix = None

    def fit(self, taxonomy: pd.DataFrame, corpus: pd.DataFrame | None = None) -> None:
        if taxonomy.empty:
            raise ValueError("taxonomy is empty; there is nothing to shortlist")
        codes = taxonomy_text(taxonomy)
        vocabulary = codes if corpus is None else codes + text_of(corpus).tolist()
        self.vectoriser.fit(vocabulary)
        self.matrix = self.vectoriser.transform(codes)

    def rank(self, texts: Sequence[str]) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("fit() before rank()")
        # Both sides are L2-normalised by TfidfVectorizer, so the product is cosine.
        similarity = (self.vectoriser.transform(list(texts)) @ self.matrix.T).toarray()
        return np.argsort(-similarity, axis=1)


def recall_at_k(ranked: np.ndarray, true_position: np.ndarray, ks: Sequence[int]) -> dict:
    """Share of records whose true code appears in the top ``k`` of the ranking.

    The ceiling on the retrieval-augmented condition: a language model handed a shortlist
    missing the answer cannot produce it, and reporting that outcome as the model's
    accuracy would measure the retriever under the model's name.
    """
    if len(ranked) != len(true_position):
        raise ValueError(f"{len(ranked)} rankings against {len(true_position)} labels")
    hit_at = np.argmax(ranked == true_position[:, None], axis=1)
    found = (ranked == true_position[:, None]).any(axis=1)
    # A record whose true code is absent from the pool entirely can never be retrieved;
    # it counts against recall rather than being quietly excluded.
    return {
        "n": int(len(ranked)),
        "recall": {int(k): float((found & (hit_at < k)).mean()) for k in ks},
        "mean_rank_when_found": float(hit_at[found].mean() + 1) if found.any() else None,
        "n_not_in_pool": int((~found).sum()),
    }


def shortlist_codes(
    record_text: str, taxonomy: pd.DataFrame, k: int = 12
) -> list[tuple[str, str]]:
    """The ``k`` candidate codes closest to ``record_text`` by embedding similarity.

    **Never send the full taxonomy to the model.** The shortlist is both the cost control
    and the retrieval half of the retrieval-augmented condition: sending every code would
    cost far more per call and would stop the condition being a retrieval one at all.
    """
    shortlister = EmbeddingShortlister()
    shortlister.fit(taxonomy)
    descriptions = taxonomy["cpv_description"].fillna("")
    return [
        (taxonomy["cpv_code"].iloc[i], descriptions.iloc[i])
        for i in shortlister.rank([record_text])[0, :k]
    ]
