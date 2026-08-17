"""CPV classification, the two classical conditions (§6.10, C7).

The language model condition is tested separately; these two carry no quota cost and are
the ones that must be finished before anything waits on the sweep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fcesreg import embed as embed_mod
from fcesreg.classify import (
    LEVELS,
    ClassificationResult,
    Classifier,
    EmbeddingLogRegClassifier,
    EmbeddingShortlister,
    Shortlister,
    TfidfShortlister,
    TfidfSvmClassifier,
    recall_at_k,
    shortlist_codes,
)


@pytest.fixture
def stub_encoder(monkeypatch):
    class Encoder:
        def encode(self, texts, **kwargs):
            out = np.zeros((len(texts), 16), dtype=np.float32)
            for i, t in enumerate(texts):
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                v = rng.normal(size=16).astype(np.float32)
                out[i] = v / np.linalg.norm(v)
            return out

    monkeypatch.setattr(embed_mod, "_load_model", lambda model_id: Encoder())


def corpus(per_class: int = 40) -> pd.DataFrame:
    codes = np.repeat(["30100000", "33100000", "42100000"], per_class)
    return pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(len(codes))],
            "title": [f"{c[:2]} widget {i}" for i, c in enumerate(codes)],
            "description": [f"description for division {c[:2]}" for c in codes],
            "cpv_code": codes,
        }
    )


class TestLevels:
    def test_only_two_levels_exist(self):
        # The leaf level is not implemented and must not be (§4.2).
        assert LEVELS == ("division", "class")

    def test_an_unknown_level_is_refused(self):
        with pytest.raises(ValueError, match="level must be one of"):
            TfidfSvmClassifier(min_df=1).fit(corpus(), "leaf")

    def test_division_and_class_are_different_targets(self):
        # A 2-digit truncation scored against 4-digit labels would look plausible and mean
        # nothing, so the two must not be interchangeable.
        m = TfidfSvmClassifier(min_df=1)
        m.fit(corpus(), "division")
        assert all(len(c) == 2 for c in m.predict(corpus().head(3)).codes)
        m.fit(corpus(), "class")
        assert all(len(c) == 4 for c in m.predict(corpus().head(3)).codes)


class TestClassificationResult:
    def test_mismatched_codes_and_scores_are_refused(self):
        with pytest.raises(ValueError, match="codes against"):
            ClassificationResult(codes=["a", "b"], scores=np.array([0.5]))

    def test_mismatched_alternatives_are_refused(self):
        with pytest.raises(ValueError, match="alternative lists"):
            ClassificationResult(
                codes=["a"], scores=np.array([0.5]), alternatives=[[], []]
            )


class TestClassicalConditions:
    @pytest.mark.parametrize("factory", [TfidfSvmClassifier, EmbeddingLogRegClassifier])
    def test_both_satisfy_the_protocol(self, factory):
        assert isinstance(factory(), Classifier)

    @pytest.mark.parametrize("factory", [TfidfSvmClassifier, EmbeddingLogRegClassifier])
    def test_predicting_before_fitting_is_a_loud_error(self, factory):
        with pytest.raises(RuntimeError, match="fit\\(\\) before predict\\(\\)"):
            factory().predict(corpus().head(2))

    def test_tfidf_learns_a_separable_signal(self, stub_encoder):
        train = corpus()
        m = TfidfSvmClassifier(min_df=1)
        m.fit(train, "division")
        got = m.predict(train)
        # A bug detector, not a performance claim: the fixture is trivially separable, so
        # anything near chance means the labels and features are misaligned.
        assert (np.array(got.codes) == train["cpv_code"].str[:2].to_numpy()).mean() > 0.9

    def test_embedding_learns_a_separable_signal(self, stub_encoder):
        train = corpus()
        m = EmbeddingLogRegClassifier()
        m.fit(train, "division")
        got = m.predict(train)
        assert (np.array(got.codes) == train["cpv_code"].str[:2].to_numpy()).mean() > 0.5

    @pytest.mark.parametrize("factory", [TfidfSvmClassifier, EmbeddingLogRegClassifier])
    def test_alternatives_are_always_returned(self, factory, stub_encoder):
        # Not optional: the review queue renders them, and a classifier that can only give
        # its first choice can be accepted or rejected but not reviewed.
        m = factory() if factory is EmbeddingLogRegClassifier else factory(min_df=1)
        m.fit(corpus(), "division")
        got = m.predict(corpus().head(4))
        assert len(got.alternatives) == 4
        assert all(len(a) > 0 for a in got.alternatives)

    @pytest.mark.parametrize("factory", [TfidfSvmClassifier, EmbeddingLogRegClassifier])
    def test_the_top_choice_outranks_its_alternatives(self, factory, stub_encoder):
        m = factory() if factory is EmbeddingLogRegClassifier else factory(min_df=1)
        m.fit(corpus(), "division")
        got = m.predict(corpus().head(5))
        for score, alts in zip(got.scores, got.alternatives, strict=True):
            assert all(score >= a for _, a in alts)

    def test_scores_are_calibrated_probabilities_not_margins(self, stub_encoder):
        # An SVM margin thresholded as if it were a confidence would make the operating
        # point meaningless.
        m = TfidfSvmClassifier(min_df=1)
        m.fit(corpus(), "division")
        got = m.predict(corpus().head(10))
        assert ((got.scores >= 0.0) & (got.scores <= 1.0)).all()

    def test_a_rare_class_does_not_break_cross_validation(self, stub_encoder):
        # A class with fewer members than the fold count would raise; the fold count drops
        # instead, so support stays as the caller set it.
        train = pd.concat([corpus(), corpus(2).assign(cpv_code="44100000")])
        TfidfSvmClassifier(min_df=1).fit(train, "division")


class TestShortlist:
    def taxonomy(self, n: int = 30) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cpv_code": [f"{30 + i:04d}" for i in range(n)],
                "cpv_description": [f"description of thing {i}" for i in range(n)],
            }
        )

    def test_returns_k_candidates(self, stub_encoder):
        assert len(shortlist_codes("a pump", self.taxonomy(), k=12)) == 12

    def test_never_returns_the_whole_taxonomy(self, stub_encoder):
        # Both the cost control and what makes this the retrieval condition.
        tax = self.taxonomy(100)
        assert len(shortlist_codes("a pump", tax, k=12)) < len(tax)

    def test_a_shorter_taxonomy_is_not_padded(self, stub_encoder):
        assert len(shortlist_codes("a pump", self.taxonomy(5), k=12)) == 5

    def test_each_candidate_carries_its_description(self, stub_encoder):
        # The model is shown what a code means, not just its digits.
        got = shortlist_codes("a pump", self.taxonomy(), k=3)
        assert all(code and description for code, description in got)

    def test_an_empty_taxonomy_is_refused(self, stub_encoder):
        with pytest.raises(ValueError, match="taxonomy is empty"):
            shortlist_codes("a pump", self.taxonomy(0))

    def test_deterministic(self, stub_encoder):
        tax = self.taxonomy()
        assert shortlist_codes("a pump", tax, k=8) == shortlist_codes("a pump", tax, k=8)


class TestShortlisters:
    def taxonomy(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cpv_code": ["3010", "3310", "4210"],
                "cpv_description": [
                    "computers and office machinery",
                    "medical equipment and consumables",
                    "industrial machinery and pumps",
                ],
            }
        )

    @pytest.mark.parametrize("factory", [EmbeddingShortlister, TfidfShortlister])
    def test_both_satisfy_the_protocol(self, factory):
        assert isinstance(factory(), Shortlister)

    @pytest.mark.parametrize("factory", [EmbeddingShortlister, TfidfShortlister])
    def test_ranking_before_fitting_is_a_loud_error(self, factory):
        with pytest.raises(RuntimeError, match="fit\\(\\) before rank\\(\\)"):
            factory().rank(["a pump"])

    @pytest.mark.parametrize("factory", [EmbeddingShortlister, TfidfShortlister])
    def test_an_empty_taxonomy_is_refused(self, factory):
        with pytest.raises(ValueError, match="taxonomy is empty"):
            factory().fit(pd.DataFrame({"cpv_code": [], "cpv_description": []}))

    @pytest.mark.parametrize("factory", [EmbeddingShortlister, TfidfShortlister])
    def test_ranking_is_a_permutation_of_the_pool(self, factory, stub_encoder):
        s = factory(min_df=1) if factory is TfidfShortlister else factory()
        s.fit(self.taxonomy())
        ranked = s.rank(["a pump", "a laptop"])
        assert ranked.shape == (2, 3)
        for row in ranked:
            assert sorted(row) == [0, 1, 2]

    def test_tfidf_ranks_a_lexical_match_first(self):
        # No stub encoder: this retriever must work on characters alone, and the point of
        # measuring it is that the embedding one may not put the answer near the top.
        s = TfidfShortlister(min_df=1)
        s.fit(self.taxonomy())
        assert s.rank(["industrial pumps"])[0][0] == 2

    def test_corpus_text_widens_the_vocabulary_without_touching_labels(self):
        # The corpus contributes document frequencies only; a retriever that saw the labels
        # would not be a retriever, it would be a classifier.
        corpus = pd.DataFrame(
            {"title": ["centrifugal pump overhaul"], "description": [""],
             "cpv_code": ["42100000"]}
        )
        s = TfidfShortlister(min_df=1)
        s.fit(self.taxonomy(), corpus=corpus)
        assert s.rank(["centrifugal pump overhaul"])[0][0] == 2


class TestRecallAtK:
    def test_counts_a_hit_only_inside_k(self):
        ranked = np.array([[2, 0, 1], [0, 1, 2]])
        got = recall_at_k(ranked, np.array([0, 0]), [1, 2, 3])
        assert got["recall"][1] == 0.5
        assert got["recall"][2] == 1.0
        assert got["recall"][3] == 1.0

    def test_a_code_absent_from_the_pool_counts_against_recall(self):
        # Never silently excluded: a true code the retriever cannot offer is exactly the
        # failure this measurement exists to expose.
        got = recall_at_k(np.array([[0, 1]]), np.array([-1]), [1, 2])
        assert got["recall"][2] == 0.0
        assert got["n_not_in_pool"] == 1
        assert got["mean_rank_when_found"] is None

    def test_mean_rank_is_one_based_and_over_found_rows_only(self):
        got = recall_at_k(np.array([[0, 1], [1, 0]]), np.array([0, -1]), [2])
        assert got["mean_rank_when_found"] == 1.0

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="rankings against"):
            recall_at_k(np.array([[0, 1]]), np.array([0, 1]), [1])
