"""C3 acceptance: both matchers run end to end and their counts reconcile.

The F1 floor of 0.40 is a bug detector, not a target: below it, the pair set and the
record set have almost certainly been built wrong. The measured value is recorded by the
runner as an observation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import CORPUS_A, CORPUS_A_PAIRS, SPLITS, requires
from fcesreg.dedup import (
    ExactMatcher,
    Matcher,
    TfidfMatcher,
    score_to_prediction,
    select_threshold,
)
from fcesreg.metrics import prf1


def records(rows):
    return pd.DataFrame(
        [{"record_id": r[0], "title": r[1], "description": r[2]} for r in rows]
    )


def pairs(rows):
    return pd.DataFrame(rows, columns=["left_id", "right_id"])


class TestProtocol:
    def test_both_matchers_satisfy_it(self):
        assert isinstance(ExactMatcher(), Matcher)
        assert isinstance(TfidfMatcher(), Matcher)

    def test_scores_align_one_per_pair(self):
        r = records([("1", "zeiss microscope", "a"), ("2", "zeiss microscope", "a")])
        p = pairs([("1", "2")])
        for m in (ExactMatcher(), TfidfMatcher()):
            assert m.score_pairs(p, r).shape == (1,)

    def test_pair_referencing_an_unknown_record_is_a_loud_error(self):
        r = records([("1", "x", "y")])
        with pytest.raises(KeyError, match="disagree"):
            ExactMatcher().score_pairs(pairs([("1", "999")]), r)


class TestExactMatcher:
    def test_ignores_casing_spacing_and_punctuation(self):
        r = records(
            [
                ("1", "Zeiss Axio-Lab A1", "Microscope."),
                ("2", "zeiss  axio lab a1", "microscope"),
            ]
        )
        assert ExactMatcher().score_pairs(pairs([("1", "2")]), r)[0] == 1.0

    def test_genuine_content_difference_is_not_a_match(self):
        r = records([("1", "zeiss microscope", "a"), ("2", "leica microscope", "a")])
        assert ExactMatcher().score_pairs(pairs([("1", "2")]), r)[0] == 0.0

    def test_scores_are_binary(self):
        r = records(
            [("1", "aa", "bb"), ("2", "aa", "bb"), ("3", "aa", "cc")]
        )
        s = ExactMatcher().score_pairs(pairs([("1", "2"), ("1", "3")]), r)
        assert set(np.unique(s)) <= {0.0, 1.0}


class TestTfidfMatcher:
    def test_similar_text_scores_higher_than_dissimilar(self):
        r = records(
            [
                ("1", "rotary vane vacuum pump", "laboratory"),
                ("2", "rotary vane vacuum pumps", "laboratory"),
                ("3", "office chair", "furniture"),
            ]
        )
        s = TfidfMatcher().score_pairs(pairs([("1", "2"), ("1", "3")]), r)
        assert s[0] > s[1]

    def test_scores_are_cosines_in_range(self):
        r = records([("1", "abc def", "x"), ("2", "abc deg", "y")])
        s = TfidfMatcher().score_pairs(pairs([("1", "2")]), r)
        assert 0.0 <= s[0] <= 1.0 + 1e-9

    def test_identical_text_scores_one(self):
        r = records([("1", "same text here", "d"), ("2", "same text here", "d")])
        assert TfidfMatcher().score_pairs(pairs([("1", "2")]), r)[0] == pytest.approx(1.0)

    def test_character_features_survive_a_typo(self):
        # A word-level model would see "micrsocope" as an unrelated token.
        r = records(
            [("1", "zeiss microscope", "x"), ("2", "zeiss micrsocope", "x"),
             ("3", "canon camera", "y")]
        )
        s = TfidfMatcher().score_pairs(pairs([("1", "2"), ("1", "3")]), r)
        # A transposition costs some similarity but nothing like as much as an unrelated
        # record does. The ordering is the claim; the absolute value depends on how many
        # documents the IDF was fitted over.
        assert s[0] > 5 * s[1]


class TestSelectThreshold:
    def _separated(self, n_pos: int, n_neg: int):
        """``n_pos`` positives scoring above ``n_neg`` negatives, every score distinct."""
        scores = np.concatenate(
            [np.linspace(1.0, 0.6, n_pos), np.linspace(0.5, 0.1, n_neg)]
        )
        return scores, np.array([1] * n_pos + [0] * n_neg)

    def test_recovers_a_known_answer(self):
        # 80 clean positives above every negative. The lowest qualifying threshold is the
        # last positive, and admitting the first negative must break it.
        scores, labels = self._separated(80, 80)
        t = select_threshold(scores, labels, 0.95)
        assert t == pytest.approx(0.6)
        assert prf1(labels, score_to_prediction(scores, t))["precision"] == 1.0

    def test_takes_the_lowest_threshold_meeting_the_target(self):
        # Among qualifying thresholds the useful one automates the most work, so the
        # selected point must be the lowest, not the highest.
        scores, labels = self._separated(80, 80)
        t = select_threshold(scores, labels, 0.95)
        admitted = int((scores >= t).sum())
        assert admitted == 80  # every positive, not just the top few

    def test_a_threshold_supported_by_too_little_evidence_does_not_qualify(self):
        # The defect this rule exists for. Three positives above everything else gives a
        # point estimate of 1.000, but three items evidence nothing.
        scores = np.concatenate([np.array([1.0, 0.99, 0.98]), np.linspace(0.5, 0.1, 200)])
        labels = np.array([1, 1, 1] + [0] * 200)
        assert select_threshold(scores, labels, 0.95) == float("inf")

    def test_the_evidence_demanded_scales_with_the_target(self):
        # 60 clean positives clears 0.95 (needs 52) but not 0.99 (needs 268).
        scores, labels = self._separated(60, 200)
        assert np.isfinite(select_threshold(scores, labels, 0.95))
        assert select_threshold(scores, labels, 0.99) == float("inf")

    def test_a_target_of_one_is_unreachable_at_any_sample_size(self):
        # A finite run of correct decisions never evidences certainty. Documented rather
        # than special-cased: 0.95 and 0.99 are the targets the paper reports.
        scores, labels = self._separated(5000, 5000)
        assert select_threshold(scores, labels, 1.0) == float("inf")

    def test_unreachable_target_returns_inf_not_a_fallback(self):
        # A finding, not an error: the caller reports that no threshold reaches it
        # rather than quietly settling for a lower precision.
        scores = np.array([0.9, 0.8])
        labels = np.array([0, 0])
        assert select_threshold(scores, labels, 0.95) == float("inf")

    def test_inf_threshold_predicts_nothing(self):
        assert score_to_prediction(np.array([0.9, 1.0]), float("inf")).sum() == 0

    def test_a_tie_group_cannot_be_split_by_a_threshold(self):
        # Regression. A sweep over individual items stops after the first 0.9 and reports
        # precision 1.0, but a threshold of 0.9 admits both 0.9s and delivers 0.5. The
        # honest answer is that no threshold reaches 0.95 on this data.
        scores = np.array([0.9, 0.9, 0.8])
        labels = np.array([1, 0, 1])
        assert select_threshold(scores, labels, 0.95) == float("inf")

    def test_the_promised_precision_is_the_precision_delivered(self):
        # ExactMatcher emits only 1.0 and 0.0, so every one of its pairs is tied with most
        # of the others -- the shape where splitting a tie group is not a corner case but
        # the normal case.
        rng = np.random.default_rng(3)
        for _ in range(30):
            scores = rng.choice([0.0, 1.0], size=60).astype(float)
            labels = rng.integers(0, 2, size=60)
            t = select_threshold(scores, labels, 0.8)
            if not np.isfinite(t):
                continue
            delivered = prf1(labels, score_to_prediction(scores, t))["precision"]
            assert delivered >= 0.8 - 1e-12


class TestPrf1:
    def test_counts_account_for_every_pair(self):
        m = prf1([1, 1, 0, 0], [1, 0, 1, 0])
        assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == m["n_pairs"] == 4

    def test_known_values(self):
        m = prf1([1, 1, 0, 0], [1, 0, 1, 0])
        assert m["precision"] == 0.5
        assert m["recall"] == 0.5
        assert m["f1"] == 0.5

    def test_no_predictions_is_zero_not_undefined(self):
        m = prf1([1, 0], [0, 0])
        assert m["precision"] == 0.0 and m["f1"] == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            prf1([1, 0], [1])


@requires(CORPUS_A, CORPUS_A_PAIRS, SPLITS)
class TestOnAbtBuyTestSplit:
    """C3's acceptance criterion, against the real supplied splits."""

    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        from fcesreg.splits import load

        rec = pd.read_parquet(CORPUS_A)
        p = pd.read_parquet(CORPUS_A_PAIRS)
        sp = load()
        return rec, sp.abtbuy(p, "dev"), sp.abtbuy(p, "test")

    def test_counts_reconcile_for_both_matchers(self, data):
        rec, _, test = data
        for m in (ExactMatcher(), TfidfMatcher()):
            scores = m.score_pairs(test, rec)
            got = prf1(test["label"], score_to_prediction(scores, 0.5))
            assert got["tp"] + got["fp"] + got["fn"] + got["tn"] == len(test)

    def test_tfidf_clears_the_bug_detector_floor(self, data):
        rec, dev, test = data
        matcher = TfidfMatcher()
        dev_scores = matcher.score_pairs(dev, rec)

        best_t, best_f1 = 0.5, 0.0
        for t in np.unique(np.round(dev_scores, 3)):
            f1 = prf1(dev["label"], score_to_prediction(dev_scores, t))["f1"]
            if f1 > best_f1:
                best_t, best_f1 = t, f1

        test_f1 = prf1(
            test["label"], score_to_prediction(matcher.score_pairs(test, rec), best_t)
        )["f1"]
        assert test_f1 >= 0.40, (
            f"F1 {test_f1:.3f} below the bug-detector floor — suspect pair construction"
        )

    def test_exact_match_finds_nothing_and_that_is_not_a_bug(self, data):
        # Abt-Buy pairs are the same product listed by two retailers under different
        # names, so nothing is trivially solvable. TF-IDF scoring above the floor on the
        # very same pairs is what distinguishes this from a construction fault.
        rec, _, test = data
        scores = ExactMatcher().score_pairs(test, rec)
        assert scores.sum() == 0
