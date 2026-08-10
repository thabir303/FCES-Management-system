"""Tests for the scoring primitives (§6.12, C8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fcesreg.metrics import confusion, macro_weighted_f1, prf1, threshold_sweep


class TestPrf1:
    def test_counts_account_for_every_pair(self):
        got = prf1([1, 1, 0, 0], [1, 0, 1, 0])
        assert got["tp"] + got["fp"] + got["fn"] + got["tn"] == got["n_pairs"] == 4

    def test_perfect_prediction(self):
        got = prf1([1, 0, 1], [1, 0, 1])
        assert (got["precision"], got["recall"], got["f1"]) == (1.0, 1.0, 1.0)

    def test_predicting_nothing_is_zero_not_undefined(self):
        got = prf1([1, 1, 0], [0, 0, 0])
        assert got["precision"] == 0.0 and got["recall"] == 0.0 and got["f1"] == 0.0

    def test_shape_mismatch_is_loud(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            prf1([1, 0], [1, 0, 1])


class TestMacroWeightedF1:
    def test_macro_and_weighted_diverge_on_a_skewed_distribution(self):
        # 8 items of class "30", 2 of class "38". The classifier learns only the head.
        y_true = ["30"] * 8 + ["38"] * 2
        y_pred = ["30"] * 10

        got = macro_weighted_f1(y_true, y_pred, ["30", "38"])
        # Head class: P=0.8, R=1.0 -> F1 = 0.888...; tail class: F1 = 0.0
        assert got["macro_f1"] == pytest.approx(0.4444, abs=1e-3)
        assert got["weighted_f1"] == pytest.approx(0.7111, abs=1e-3)
        # The divergence is the point: reporting either alone would mislead.
        assert got["weighted_f1"] > got["macro_f1"]

    def test_perfect_prediction_scores_one_on_both(self):
        got = macro_weighted_f1(["a", "b", "a"], ["a", "b", "a"], ["a", "b"])
        assert got["macro_f1"] == 1.0 and got["weighted_f1"] == 1.0

    def test_a_declared_label_with_no_support_drags_macro_down_and_is_counted(self):
        # "99" is in the supported label set but absent from this split entirely.
        got = macro_weighted_f1(["a", "b"], ["a", "b"], ["a", "b", "99"])
        assert got["labels_without_support"] == 1
        assert got["macro_f1"] == pytest.approx(2 / 3)
        # Weighted ignores it, because it weights by support and its support is zero.
        assert got["weighted_f1"] == 1.0

    def test_a_true_label_outside_the_declared_set_is_refused(self):
        # Silently dropping it would remove real items and improve the score.
        with pytest.raises(ValueError, match=r"y_true contains 1 label"):
            macro_weighted_f1(["a", "zz"], ["a", "a"], ["a"])

    def test_a_predicted_label_outside_the_declared_set_is_refused(self):
        with pytest.raises(ValueError, match=r"y_pred contains 1 label"):
            macro_weighted_f1(["a", "a"], ["a", "zz"], ["a"])

    def test_duplicate_labels_are_refused(self):
        with pytest.raises(ValueError, match="duplicates"):
            macro_weighted_f1(["a"], ["a"], ["a", "a"])

    def test_empty_label_set_is_refused(self):
        with pytest.raises(ValueError, match="nothing to average"):
            macro_weighted_f1(["a"], ["a"], [])

    def test_per_class_support_sums_to_the_item_count(self):
        y_true = ["a", "a", "b", "c"]
        got = macro_weighted_f1(y_true, ["a", "b", "b", "c"], ["a", "b", "c"])
        assert sum(v["support"] for v in got["per_class"].values()) == got["n_items"] == 4


class TestConfusion:
    def test_diagonal_carries_the_correct_predictions(self):
        matrix = confusion(["a", "b", "a"], ["a", "b", "b"], ["a", "b"])
        assert matrix.loc["a", "a"] == 1
        assert matrix.loc["a", "b"] == 1
        assert matrix.loc["b", "b"] == 1
        assert matrix.loc["b", "a"] == 0

    def test_total_equals_the_item_count(self):
        matrix = confusion(["a", "b", "c", "a"], ["b", "b", "c", "a"], ["a", "b", "c"])
        assert matrix.to_numpy().sum() == 4

    def test_rows_are_true_and_columns_are_predicted(self):
        # One item, truly "a", predicted "b". If the axes were swapped this would land
        # at [b, a] instead -- a confusion matrix read the wrong way round inverts every
        # claim made from it.
        matrix = confusion(["a"], ["b"], ["a", "b"])
        assert matrix.loc["a", "b"] == 1
        assert matrix.loc["b", "a"] == 0

    def test_label_order_is_preserved_not_sorted(self):
        matrix = confusion(["b"], ["b"], ["b", "a"])
        assert list(matrix.index) == ["b", "a"]
        assert list(matrix.columns) == ["b", "a"]


class TestThresholdSweep:
    """The tie-collapsing property, which is what makes a promised precision deliverable."""

    def test_one_row_per_distinct_score_not_per_item(self):
        sweep = threshold_sweep([1.0, 1.0, 1.0, 0.0], [1, 1, 0, 0])
        assert len(sweep.threshold) == 2
        assert list(sweep.threshold) == [1.0, 0.0]

    def test_a_tie_group_is_admitted_whole(self):
        # Three items score 1.0; the only attainable point over them admits all three.
        sweep = threshold_sweep([1.0, 1.0, 1.0, 0.0], [1, 1, 0, 0])
        assert sweep.n_selected[0] == 3
        assert sweep.tp[0] == 2
        assert sweep.precision[0] == pytest.approx(2 / 3)

    def test_precision_reported_is_precision_delivered(self):
        # The regression guard. For every attainable threshold, applying it must reproduce
        # the precision the sweep reported -- this is exactly what a per-index sweep breaks.
        rng = np.random.default_rng(0)
        scores = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=200)
        labels = rng.integers(0, 2, size=200)

        sweep = threshold_sweep(scores, labels)
        for t, reported in zip(sweep.threshold, sweep.precision, strict=True):
            delivered = prf1(labels, (scores >= t).astype(int))["precision"]
            assert delivered == pytest.approx(reported)

    def test_counts_are_cumulative_and_monotone(self):
        sweep = threshold_sweep([0.9, 0.8, 0.7], [1, 0, 1])
        assert list(sweep.n_selected) == [1, 2, 3]
        assert list(sweep.tp) == [1, 1, 2]

    def test_empty_input_is_refused(self):
        with pytest.raises(ValueError, match="no items"):
            threshold_sweep([], [])

    def test_nan_scores_are_refused(self):
        with pytest.raises(ValueError, match="nan or inf"):
            threshold_sweep([0.5, np.nan], [1, 0])

    def test_non_binary_labels_are_refused(self):
        with pytest.raises(ValueError, match="must be 0 or 1"):
            threshold_sweep([0.5, 0.6], [1, 2])

    def test_shape_mismatch_is_loud(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            threshold_sweep([0.5, 0.6], [1])
