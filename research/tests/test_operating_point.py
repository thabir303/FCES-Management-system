"""Tests for the precision–automation trade-off (§6.13, C8).

C8's acceptance criterion is that ``automated_share_at_precision`` recovers a known answer
on a synthetic fixture. ``TestKnownFixture`` is that criterion; the rest guard the
properties the headline result rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

from fcesreg.metrics import prf1
from fcesreg.operating_point import (
    DEFAULT_TARGET,
    automated_share_at_precision,
    precision_automation_curve,
    residual_effort,
)


def _hand_built(n_head_pos: int, n_mid_neg: int, n_tail_neg: int):
    """A well-separated scorer whose operating points are computable by hand.

    ``n_head_pos`` positives score highest, then ``n_mid_neg`` negatives, then
    ``n_tail_neg`` negatives lower still. Every score is distinct, so each item is its own
    attainable threshold and precision falls monotonically as the threshold drops — which
    makes the answer to "the lowest threshold holding target P" arithmetic rather than a
    search.
    """
    scores = np.concatenate([
        np.linspace(1.00, 0.90, n_head_pos),
        np.linspace(0.85, 0.81, n_mid_neg),
        np.linspace(0.80, 0.10, n_tail_neg),
    ])
    labels = np.array([1] * n_head_pos + [0] * (n_mid_neg + n_tail_neg))
    return scores, labels


class TestKnownFixture:
    """C8's stated acceptance criterion, under the Wilson floor."""

    def test_recovers_a_hand_computed_answer(self):
        # 200 items: 100 positives scoring above 100 negatives. The floor is the lower
        # bound of a one-sided 95% Wilson interval, so the selected point is the lowest one
        # whose bound still clears 0.95. That admits 101 items -- all 100 positives plus the
        # first negative, point estimate 100/101 = 0.9901, bound 0.9568. Admitting one more
        # gives 100/102 = 0.9804 and a bound of 0.9425, which fails. So the answer is
        # 101/200 = 0.505, and it tolerates exactly one false positive.
        scores, labels = _hand_built(n_head_pos=100, n_mid_neg=100, n_tail_neg=0)

        threshold, share = automated_share_at_precision(scores, labels, target=0.95)

        assert share == pytest.approx(101 / 200)
        delivered = prf1(labels, (scores >= threshold).astype(int))
        assert delivered["tp"] == 100 and delivered["fp"] == 1

    def test_a_stricter_target_automates_less(self):
        scores, labels = _hand_built(n_head_pos=100, n_mid_neg=100, n_tail_neg=0)
        _, at_95 = automated_share_at_precision(scores, labels, target=0.95)
        threshold_99, at_99 = automated_share_at_precision(scores, labels, target=0.99)
        # 0.99 needs 268 accepted items even if every one is correct, and only 100 positives
        # exist here. Unreachable is the honest answer, not a smaller share.
        assert np.isnan(threshold_99) and at_99 == 0.0
        assert at_95 > 0.0

    def test_a_point_estimate_that_clears_the_target_is_not_enough(self):
        # The whole reason for the rule. Three positives above everything else give a point
        # estimate of 1.000 at a share of 3/203, which the old rule quoted as an operating
        # point. Three items evidence nothing.
        scores = np.concatenate([np.array([1.0, 0.99, 0.98]), np.linspace(0.5, 0.1, 200)])
        labels = np.array([1, 1, 1] + [0] * 200)
        threshold, share = automated_share_at_precision(scores, labels, target=0.95)
        assert np.isnan(threshold) and share == 0.0

    def test_default_target_is_the_headline_floor(self):
        assert DEFAULT_TARGET == 0.95


class TestPromisedPrecisionIsDelivered:
    """The property the whole operating point rests on."""

    def test_threshold_delivers_at_least_the_target(self):
        rng = np.random.default_rng(7)
        for _ in range(25):
            scores = rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=120)
            labels = rng.integers(0, 2, size=120)
            threshold, share = automated_share_at_precision(scores, labels, target=0.7)
            if np.isnan(threshold):
                continue
            delivered = prf1(labels, (scores >= threshold).astype(int))["precision"]
            assert delivered >= 0.7 - 1e-12

    def test_tied_scores_cannot_be_split_by_a_threshold(self):
        # The exact matcher's shape: only 1.0 and 0.0. A threshold of 1.0 admits all three
        # top items, delivering 2/3, so no threshold reaches 0.95 and the honest answer is
        # that this method cannot be operated there.
        scores = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        labels = np.array([1, 1, 0, 0, 0])
        threshold, share = automated_share_at_precision(scores, labels, target=0.95)
        assert np.isnan(threshold) and share == 0.0

    def test_unreachable_target_is_a_finding_not_an_error(self):
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array([0, 0, 0])
        threshold, share = automated_share_at_precision(scores, labels, target=0.95)
        assert np.isnan(threshold)
        assert share == 0.0

    def test_target_outside_the_unit_interval_is_refused(self):
        scores, labels = _hand_built(5, 1, 5)
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="target must be in"):
                automated_share_at_precision(scores, labels, target=bad)


class TestCurve:
    def test_columns_are_as_specified(self):
        scores, labels = _hand_built(5, 2, 3)
        curve = precision_automation_curve(scores, labels)
        assert list(curve.columns) == [
            "threshold", "precision", "precision_lower", "recall", "automated_share",
        ]

    def test_the_lower_bound_never_exceeds_the_point_estimate(self):
        scores, labels = _hand_built(20, 10, 10)
        curve = precision_automation_curve(scores, labels)
        assert (curve["precision_lower"] <= curve["precision"] + 1e-12).all()

    def test_the_selected_point_is_the_lowest_row_clearing_the_bound(self):
        # Ties the curve's precision_lower column to what the selector actually does, so a
        # reader of the figure can see the floor being applied rather than inferring it.
        scores, labels = _hand_built(n_head_pos=100, n_mid_neg=100, n_tail_neg=0)
        threshold, _ = automated_share_at_precision(scores, labels, target=0.95)
        curve = precision_automation_curve(scores, labels)
        qualifying = curve[curve["precision_lower"] >= 0.95]
        assert qualifying["threshold"].min() == pytest.approx(threshold)

    def test_automated_share_rises_as_the_threshold_falls(self):
        scores, labels = _hand_built(5, 2, 3)
        curve = precision_automation_curve(scores, labels)
        assert curve["threshold"].is_monotonic_decreasing
        assert curve["automated_share"].is_monotonic_increasing

    def test_the_curve_ends_at_full_automation(self):
        scores, labels = _hand_built(5, 2, 3)
        curve = precision_automation_curve(scores, labels)
        assert curve["automated_share"].iloc[-1] == pytest.approx(1.0)
        assert curve["recall"].iloc[-1] == pytest.approx(1.0)

    def test_no_zero_coverage_row(self):
        # The precision of an empty selection is undefined, not perfect. A row claiming
        # precision 1.0 at automated_share 0.0 would be read off the figure as an
        # achievable operating point.
        scores, labels = _hand_built(5, 2, 3)
        curve = precision_automation_curve(scores, labels)
        assert (curve["automated_share"] > 0).all()

    def test_curve_and_point_agree(self):
        scores, labels = _hand_built(n_head_pos=100, n_mid_neg=100, n_tail_neg=0)
        threshold, share = automated_share_at_precision(scores, labels, target=0.95)
        curve = precision_automation_curve(scores, labels)
        row = curve[np.isclose(curve["threshold"], threshold)]
        assert len(row) == 1
        assert row["automated_share"].iloc[0] == pytest.approx(share)
        assert row["precision"].iloc[0] >= 0.95

    def test_recall_is_nan_when_there_is_nothing_to_recall(self):
        # Unmeasured is not estimated: an inferred 0.0 would read as a measurement.
        curve = precision_automation_curve([0.9, 0.5], [0, 0])
        assert curve["recall"].isna().all()

    def test_every_row_reports_a_precision_a_threshold_delivers(self):
        rng = np.random.default_rng(11)
        scores = rng.choice([0.1, 0.4, 0.9], size=150)
        labels = rng.integers(0, 2, size=150)
        curve = precision_automation_curve(scores, labels)
        for _, row in curve.iterrows():
            delivered = prf1(labels, (scores >= row["threshold"]).astype(int))
            assert delivered["precision"] == pytest.approx(row["precision"])
            assert delivered["recall"] == pytest.approx(row["recall"])


class TestResidualEffort:
    def test_hours_reconcile(self):
        got = residual_effort(1000, 0.75, 60.0)
        assert got["baseline_hours"] == pytest.approx(1000 * 60 / 3600)
        assert got["residual_hours"] == pytest.approx(got["baseline_hours"] * 0.25)
        assert got["hours_saved"] + got["residual_hours"] == pytest.approx(
            got["baseline_hours"]
        )

    def test_full_automation_leaves_nothing(self):
        got = residual_effort(500, 1.0, 45.0)
        assert got["residual_hours"] == pytest.approx(0.0)
        assert got["hours_saved"] == pytest.approx(got["baseline_hours"])

    def test_no_automation_saves_nothing(self):
        got = residual_effort(500, 0.0, 45.0)
        assert got["hours_saved"] == pytest.approx(0.0)
        assert got["residual_hours"] == pytest.approx(got["baseline_hours"])

    def test_the_inputs_are_carried_into_the_result(self):
        # A run record must be able to show what produced the number, not just the number.
        got = residual_effort(1000, 0.75, 60.0)
        assert got["n_records"] == 1000
        assert got["automated_share"] == 0.75
        assert got["mean_seconds_per_item"] == 60.0

    def test_handling_time_has_no_default(self):
        with pytest.raises(TypeError):
            residual_effort(1000, 0.75)  # type: ignore[call-arg]

    def test_a_non_positive_handling_time_is_refused(self):
        with pytest.raises(ValueError, match="measured by the timed annotation"):
            residual_effort(1000, 0.75, 0.0)

    def test_a_share_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValueError, match="automated_share must be in"):
            residual_effort(1000, 1.5, 60.0)

    def test_a_negative_record_count_is_refused(self):
        with pytest.raises(ValueError, match="n_records must be non-negative"):
            residual_effort(-1, 0.5, 60.0)
