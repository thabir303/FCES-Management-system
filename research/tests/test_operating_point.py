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
    band_operating_point,
    precision_automation_curve,
    reject_bound,
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


class TestBandOperatingPoint:
    """The two-bound rule the duplicate pipeline actually uses.

    The single-threshold model counted every obvious non-duplicate as outstanding human
    work and put the automated share at 1.7% where it is above 99%. These tests pin the
    difference so it cannot come back.
    """

    def separable(self, n_pos: int = 200, n_neg: int = 800):
        # Positives high, negatives low, well separated: both bounds are attainable, so
        # the band is small and both ends of the automation are exercised.
        scores = np.concatenate(
            [np.linspace(0.75, 1.0, n_pos), np.linspace(0.0, 0.25, n_neg)]
        )
        labels = np.concatenate([np.ones(n_pos, int), np.zeros(n_neg, int)])
        return scores, labels

    def test_automation_counts_both_ends(self):
        got = band_operating_point(*self.separable())
        # The point of the rule: rejects dominate, and ignoring them loses almost all of it.
        assert got["n_auto_rejected"] > got["n_auto_accepted"]
        assert got["automated_share"] > 0.9

    def test_the_three_outcomes_partition_the_items(self):
        got = band_operating_point(*self.separable())
        assert (
            got["n_auto_accepted"] + got["n_auto_rejected"] + got["n_band"]
            == got["n_items"]
        )

    def test_automated_share_is_one_minus_the_band(self):
        got = band_operating_point(*self.separable())
        assert got["automated_share"] == pytest.approx(1.0 - got["band_fraction"])

    def test_both_ends_hold_the_floor_they_were_fitted_at(self):
        got = band_operating_point(*self.separable(), 0.95)
        assert got["precision_auto_accepted"] >= 0.95
        assert got["purity_auto_rejected"] >= 0.95

    def test_crossed_bounds_are_flagged_not_silently_resolved(self):
        # A well-separated scorer makes both rules certify the same middle region: each is
        # computed against its own denominator, so both can hold at once. The overlap is
        # resolved to a single cut and the crossing is recorded, because an automated share
        # computed from contradictory bounds would exceed the number of items.
        got = band_operating_point(*self.separable(), 0.95)
        assert got["bounds_crossed"]
        assert got["lower"] == got["upper"]
        assert got["n_band"] == 0

    def test_uncrossed_bounds_are_not_flagged(self):
        # Guards the guard: a fixture where the rules genuinely leave a band must not trip
        # the flag, or the flag would say nothing.
        scores = np.concatenate([np.linspace(0.4, 1.0, 200), np.linspace(0.0, 0.6, 800)])
        labels = np.concatenate([np.ones(200, int), np.zeros(800, int)])
        got = band_operating_point(scores, labels, 0.95)
        assert not got["bounds_crossed"]
        assert got["n_band"] > 0

    def test_a_higher_floor_never_automates_more(self):
        scores, labels = self.separable()
        shares = [
            band_operating_point(scores, labels, t)["automated_share"]
            for t in (0.80, 0.90, 0.95, 0.99)
        ]
        assert shares == sorted(shares, reverse=True)

    def test_an_unreachable_end_is_reported_not_faked(self):
        # Nothing separable: no threshold can confidently accept, so upper is undefined and
        # the accepted set is empty rather than being filled to make the number look better.
        scores = np.full(100, 0.5)
        labels = np.array([i % 2 for i in range(100)])
        got = band_operating_point(scores, labels, 0.99)
        assert got["upper_undefined"]
        assert got["n_auto_accepted"] == 0
        assert got["precision_auto_accepted"] is None

    def test_automation_can_be_entirely_rejection(self):
        # The headline shape of the real result: at a high floor nothing is confidently a
        # duplicate, yet most pairs are still resolved without a human.
        scores = np.concatenate([np.linspace(0.4, 0.6, 50), np.zeros(950)])
        labels = np.concatenate([np.ones(50, int), np.zeros(950, int)])
        got = band_operating_point(scores, labels, 0.95)
        assert got["n_auto_accepted"] == 0
        assert got["n_auto_rejected"] > 0
        assert got["automated_share"] > 0.0


class TestRejectBound:
    def test_nothing_confidently_negative_returns_minus_inf(self):
        # Half positive at every score: no rejected set is confidently negative, and an
        # invented bound would silently discard real duplicates.
        labels = np.array([i % 2 for i in range(100)])
        assert reject_bound(np.full(100, 0.5), labels, 0.95) == float("-inf")

    def test_the_rejected_set_meets_the_floor(self):
        scores = np.concatenate([np.linspace(0.8, 1.0, 100), np.linspace(0.0, 0.2, 400)])
        labels = np.concatenate([np.ones(100, int), np.zeros(400, int)])
        bound = reject_bound(scores, labels, 0.95)
        rejected = scores < bound
        assert rejected.any()
        assert 1.0 - labels[rejected].mean() >= 0.95


class TestResidualEffort:
    """Review VOLUME, not hours (ruled 2026-08-17). Handling time is not measured anywhere
    in this project, so it must not be an input here — and it must not be reachable through
    a default either, which is what these tests are mostly guarding."""

    def test_volumes_reconcile(self):
        got = residual_effort(1000, 0.75)
        assert got["n_review"] == 250
        assert got["n_automated"] == 750
        assert got["n_automated"] + got["n_review"] == got["baseline_review"] == 1000

    def test_full_automation_leaves_nothing_to_review(self):
        got = residual_effort(500, 1.0)
        assert got["n_review"] == 0
        assert got["n_automated"] == 500

    def test_no_automation_reviews_everything(self):
        got = residual_effort(500, 0.0)
        assert got["n_review"] == 500
        assert got["n_automated"] == 0

    def test_it_takes_no_handling_time_in_any_position(self):
        # The whole point of the ruling: there is no argument to pass one into.
        with pytest.raises(TypeError):
            residual_effort(1000, 0.75, 60.0)  # type: ignore[call-arg]

    def test_it_reports_no_hours_under_any_key(self):
        # A key named *hours would be an assumed handling time reaching a result, which is
        # exactly what reporting volume exists to prevent.
        got = residual_effort(1000, 0.75)
        assert not [k for k in got if "hour" in k or "second" in k]

    def test_it_carries_the_conversion_as_a_formula(self):
        # Declined, not forgotten: a reader with their own handling time can convert, and
        # can see that this project did not.
        got = residual_effort(1000, 0.75)
        assert "n_review" in got["effort_formula"] and "3600" in got["effort_formula"]

    def test_the_reduction_is_a_ratio_and_needs_no_handling_time(self):
        # Why the volume framing costs nothing: the proportional saving is the same number
        # whatever t turns out to be.
        assert residual_effort(1000, 0.75)["volume_reduction"] == 0.75
        assert residual_effort(37, 0.75)["volume_reduction"] == 0.75

    def test_the_inputs_are_carried_into_the_result(self):
        # A run record must be able to show what produced the number, not just the number.
        got = residual_effort(1000, 0.75)
        assert got["n_records"] == 1000
        assert got["automated_share"] == 0.75

    def test_a_share_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValueError, match="automated_share must be in"):
            residual_effort(1000, 1.5)

    def test_a_negative_record_count_is_refused(self):
        with pytest.raises(ValueError, match="n_records must be non-negative"):
            residual_effort(-1, 0.5)
