"""Tests for the parts of judge_distractors.py that are not the interactive loop.

The 50-pair sample and its Wilson interval are numbers that reach the paper, so — unlike
the earlier full-verification design, which was smoke-tested only — the sampling and the
interval math get real tests. The interactive `input()` loop is not covered here; it is
exercised by hand, as it always has been.

Not wired into `make test`: `annotation/` is not a package under `research/tests`'
discovery scope, and nothing else in the repo runs tests from here. Invoke directly:

    .venv/bin/pytest annotation/test_judge_distractors.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from judge_distractors import CONTAMINATED_VERDICTS, _summary, draw_sample, wilson_interval


class TestWilsonInterval:
    def test_matches_a_known_reference_value(self):
        # n=50, x=17 (34%) is a standard textbook check for the Wilson interval.
        rate, lower, upper = wilson_interval(17, 50)
        assert rate == pytest.approx(0.34)
        assert lower == pytest.approx(0.2243, abs=1e-3)
        assert upper == pytest.approx(0.4785, abs=1e-3)

    def test_bounds_never_escape_zero_one(self):
        rate, lower, upper = wilson_interval(0, 50)
        assert lower == 0.0
        assert 0.0 <= upper <= 1.0

        rate, lower, upper = wilson_interval(50, 50)
        assert upper == 1.0
        assert 0.0 <= lower <= 1.0

    def test_interval_widens_as_n_shrinks(self):
        # Same observed rate, less evidence — the interval should not get more confident.
        _, lower_50, upper_50 = wilson_interval(17, 50)
        _, lower_10, upper_10 = wilson_interval(3, 10)  # ~30%, close to 34%
        assert (upper_10 - lower_10) > (upper_50 - lower_50)

    def test_zero_judged_pairs_is_an_error_not_a_silent_nan(self):
        with pytest.raises(ValueError, match="zero judged pairs"):
            wilson_interval(0, 0)

    def test_wald_would_have_gone_negative_here_wilson_does_not(self):
        # The documented reason Wilson is used over the naive normal approximation: a low
        # rate at small n pushes the crude interval's lower bound below zero.
        rate, lower, _ = wilson_interval(2, 50)  # 4%
        wald_lower = rate - 1.96 * ((rate * (1 - rate) / 50) ** 0.5)
        assert wald_lower < 0
        assert lower >= 0.0


class TestDrawSample:
    def _pool(self, n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {"left_id": [f"l{i}" for i in range(n)], "right_id": [f"r{i}" for i in range(n)]}
        )

    def test_same_seed_draws_the_same_sample(self):
        pool = self._pool(500)
        first = draw_sample(pool, 50, seed=7)
        second = draw_sample(pool, 50, seed=7)
        pd.testing.assert_frame_equal(first, second)

    def test_different_seeds_draw_different_samples(self):
        pool = self._pool(500)
        a = draw_sample(pool, 50, seed=1)
        b = draw_sample(pool, 50, seed=2)
        assert set(a["left_id"]) != set(b["left_id"])

    def test_sample_size_is_respected(self):
        pool = self._pool(500)
        assert len(draw_sample(pool, 50, seed=0)) == 50

    def test_a_pool_smaller_than_the_sample_size_returns_the_whole_pool(self):
        pool = self._pool(10)
        sample = draw_sample(pool, 50, seed=0)
        assert len(sample) == 10

    def test_drawn_without_replacement(self):
        pool = self._pool(500)
        sample = draw_sample(pool, 50, seed=3)
        assert sample["left_id"].is_unique


class TestSampleScopedSummary:
    """The bug this guards against: a judgements file accumulates rows across every seed
    ever run against it. A later --seed drawing a different sample must not have its
    contamination rate diluted — or its completion state corrupted — by rows left over
    from an earlier seed's sample."""

    def _sample(self, ids: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"left_id": ids, "right_id": [f"r-{i}" for i in ids]})

    def _pool(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({"left_id": range(n), "right_id": range(n)})

    def test_judgements_outside_the_current_sample_do_not_count_toward_completion(
        self, capsys
    ):
        sample = self._sample(["a", "b"])
        # Simulates what main() passes in: judged has already been filtered to sample_keys
        # before _summary is called, so a leftover row from a different seed's sample (a
        # key not in the current sample) must not appear here at all.
        judged = {
            ("a", "r-a"): {"verdict": "distinct"},
            ("b", "r-b"): {"verdict": "same_procurement"},
        }

        class Args:
            seed = 0
            timings = Path("/nonexistent")

        _summary(self._pool(500), sample, judged, Args())
        out = capsys.readouterr().out
        assert "contamination:" in out  # both members of this 2-pair sample were judged
        assert "50.0%" in out

    def test_reports_incomplete_when_the_sample_is_only_partly_judged(self, capsys):
        sample = self._sample(["a", "b", "c"])
        judged = {("a", "r-a"): {"verdict": "distinct"}}

        class Args:
            seed = 0
            timings = Path("/nonexistent")

        _summary(self._pool(500), sample, judged, Args())
        out = capsys.readouterr().out
        assert "incomplete" in out
        assert "contamination:" not in out


def test_contaminated_verdicts_includes_unsure():
    # An uncertain pair is not evidence of cleanliness — the conservative reading carried
    # over from the earlier full-verification tool, which dropped both same_procurement and
    # unsure rather than keeping unsure as a negative.
    assert set(CONTAMINATED_VERDICTS) == {"same_procurement", "unsure"}
