"""Tests for the parts of annotate.py that are not the interactive loop.

The label-noise rate reaches the paper, so the sampling, the rate arithmetic and the
exclusion accounting are tested. Handling time no longer reaches anything, and there are
tests for that too. The `input()` loop is exercised by hand, as the distractor tool's is.

Not wired into `make test`: `annotation/` is outside `research/tests`' discovery scope.

    .venv/bin/pytest annotation/test_annotate.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from annotate import KEYS, NOISE_VERDICTS, _summary, render


class Args:
    seed = 0
    timings = Path("/nonexistent")


def _sample(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(n)],
            "title": [f"t{i}" for i in range(n)],
            "cpv_code": ["30100000"] * n,
        }
    )


def _judged(**counts: int) -> dict:
    out, i = {}, 0
    for verdict, k in counts.items():
        for _ in range(k):
            out[f"r{i}"] = {"verdict": verdict}
            i += 1
    return out


class TestNoiseRate:
    def test_rate_is_over_decided_items_not_the_whole_sample(self):
        # 2 disagree, 6 agree, 2 unsure -> 2/8, not 2/10. Counting `unsure` in the
        # denominator as if it were agreement would assert something the annotator
        # explicitly declined to assert.
        _summary(_sample(10), _judged(agree=6, disagree=2, unsure=2), Args())

    def test_reports_rate_and_interval_when_complete(self, capsys):
        _summary(_sample(10), _judged(agree=8, disagree=2), Args())
        out = capsys.readouterr().out
        assert "label noise: 20.0%" in out
        assert "Wilson 95% CI" in out
        assert "n=10 decided" in out

    def test_unsure_is_excluded_and_reported_separately(self, capsys):
        _summary(_sample(10), _judged(agree=6, disagree=2, unsure=2), Args())
        out = capsys.readouterr().out
        assert "n=8 decided" in out
        assert "label noise: 25.0%" in out          # 2 of 8, not 2 of 10
        assert "unsure     : 2 of 10" in out
        assert "excluded from the rate" in out

    def test_incomplete_sample_reports_no_rate(self, capsys):
        _summary(_sample(10), _judged(agree=3), Args())
        out = capsys.readouterr().out
        assert "incomplete" in out
        assert "label noise" not in out

    def test_all_unsure_reports_no_rate_rather_than_zero(self, capsys):
        _summary(_sample(4), _judged(unsure=4), Args())
        out = capsys.readouterr().out
        assert "no rate can be reported" in out
        assert "label noise" not in out

    def test_the_upper_bound_caveat_is_stated_with_the_rate(self, capsys):
        # The rate is measured at the published 8-digit level, finer than either level RQ2
        # evaluates. Reporting it without that caveat would overstate the noise affecting
        # the results.
        _summary(_sample(4), _judged(agree=3, disagree=1), Args())
        assert "UPPER BOUND" in capsys.readouterr().out


class TestHandlingTimeIsNotAResult:
    """Handling time is no longer measured (ruled 2026-08-17).

    RQ3 reports residual review *volume*, with total effort left as a formula a reader
    substitutes their own handling time into. These guard the two ways the old figure could
    come back: the mode that produced it, and the hand-off line that piped it into the
    headline result.
    """

    def test_the_timing_only_mode_is_gone(self):
        import annotate as mod

        assert not hasattr(mod, "timing_run")

    def test_no_timing_only_flag_is_offered(self):
        # It would otherwise cost the author eight minutes on a figure with nowhere to go.
        import annotate as mod

        assert "--timing-only" not in (mod.main.__doc__ or "") + Path(
            mod.__file__
        ).read_text(encoding="utf-8")

    def test_the_summary_pipes_nothing_into_the_operating_point(self, capsys, tmp_path):
        import annotate as mod

        class Args:
            seed = 0
            timings = tmp_path / "absent.jsonl"

        mod._summary(_sample(4), _judged(agree=3, disagree=1), Args())
        out = capsys.readouterr().out
        assert "mean_seconds_per_item" not in out
        assert "operating_point" not in out


class TestVerdicts:
    def test_only_disagree_counts_as_noise(self):
        # Deliberately unlike the distractor tool, where `unsure` counted against purity.
        # There an unresolved pair still contaminated the negative set; here an item the
        # annotator cannot judge is evidence about the notice, not about the code.
        assert NOISE_VERDICTS == ("disagree",)

    def test_three_responses_are_offered(self):
        assert set(KEYS) == {"a", "d", "u"}
        assert {v[0] for v in KEYS.values()} == {"agree", "disagree", "unsure"}


class TestRender:
    def _record(self, **over) -> pd.Series:
        base = {
            "title": "Vacuum pump",
            "description": "rotary vane, 230V",
            "cpv_code": "42123400",
            "_division_desc": "Industrial machinery",
            "_class_desc": "Compressors",
        }
        return pd.Series({**base, **over})

    def test_shows_the_published_code_at_both_evaluated_levels(self, capsys):
        render(1, 40, self._record())
        out = capsys.readouterr().out
        assert "42123400" in out
        assert "division 42" in out and "Industrial machinery" in out
        assert "class    4212" in out and "Compressors" in out

    def test_asks_the_fair_description_question_not_the_best_code_question(self, capsys):
        render(1, 40, self._record())
        out = capsys.readouterr().out
        assert "FAIRLY DESCRIBE" in out
        assert "not: is it the best possible code" in out

    def test_a_missing_class_description_is_explained_not_printed_as_nan(self, capsys):
        # A code published at division level (30000000) has no four-digit class. The same
        # truthy-nan defect that put "nan" into degraded titles reaches the annotator's
        # screen here if `or` is used as the guard.
        render(1, 40, self._record(cpv_code="30000000", _class_desc=float("nan")))
        out = capsys.readouterr().out
        assert "nan" not in out
        assert "published at division level" in out

    def test_a_null_description_does_not_print_as_nan(self, capsys):
        render(1, 40, self._record(description=float("nan")))
        assert "nan" not in capsys.readouterr().out

    def test_shows_no_model_opinion_and_no_prefilled_answer(self, capsys):
        # The whole value of the exercise is that it is not a model's judgement.
        out = capsys.readouterr().out
        render(1, 40, self._record())
        out = capsys.readouterr().out
        for leak in ("suggest", "predict", "model", "recommend", "confidence", "likely"):
            assert leak not in out.lower()
