"""A4 acceptance: monotonic per-item timing, abandonment excluded, mean refused below 30."""

from __future__ import annotations

import time

import pytest

from fcesreg.timing import (
    IDLE_CUTOFF_S,
    MIN_ITEMS_FOR_MEAN,
    ItemTiming,
    TooFewTimings,
    read_timings,
    summarise,
    time_item,
    write_timings,
)


def _timings(n, seconds=10.0, abandoned=False):
    return [
        ItemTiming(f"i{i}", seconds, "2026-08-01T00:00:00+00:00", abandoned)
        for i in range(n)
    ]


class TestTimeItem:
    def test_measures_elapsed_time(self):
        sink = []
        with time_item("i1", sink):
            time.sleep(0.05)
        assert len(sink) == 1
        assert sink[0].item_id == "i1"
        assert sink[0].seconds >= 0.05
        assert sink[0].abandoned is False

    def test_records_even_when_the_body_raises(self):
        # An interrupted session must not lose the item that was on screen.
        sink = []
        with pytest.raises(KeyboardInterrupt):
            with time_item("i1", sink):
                raise KeyboardInterrupt
        assert len(sink) == 1

    def test_explicit_abandonment(self):
        sink = []
        with time_item("i1", sink) as abandon:
            abandon()
        assert sink[0].abandoned is True

    def test_idle_cutoff_marks_abandoned_automatically(self):
        sink = []
        with time_item("i1", sink, idle_cutoff_s=0.01):
            time.sleep(0.02)
        assert sink[0].abandoned is True

    def test_default_cutoff_is_two_minutes(self):
        assert IDLE_CUTOFF_S == 120.0

    def test_reads_the_clock_through_time_monotonic(self, monkeypatch):
        # A wall-clock implementation would report a negative duration if the system
        # clock were stepped backwards mid-item. Pinning monotonic proves which clock
        # the duration comes from.
        seq = iter([1000.0, 1000.25])
        monkeypatch.setattr(time, "monotonic", lambda: next(seq))
        sink = []
        with time_item("i1", sink):
            pass
        assert sink[0].seconds == pytest.approx(0.25)


class TestSummarise:
    def test_excludes_abandoned_from_the_mean_and_counts_them(self):
        timings = _timings(30, seconds=10.0) + _timings(5, seconds=900.0, abandoned=True)
        got = summarise(timings)
        assert got["n"] == 30
        assert got["n_abandoned"] == 5
        assert got["mean_seconds"] == pytest.approx(10.0)
        assert got["total_seconds"] == pytest.approx(300.0)

    def test_refuses_a_mean_below_the_minimum_sample(self):
        with pytest.raises(TooFewTimings, match="need at least 30"):
            summarise(_timings(MIN_ITEMS_FOR_MEAN - 1))

    def test_refuses_when_abandonment_drops_it_below_the_minimum(self):
        timings = _timings(20) + _timings(50, abandoned=True)
        with pytest.raises(TooFewTimings):
            summarise(timings)

    def test_empty_is_refused_not_zero(self):
        # Returning 0.0 here would put a fabricated number into the residual-effort
        # calculation, which is the one thing this must never do.
        with pytest.raises(TooFewTimings):
            summarise([])

    def test_median_and_p90(self):
        timings = [
            ItemTiming(f"i{i}", float(i), "2026-08-01T00:00:00+00:00", False)
            for i in range(1, 101)
        ]
        got = summarise(timings)
        assert got["median_seconds"] == pytest.approx(50.5)
        assert got["p90_seconds"] == pytest.approx(90.1)
        assert got["mean_seconds"] == pytest.approx(50.5)


class TestPersistence:
    def test_round_trips_through_jsonl(self, tmp_path):
        path = tmp_path / "labels" / "t.jsonl"
        original = _timings(3)
        write_timings(path, original)
        assert read_timings(path) == original

    def test_appends_rather_than_overwrites(self, tmp_path):
        path = tmp_path / "t.jsonl"
        write_timings(path, _timings(2))
        write_timings(path, _timings(3))
        assert len(read_timings(path)) == 5
