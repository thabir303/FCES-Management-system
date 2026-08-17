"""Cost aggregation over the ledger (§10, `T8_cost`).

The load-bearing property is invariance under replay. A resumable sweep appends a row every
time it re-reads a cached call, so an aggregation that counts rows reports a cost that grows
with the number of days the quota took rather than with what the method costs.
"""

from __future__ import annotations

import pytest

from fcesreg.costs import EXCLUDED_CONDITIONS, summarise_costs, throughput_per_day


def row(prompt: str, condition: str = "cascade", cache_hit: bool = False, **over) -> dict:
    base = {
        "prompt_sha256": prompt,
        "condition": condition,
        "cache_hit": cache_hit,
        "input_tokens": 400,
        "output_tokens": 200,
        "usd": 0.0 if cache_hit else 0.00018,
        "usd_uncached": 0.00018,
        "latency_ms": 1 if cache_hit else 900,
    }
    return {**base, **over}


class TestInvarianceUnderReplay:
    """The property the whole module exists for."""

    def test_replaying_every_call_changes_nothing(self):
        live = [row(f"p{i}") for i in range(10)]
        replay = [row(f"p{i}", cache_hit=True) for i in range(10)]

        once = summarise_costs(live)["cascade"]
        twice = summarise_costs(live + replay)["cascade"]

        assert twice.usd == pytest.approx(once.usd)
        assert twice.n_calls == once.n_calls == 10
        assert twice.total_tokens == once.total_tokens

    def test_seven_daily_resumes_do_not_multiply_the_cost(self):
        # The real shape: each day replays everything already done, then adds a little.
        rows, done = [], 0
        for _ in range(7):
            rows += [row(f"p{i}", cache_hit=True) for i in range(done)]
            rows += [row(f"p{i}") for i in range(done, done + 20)]
            done += 20

        got = summarise_costs(rows)["cascade"]
        assert got.n_calls == 140
        assert got.usd == pytest.approx(140 * 0.00018)
        # And the inflation that would otherwise be reported is visible, not hidden.
        assert got.n_replayed_rows == len(rows) - 140 > 0

    def test_summing_rows_naively_would_have_inflated_it(self):
        # Pins the defect this replaces, so the guard cannot quietly become a no-op.
        rows = [row("p0")] + [row("p0", cache_hit=True)] * 5
        naive = sum(r["usd_uncached"] for r in rows)
        assert summarise_costs(rows)["cascade"].usd == pytest.approx(naive / 6)


class TestWhatCounts:
    def test_probe_and_pilot_conditions_are_excluded(self):
        rows = [row("p0"), row("q0", condition="c5_pilot"), row("q1", condition="rq2_probe")]
        got = summarise_costs(rows)
        assert set(got) == {"cascade"}
        assert got["cascade"].n_calls == 1

    def test_the_exclusion_list_covers_every_probe_used_so_far(self):
        assert {"c5_pilot", "cost_probe", "schema_probe", "rq2_probe"} <= EXCLUDED_CONDITIONS

    def test_conditions_are_kept_apart(self):
        rows = [row("p0", condition="cascade"), row("p1", condition="rq2_incontext")]
        assert set(summarise_costs(rows)) == {"cascade", "rq2_incontext"}

    def test_the_same_prompt_under_two_conditions_counts_once_each(self):
        # Deduplication is per condition: the identical prompt used by two methods is two
        # costs, because each method would pay it in its own clean run.
        rows = [row("same", condition="a"), row("same", condition="b")]
        got = summarise_costs(rows)
        assert got["a"].n_calls == 1 and got["b"].n_calls == 1

    def test_a_condition_filter_narrows_the_result(self):
        rows = [row("p0", condition="cascade"), row("p1", condition="other")]
        assert set(summarise_costs(rows, conditions={"cascade"})) == {"cascade"}


class TestTokensAndLatency:
    def test_tokens_are_deduplicated_like_cost(self):
        rows = [row("p0")] + [row("p0", cache_hit=True)] * 4
        got = summarise_costs(rows)["cascade"]
        assert got.total_tokens == 600
        assert got.mean_tokens == pytest.approx(600.0)

    def test_latency_excludes_cache_hits(self):
        # A cache hit's elapsed time measures a disk read. Averaging it in would report the
        # method as getting faster the more often it was replayed.
        rows = [row("p0", latency_ms=900)] + [row("p0", cache_hit=True, latency_ms=1)] * 9
        assert summarise_costs(rows)["cascade"].mean_latency_ms == pytest.approx(900.0)

    def test_latency_is_none_when_nothing_was_measured_live(self):
        # Unmeasured is not estimated: a fully replayed run reports no latency at all.
        rows = [row("p0", cache_hit=True)]
        assert summarise_costs(rows)["cascade"].mean_latency_ms is None

    def test_per_thousand_scales_by_records_processed(self):
        got = summarise_costs([row(f"p{i}") for i in range(50)])["cascade"]
        scaled = got.per_thousand(500)
        assert scaled["calls_per_1000"] == pytest.approx(100.0)
        assert scaled["usd_per_1000"] == pytest.approx(got.usd * 2)

    def test_per_thousand_refuses_a_meaningless_denominator(self):
        got = summarise_costs([row("p0")])["cascade"]
        with pytest.raises(ValueError, match="must be positive"):
            got.per_thousand(0)


class TestThroughput:
    def test_tokens_bind_before_requests_on_the_free_tier(self):
        got = throughput_per_day(638, tokens_per_day=200_000, requests_per_day=1_000)
        assert got["binding_limit"] == "tokens"
        assert got["calls_per_day"] == pytest.approx(200_000 / 638)

    def test_requests_can_bind_for_a_cheap_call(self):
        got = throughput_per_day(50, tokens_per_day=200_000, requests_per_day=1_000)
        assert got["binding_limit"] == "requests"
        assert got["calls_per_day"] == 1_000.0

    def test_zero_tokens_is_refused_rather_than_dividing(self):
        with pytest.raises(ValueError, match="must be positive"):
            throughput_per_day(0, 200_000, 1_000)
