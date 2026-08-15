"""C5 acceptance, and the quota governance that replaced the money constraint (§6.11, G1).

The criterion is about a *re-run*: it must issue zero HTTP requests, consume zero tokens,
sum ``usd`` to exactly $0.00, and log ``cache_hit=true`` on every row. All four are asserted
directly here against a stub transport that counts calls, so the property is checked rather
than inferred from a wall-clock or a bill.

Everything that touches the outside world is injected. That is deliberate: under a free tier
the expensive mistakes are a blind retry storm and a run that blows the daily allowance, and
neither is testable against a real endpoint without spending the very quota being protected.
"""

from __future__ import annotations

import json

import pytest

from fcesreg.llm import (
    DEFAULT_CACHE_DIR,
    DEFAULT_LEDGER_PATH,
    BudgetExceeded,
    DailyQuotaExhausted,
    LLMClient,
    LLMError,
    LLMRequest,
    Limits,
    RateCard,
    cache_key,
    read_ledger,
)

CARD = RateCard(
    model="test-model",
    usd_per_m_input=0.15,
    usd_per_m_output=0.60,
    source="test",
    checked="2026-08-08",
)


class StubTransport:
    """Scriptable stand-in for the endpoint. Counts every HTTP call it is asked to make."""

    def __init__(self, script=None, input_tokens: int = 700, output_tokens: int = 10):
        self.calls = 0
        self.payloads: list[dict] = []
        self.script = list(script or [])
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def _ok(self):
        return (
            200,
            {
                "choices": [{"message": {"content": "duplicate"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": self.input_tokens,
                    "completion_tokens": self.output_tokens,
                },
            },
            {"x-ratelimit-remaining-tokens": "7000", "x-ratelimit-remaining-requests": "900"},
        )

    def __call__(self, payload):
        self.calls += 1
        self.payloads.append(payload)
        if self.script:
            return self.script.pop(0)
        return self._ok()


class FakeClock:
    """Monotonic time that only advances when something sleeps. No test ever waits."""

    def __init__(self):
        self.t = 0.0
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds

    def monotonic(self) -> float:
        self.t += 0.001  # so latency_ms is positive and calls are ordered
        return self.t


def make_client(tmp_path, transport=None, clock=None, **kwargs) -> LLMClient:
    clock = clock or FakeClock()
    return LLMClient(
        model="test-model",
        cache_dir=tmp_path / "cache",
        ledger_path=tmp_path / "ledger.jsonl",
        run_id=kwargs.pop("run_id", "run-1"),
        rate_card=CARD,
        transport=transport or StubTransport(),
        sleep=clock.sleep,
        monotonic=clock.monotonic,
        **kwargs,
    )


class TestPathsAreAnchored:
    def test_package_constants_do_not_resolve_against_the_cwd(self):
        # A relative cache dir would mean a run from research/ silently missed every entry
        # a run from the root had written.
        assert DEFAULT_CACHE_DIR.is_absolute()
        assert DEFAULT_LEDGER_PATH.is_absolute()


class TestCacheKey:
    def test_stable_for_identical_input(self):
        assert cache_key("m", "s", "p", None) == cache_key("m", "s", "p", None)

    def test_model_is_inside_the_digest(self):
        # Otherwise switching checkpoints silently reuses another model's completions.
        assert cache_key("a", "s", "p") != cache_key("b", "s", "p")

    def test_schema_is_inside_the_digest(self):
        assert cache_key("m", "s", "p", {"a": 1}) != cache_key("m", "s", "p", {"a": 2})


class TestAcceptanceCriterion:
    """A pilot runs; re-running the identical set consumes nothing."""

    def test_rerun_issues_no_request_consumes_no_tokens_and_costs_exactly_zero(self, tmp_path):
        transport = StubTransport()
        client = make_client(tmp_path, transport)

        first = [client.complete("sys", f"pair {i}", condition="pilot") for i in range(20)]
        assert transport.calls == 20
        assert all(not r.cache_hit for r in first)
        assert all(r.input_tokens > 0 for r in first)
        assert sum(r.usd for r in first) > 0

        calls_after_pilot = transport.calls
        second = [client.complete("sys", f"pair {i}", condition="pilot") for i in range(20)]

        assert transport.calls == calls_after_pilot, "a re-run issued an HTTP request"
        assert all(r.cache_hit for r in second)
        assert sum(r.input_tokens + r.output_tokens for r in second if not r.cache_hit) == 0
        assert sum(r.usd for r in second) == 0.0

        rows = read_ledger(tmp_path / "ledger.jsonl")
        replay = rows[20:]
        assert len(replay) == 20
        assert all(row["cache_hit"] is True for row in replay)
        assert sum(row["usd"] for row in replay) == 0.0

    def test_every_row_carries_run_id_and_lands_in_the_one_global_ledger(self, tmp_path):
        client = make_client(tmp_path, run_id="run-abc")
        client.complete("sys", "p", condition="rq2_incontext")
        rows = read_ledger(tmp_path / "ledger.jsonl")
        assert [r["run_id"] for r in rows] == ["run-abc"]
        assert rows[0]["condition"] == "rq2_incontext"
        assert rows[0]["provider"] == "groq"
        assert not (tmp_path / "runs").exists(), "no per-run ledger may be written"

    def test_the_cache_survives_a_new_client_so_a_resumed_run_is_free(self, tmp_path):
        first = StubTransport()
        make_client(tmp_path, first).complete("sys", "p")
        second = StubTransport()
        response = make_client(tmp_path, second).complete("sys", "p")
        assert second.calls == 0
        assert response.cache_hit


class TestCostAccounting:
    def test_usd_is_consumed_cost_and_usd_uncached_is_method_cost(self, tmp_path):
        client = make_client(tmp_path)
        live = client.complete("sys", "p")
        cached = client.complete("sys", "p")

        # The re-run consumed nothing, but the work still costs what it costs: reporting the
        # method's cost from a cached run must not understate it.
        assert cached.usd == 0.0
        assert cached.usd_uncached == pytest.approx(live.usd_uncached)
        assert cached.usd_uncached > 0

    def test_token_counts_are_real_on_a_cache_hit(self, tmp_path):
        client = make_client(tmp_path, StubTransport(input_tokens=712, output_tokens=9))
        client.complete("sys", "p")
        cached = client.complete("sys", "p")
        assert (cached.input_tokens, cached.output_tokens) == (712, 9)

    def test_rate_card_travels_with_every_row(self, tmp_path):
        # A cost figure is meaningless without the rates and the date that produced it.
        client = make_client(tmp_path)
        client.complete("sys", "p")
        row = read_ledger(tmp_path / "ledger.jsonl")[0]
        assert row["rate_usd_per_m_input"] == 0.15
        assert row["rate_usd_per_m_output"] == 0.60
        assert row["rate_card_checked"] == "2026-08-08"

    def test_notional_cost_matches_the_rate_card(self, tmp_path):
        client = make_client(tmp_path, StubTransport(input_tokens=1_000_000, output_tokens=0))
        assert client.complete("sys", "p").usd == pytest.approx(0.15)


class TestLedgerFormat:
    def test_rows_are_one_json_object_per_line(self, tmp_path):
        client = make_client(tmp_path)
        for i in range(3):
            client.complete("sys", f"p{i}")
        text = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
        assert text.endswith("\n")
        lines = text.splitlines()
        assert len(lines) == 3
        assert all(json.loads(line) for line in lines)

    def test_a_torn_line_does_not_make_the_ledger_unreadable(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        path.write_text('{"usd": 1.0}\n{"usd": 2.0\n', encoding="utf-8")
        assert len(read_ledger(path)) == 1


class TestBudgetGuard:
    """The cap is runaway protection on notional spend. Quota is the real constraint, so these
    tests raise the daily allowance out of the way to isolate the cap from it."""

    ROOMY = Limits(tokens_per_day=10_000_000, tokens_per_minute=10_000_000)

    def test_cap_raises_once_notional_spend_crosses(self, tmp_path):
        transport = StubTransport(input_tokens=1_000_000, output_tokens=0)  # $0.15 a call
        client = make_client(tmp_path, transport, cap_usd=0.30, limits=self.ROOMY)
        client.complete("sys", "a")
        client.complete("sys", "b")
        with pytest.raises(BudgetExceeded):
            client.complete("sys", "c")

    def test_spend_aggregates_across_runs_through_the_shared_ledger(self, tmp_path):
        # A cap that reset with every run would not be a cap. This is why the ledger is global.
        transport = StubTransport(input_tokens=1_000_000, output_tokens=0)
        make_client(
            tmp_path, transport, cap_usd=0.20, limits=self.ROOMY, run_id="r1"
        ).complete("sys", "a")
        later = make_client(
            tmp_path,
            StubTransport(input_tokens=1_000_000, output_tokens=0),
            cap_usd=0.20,
            limits=self.ROOMY,
            run_id="r2",
        )
        with pytest.raises(BudgetExceeded):
            later.complete("sys", "b")

    def test_cache_hits_do_not_push_toward_the_cap(self, tmp_path):
        transport = StubTransport(input_tokens=1_000_000, output_tokens=0)
        client = make_client(tmp_path, transport, cap_usd=0.20, limits=self.ROOMY)
        client.complete("sys", "a")
        for _ in range(10):
            client.complete("sys", "a")  # free replays must never trip runaway protection


class TestRateLimitHeaders:
    def test_headers_overwrite_configured_limits(self, tmp_path):
        transport = StubTransport(
            script=[
                (
                    200,
                    {
                        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 1},
                    },
                    {"x-ratelimit-limit-tokens": "6000", "x-ratelimit-limit-requests": "500"},
                )
            ]
        )
        client = make_client(tmp_path, transport)
        client.complete("sys", "p")
        # limit-tokens is per MINUTE and limit-requests is per DAY. Reading that pair the wrong
        # way round would pace at a thirtieth of the real rate or exhaust the day in a minute.
        assert client.limits.tokens_per_minute == 6000
        assert client.limits.requests_per_day == 500

    def test_a_429_honours_retry_after_rather_than_retrying_blindly(self, tmp_path):
        clock = FakeClock()
        transport = StubTransport(
            script=[(429, {"error": "slow down"}, {"retry-after": "7", "x-ratelimit-remaining-requests": "5"})]
        )
        client = make_client(tmp_path, transport, clock=clock)
        client.complete("sys", "p")
        assert transport.calls == 2, "should retry exactly once after honouring the header"
        assert 7.0 in clock.slept

    def test_repeated_429s_stop_rather_than_looping_forever(self, tmp_path):
        script = [
            (429, {}, {"retry-after": "1", "x-ratelimit-remaining-requests": "5"})
            for _ in range(10)
        ]
        client = make_client(tmp_path, StubTransport(script=script), max_retries=3)
        with pytest.raises(LLMError, match="rate limited"):
            client.complete("sys", "p")

    def test_a_429_with_no_requests_left_is_a_clean_stop_not_a_retry(self, tmp_path):
        transport = StubTransport(
            script=[(429, {}, {"x-ratelimit-remaining-requests": "0"})]
        )
        client = make_client(tmp_path, transport)
        with pytest.raises(DailyQuotaExhausted):
            client.complete("sys", "p")
        assert transport.calls == 1, "burned quota retrying a request that cannot succeed"

    def test_server_errors_back_off_and_then_give_up(self, tmp_path):
        script = [(500, {}, {}) for _ in range(10)]
        client = make_client(tmp_path, StubTransport(script=script), max_retries=2)
        with pytest.raises(LLMError, match="server error"):
            client.complete("sys", "p")

    def test_an_unexpected_status_is_not_retried(self, tmp_path):
        transport = StubTransport(script=[(400, {"error": "bad schema"}, {})])
        client = make_client(tmp_path, transport)
        with pytest.raises(LLMError):
            client.complete("sys", "p")
        assert transport.calls == 1


class TestStructuredOutputFailures:
    """A 400 that reports a bad *generation* is transient; a 400 reporting a bad request
    is not. On a sweep of thousands of calls, treating the first as fatal kills a multi-day
    run, and treating the second as retryable only burns quota."""

    FAILED = (
        400,
        {"error": {"code": "json_validate_failed", "failed_generation": ""}},
        {},
    )

    def test_a_failed_generation_is_retried_and_succeeds(self, tmp_path):
        # The stub returns a success once its script is exhausted.
        transport = StubTransport(script=[self.FAILED, self.FAILED])
        client = make_client(tmp_path, transport, max_retries=3)
        assert client.complete("sys", "p").text == "duplicate"
        assert transport.calls == 3

    def test_a_persistent_failure_gives_up_and_says_it_is_pathological(self, tmp_path):
        transport = StubTransport(script=[self.FAILED] * 10)
        client = make_client(tmp_path, transport, max_retries=2)
        with pytest.raises(LLMError, match="pathological"):
            client.complete("sys", "p")

    def test_a_malformed_request_400_is_still_fatal_on_the_first_try(self, tmp_path):
        # Distinguished by error code, not by the prose message, which gets reworded.
        transport = StubTransport(
            script=[(400, {"error": {"code": "invalid_request_error"}}, {})] * 5
        )
        client = make_client(tmp_path, transport, max_retries=3)
        with pytest.raises(LLMError):
            client.complete("sys", "p")
        assert transport.calls == 1

    def test_a_bare_schema_is_refused_before_a_request_is_issued(self, tmp_path):
        # The endpoint wants the whole response_format.json_schema object; passing the bare
        # JSON Schema earns a 400 that costs a request to discover.
        transport = StubTransport()
        client = make_client(tmp_path, transport)
        with pytest.raises(LLMError, match="must be the response_format.json_schema object"):
            client.complete("sys", "p", json_schema={"type": "object"})
        assert transport.calls == 0


class TestQuotaGovernance:
    def test_daily_token_allowance_stops_before_the_request_is_issued(self, tmp_path):
        transport = StubTransport()
        client = make_client(
            tmp_path, transport, limits=Limits(tokens_per_day=100, tokens_per_minute=10_000)
        )
        with pytest.raises(DailyQuotaExhausted):
            client.complete("sys", "x" * 4000)
        assert transport.calls == 0, "quota was checked after spending it"

    def test_todays_consumption_is_read_back_from_the_shared_ledger(self, tmp_path):
        transport = StubTransport(input_tokens=400, output_tokens=10)
        limits = Limits(tokens_per_day=450, tokens_per_minute=10_000)
        make_client(tmp_path, transport, limits=limits, run_id="r1").complete("sys", "a")
        resumed = make_client(tmp_path, StubTransport(), limits=limits, run_id="r2")
        with pytest.raises(DailyQuotaExhausted):
            resumed.complete("sys", "b")

    def test_the_daily_guard_can_overshoot_by_at_most_one_call(self, tmp_path):
        """The pre-flight check uses an estimate, because actual usage is only known after
        the call. So the local guard can cross the line by one call's worth and no more —
        the next pre-flight sees the real total. The authoritative check is the endpoint's
        own remaining-tokens header; this one exists to stop a run before it gets there."""
        transport = StubTransport(input_tokens=400, output_tokens=0)
        client = make_client(
            tmp_path, transport, limits=Limits(tokens_per_day=500, tokens_per_minute=10_000)
        )
        client.complete("sys", "a")  # estimate 64 passes; actual 400 lands
        client.complete("sys", "b")  # 400 + 64 still under 500; actual takes it to 800
        with pytest.raises(DailyQuotaExhausted):
            client.complete("sys", "c")
        assert transport.calls == 2

    def test_a_call_larger_than_a_minutes_allowance_fails_loudly(self, tmp_path):
        # Waiting cannot help here, so spinning in the pacing loop would hang the run.
        client = make_client(tmp_path, limits=Limits(tokens_per_minute=50))
        with pytest.raises(LLMError, match="per-minute allowance"):
            client.complete("sys", "x" * 4000)

    def test_pacing_waits_instead_of_exceeding_tokens_per_minute(self, tmp_path):
        clock = FakeClock()
        client = make_client(
            tmp_path,
            StubTransport(input_tokens=600, output_tokens=0),
            clock=clock,
            limits=Limits(tokens_per_minute=1000, requests_per_minute=600),
        )
        client.complete("sys", "a")
        client.complete("sys", "b")
        assert clock.slept, "second call should have waited for the window to clear"

    def test_requests_per_minute_paces_consecutive_calls(self, tmp_path):
        clock = FakeClock()
        client = make_client(
            tmp_path, clock=clock, limits=Limits(requests_per_minute=30, tokens_per_minute=100_000)
        )
        client.complete("sys", "a")
        client.complete("sys", "b")
        assert any(s == pytest.approx(2.0, abs=0.1) for s in clock.slept)


class TestCompleteMany:
    def test_results_are_keyed_by_custom_id(self, tmp_path):
        client = make_client(tmp_path)
        requests = [LLMRequest(custom_id=f"pair-{i}", system="s", prompt=f"p{i}") for i in range(3)]
        results = client.complete_many(requests)
        assert set(results) == {"pair-0", "pair-1", "pair-2"}

    def test_cached_requests_are_served_before_any_quota_is_spent(self, tmp_path):
        client = make_client(tmp_path)
        client.complete("s", "already-done")

        transport = StubTransport()
        resumed = make_client(tmp_path, transport)
        requests = [
            LLMRequest(custom_id="new", system="s", prompt="fresh"),
            LLMRequest(custom_id="old", system="s", prompt="already-done"),
        ]
        resumed.complete_many(requests)
        # The cached one is replayed first, so a quota stop can never strand a free result.
        rows = read_ledger(tmp_path / "ledger.jsonl")
        assert rows[1]["cache_hit"] is True
        assert transport.calls == 1

    def test_quota_exhaustion_carries_the_work_already_done(self, tmp_path):
        transport = StubTransport(input_tokens=400, output_tokens=0)
        client = make_client(
            tmp_path, transport, limits=Limits(tokens_per_day=900, tokens_per_minute=10_000)
        )
        requests = [LLMRequest(custom_id=f"p{i}", system="s", prompt=f"{i}") for i in range(5)]
        with pytest.raises(DailyQuotaExhausted) as caught:
            client.complete_many(requests)
        assert caught.value.completed, "a day's limit discarded the work already paid for"
        # Three land before the guard trips: the pre-flight estimate is 64 tokens while each
        # call actually consumes 400, so the wall is crossed by one call (see the overshoot
        # test above). What matters is that the completed work is carried out, not lost.
        assert len(caught.value.completed) == 3
        assert set(caught.value.completed) == {"p0", "p1", "p2"}

    def test_the_run_resumes_from_cache_the_next_day(self, tmp_path):
        transport = StubTransport(input_tokens=400, output_tokens=0)
        limits = Limits(tokens_per_day=900, tokens_per_minute=10_000)
        client = make_client(tmp_path, transport, limits=limits)
        requests = [LLMRequest(custom_id=f"p{i}", system="s", prompt=f"{i}") for i in range(5)]
        with pytest.raises(DailyQuotaExhausted):
            client.complete_many(requests)

        # Tomorrow: a fresh allowance, and yesterday's completed calls replay for nothing.
        tomorrow = make_client(
            tmp_path, StubTransport(input_tokens=400), limits=Limits(tokens_per_day=100_000)
        )
        tomorrow._tokens_today = 0
        results = tomorrow.complete_many(requests)
        assert len(results) == 5
        assert sum(1 for r in results.values() if r.cache_hit) == 3
        assert sum(r.usd for r in results.values() if r.cache_hit) == 0.0


class TestRequestShape:
    def test_json_schema_is_sent_as_response_format(self, tmp_path):
        transport = StubTransport()
        client = make_client(tmp_path, transport)
        client.complete("sys", "p", json_schema={"name": "verdict", "schema": {}})
        assert transport.payloads[0]["response_format"]["type"] == "json_schema"

    def test_temperature_is_pinned_for_determinism(self, tmp_path):
        transport = StubTransport()
        make_client(tmp_path, transport).complete("sys", "p")
        assert transport.payloads[0]["temperature"] == 0.0
