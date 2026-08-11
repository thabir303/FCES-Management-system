"""Tests for the cascade and its adjudicator (§6.9, C6).

C6's acceptance criterion is about the band's integrity, not about accuracy:
``stats`` populated with all three keys, the counts mutually consistent, and **every pair
sent to the adjudicator strictly inside ``(lower, upper)`` with no pair outside it
adjudicated**. ``TestBandIntegrity`` is that criterion.

The adjudicator is stubbed throughout. The cascade's band logic must be checkable without a
key, a network or a quota, which is the reason `Adjudicator` is a protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fcesreg import embed as embed_mod
from fcesreg.adjudicate import AdjudicationFailed, _verdict, render_prompt
from fcesreg.dedup import (
    AdjudicationBudgetExceeded,
    Adjudicator,
    CascadeMatcher,
    EmbeddingMatcher,
    Matcher,
)


class FixedBase:
    """A base matcher returning scores set by the test, so the band is exactly known."""

    name = "fixed"

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def score_pairs(self, pairs, records):
        return self.scores[: len(pairs)]


class RecordingAdjudicator:
    """Says duplicate to everything, and records exactly what it was asked."""

    def __init__(self, verdict: float = 1.0):
        self.verdict = verdict
        self.seen: list[pd.DataFrame] = []

    def adjudicate(self, pairs, records):
        self.seen.append(pairs.copy())
        return np.full(len(pairs), self.verdict, dtype=float)


def _pairs(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"left_id": [f"l{i}" for i in range(n)], "right_id": [f"r{i}" for i in range(n)]}
    )


def _records(n: int) -> pd.DataFrame:
    ids = [f"l{i}" for i in range(n)] + [f"r{i}" for i in range(n)]
    return pd.DataFrame(
        {"record_id": ids, "title": [f"t{i}" for i in ids], "description": ["d"] * len(ids)}
    )


class TestBandIntegrity:
    """C6's stated acceptance criterion."""

    def test_stats_carries_all_three_keys(self):
        scores = [0.1, 0.5, 0.9]
        c = CascadeMatcher(FixedBase(scores), 0.3, 0.7, RecordingAdjudicator())
        c.score_pairs(_pairs(3), _records(3))
        assert {"n_pairs", "n_adjudicated", "band_fraction"} <= set(c.stats)

    def test_counts_are_mutually_consistent(self):
        scores = [0.1, 0.4, 0.5, 0.6, 0.9]
        c = CascadeMatcher(FixedBase(scores), 0.3, 0.7, RecordingAdjudicator())
        c.score_pairs(_pairs(5), _records(5))
        assert c.stats["n_adjudicated"] <= c.stats["n_pairs"]
        assert c.stats["band_fraction"] == pytest.approx(
            c.stats["n_adjudicated"] / c.stats["n_pairs"]
        )

    def test_only_pairs_strictly_inside_the_band_are_adjudicated(self):
        # 0.3 and 0.7 sit exactly on the bounds and must be decided by the cheap tier.
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        adj = RecordingAdjudicator()
        c = CascadeMatcher(FixedBase(scores), 0.3, 0.7, adj)
        c.score_pairs(_pairs(5), _records(5))

        assert len(adj.seen) == 1
        sent = adj.seen[0]
        assert list(sent["left_id"]) == ["l2"]  # only the 0.5
        assert c.stats["n_adjudicated"] == 1

    def test_no_pair_outside_the_band_reaches_the_adjudicator(self):
        rng = np.random.default_rng(0)
        scores = rng.random(200)
        lower, upper = 0.4, 0.6
        adj = RecordingAdjudicator()
        c = CascadeMatcher(FixedBase(scores), lower, upper, adj)
        pairs = _pairs(200)
        c.score_pairs(pairs, _records(200))

        sent_positions = [int(i[1:]) for i in adj.seen[0]["left_id"]]
        for position in sent_positions:
            assert lower < scores[position] < upper
        # And the converse: everything in the band was sent.
        assert set(sent_positions) == set(np.flatnonzero((scores > lower) & (scores < upper)))

    def test_a_band_fraction_of_any_size_is_a_finding_not_an_error(self):
        # Criterion wording: "a fraction of 0.30 is a fact about the method".
        scores = np.linspace(0.0, 1.0, 100)
        c = CascadeMatcher(FixedBase(scores), 0.35, 0.65, RecordingAdjudicator())
        c.score_pairs(_pairs(100), _records(100))
        assert 0.0 < c.stats["band_fraction"] < 1.0


class TestDecisions:
    def test_above_upper_is_accepted_without_a_call(self):
        adj = RecordingAdjudicator(verdict=0.0)
        c = CascadeMatcher(FixedBase([0.9]), 0.3, 0.7, adj)
        out = c.score_pairs(_pairs(1), _records(1))
        assert out[0] == 1.0
        assert adj.seen == []

    def test_below_lower_is_rejected_without_a_call(self):
        adj = RecordingAdjudicator(verdict=1.0)
        c = CascadeMatcher(FixedBase([0.1]), 0.3, 0.7, adj)
        out = c.score_pairs(_pairs(1), _records(1))
        assert out[0] == 0.0
        assert adj.seen == []

    def test_the_adjudicator_decides_the_band(self):
        c = CascadeMatcher(FixedBase([0.5]), 0.3, 0.7, RecordingAdjudicator(verdict=1.0))
        assert c.score_pairs(_pairs(1), _records(1))[0] == 1.0
        c = CascadeMatcher(FixedBase([0.5]), 0.3, 0.7, RecordingAdjudicator(verdict=0.0))
        assert c.score_pairs(_pairs(1), _records(1))[0] == 0.0

    def test_verdicts_land_on_the_right_pairs(self):
        # A verdict applied at the wrong position is the failure a positional API invites.
        class Alternating:
            def adjudicate(self, pairs, records):
                return np.array([1.0, 0.0, 1.0])

        scores = [0.9, 0.5, 0.5, 0.5, 0.1]
        c = CascadeMatcher(FixedBase(scores), 0.3, 0.7, Alternating())
        out = c.score_pairs(_pairs(5), _records(5))
        assert list(out) == [1.0, 1.0, 0.0, 1.0, 0.0]

    def test_output_is_binary(self):
        # The cascade emits decisions, not a ranking.
        scores = np.linspace(0, 1, 50)
        c = CascadeMatcher(FixedBase(scores), 0.3, 0.7, RecordingAdjudicator(verdict=1.0))
        out = c.score_pairs(_pairs(50), _records(50))
        assert set(np.unique(out)) <= {0.0, 1.0}


class TestUndefinedUpperThreshold:
    """Where no threshold meets the precision target, upper is inf. That is a result."""

    def test_infinite_upper_accepts_nothing_and_bands_the_rest(self):
        scores = np.array([0.1, 0.5, 0.99])
        adj = RecordingAdjudicator()
        c = CascadeMatcher(FixedBase(scores), 0.3, float("inf"), adj)
        c.score_pairs(_pairs(3), _records(3))
        # Everything above lower is adjudicated; nothing is auto-accepted.
        assert c.stats["n_adjudicated"] == 2
        assert c.stats["upper_undefined"] is True

    def test_a_finite_upper_is_not_flagged_undefined(self):
        c = CascadeMatcher(FixedBase([0.5]), 0.3, 0.7, RecordingAdjudicator())
        c.score_pairs(_pairs(1), _records(1))
        assert c.stats["upper_undefined"] is False


class TestGuards:
    def test_a_band_over_the_cap_raises_rather_than_adjudicating_a_prefix(self):
        scores = np.full(10, 0.5)
        adj = RecordingAdjudicator()
        c = CascadeMatcher(FixedBase(scores), 0.3, 0.7, adj, max_adjudications=5)
        with pytest.raises(AdjudicationBudgetExceeded, match="will not adjudicate part"):
            c.score_pairs(_pairs(10), _records(10))
        assert adj.seen == []  # nothing was spent before the refusal

    def test_inverted_bounds_are_refused(self):
        with pytest.raises(ValueError, match="lower .* is above upper"):
            CascadeMatcher(FixedBase([0.5]), 0.8, 0.2, RecordingAdjudicator())

    def test_a_wrong_length_verdict_array_is_refused(self):
        class Wrong:
            def adjudicate(self, pairs, records):
                return np.array([1.0, 1.0])

        c = CascadeMatcher(FixedBase([0.5]), 0.3, 0.7, Wrong())
        with pytest.raises(ValueError, match="adjudicator returned"):
            c.score_pairs(_pairs(1), _records(1))

    def test_satisfies_both_protocols(self):
        c = CascadeMatcher(FixedBase([0.5]), 0.3, 0.7, RecordingAdjudicator())
        assert isinstance(c, Matcher)
        assert isinstance(RecordingAdjudicator(), Adjudicator)


class TestEmbeddingMatcher:
    """Scored against a stub encoder, so no model is downloaded and no network is touched."""

    @pytest.fixture
    def stub(self, monkeypatch):
        class Encoder:
            def encode(self, texts, **kwargs):
                out = np.zeros((len(texts), 8), dtype=np.float32)
                for i, t in enumerate(texts):
                    rng = np.random.default_rng(abs(hash(t)) % (2**32))
                    v = rng.normal(size=8).astype(np.float32)
                    out[i] = v / np.linalg.norm(v)
                return out

        monkeypatch.setattr(embed_mod, "_load_model", lambda model_id: Encoder())

    def test_identical_text_scores_one(self, stub, tmp_path):
        records = pd.DataFrame(
            {
                "record_id": ["a", "b"],
                "title": ["vacuum pump", "vacuum pump"],
                "description": ["rotary", "rotary"],
            }
        )
        pairs = pd.DataFrame({"left_id": ["a"], "right_id": ["b"]})
        m = EmbeddingMatcher(cache_dir=tmp_path)
        assert m.score_pairs(pairs, records)[0] == pytest.approx(1.0, abs=1e-5)

    def test_scores_are_cosines_in_range(self, stub, tmp_path):
        records = pd.DataFrame(
            {
                "record_id": [f"r{i}" for i in range(6)],
                "title": [f"item {i}" for i in range(6)],
                "description": ["d"] * 6,
            }
        )
        pairs = pd.DataFrame(
            {"left_id": ["r0", "r1", "r2"], "right_id": ["r3", "r4", "r5"]}
        )
        scores = EmbeddingMatcher(cache_dir=tmp_path).score_pairs(pairs, records)
        assert ((scores >= -1.0001) & (scores <= 1.0001)).all()

    def test_scores_align_one_per_pair(self, stub, tmp_path):
        records = pd.DataFrame(
            {"record_id": ["a", "b", "c"], "title": ["x", "y", "z"], "description": [""] * 3}
        )
        pairs = pd.DataFrame({"left_id": ["a", "b"], "right_id": ["b", "c"]})
        assert EmbeddingMatcher(cache_dir=tmp_path).score_pairs(pairs, records).shape == (2,)

    def test_satisfies_the_matcher_protocol(self, tmp_path):
        assert isinstance(EmbeddingMatcher(cache_dir=tmp_path), Matcher)

    def test_a_null_description_does_not_become_the_string_nan(self, stub, tmp_path):
        # text_of uses fillna(""), so the vector must match a record whose description is
        # genuinely empty -- not one carrying a literal "nan" token.
        records = pd.DataFrame(
            {
                "record_id": ["a", "b"],
                "title": ["pump", "pump"],
                "description": [float("nan"), ""],
            }
        )
        pairs = pd.DataFrame({"left_id": ["a"], "right_id": ["b"]})
        assert EmbeddingMatcher(cache_dir=tmp_path).score_pairs(pairs, records)[0] == (
            pytest.approx(1.0, abs=1e-5)
        )


class TestVerdictParsing:
    def test_reads_a_well_formed_reply(self):
        assert _verdict('{"same": true, "reason": "identical model"}', "x") is True
        assert _verdict('{"same": false, "reason": "different capacity"}', "x") is False

    def test_unparseable_reply_raises_rather_than_defaulting_to_false(self):
        # Defaulting would turn an infrastructure failure into a measurement, and the pairs
        # that fail to parse are not a random sample of the band.
        with pytest.raises(AdjudicationFailed, match="not JSON"):
            _verdict("I think they are the same", "x")

    def test_missing_field_raises(self):
        with pytest.raises(AdjudicationFailed, match="no `same` field"):
            _verdict('{"reason": "hmm"}', "x")

    def test_non_boolean_verdict_raises(self):
        with pytest.raises(AdjudicationFailed, match="not a boolean"):
            _verdict('{"same": "yes", "reason": "hmm"}', "x")


class TestPromptRendering:
    def test_shows_both_records_with_fields_kept_apart(self):
        left = pd.Series({"title": "Sony TV", "description": "40 inch"})
        right = pd.Series({"title": "Sony Television", "description": "40in"})
        prompt = render_prompt(left, right)
        assert "[A]" in prompt and "[B]" in prompt
        assert "title: Sony TV" in prompt
        assert "description: 40 inch" in prompt

    def test_a_null_description_does_not_reach_the_prompt_as_nan(self):
        # The same defect class that put "nan" into degraded titles.
        left = pd.Series({"title": "Pump", "description": float("nan")})
        right = pd.Series({"title": "Pump", "description": "brass"})
        assert "nan" not in render_prompt(left, right)
