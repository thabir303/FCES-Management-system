"""C2 acceptance: three schemes, availability decided by data, metrics recoverable.

Per Amendment 2 the criterion tests the implementation, not the outcome. Pair completeness
and reduction ratio are checked against a fixture whose answer is known by construction,
and on the real corpora only ranges and internal consistency are asserted. Whatever the
measured values turn out to be, they are recorded by `run_blocking.py`, not gated here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import CORPUS_A, CORPUS_A_PAIRS, requires
from fcesreg.blocking import (
    LEADING_STOPWORDS,
    SCHEMES,
    SchemeUnavailable,
    applicable_schemes,
    block_by_buyer,
    block_by_leading_token,
    block_by_sorted_ngrams,
    candidate_pairs,
    evaluate_blocking,
    ngram_overlap_candidates,
)


def frame(rows):
    return pd.DataFrame(rows)


class TestSchemeSet:
    def test_exactly_three(self):
        assert SCHEMES == ("sorted_ngrams", "leading_token", "buyer")

    def test_no_manufacturer_scheme_exists(self):
        # manufacturer is null across both corpora; keying on it yields one enormous
        # block or none at all.
        import fcesreg.blocking as b

        assert not hasattr(b, "block_by_manufacturer")


class TestSortedNgrams:
    def test_word_order_does_not_change_the_key(self):
        df = frame(
            [
                {"record_id": "1", "title": "microscope zeiss"},
                {"record_id": "2", "title": "zeiss microscope"},
            ]
        )
        blocks = block_by_sorted_ngrams(df)
        assert len(blocks) == 1
        assert sorted(next(iter(blocks.values()))) == ["1", "2"]

    def test_short_tokens_contribute_themselves(self):
        # A short but distinctive token ("a4", "ph") must not be silently dropped.
        blocks = block_by_sorted_ngrams(frame([{"record_id": "1", "title": "ab"}]), n=3)
        assert blocks == {"ab": ["1"]}

    def test_only_an_empty_title_is_unblocked(self):
        df = frame([{"record_id": "1", "title": "   "}])
        assert block_by_sorted_ngrams(df, n=3) == {}

    def test_cross_word_grams_are_not_generated(self):
        # "eze" would exist only if the title were concatenated before gramming, and its
        # presence would make the key depend on word order.
        blocks = block_by_sorted_ngrams(
            frame([{"record_id": "1", "title": "microscope zeiss"}]), n=3, k=100
        )
        assert "eze" not in next(iter(blocks)).split("|")

    def test_applies_to_records_with_no_buyer(self):
        df = frame([{"record_id": "A:1", "title": "canon powershot camera"}])
        assert len(block_by_sorted_ngrams(df)) == 1


class TestNgramOverlap:
    def test_threshold_of_one_is_plain_qgram_indexing(self):
        df = frame(
            [
                {"record_id": "1", "title": "zeiss microscope"},
                {"record_id": "2", "title": "zeiss telescope"},
            ]
        )
        pairs, _ = ngram_overlap_candidates(df, n=3, min_overlap=1)
        assert len(pairs) == 1

    def test_raising_the_threshold_prunes(self):
        df = frame(
            [
                {"record_id": "1", "title": "zeiss microscope"},
                {"record_id": "2", "title": "zeiss telescope"},
            ]
        )
        # They share the "zeiss" grams and "scope"; demanding more separates them.
        assert len(ngram_overlap_candidates(df, n=3, min_overlap=50)[0]) == 0

    def test_pairs_are_upper_triangle_only(self):
        df = frame(
            [{"record_id": str(i), "title": "zeiss microscope unit"} for i in range(4)]
        )
        pairs, _ = ngram_overlap_candidates(df, n=3, min_overlap=1)
        assert len(pairs) == 6  # 4 choose 2, each once
        assert (pairs["left_id"] < pairs["right_id"]).all()

    def test_identical_records_always_survive_any_threshold_they_can_meet(self):
        df = frame(
            [
                {"record_id": "1", "title": "rotary vane vacuum pump"},
                {"record_id": "2", "title": "rotary vane vacuum pump"},
            ]
        )
        pairs, _ = ngram_overlap_candidates(df, n=3, min_overlap=8)
        assert len(pairs) == 1

    def test_oversized_gram_blocks_are_dropped_and_counted(self):
        # A gram shared by everything carries no evidence of identity.
        df = frame([{"record_id": str(i), "title": "aaa"} for i in range(20)])
        _, report = ngram_overlap_candidates(df, n=3, min_overlap=1, max_block_size=5)
        assert report.blocks_dropped == 1
        assert report.records_in_dropped_blocks == 20
        assert report.n_unblocked_records == 20

    def test_report_carries_the_threshold_it_used(self):
        df = frame([{"record_id": "1", "title": "zeiss microscope"}])
        _, report = ngram_overlap_candidates(df, n=3, min_overlap=4)
        assert report.extras["min_overlap"] == 4
        assert report.extras["mode"] == "per_gram"

    def test_chunking_does_not_change_the_result(self):
        df = frame(
            [{"record_id": str(i), "title": f"zeiss microscope model {i}"} for i in range(30)]
        )
        a, _ = ngram_overlap_candidates(df, n=3, min_overlap=3, chunk_rows=5)
        b, _ = ngram_overlap_candidates(df, n=3, min_overlap=3, chunk_rows=1000)
        assert len(a) == len(b)
        assert set(map(tuple, a.values)) == set(map(tuple, b.values))


class TestLeadingToken:
    def test_skips_procurement_stopwords(self):
        df = frame(
            [
                {"record_id": "1", "title": "Supply of Zeiss microscopes"},
                {"record_id": "2", "title": "Zeiss microscope units"},
            ]
        )
        blocks = block_by_leading_token(df)
        assert set(blocks) == {"zeiss"}
        assert sorted(blocks["zeiss"]) == ["1", "2"]

    def test_skips_bare_quantities(self):
        df = frame([{"record_id": "1", "title": "10 microscopes"}])
        assert set(block_by_leading_token(df)) == {"microscopes"}

    def test_title_of_only_stopwords_is_unblocked_and_counted(self):
        df = frame(
            [
                {"record_id": "1", "title": "supply and provision of the contract"},
                {"record_id": "2", "title": "Zeiss microscope"},
            ]
        )
        blocks = block_by_leading_token(df)
        assert set(blocks) == {"zeiss"}

        _, reports = candidate_pairs(df, ["leading_token"])
        assert reports[0].n_unblocked_records == 1

    def test_stopword_list_is_fixed_not_tunable(self):
        # Documented as part of the scheme and fixed before measurement.
        assert "supply" in LEADING_STOPWORDS
        assert "zeiss" not in LEADING_STOPWORDS


class TestBuyerScheme:
    def test_groups_by_buyer(self):
        df = frame(
            [
                {"record_id": "1", "title": "x", "buyer_id": "GB-1"},
                {"record_id": "2", "title": "y", "buyer_id": "GB-1"},
                {"record_id": "3", "title": "z", "buyer_id": "GB-2"},
            ]
        )
        blocks = block_by_buyer(df)
        assert sorted(blocks["GB-1"]) == ["1", "2"]
        assert blocks["GB-2"] == ["3"]

    def test_raises_when_the_column_is_wholly_null(self):
        df = frame([{"record_id": "A:1", "title": "x", "buyer_id": None}])
        with pytest.raises(SchemeUnavailable, match="reported result"):
            block_by_buyer(df)

    def test_raises_when_the_column_is_absent(self):
        with pytest.raises(SchemeUnavailable):
            block_by_buyer(frame([{"record_id": "1", "title": "x"}]))


class TestApplicableSchemes:
    def test_corpus_with_buyers_supports_all_three(self):
        df = frame([{"record_id": "1", "title": "x", "buyer_id": "GB-1"}])
        assert applicable_schemes(df) == ["sorted_ngrams", "leading_token", "buyer"]

    def test_corpus_without_buyers_supports_two(self):
        df = frame([{"record_id": "A:1", "title": "x", "buyer_id": None}])
        assert applicable_schemes(df) == ["sorted_ngrams", "leading_token"]

    def test_decided_by_data_not_by_corpus_name(self):
        # Nothing here inspects `source`; availability follows the column.
        df = frame([{"record_id": "1", "title": "x", "source": "contractsfinder"}])
        assert "buyer" not in applicable_schemes(df)


class TestCandidatePairs:
    def test_pairs_are_ordered_and_deduplicated_across_schemes(self):
        df = frame(
            [
                {"record_id": "b", "title": "zeiss microscope", "buyer_id": "GB-1"},
                {"record_id": "a", "title": "zeiss microscope", "buyer_id": "GB-1"},
            ]
        )
        pairs, _ = candidate_pairs(df, ["sorted_ngrams", "leading_token", "buyer"])
        assert len(pairs) == 1
        assert pairs.iloc[0]["left_id"] < pairs.iloc[0]["right_id"]

    def test_oversized_blocks_are_dropped_and_counted(self):
        df = frame(
            [{"record_id": str(i), "title": "identical title", "buyer_id": "GB-1"}
             for i in range(10)]
        )
        pairs, reports = candidate_pairs(df, ["buyer"], max_block_size=5)
        assert len(pairs) == 0
        assert reports[0].blocks_dropped == 1
        assert reports[0].records_in_dropped_blocks == 10

    def test_report_records_the_largest_block(self):
        df = frame(
            [{"record_id": str(i), "title": "same title here", "buyer_id": "GB-1"}
             for i in range(4)]
        )
        _, reports = candidate_pairs(df, ["leading_token"])
        assert reports[0].largest_block == 4
        assert reports[0].n_candidates == 6  # 4 choose 2

    def test_unknown_scheme_rejected(self):
        with pytest.raises(ValueError, match="unknown scheme"):
            candidate_pairs(frame([{"record_id": "1", "title": "x"}]), ["manufacturer"])


class TestEvaluateBlocking:
    def test_recovers_a_known_answer(self):
        # Four records, one true match (1,2), which blocking retains.
        candidates = pd.DataFrame(
            {"left_id": ["1", "1"], "right_id": ["2", "3"]}
        )
        truth = pd.DataFrame(
            {"left_id": ["1", "1"], "right_id": ["2", "4"], "label": [1, 1]}
        )
        got = evaluate_blocking(candidates, truth, n_records=4)

        assert got["pair_completeness"] == 0.5  # kept (1,2), lost (1,4)
        assert got["n_possible"] == 6
        assert got["reduction_ratio"] == pytest.approx(1 - 2 / 6)
        assert got["n_true_positives_lost"] == 1

    def test_perfect_blocking(self):
        candidates = pd.DataFrame({"left_id": ["1"], "right_id": ["2"]})
        truth = pd.DataFrame({"left_id": ["1"], "right_id": ["2"], "label": [1]})
        got = evaluate_blocking(candidates, truth, n_records=2)
        assert got["pair_completeness"] == 1.0
        assert got["reduction_ratio"] == 0.0  # only one possible pair, and we made it

    def test_pair_order_does_not_affect_matching(self):
        candidates = pd.DataFrame({"left_id": ["2"], "right_id": ["1"]})
        truth = pd.DataFrame({"left_id": ["1"], "right_id": ["2"], "label": [1]})
        assert evaluate_blocking(candidates, truth, n_records=2)["pair_completeness"] == 1.0

    def test_negatives_do_not_count_towards_completeness(self):
        candidates = pd.DataFrame({"left_id": ["1"], "right_id": ["2"]})
        truth = pd.DataFrame(
            {"left_id": ["1", "3"], "right_id": ["2", "4"], "label": [1, 0]}
        )
        got = evaluate_blocking(candidates, truth, n_records=4)
        assert got["n_true_positives"] == 1
        assert got["pair_completeness"] == 1.0


@requires(CORPUS_A, CORPUS_A_PAIRS)
class TestOnRealCorpusA:
    """Ranges and consistency only. The measured values are run_blocking's to report."""

    @pytest.fixture(scope="class")
    @classmethod
    def records(cls):
        return pd.read_parquet(CORPUS_A)

    def test_buyer_scheme_is_unavailable_here(self, records):
        # The asymmetry between the corpora, asserted rather than averaged away.
        assert applicable_schemes(records) == ["sorted_ngrams", "leading_token"]

    def test_metrics_are_in_range_and_consistent(self, records):
        pairs, reports = candidate_pairs(records, applicable_schemes(records))
        truth = pd.read_parquet(CORPUS_A_PAIRS)
        got = evaluate_blocking(pairs, truth, n_records=len(records))

        assert 0.0 <= got["pair_completeness"] <= 1.0
        assert 0.0 <= got["reduction_ratio"] <= 1.0
        assert got["n_candidates"] <= got["n_possible"]
        assert (
            got["n_true_positives_retained"] + got["n_true_positives_lost"]
            == got["n_true_positives"]
        )
        for r in reports:
            assert r.n_candidates >= 0
            assert r.n_unblocked_records <= r.n_records
