"""B5 acceptance: the partitions are disjoint, frozen, and temporal where they must be.

The guarantee differs by corpus, and deliberately so:

* Contracts Finder — **record-level**. We control the split, so no record_id may appear
  on both sides.
* Abt-Buy — **pair-level**. The splits are supplied and are defined over pairs drawn from
  a fixed record pool, so records recur across sides by the benchmark's own design.
  Re-splitting to force a record-level guarantee would break comparability with every
  published figure on this benchmark, which §4.4 forbids.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from conftest import CORPUS_A_PAIRS, CORPUS_B, SPLITS, requires
from fcesreg.splits import SPLITS_PATH, SplitOverlap, Splits, freeze, load

# Everything below except TestFreezing's pure-logic cases reads the frozen assignment and
# the ingested corpus, neither of which exists on a clean clone.
pytestmark = requires(SPLITS, CORPUS_B, CORPUS_A_PAIRS)


@pytest.fixture(scope="module")
def splits():
    return load()


@pytest.fixture(scope="module")
def corpus():
    return pd.read_parquet(CORPUS_B)


class TestContractsFinder:
    def test_zero_record_id_overlap(self, splits):
        assert splits.cf_dev & splits.cf_test == set()

    def test_partitions_account_for_the_whole_corpus(self, splits, corpus):
        assert splits.cf_dev | splits.cf_test == set(corpus["record_id"])

    def test_both_partitions_are_populated(self, splits):
        assert len(splits.cf_dev) > 0
        assert len(splits.cf_test) > 0

    def test_the_split_is_temporal_not_random(self, splits, corpus):
        dev = splits.cf(corpus, "dev")
        test = splits.cf(corpus, "test")
        assert pd.to_datetime(dev["release_date"]).dt.date.max() < splits.cutoff
        assert pd.to_datetime(test["release_date"]).dt.date.min() >= splits.cutoff

    def test_no_publication_month_straddles_the_split(self, splits, corpus):
        # Near-identical repeat notices from one buyer must not appear on both sides.
        dev_months = set(pd.to_datetime(splits.cf(corpus, "dev")["release_date"]).dt.to_period("M"))
        test_months = set(pd.to_datetime(splits.cf(corpus, "test")["release_date"]).dt.to_period("M"))
        assert dev_months & test_months == set()


class TestAbtBuy:
    def test_zero_pair_overlap(self, splits):
        assert splits.abtbuy_dev_pairs & splits.abtbuy_test_pairs == set()

    def test_test_partition_is_the_supplied_one(self, splits):
        assert len(splits.abtbuy_test_pairs) == 1916

    def test_records_do_recur_across_sides_and_that_is_expected(self, splits):
        # Documenting the benchmark's design rather than asserting a property we want:
        # a record-level guarantee is unobtainable without re-splitting, and §4.4
        # forbids re-splitting.
        dev_records = {r for p in splits.abtbuy_dev_pairs for r in p}
        test_records = {r for p in splits.abtbuy_test_pairs for r in p}
        assert dev_records & test_records, "expected shared records in a pair-level split"


class TestFreezing:
    def test_refuses_to_overwrite(self, tmp_path, corpus):
        target = tmp_path / "splits.json"
        target.write_text("{}", encoding="utf-8")
        pairs = pd.read_parquet(CORPUS_A_PAIRS)
        with pytest.raises(FileExistsError, match="frozen"):
            freeze(corpus.head(10), pairs, path=target)

    def test_overlap_is_rejected_at_construction(self):
        bad = Splits(
            cf_dev={"a", "b"},
            cf_test={"b", "c"},
            abtbuy_dev_pairs=set(),
            abtbuy_test_pairs=set(),
            cutoff=date(2025, 1, 1),
        )
        from fcesreg.splits import _check

        with pytest.raises(SplitOverlap, match="both dev"):
            _check(bad)

    def test_the_frozen_file_is_where_everything_loads_it_from(self):
        assert SPLITS_PATH.exists()
