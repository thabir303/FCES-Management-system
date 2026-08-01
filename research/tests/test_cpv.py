"""B3 acceptance: two levels only, hierarchy by truncation, coverage reported."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import TAXONOMY, requires
from fcesreg.cpv import (
    LEVELS,
    division,
    cpv_class,
    label_series,
    leaf_sparsity,
    supported_labels,
)


def corpus(codes):
    return pd.DataFrame({"cpv_code": codes})


class TestTruncation:
    def test_division_is_two_digits(self):
        assert division("38510000") == "38"

    def test_class_is_four_digits(self):
        assert cpv_class("38510000") == "3851"

    def test_only_two_levels_exist(self):
        assert LEVELS == ("division", "class")

    def test_leaf_level_is_rejected(self):
        # §4.2: eight-digit classification is not viable and must not be built.
        with pytest.raises(ValueError, match="level must be one of"):
            label_series(corpus(["38510000"]), "leaf")


class TestSupportedLabels:
    def test_applies_the_minimum_example_floor(self):
        df = corpus(["3851" + "0000"] * 60 + ["4212" + "0000"] * 10)
        labels, coverage = supported_labels(df, "class", min_examples=50)
        assert labels == {"3851"}
        assert coverage == pytest.approx(60 / 70)

    def test_coverage_is_a_fraction(self):
        df = corpus(["38510000"] * 100)
        _, coverage = supported_labels(df, "division", min_examples=50)
        assert 0.0 <= coverage <= 1.0
        assert coverage == 1.0

    def test_no_label_meets_the_floor(self):
        labels, coverage = supported_labels(corpus(["38510000"] * 3), "class", 50)
        assert labels == set()
        assert coverage == 0.0

    def test_empty_frame(self):
        labels, coverage = supported_labels(corpus([]), "class", 50)
        assert labels == set()
        assert coverage == 0.0


class TestLeafSparsity:
    def test_reports_the_counts_the_paper_needs(self):
        df = corpus(["38510000"] * 60 + ["38520000"] * 25 + ["38530000"] + ["38540000"])
        got = leaf_sparsity(df)
        assert got["n_distinct_leaves"] == 4
        assert got["n_leaves_ge_20"] == 2
        assert got["n_leaves_ge_50"] == 1
        assert got["n_singleton_leaves"] == 2
        assert got["n_records"] == 87


@requires(TAXONOMY)
class TestBuiltTaxonomy:
    """Against the real taxonomy built from all four bundles."""

    @pytest.fixture(scope="class")
    @classmethod
    def tax(cls):
        return pd.read_parquet("data/processed/cpv_taxonomy.parquet")

    def test_only_levels_2_4_and_8_exist(self, tax):
        assert set(tax["level"]) == {2, 4, 8}

    def test_code_length_matches_level(self, tax):
        for lvl, width in ((2, 2), (4, 4), (8, 8)):
            sub = tax[tax["level"] == lvl]
            assert (sub["cpv_code"].str.len() == width).all()

    def test_every_parent_exists(self, tax):
        # A class with no division row would break the categories foreign key.
        known = set(tax["cpv_code"])
        parents = set(tax["parent_code"].dropna())
        assert parents <= known

    def test_divisions_have_no_parent(self, tax):
        assert tax.loc[tax["level"] == 2, "parent_code"].isna().all()

    def test_parent_is_a_truncation_of_the_child(self, tax):
        child = tax[tax["level"] != 2]
        assert (
            child.apply(lambda r: r["cpv_code"].startswith(r["parent_code"]), axis=1)
        ).all()

    def test_codes_are_unique(self, tax):
        assert tax["cpv_code"].is_unique

    def test_the_hazard_designated_divisions_are_present(self, tax):
        # 33, 38 and 42 carry the illustrative hazard_class in seed_categories.py (§5.4).
        divisions = set(tax.loc[tax["level"] == 2, "cpv_code"])
        assert {"33", "38", "42"} <= divisions
