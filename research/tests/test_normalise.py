"""A2 acceptance: normalisation edge cases, with mojibake and control characters covered."""

from __future__ import annotations

import pandas as pd
import pytest

from fcesreg.normalise import (
    fix_mojibake,
    normalise_frame,
    normalise_key,
    normalise_text,
    strip_control,
)


class TestFixMojibake:
    @pytest.mark.parametrize(
        ("damaged", "expected"),
        [
            ("Weâ€™re supplying", "We’re supplying"),
            ("â€œquotedâ€\x9d", "“quoted”"),
            ("Ã©lectrique", "électrique"),
            ("5Â°C incubator", "5°C incubator"),
        ],
    )
    def test_repairs_cp1252_round_trip(self, damaged, expected):
        assert fix_mojibake(damaged) == expected

    def test_repairs_closing_quote_through_undefined_cp1252_slot(self):
        # U+009D is one of the five slots cp1252 leaves undefined. A strict encode would
        # raise here and leave the commonest damaged character in procurement text — the
        # closing smart quote — unrepaired.
        assert fix_mojibake("â€\x9d") == "”"

    def test_strips_replacement_character(self):
        assert fix_mojibake("micro�scope") == "microscope"

    def test_leaves_clean_text_untouched(self):
        for clean in ["Zeiss microscope", "café", "5°C", "±0.1mm", ""]:
            assert fix_mojibake(clean) == clean

    def test_does_not_raise_on_undecodable_sequence(self):
        # Contains a signal character but is not a valid round trip.
        assert isinstance(fix_mojibake("Â at the end Â"), str)


class TestStripControl:
    def test_drops_c0_controls(self):
        assert strip_control("micro\x00scope\x07") == "microscope"

    def test_keeps_newline_tab_return(self):
        assert strip_control("a\nb\tc\rd") == "a\nb\tc\rd"

    def test_empty(self):
        assert strip_control("") == ""


class TestNormaliseText:
    def test_none_and_nan_become_empty(self):
        assert normalise_text(None) == ""
        assert normalise_text(float("nan")) == ""

    def test_casefold_and_whitespace_collapse(self):
        assert normalise_text("  ZEISS   Axio   Lab  ") == "zeiss axio lab"

    def test_punctuation_becomes_a_space_not_nothing(self):
        # 'pump/valve' must yield two tokens, otherwise character n-grams see a word
        # that does not exist.
        assert normalise_text("pump/valve") == "pump valve"
        assert normalise_text("Model: XR-200.") == "model xr 200"

    def test_symbol_categories_survive_punctuation_does_not(self):
        # ° and ± are Unicode symbols and carry meaning in equipment descriptions.
        # % is category Po, not a symbol, so it goes with the rest of the punctuation.
        assert normalise_text("5°C ±0.1 at 95%") == "5°c ±0 1 at 95"

    def test_apostrophes_are_deleted_not_spaced(self):
        # Intra-word punctuation: "buyer's" must not become two tokens.
        assert normalise_text("the buyer's own equipment") == "the buyers own equipment"
        assert normalise_text("We’re supplying") == "were supplying"

    def test_nfkc_folds_compatibility_forms(self):
        assert normalise_text("ﬁlter") == "filter"
        assert normalise_text("ＭＩＣＲＯＳＣＯＰＥ") == "microscope"

    def test_mojibake_repaired_before_nfkc(self):
        # 'â„¢' is '™' through a cp1252 round trip. NFKC-first would expand the '™'
        # inside the damaged sequence to 'TM' and the repair would no longer round trip.
        assert normalise_text("Acmeâ„¢ pump") == "acmetm pump"

    def test_control_characters_removed(self):
        assert normalise_text("micro\x00scope\x1b") == "microscope"

    def test_idempotent(self):
        once = normalise_text("  Weâ€™re  SUPPLYING\x00 pumps/valves ")
        assert normalise_text(once) == once


class TestNormaliseKey:
    def test_removes_every_non_alphanumeric(self):
        assert normalise_key("FCES-000123") == "fces000123"
        assert normalise_key("5°C ±0.1") == "5c01"

    def test_collapses_spacing_and_punctuation_differences(self):
        # The whole point of the exact baseline: two people, same item, different typing.
        assert normalise_key("Zeiss Axio-Lab A1") == normalise_key("zeiss axio lab a1")
        assert normalise_key("230V") == normalise_key("230 v")

    def test_none_becomes_empty(self):
        assert normalise_key(None) == ""


class TestNormaliseFrame:
    def test_adds_columns_without_destroying_originals(self):
        df = pd.DataFrame(
            {"title": ["Weâ€™re SUPPLYING"], "description": ["a\x00 pump"]}
        )
        out = normalise_frame(df)

        assert out["title"].iloc[0] == "Weâ€™re SUPPLYING"
        assert out["description"].iloc[0] == "a\x00 pump"
        assert out["title_norm"].iloc[0] == "were supplying"
        assert out["description_norm"].iloc[0] == "a pump"
        assert out["title_key"].iloc[0] == "weresupplying"

    def test_missing_column_is_skipped_not_fatal(self):
        out = normalise_frame(pd.DataFrame({"title": ["x"]}))
        assert "title_norm" in out.columns
        assert "description_norm" not in out.columns

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"title": ["x"]})
        normalise_frame(df)
        assert list(df.columns) == ["title"]
