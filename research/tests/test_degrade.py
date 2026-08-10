"""C4 acceptance: seven error classes, each with its own test, and byte-identical replay."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fcesreg.degrade import (
    ERROR_CLASSES,
    procurement_ref,
    DegradationConfig,
    abbreviate,
    char_noise,
    degrade_frame,
    degrade_record,
    load_lexicon,
    make_distractors,
    make_duplicate_pairs,
    merge_fields,
    omit_field,
    perturb_whitespace,
    vary_case,
    vary_units,
)

LEX = {"laboratory": "lab", "equipment": "equip", "temperature": "temp"}


def rng(seed=0):
    return np.random.default_rng(seed)


def records(n=6):
    return pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(n)],
            "title": [f"Laboratory microscope model {i} 230V" for i in range(n)],
            "description": [f"Supply of laboratory equipment item {i}" for i in range(n)],
            "cpv_code": ["38510000"] * n,
            "buyer_id": ["GB-1"] * n,
        }
    )


class TestConfig:
    def test_seven_classes_and_seven_knobs(self):
        assert len(ERROR_CLASSES) == 7
        cfg = DegradationConfig(severity=0.5)
        for cls in ERROR_CLASSES:
            assert hasattr(cfg, f"p_{cls}")

    def test_rate_is_severity_times_multiplier(self):
        cfg = DegradationConfig(severity=0.4, p_charnoise=0.5)
        assert cfg.rate("charnoise") == pytest.approx(0.2)

    def test_rate_is_capped_at_one(self):
        assert DegradationConfig(severity=0.9, p_case=4.0).rate("case") == 1.0

    def test_severity_out_of_range_raises(self):
        with pytest.raises(ValueError, match="severity"):
            DegradationConfig(severity=1.5)

    def test_unknown_class_raises(self):
        with pytest.raises(ValueError, match="unknown error class"):
            DegradationConfig(severity=0.5).rate("gremlins")

    def test_zero_severity_changes_nothing(self):
        cfg = DegradationConfig(severity=0.0)
        rec = {"title": "Laboratory microscope 230V", "description": "Supply of equipment"}
        assert degrade_record(rec, cfg, rng(), LEX) == rec


class TestAbbreviate:
    def test_replaces_whole_words_from_the_lexicon(self):
        assert abbreviate("laboratory equipment", rng(), LEX, 1.0) == "lab equip"

    def test_leaves_words_absent_from_the_lexicon(self):
        assert abbreviate("microscope stand", rng(), LEX, 1.0) == "microscope stand"

    def test_does_not_corrupt_a_longer_word_containing_a_key(self):
        # "laboratories" must not become "labies".
        assert abbreviate("laboratories", rng(), LEX, 1.0) == "laboratories"

    def test_preserves_all_caps(self):
        assert abbreviate("LABORATORY", rng(), LEX, 1.0) == "LAB"

    def test_rate_zero_is_identity(self):
        assert abbreviate("laboratory", rng(), LEX, 0.0) == "laboratory"

    def test_the_shipped_lexicon_loads(self):
        lex = load_lexicon()
        assert lex["laboratory"] == "lab"
        assert all(k == k.lower() for k in lex)


class TestCharNoise:
    def test_changes_text_at_full_rate(self):
        assert char_noise("microscope", rng(1), 1.0) != "microscope"

    def test_rate_zero_is_identity(self):
        assert char_noise("microscope", rng(), 0.0) == "microscope"

    def test_never_edits_whitespace(self):
        # Whitespace is the whitespace class's job; both touching it would double-count
        # the class in the degradation check.
        out = char_noise("aaaa bbbb cccc", rng(2), 1.0)
        assert out.count(" ") == 2

    def test_all_four_operations_occur_over_many_draws(self):
        source = "abcdefghij"
        outs = {char_noise(source, rng(s), 0.5) for s in range(60)}
        assert any(len(o) > len(source) for o in outs), "no insertion"
        assert any(len(o) < len(source) for o in outs), "no deletion"
        assert any(len(o) == len(source) and o != source for o in outs), "no sub/transpose"


class TestVaryCase:
    def test_changes_case_at_full_rate(self):
        assert vary_case("microscope stand unit", rng(3), 1.0) != "microscope stand unit"

    def test_preserves_the_letters_themselves(self):
        out = vary_case("microscope stand", rng(3), 1.0)
        assert out.lower() == "microscope stand"

    def test_rate_zero_is_identity(self):
        assert vary_case("Microscope", rng(), 0.0) == "Microscope"


class TestPerturbWhitespace:
    def test_changes_spacing_at_full_rate(self):
        assert perturb_whitespace("a b c", rng(4), 1.0) != "a b c"

    def test_preserves_the_non_space_characters(self):
        out = perturb_whitespace("abc def ghi", rng(4), 1.0)
        assert out.replace(" ", "") == "abcdefghi"

    def test_rate_zero_is_identity(self):
        assert perturb_whitespace("a b", rng(), 0.0) == "a b"


class TestVaryUnits:
    def test_alters_unit_notation(self):
        # The identity is one of the reachable forms — "230V" can be redrawn as "230V" —
        # so the claim is that variation occurs across draws, not on every draw.
        outs = {vary_units("230V supply", rng(s), 1.0) for s in range(20)}
        assert len(outs) > 1
        assert "230 v supply" in {o.lower() for o in outs}

    def test_keeps_the_numeric_value(self):
        out = vary_units("motor rated 1.5kW", rng(5), 1.0)
        assert "1.5" in out or "1,5" in out

    def test_leaves_text_without_units_alone(self):
        assert vary_units("microscope stand", rng(), 1.0) == "microscope stand"

    def test_rate_zero_is_identity(self):
        assert vary_units("230V", rng(), 0.0) == "230V"


class TestMergeFields:
    def test_moves_description_into_title(self):
        out = merge_fields({"title": "Pump", "description": "vacuum, rotary"}, rng(), 1.0)
        assert out["title"] == "Pump vacuum, rotary"
        assert out["description"] is None

    def test_rate_zero_is_identity(self):
        rec = {"title": "Pump", "description": "d"}
        assert merge_fields(rec, rng(), 0.0) == rec

    def test_does_not_mutate_the_input(self):
        rec = {"title": "Pump", "description": "d"}
        merge_fields(rec, rng(), 1.0)
        assert rec["description"] == "d"


class TestNullDescriptions:
    """pandas stores a missing string as float nan, which is truthy.

    20% of Abt-Buy records carry a null description, so every path here is the common
    case on Corpus A rather than an edge case.
    """

    def test_merge_does_not_plant_the_literal_string_nan(self):
        # The dangerous failure: it does not raise, it writes "Pump nan" and carries on.
        # Both copies of a degraded pair would receive the same spurious token, making
        # duplicates easier to match and flattering every figure downstream.
        out = merge_fields({"title": "Pump", "description": float("nan")}, rng(), 1.0)
        assert out["title"] == "Pump"
        assert "nan" not in out["title"]

    def test_merge_leaves_a_null_description_alone(self):
        out = merge_fields({"title": "Pump", "description": None}, rng(), 1.0)
        assert out["title"] == "Pump"

    def test_degrade_record_survives_a_null_description(self):
        out = degrade_record(
            {"title": "Pump", "description": float("nan")}, DegradationConfig(0.5), rng()
        )
        assert out["title"]
        assert not isinstance(out["description"], str)

    def test_degrade_frame_survives_a_frame_of_them(self):
        frame = pd.DataFrame(
            {
                "record_id": ["a", "b"],
                "title": ["Pump", "Valve"],
                "description": [float("nan"), "brass"],
            }
        )
        out = degrade_frame(frame, DegradationConfig(0.5), seed=0)
        assert len(out) == 2
        assert not out["title"].str.contains(r"\bnan\b", na=False).any()


class TestOmitField:
    def test_drops_the_description(self):
        assert omit_field({"title": "Pump", "description": "d"}, rng(), 1.0)["description"] is None

    def test_never_drops_the_title(self):
        # A record with no title cannot be blocked, scored or reviewed.
        out = omit_field({"title": "Pump", "description": "d"}, rng(), 1.0)
        assert out["title"] == "Pump"

    def test_rate_zero_is_identity(self):
        rec = {"title": "Pump", "description": "d"}
        assert omit_field(rec, rng(), 0.0) == rec


class TestDeterminism:
    def test_same_seed_gives_byte_identical_frames(self):
        cfg = DegradationConfig(severity=0.6)
        a = degrade_frame(records(), cfg, seed=7)
        b = degrade_frame(records(), cfg, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_gives_different_output(self):
        cfg = DegradationConfig(severity=0.6)
        a = degrade_frame(records(), cfg, seed=7)
        b = degrade_frame(records(), cfg, seed=8)
        assert not a["title"].equals(b["title"])

    def test_no_module_level_random_state(self):
        # Drawing in between must not shift the result.
        cfg = DegradationConfig(severity=0.6)
        a = degrade_frame(records(), cfg, seed=7)
        np.random.default_rng(999).random(1000)
        b = degrade_frame(records(), cfg, seed=7)
        pd.testing.assert_frame_equal(a, b)


class TestDuplicatePairs:
    def test_one_positive_pair_per_source_record(self):
        degraded, pairs = make_duplicate_pairs(records(6), DegradationConfig(0.5), seed=1)
        assert len(pairs) == 6
        assert (pairs["label"] == 1).all()
        assert len(degraded) == 12

    def test_the_two_copies_are_drawn_independently(self):
        # Two staff entering the same item without reference to one another: the copies
        # must differ from each other, not merely from the source.
        degraded, pairs = make_duplicate_pairs(records(8), DegradationConfig(0.7), seed=1)
        by_id = degraded.set_index("record_id")["title"]
        differing = sum(
            by_id[l] != by_id[r] for l, r in zip(pairs["left_id"], pairs["right_id"], strict=True)
        )
        assert differing > 0

    def test_ids_are_suffixed_and_unique(self):
        degraded, pairs = make_duplicate_pairs(records(4), DegradationConfig(0.5), seed=1)
        assert degraded["record_id"].is_unique
        assert pairs["left_id"].str.endswith("::a").all()
        assert pairs["right_id"].str.endswith("::b").all()


class TestDistractors:
    def test_cf_rule_mines_within_a_class(self):
        df = pd.DataFrame(
            {
                "record_id": ["a", "b", "c"],
                "title": [
                    "rotary vane vacuum pump",
                    "rotary vane vacuum pumps",
                    "office chair",
                ],
                "cpv_code": ["38510000", "38510000", "39110000"],
                "buyer_id": ["GB-1", "GB-2", "GB-3"],
                "tender_ref": ["REF-1", "REF-2", "REF-3"],
            }
        )
        out = make_distractors(df, DegradationConfig(0.3), seed=0, corpus="cf", sim_threshold=0.6)
        got = {tuple(sorted(p)) for p in zip(out["left_id"], out["right_id"], strict=True)}
        assert ("a", "b") in got
        assert ("a", "c") not in got  # different class

    def test_abtbuy_rule_mines_by_leading_token(self):
        df = pd.DataFrame(
            {
                "record_id": ["A:1", "A:2", "A:3"],
                "title": ["Canon PowerShot A", "Canon EOS 400", "Nikon D40"],
                "cpv_code": [None, None, None],
            }
        )
        out = make_distractors(df, DegradationConfig(0.3), seed=0, corpus="abtbuy")
        got = {tuple(sorted(p)) for p in zip(out["left_id"], out["right_id"], strict=True)}
        assert ("A:1", "A:2") in got
        assert ("A:1", "A:3") not in got

    def test_all_distractors_are_negatives(self):
        out = make_distractors(records(6), DegradationConfig(0.3), seed=0, corpus="cf",
                               sim_threshold=0.5)
        assert (out["label"] == 0).all()

    def test_never_pairs_a_record_with_itself(self):
        out = make_distractors(records(6), DegradationConfig(0.3), seed=0, corpus="cf",
                               sim_threshold=0.5)
        assert (out["left_id"] != out["right_id"]).all()

    def test_touches_none_of_the_three_null_columns(self):
        import inspect

        import fcesreg.degrade as d

        source = inspect.getsource(d)
        for column in ("manufacturer", "model", "serial_number"):
            assert f'"{column}"' not in source

    def test_pairs_sharing_a_procurement_reference_are_excluded(self):
        # An award notice and its tender notice are one procurement, and labelling that
        # pair 0 puts a positive in the negative set.
        df = pd.DataFrame(
            {
                "record_id": ["a", "b"],
                "title": ["rotary vane vacuum pump", "rotary vane vacuum pump"],
                "cpv_code": ["38510000", "38510000"],
                "buyer_id": ["GB-1", "GB-2"],
                "tender_ref": ["IT-368-17809", "IT-368-17809 - AWARD"],
            }
        )
        out = make_distractors(df, DegradationConfig(0.0), seed=0, corpus="cf",
                               sim_threshold=0.5)
        assert out.empty

    def test_same_buyer_with_an_identical_title_is_excluded(self):
        df = pd.DataFrame(
            {
                "record_id": ["a", "b"],
                "title": ["Grounds Maintenance Equipment", "grounds  maintenance equipment"],
                "cpv_code": ["38510000", "38510000"],
                "buyer_id": ["GB-1", "GB-1"],
                "tender_ref": ["REF-1", "REF-2"],
            }
        )
        out = make_distractors(df, DegradationConfig(0.0), seed=0, corpus="cf",
                               sim_threshold=0.5)
        assert out.empty

    def test_missing_tender_ref_warns_rather_than_mining_silently(self):
        df = pd.DataFrame(
            {
                "record_id": ["a", "b"],
                "title": ["rotary vane pump", "rotary vane pumps"],
                "cpv_code": ["38510000", "38510000"],
                "buyer_id": ["GB-1", "GB-2"],
            }
        )
        with pytest.warns(UserWarning, match="48%"):
            make_distractors(df, DegradationConfig(0.0), seed=0, corpus="cf",
                             sim_threshold=0.5)

    def test_procurement_ref_strips_stage_suffixes(self):
        assert procurement_ref("IT-368-17809 - AWARD") == procurement_ref("IT-368-17809")
        assert procurement_ref("ABC/1 — Cancellation") == procurement_ref("ABC/1")
        assert procurement_ref(None) == ""

    def test_unknown_corpus_raises(self):
        with pytest.raises(ValueError, match="corpus must be"):
            make_distractors(records(), DegradationConfig(0.3), seed=0, corpus="walmart")
