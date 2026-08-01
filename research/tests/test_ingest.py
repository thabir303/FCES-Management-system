"""B2/B4 acceptance: discard accounting, loud schema failure, supplied splits."""

from __future__ import annotations

import csv

import pandas as pd
import pytest

from fcesreg.ingest_abtbuy import load as load_abtbuy
from fcesreg.ingest_contractsfinder import (
    BOILERPLATE_BLOCKLIST,
    EXPECTED_COLUMNS,
    FIELD_MAP,
    DiscardReport,
    SchemaMismatch,
    apply_filters,
    load_bundle,
    to_records,
)
from conftest import ABTBUY_RAW, requires
from fcesreg.schema import RECORD_COLUMNS

LONG = "supply and installation of one laboratory microscope with warranty and training"


def write_bundle(tmp_path, rows, columns=None):
    """A minimal main.csv carrying the full verified header."""
    d = tmp_path / "2099"
    d.mkdir(parents=True, exist_ok=True)
    cols = sorted(columns if columns is not None else EXPECTED_COLUMNS)
    with (d / "main.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return d


def row(i, *, cpv="38510000", title="Zeiss microscope", desc=LONG, buyer="GB-1"):
    return {
        "id": f"cf-{i}",
        "tender_title": title,
        "tender_description": desc,
        "buyer_id": buyer,
        "tender_classification_id": cpv,
        "date": "2025-03-04T09:00:00Z",
    }


class TestSchemaVerification:
    def test_accepts_the_verified_header(self, tmp_path):
        df = load_bundle(write_bundle(tmp_path, [row(1)]))
        assert len(df) == 1

    def test_renamed_column_fails_loudly_and_names_it(self, tmp_path):
        # §13.6: never coerce. A rename must stop the build and say what changed.
        cols = (EXPECTED_COLUMNS - {"tender_title"}) | {"tender_titleText"}
        d = write_bundle(tmp_path, [], columns=cols)
        with pytest.raises(SchemaMismatch) as e:
            load_bundle(d)
        assert "tender_title" in str(e.value)
        assert "tender_titleText" in str(e.value)
        assert "Do not coerce" in str(e.value)

    def test_extra_column_also_fails(self, tmp_path):
        d = write_bundle(tmp_path, [], columns=EXPECTED_COLUMNS | {"tender_newField"})
        with pytest.raises(SchemaMismatch, match="tender_newField"):
            load_bundle(d)

    def test_column_order_does_not_matter(self, tmp_path):
        # The four real bundles differ in order only; every read is by name.
        d = write_bundle(tmp_path, [row(1)])
        header = (d / "main.csv").read_text(encoding="utf-8").splitlines()[0]
        assert set(header.split(",")) == EXPECTED_COLUMNS
        assert len(load_bundle(d)) == 1

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_bundle(tmp_path / "nope")


class TestToRecords:
    def test_produces_record_columns(self, tmp_path):
        out = to_records(load_bundle(write_bundle(tmp_path, [row(1)])), 2099)
        assert set(RECORD_COLUMNS).issubset(out.columns)
        assert out["source"].iloc[0] == "contractsfinder"
        assert out["bundle_year"].iloc[0] == 2099

    def test_the_three_absent_fields_are_null(self, tmp_path):
        out = to_records(load_bundle(write_bundle(tmp_path, [row(1)])), 2099)
        for col in ("manufacturer", "model", "serial_number"):
            assert out[col].isna().all()

    def test_release_date_comes_from_date_not_date_published(self):
        # §4.1: tender_datePublished is populated for a minority of rows.
        assert FIELD_MAP["release_date"] == "date"

    def test_parses_iso_timestamps(self, tmp_path):
        out = to_records(load_bundle(write_bundle(tmp_path, [row(1)])), 2099)
        assert out["release_date"].iloc[0].isoformat() == "2025-03-04"


class TestDiscardAccounting:
    def test_counts_reconcile(self, tmp_path):
        rows = [
            row(1),
            row(2, cpv="71220000"),  # out of scope
            row(3, title="Pump", desc="Short"),  # short
            row(4, title="Rotary vane pump for vacuum line in the physics laboratory",
                desc="Rotary vane pump for vacuum line in the physics laboratory"),
            # Title long enough to clear the 60-character rule, so this row reaches the
            # boilerplate rule rather than being dropped as short first.
            row(5, title="Supply of laboratory microscopes to the school of engineering",
                desc="As per tender"),
        ]
        df = to_records(load_bundle(write_bundle(tmp_path, rows)), 2099)
        kept, report = apply_filters(df)

        assert report.total_in == 5
        assert report.dropped_out_of_scope == 1
        assert report.dropped_short == 1
        assert report.dropped_desc_equals_title == 1
        assert report.dropped_boilerplate == 1
        assert report.total_out == 1
        assert len(kept) == 1
        report.check()

    def test_check_catches_a_broken_tally(self):
        bad = DiscardReport(100, 10, 10, 10, 10, 99)
        with pytest.raises(AssertionError, match="do not reconcile"):
            bad.check()

    def test_each_row_is_attributed_to_the_first_rule_it_fails(self, tmp_path):
        # A row that is BOTH short and desc==title counts once, as short. This is what
        # makes the five counts sum to the total dropped.
        rows = [row(9, title="Pump", desc="Pump")]
        df = to_records(load_bundle(write_bundle(tmp_path, rows)), 2099)
        _, report = apply_filters(df)
        assert report.dropped_short == 1
        assert report.dropped_desc_equals_title == 0
        report.check()

    def test_blocklist_entries_are_stored_normalised(self):
        # They are compared against normalise_text(description), so an entry carrying
        # raw punctuation would never match anything.
        from fcesreg.normalise import normalise_text

        for entry in BOILERPLATE_BLOCKLIST:
            assert normalise_text(entry) == entry

    def test_division_filter_uses_the_first_two_digits(self, tmp_path):
        rows = [row(1, cpv="38510000"), row(2, cpv="48000000")]
        df = to_records(load_bundle(write_bundle(tmp_path, rows)), 2099)
        kept, _ = apply_filters(df, divisions={"38"})
        assert kept["record_id"].tolist() == ["cf-1"]


@requires(ABTBUY_RAW / "tableA.csv")
class TestAbtBuy:
    @pytest.fixture(scope="class")
    @classmethod
    def loaded(cls):
        return load_abtbuy("data/raw/abtbuy")

    def test_record_ids_are_prefixed_into_one_namespace(self, loaded):
        records, _ = loaded
        assert records["record_id"].str.startswith(("A:", "B:")).all()
        assert records["record_id"].is_unique

    def test_shapes_match_the_published_benchmark(self, loaded):
        records, pairs = loaded
        assert (records["table"] == "A").sum() == 1081
        assert (records["table"] == "B").sum() == 1092
        assert len(pairs) == 9575

    def test_supplied_splits_are_used_as_given(self, loaded):
        _, pairs = loaded
        sizes = pairs["split"].value_counts()
        assert sizes["train"] == 5743
        assert sizes["valid"] == 1916
        assert sizes["test"] == 1916

    def test_test_split_positive_rate(self, loaded):
        _, pairs = loaded
        rate = pairs.loc[pairs["split"] == "test", "label"].mean()
        assert 0.09 < rate < 0.13, f"positive rate {rate:.3f} is far from the published 10.7%"

    def test_buyer_id_is_null_so_the_buyer_scheme_cannot_apply(self, loaded):
        # This is what blocking.applicable_schemes keys on for Abt-Buy.
        records, _ = loaded
        assert records["buyer_id"].isna().all()
