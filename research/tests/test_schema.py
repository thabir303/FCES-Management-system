"""A2 acceptance: the canonical Record frame, and the null-column warning."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from fcesreg.schema import (
    NULL_IN_BOTH_CORPORA,
    RECORD_COLUMNS,
    Record,
    SchemaError,
    text_of,
    validate_frame,
)


def frame(**overrides) -> pd.DataFrame:
    base = {
        "record_id": ["cf-1", "cf-2"],
        "title": ["Zeiss microscope", "Rotary vane pump"],
        "description": ["Supply of one microscope", "Supply of one pump"],
        "manufacturer": [None, None],
        "model": [None, None],
        "serial_number": [None, None],
        "buyer_id": ["GB-1", "GB-2"],
        "cpv_code": ["38510000", "42123000"],
        "release_date": [date(2025, 1, 5), date(2025, 6, 9)],
        "source": ["contractsfinder", "contractsfinder"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestValidateFrame:
    def test_accepts_a_well_formed_frame(self):
        with pytest.warns(UserWarning):  # the three null columns
            validate_frame(frame())

    def test_missing_columns_raise(self):
        with pytest.raises(SchemaError, match="missing columns"):
            validate_frame(frame().drop(columns=["buyer_id", "cpv_code"]))

    def test_duplicate_record_id_raises(self):
        with pytest.raises(SchemaError, match="not unique"):
            validate_frame(frame(record_id=["cf-1", "cf-1"]))

    def test_null_record_id_raises(self):
        with pytest.raises(SchemaError, match="record_id contains nulls"):
            validate_frame(frame(record_id=["cf-1", None]))

    def test_null_title_raises(self):
        with pytest.raises(SchemaError, match="title contains nulls"):
            validate_frame(frame(title=["ok", None]))

    def test_unknown_source_raises(self):
        with pytest.raises(SchemaError, match="unknown source"):
            validate_frame(frame(source=["contractsfinder", "sharepoint"]))

    def test_bad_release_date_type_raises(self):
        with pytest.raises(SchemaError, match="release_date"):
            validate_frame(frame(release_date=["2025-01-05", "2025-06-09"]))

    def test_datetime_dtype_release_date_is_accepted(self):
        df = frame(release_date=pd.to_datetime(["2025-01-05", "2025-06-09"]))
        with pytest.warns(UserWarning):
            validate_frame(df)

    def test_empty_frame_with_right_columns_is_fine(self):
        validate_frame(pd.DataFrame(columns=RECORD_COLUMNS))


class TestNullColumnWarning:
    def test_warns_when_the_three_fields_are_wholly_null(self):
        # This is the expected condition on BOTH research corpora. It warns rather than
        # raising so the condition is visible in logs instead of assumed.
        with pytest.warns(UserWarning, match="wholly null and unusable for keying"):
            validate_frame(frame())

    def test_names_every_null_column(self):
        with pytest.warns(UserWarning) as rec:
            validate_frame(frame())
        message = str(rec[0].message)
        for col in NULL_IN_BOTH_CORPORA:
            assert col in message

    def test_silent_when_an_upload_populates_them(self):
        df = frame(
            manufacturer=["Zeiss", "Edwards"],
            model=["Axio Lab A1", "RV12"],
            serial_number=["SN-1", "SN-2"],
            source=["upload", "upload"],
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_frame(df)


class TestTextOf:
    def test_concatenates_title_and_description(self):
        assert text_of(frame()).tolist() == [
            "Zeiss microscope Supply of one microscope",
            "Rotary vane pump Supply of one pump",
        ]

    def test_null_description_does_not_produce_the_string_nan(self):
        got = text_of(frame(description=[None, "Supply of one pump"]))
        assert got.iloc[0] == "Zeiss microscope"
        assert "nan" not in got.iloc[0]


class TestRecordModel:
    def test_minimal_record(self):
        r = Record(record_id="A:1", title="Pump", source="abtbuy")
        assert r.description is None
        assert r.cpv_code is None

    def test_source_is_constrained(self):
        with pytest.raises(ValueError):
            Record(record_id="x", title="y", source="sharepoint")

    def test_field_set_matches_record_columns(self):
        assert set(Record.model_fields) == set(RECORD_COLUMNS)
