"""The canonical Record shape (§6.1).

Both corpora and the spreadsheet importer map into this one shape. Anything downstream of
ingest only ever sees ``RECORD_COLUMNS``.
"""

from __future__ import annotations

import warnings
from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel

__all__ = [
    "RECORD_COLUMNS",
    "NULL_IN_BOTH_CORPORA",
    "Record",
    "SchemaError",
    "validate_frame",
    "text_of",
    "present_text",
]

RECORD_COLUMNS = [
    "record_id",
    "title",
    "description",
    "manufacturer",
    "model",
    "serial_number",
    "buyer_id",
    "cpv_code",
    "release_date",
    "source",
]

#: Fields that are null across BOTH research corpora. Contracts Finder supplies
#: record_id, title, description, buyer_id, cpv_code and release_date; Abt-Buy supplies
#: name, description and price. These three survive in Record only because
#: ``source="upload"`` — a real FCES spreadsheet — can populate them, and the system's
#: ``assets`` table carries them.
#:
#: **Nothing in the research path may key, block, group or score on these.** A blocking
#: scheme or distractor rule that reads them is keying on an always-null column and will
#: silently produce one enormous block, no blocks at all, or an empty negative set. Where
#: a brand proxy is needed, use ``blocking.block_by_leading_token``, which the paper
#: states as an approximation and evaluates as one.
NULL_IN_BOTH_CORPORA = ("manufacturer", "model", "serial_number")

_SOURCES = ("contractsfinder", "abtbuy", "upload")


class SchemaError(ValueError):
    """A frame does not conform to RECORD_COLUMNS."""


class Record(BaseModel):
    record_id: str
    title: str
    description: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None
    buyer_id: str | None = None
    cpv_code: str | None = None
    release_date: date | None = None
    source: Literal["contractsfinder", "abtbuy", "upload"]


def validate_frame(df: pd.DataFrame) -> None:
    """Raise ``SchemaError`` if ``df`` is not a valid Record frame.

    Warns — deliberately does not raise — when a column in ``NULL_IN_BOTH_CORPORA`` is
    wholly null. That is the expected condition on both research corpora, and the warning
    exists so it is visible in logs rather than assumed.
    """
    missing = [c for c in RECORD_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")

    if df.empty:
        return

    if df["record_id"].isna().any():
        raise SchemaError("record_id contains nulls")
    n_dup = int(df["record_id"].duplicated().sum())
    if n_dup:
        raise SchemaError(f"record_id is not unique: {n_dup} duplicates")
    if df["title"].isna().any():
        raise SchemaError("title contains nulls")

    bad_sources = sorted(set(df["source"].dropna().unique()) - set(_SOURCES))
    if bad_sources:
        raise SchemaError(f"unknown source values: {bad_sources}")

    if "release_date" in df and not df["release_date"].isna().all():
        col = df["release_date"].dropna()
        if not (
            pd.api.types.is_datetime64_any_dtype(col)
            or col.map(lambda v: isinstance(v, date)).all()
        ):
            raise SchemaError("release_date is neither datetime nor date objects")

    wholly_null = [c for c in NULL_IN_BOTH_CORPORA if df[c].isna().all()]
    if wholly_null:
        warnings.warn(
            f"columns wholly null and unusable for keying: {wholly_null} "
            "(expected on both research corpora — see schema.NULL_IN_BOTH_CORPORA)",
            stacklevel=2,
        )


def text_of(df: pd.DataFrame) -> pd.Series:
    """Concatenated field used for embedding and TF-IDF: title + ' ' + description."""
    title = df["title"].fillna("")
    description = df["description"].fillna("") if "description" in df else ""
    return (title + " " + description).str.strip()


def present_text(value) -> str | None:
    """``value`` as text, or ``None`` where the field is absent.

    **pandas stores a missing string as float ``nan``, which is truthy.** A bare
    ``if value:`` guard therefore admits it, and what follows either raises — string
    functions reject a float — or formats it into the output as the literal ``"nan"``. The
    second is the dangerous one, because it does not stop anything: it has already put
    ``"nan"`` into a degraded title (where both copies of a pair got the same spurious
    token, making duplicates easier to match) and into an annotator's screen.

    This is the single scalar-level companion to :func:`text_of`, which does the same job
    column-wise through ``fillna("")``. Prefer either to an ad-hoc truthiness check.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return value or None
