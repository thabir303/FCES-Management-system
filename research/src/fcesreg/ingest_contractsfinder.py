"""Contracts Finder ingest (§6.3).

Reads the OCDS CSV bundles published at
``https://data.open-contracting.org/en/publication/128/download?name=<YEAR>.csv.tar.gz``
and maps them into the canonical Record shape.

Two properties of this data drive the implementation (§4.1):

* Descriptions contain embedded newlines inside quoted fields, so ``wc -l`` overcounts and
  every read goes through a real CSV reader with the field size limit raised.
* ``tender_datePublished`` is populated for only 15–31% of rows depending on the year.
  ``date`` is 100% populated in all four bundles and is what ``release_date`` maps to.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from fcesreg.normalise import normalise_text
from fcesreg.schema import RECORD_COLUMNS, validate_frame

__all__ = [
    "FIELD_MAP",
    "EXTRA_FIELDS",
    "EXPECTED_COLUMNS",
    "CANDIDATE_DIVISIONS",
    "BOILERPLATE_BLOCKLIST",
    "MIN_CHARS",
    "SchemaMismatch",
    "DiscardReport",
    "load_bundle",
    "to_records",
    "apply_filters",
]

csv.field_size_limit(10**9)

FIELD_MAP = {
    "record_id": "id",
    "title": "tender_title",
    "description": "tender_description",
    "buyer_id": "buyer_id",
    "cpv_code": "tender_classification_id",
    "release_date": "date",  # NOT tender_datePublished — see §4.1
}

#: Carried alongside RECORD_COLUMNS, not inside them: it is Contracts Finder specific.
#: `tender_id` is the procurement reference, and it is what establishes that two notices
#: describe *different* procurements. The record identifier does not serve, because the
#: publisher mints one per notice rather than per contracting process — an award notice
#: and its tender notice carry different `id` values and different `ocid` values while
#: describing one procurement. See `degrade.procurement_ref`.
EXTRA_FIELDS = {"tender_ref": "tender_id"}

#: The full main.csv header, verified identical across the 2022–2025 bundles. The bundles
#: differ in column *order* only, which is immaterial because every read is by name.
#: Any difference in the set of names is a hard failure (§13.6): a renamed column must be
#: reported and reconciled, never coerced or silently filled.
EXPECTED_COLUMNS = frozenset(
    {
        "id",
        "tag",
        "date",
        "ocid",
        "language",
        "initiationType",
        "buyer_id",
        "buyer_name",
        "tender_id",
        "tender_mainProcurementCategory",
        "tender_title",
        "tender_status",
        "tender_procurementMethodDetails",
        "tender_description",
        "tender_procurementMethod",
        "tender_datePublished",
        "tender_suitability_sme",
        "tender_suitability_vcse",
        "tender_tenderPeriod_endDate",
        "tender_classification_id",
        "tender_classification_scheme",
        "tender_classification_description",
        "tender_contractPeriod_endDate",
        "tender_contractPeriod_startDate",
        "tender_value_amount",
        "tender_value_currency",
        "tender_minValue_amount",
        "tender_minValue_currency",
        "tender_procedure_isAccelerated",
        "title",
        "tender_communication_futureNoticeDate",
        "tender_procedure_acceleratedRationale",
    }
)

#: The candidate division set (§4.2). Ingest retains ALL of these, including 39
#: (furniture) and 48 (software), so that `run_profile.py` can measure corpus size and
#: per-class support both with and without them from one parquet. Which set the study
#: finally adopts is a reported decision, not an ingest-time one.
CANDIDATE_DIVISIONS = frozenset({"30", "31", "32", "33", "38", "39", "42", "43", "44", "48"})

MIN_CHARS = 60

#: Descriptions carrying no information about the item (§4.3).
BOILERPLATE_BLOCKLIST = frozenset(
    {
        "as per tender",
        "contract award notice",
        "transparency only",
        "notice of awarded contract following a mini competition",
        "call off from fcdo services ref xly120 121 21cc",
    }
)


class SchemaMismatch(RuntimeError):
    """A bundle's columns differ from the verified schema. Reconcile; never coerce."""


@dataclass
class DiscardReport:
    total_in: int
    dropped_out_of_scope: int
    dropped_short: int
    dropped_desc_equals_title: int
    dropped_boilerplate: int
    total_out: int

    def check(self) -> None:
        """The five counts must account for every input row exactly once."""
        dropped = (
            self.dropped_out_of_scope
            + self.dropped_short
            + self.dropped_desc_equals_title
            + self.dropped_boilerplate
        )
        if self.total_in - dropped != self.total_out:
            raise AssertionError(
                f"discard counts do not reconcile: {self.total_in} in "
                f"- {dropped} dropped != {self.total_out} out"
            )


def load_bundle(year_dir: Path) -> pd.DataFrame:
    """Read ``<year_dir>/main.csv`` and verify its columns against the known schema.

    Raises :class:`SchemaMismatch` naming the exact difference if the header does not
    match. Column *order* is allowed to differ, since every read is by name.
    """
    year_dir = Path(year_dir)
    path = year_dir / "main.csv"
    if not path.exists():
        raise FileNotFoundError(f"no main.csv in {year_dir}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        _verify_columns(header, path)
        rows = list(reader)

    return pd.DataFrame(rows, columns=sorted(header))


def _verify_columns(header: set[str], path: Path) -> None:
    missing = EXPECTED_COLUMNS - header
    extra = header - EXPECTED_COLUMNS
    if missing or extra:
        raise SchemaMismatch(
            f"{path} does not match the verified column schema.\n"
            f"  missing: {sorted(missing) or 'none'}\n"
            f"  extra:   {sorted(extra) or 'none'}\n"
            "Reconcile this by hand and report it. Do not coerce, rename or fill."
        )
    needed = set(FIELD_MAP.values()) - header
    if needed:
        raise SchemaMismatch(f"{path} is missing mapped source columns: {sorted(needed)}")


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return datetime.strptime(value.strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def to_records(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Map a raw bundle frame into RECORD_COLUMNS.

    ``manufacturer``, ``model`` and ``serial_number`` are created as all-null: Contracts
    Finder has no such fields. Nothing downstream may key on them (see
    ``schema.NULL_IN_BOTH_CORPORA``).
    """
    out = pd.DataFrame(index=df.index)
    for target, source in {**FIELD_MAP, **EXTRA_FIELDS}.items():
        out[target] = df[source]

    out["release_date"] = out["release_date"].map(_parse_date)
    out["record_id"] = out["record_id"].astype(str)
    for col in ("title", "description", "buyer_id", "cpv_code", "tender_ref"):
        out[col] = out[col].replace("", None)

    out["manufacturer"] = None
    out["model"] = None
    out["serial_number"] = None
    out["source"] = "contractsfinder"
    out["bundle_year"] = year

    out = out.dropna(subset=["title"])
    return out[[*RECORD_COLUMNS, "tender_ref", "bundle_year"]]


def apply_filters(
    df: pd.DataFrame,
    divisions: frozenset[str] | set[str] = CANDIDATE_DIVISIONS,
    min_chars: int = MIN_CHARS,
    blocklist: frozenset[str] | set[str] | None = None,
) -> tuple[pd.DataFrame, DiscardReport]:
    """Apply the §4.3 discard rules and report how many rows each removed.

    The rules are applied in order and each dropped row is attributed to the *first* rule
    it fails, so the counts sum to the total dropped exactly. Those counts are a reported
    result, not diagnostics.
    """
    blocklist = BOILERPLATE_BLOCKLIST if blocklist is None else blocklist
    total_in = len(df)

    div = df["cpv_code"].fillna("").str.slice(0, 2)
    in_scope = div.isin(set(divisions))
    dropped_out_of_scope = int((~in_scope).sum())
    work = df[in_scope].copy()

    title_norm = work["title"].map(normalise_text)
    desc_norm = work["description"].map(normalise_text)
    combined_len = (title_norm + " " + desc_norm).str.strip().str.len()

    short = combined_len < min_chars
    dropped_short = int(short.sum())
    keep = ~short

    desc_equals_title = keep & (desc_norm == title_norm)
    dropped_desc_equals_title = int(desc_equals_title.sum())
    keep &= ~desc_equals_title

    boilerplate = keep & desc_norm.isin(set(blocklist))
    dropped_boilerplate = int(boilerplate.sum())
    keep &= ~boilerplate

    out = work[keep].reset_index(drop=True)
    report = DiscardReport(
        total_in=total_in,
        dropped_out_of_scope=dropped_out_of_scope,
        dropped_short=dropped_short,
        dropped_desc_equals_title=dropped_desc_equals_title,
        dropped_boilerplate=dropped_boilerplate,
        total_out=len(out),
    )
    report.check()
    return out, report


def build(
    raw: Path, years: list[int], out_path: Path
) -> tuple[pd.DataFrame, dict[str, DiscardReport]]:
    frames, reports = [], {}
    for year in years:
        raw_df = load_bundle(Path(raw) / str(year))
        records = to_records(raw_df, year)
        kept, report = apply_filters(records)
        reports[str(year)] = report
        frames.append(kept)
        print(
            f"  {year}: {report.total_in:>7,} in -> {report.total_out:>6,} out "
            f"(scope {report.dropped_out_of_scope:,}, short {report.dropped_short:,}, "
            f"desc=title {report.dropped_desc_equals_title:,}, "
            f"boilerplate {report.dropped_boilerplate:,})",
            file=sys.stderr,
        )

    corpus = pd.concat(frames, ignore_index=True)
    validate_frame(corpus)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(out_path, index=False)

    combined = DiscardReport(
        total_in=sum(r.total_in for r in reports.values()),
        dropped_out_of_scope=sum(r.dropped_out_of_scope for r in reports.values()),
        dropped_short=sum(r.dropped_short for r in reports.values()),
        dropped_desc_equals_title=sum(r.dropped_desc_equals_title for r in reports.values()),
        dropped_boilerplate=sum(r.dropped_boilerplate for r in reports.values()),
        total_out=len(corpus),
    )
    combined.check()

    report_path = out_path.with_suffix(".discard_report.json")
    report_path.write_text(
        json.dumps(
            {
                "per_year": {y: asdict(r) for y, r in reports.items()},
                "combined": asdict(combined),
                "divisions": sorted(CANDIDATE_DIVISIONS),
                "min_chars": MIN_CHARS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return corpus, reports


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingest Contracts Finder bundles.")
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    p.add_argument("--out", type=Path, default=Path("data/processed/corpus_b_contractsfinder.parquet"))
    args = p.parse_args(argv)

    try:
        corpus, _ = build(args.raw, args.years, args.out)
    except SchemaMismatch as e:
        print(f"\nSCHEMA MISMATCH — stopping.\n{e}\n", file=sys.stderr)
        return 2

    print(f"wrote {len(corpus):,} records to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
