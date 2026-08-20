"""Seeds `categories` from `data/processed/cpv_taxonomy.parquet` (§5.4).

**`hazard_class` is an illustrative designation, not a hazard taxonomy.** It stands in for
a faculty health-and-safety schedule that does not yet exist, and no claim is made that
these three divisions or intervals are the faculty's real policy -- stated here, at the
constant, and again in the log line on every run, per the plan's explicit requirement that
this not be presented as more than it is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from fcesapi.db import get_sessionmaker
from fcesapi.models import Category

#: ILLUSTRATIVE ONLY. See module docstring.
_HAZARD_DIVISIONS = {
    "33": ("regulated", 365),            # medical equipment
    "38": ("calibration_required", 365),  # laboratory / optical / precision instruments
    "42": ("mechanical", 180),            # industrial machinery
}


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "processed").exists():
            return parent
    raise RuntimeError("could not locate repo root from " + str(here))


def main() -> int:
    taxonomy_path = _repo_root() / "data" / "processed" / "cpv_taxonomy.parquet"
    if not taxonomy_path.exists():
        print(f"seed_categories: {taxonomy_path} not found -- run `make data` first",
              file=sys.stderr)
        return 1

    taxonomy = pd.read_parquet(taxonomy_path)
    print(
        "seed_categories: hazard_class and default_service_interval_days below are an "
        "ILLUSTRATIVE designation standing in for a faculty health-and-safety schedule "
        "that does not exist yet -- not the faculty's real policy."
    )

    db = get_sessionmaker()()
    try:
        n_created = 0
        for _, row in taxonomy.iterrows():
            if db.get(Category, row["cpv_code"]) is not None:
                continue
            hazard_class, interval = _HAZARD_DIVISIONS.get(row["cpv_code"][:2], (None, None))
            parent = row.get("parent_code")
            # pandas stores a missing string as float NaN, which is truthy -- `parent or
            # None` would pass NaN straight through and violate the self-referencing FK.
            parent_code = parent if isinstance(parent, str) else None
            db.add(
                Category(
                    cpv_code=row["cpv_code"],
                    cpv_description=row["cpv_description"],
                    level=int(row["level"]),
                    parent_code=parent_code,
                    hazard_class=hazard_class,
                    default_service_interval_days=interval,
                )
            )
            n_created += 1
        db.commit()
        print(f"seed_categories: created {n_created} categories")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
