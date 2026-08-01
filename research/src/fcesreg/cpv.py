"""CPV taxonomy (§6.5).

The hierarchy is reconstructed from the (code, description) pairs the notices already
carry, with parents derived by truncation. No external CPV vocabulary file is needed, so
the taxonomy is regenerable from the released bundles alone.

**Two levels, and only two: division and class.** Eight-digit leaf classification is not
viable on this data and must not be built (§4.2) — the measured sparsity is a reported
result, produced by :func:`leaf_sparsity`.

Both terms are used in their official CPV sense: a division is the first two digits and a
class is the first four. (The intervening level, the first three digits, is the official
"group"; it is not evaluated here.)
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import pandas as pd

__all__ = [
    "LEVELS",
    "Level",
    "division",
    "cpv_class",
    "label_series",
    "build_taxonomy",
    "supported_labels",
    "leaf_sparsity",
]

Level = Literal["division", "class"]
LEVELS: tuple[Level, ...] = ("division", "class")

_CLASSIFICATION_SOURCES = (
    ("main.csv", "tender_classification_id", "tender_classification_description"),
    ("tender_additionalClassifications.csv", "id", "description"),
)


def division(code: str) -> str:
    """First two digits of a CPV code."""
    return str(code)[:2]


def cpv_class(code: str) -> str:
    """First four digits of a CPV code — the CPV *class* level."""
    return str(code)[:4]


def label_series(df: pd.DataFrame, level: Level, column: str = "cpv_code") -> pd.Series:
    """The label each record carries at ``level`` (``"division"`` or ``"class"``)."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    fn = division if level == "division" else cpv_class
    return df[column].fillna("").map(fn)


def _level_of(code: str) -> int:
    """Level implied by a code's trailing zeros: 30000000 -> 2, 30190000 -> 4, else 8."""
    if code.endswith("000000"):
        return 2
    if code.endswith("0000"):
        return 4
    return 8


def _short_form(code: str, level: int) -> str:
    """The 2-, 4- or 8-digit key used by ``categories.cpv_code`` (§5.4)."""
    return code[:2] if level == 2 else code[:4] if level == 4 else code


def build_taxonomy(bundle_dirs: Iterable[Path | str]) -> pd.DataFrame:
    """Harvest distinct (code, description) pairs and derive the hierarchy by truncation.

    Returns columns ``cpv_code`` (2, 4 or 8 digit — the ``categories`` primary key),
    ``cpv_code_full`` (the canonical 8-digit form), ``cpv_description``, ``level`` and
    ``parent_code``.

    Where several descriptions are observed for one code the most frequent wins, and the
    number of codes with competing descriptions is available as an attribute on the result
    (``df.attrs["codes_with_conflicting_descriptions"]``) so the ambiguity is countable
    rather than hidden.
    """
    pairs: list[tuple[str, str]] = []
    for bundle in bundle_dirs:
        bundle = Path(bundle)
        for filename, code_col, desc_col in _CLASSIFICATION_SOURCES:
            path = bundle / filename
            if not path.exists():
                continue
            frame = pd.read_csv(
                path,
                usecols=[code_col, desc_col],
                dtype=str,
                keep_default_na=False,
                engine="python",
            )
            pairs.extend(
                (c.strip(), d.strip())
                for c, d in zip(frame[code_col], frame[desc_col], strict=True)
                if c.strip()
            )

    if not pairs:
        raise ValueError("no CPV codes harvested — check the bundle directories")

    observed = pd.DataFrame(pairs, columns=["cpv_code_full", "cpv_description"])
    counts = (
        observed.groupby(["cpv_code_full", "cpv_description"], sort=False)
        .size()
        .reset_index(name="n")
        .sort_values(["cpv_code_full", "n"], ascending=[True, False])
    )
    n_conflicting = int(
        (counts.groupby("cpv_code_full").size() > 1).sum()
    )
    best = counts.drop_duplicates("cpv_code_full", keep="first")

    rows: dict[str, dict] = {}
    for code, desc in zip(best["cpv_code_full"], best["cpv_description"], strict=True):
        if not code.isdigit() or len(code) != 8:
            continue
        lvl = _level_of(code)
        key = _short_form(code, lvl)
        rows.setdefault(
            key,
            {
                "cpv_code": key,
                "cpv_code_full": code,
                "cpv_description": desc,
                "level": lvl,
            },
        )

    # Ensure every ancestor exists even if it was never published in its own right; a
    # class with no division row would otherwise break the foreign key in `categories`.
    for key in list(rows):
        lvl = rows[key]["level"]
        if lvl == 8:
            rows.setdefault(
                key[:4],
                {
                    "cpv_code": key[:4],
                    "cpv_code_full": key[:4] + "0000",
                    "cpv_description": None,
                    "level": 4,
                },
            )
        if lvl in (4, 8):
            rows.setdefault(
                key[:2],
                {
                    "cpv_code": key[:2],
                    "cpv_code_full": key[:2] + "000000",
                    "cpv_description": None,
                    "level": 2,
                },
            )

    taxonomy = pd.DataFrame(rows.values())
    taxonomy["parent_code"] = [
        None if lvl == 2 else (code[:2] if lvl == 4 else code[:4])
        for code, lvl in zip(taxonomy["cpv_code"], taxonomy["level"], strict=True)
    ]
    taxonomy = taxonomy.sort_values(["level", "cpv_code"]).reset_index(drop=True)
    taxonomy.attrs["codes_with_conflicting_descriptions"] = n_conflicting
    taxonomy.attrs["codes_without_description"] = int(
        taxonomy["cpv_description"].isna().sum()
    )
    return taxonomy


def supported_labels(
    train: pd.DataFrame, level: Level, min_examples: int = 50
) -> tuple[set[str], float]:
    """Labels with at least ``min_examples`` training records, and their coverage.

    ``coverage`` is the fraction of ``train`` whose label is supported. Callers must
    report it: a high macro F1 over labels covering half the corpus is not the same
    result as one covering nearly all of it.
    """
    labels = label_series(train, level)
    counts = labels.value_counts()
    supported = set(counts[counts >= min_examples].index)
    coverage = float(labels.isin(supported).mean()) if len(labels) else 0.0
    return supported, coverage


def leaf_sparsity(df: pd.DataFrame, column: str = "cpv_code") -> dict[str, int]:
    """Why the eight-digit level is not evaluated (§4.2, RQ2).

    The paper states that leaf-level classification is out of scope and that the measured
    sparsity appears in Section V, so this is a reported result rather than a note.
    """
    counts = df[column].dropna().value_counts()
    return {
        "n_distinct_leaves": int(len(counts)),
        "n_leaves_ge_20": int((counts >= 20).sum()),
        "n_leaves_ge_50": int((counts >= 50).sum()),
        "n_singleton_leaves": int((counts == 1).sum()),
        "n_records": int(counts.sum()),
    }
