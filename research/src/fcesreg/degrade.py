"""The degradation model (§6.6).

Published data is cleaner than a working spreadsheet, so this reintroduces the error
classes the corpora lack. The model extends the dirty-data protocol of Mudgal et al.,
which relocates attribute values into a free text field, with six further classes drawn
from the quality issues the Contracts Finder publisher documents and from a manual audit
of the raw data: abbreviation from a domain lexicon, character-level noise (insertion,
deletion, substitution and transposition), inconsistent casing, whitespace perturbation,
field omission, and variation in the notation of units and voltages.

Seven classes in total, matching the seven knobs on :class:`DegradationConfig`.

**Every function takes an explicit rng and there is no module-level random state.** The
same seed must reproduce byte-identical output; there is a test for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fcesreg.cpv import cpv_class
from fcesreg.normalise import normalise_text

__all__ = [
    "ERROR_CLASSES",
    "DEFAULT_LEXICON_PATH",
    "DegradationConfig",
    "load_lexicon",
    "abbreviate",
    "char_noise",
    "vary_case",
    "perturb_whitespace",
    "vary_units",
    "merge_fields",
    "omit_field",
    "degrade_record",
    "degrade_frame",
    "make_duplicate_pairs",
    "make_distractors",
]

#: The seven classes, in the order the paper lists them. `merge` is Mudgal's.
ERROR_CLASSES = (
    "abbreviate",
    "charnoise",
    "case",
    "whitespace",
    "units",
    "omit",
    "merge",
)

DEFAULT_LEXICON_PATH = Path("data/lexicon/abbreviations.yaml")

_TEXT_FIELDS = ("title", "description")
_OMITTABLE = ("description",)


@dataclass
class DegradationConfig:
    """One severity knob, with a per-class multiplier applied on top of it.

    ``severity`` spans 0.0 (untouched) to 1.0 (heavily degraded). The multipliers exist so
    a single class can be isolated for the degradation check without rebuilding the model.
    """

    severity: float
    p_abbreviate: float = 1.0
    p_charnoise: float = 1.0
    p_case: float = 1.0
    p_whitespace: float = 1.0
    p_units: float = 1.0
    p_omit: float = 1.0
    p_merge: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {self.severity}")

    def rate(self, error_class: str) -> float:
        if error_class not in ERROR_CLASSES:
            raise ValueError(f"unknown error class {error_class!r}")
        return min(1.0, self.severity * getattr(self, f"p_{error_class}"))


def load_lexicon(path: Path = DEFAULT_LEXICON_PATH) -> dict[str, str]:
    """Expansion to abbreviation. Keys are lowercased; values are used verbatim."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(k).lower(): str(v) for k, v in raw.items()}


# --------------------------------------------------------------------------- classes


def abbreviate(s: str, rng: np.random.Generator, lexicon: dict[str, str], rate: float) -> str:
    """Replace whole words with their lexicon abbreviation, each with probability ``rate``."""
    if not s or rate <= 0:
        return s

    def repl(match: re.Match) -> str:
        word = match.group(0)
        short = lexicon.get(word.lower())
        if short is None or rng.random() >= rate:
            return word
        return short.upper() if word.isupper() else short

    return re.sub(r"[A-Za-z]+", repl, s)


def char_noise(s: str, rng: np.random.Generator, rate: float) -> str:
    """Insert, delete, substitute or transpose characters, each position with ``rate``.

    ``rate`` is per character, so it is the one class whose effect scales with field
    length. Whitespace is never chosen as an edit site: that is the whitespace class's
    job, and letting both touch it would double-count it in the degradation check.
    """
    if not s or rate <= 0:
        return s

    letters = "abcdefghijklmnopqrstuvwxyz"
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isspace() or rng.random() >= rate:
            out.append(ch)
            i += 1
            continue

        op = rng.integers(0, 4)
        if op == 0:  # insert
            out.append(letters[rng.integers(0, 26)])
            out.append(ch)
            i += 1
        elif op == 1:  # delete
            i += 1
        elif op == 2:  # substitute
            out.append(letters[rng.integers(0, 26)])
            i += 1
        else:  # transpose with the next character
            if i + 1 < len(s) and not s[i + 1].isspace():
                out.append(s[i + 1])
                out.append(ch)
                i += 2
            else:
                out.append(ch)
                i += 1
    return "".join(out)


def vary_case(s: str, rng: np.random.Generator, rate: float) -> str:
    """Recase whole words: upper, lower or title, each word with probability ``rate``."""
    if not s or rate <= 0:
        return s
    words = s.split(" ")
    for idx, word in enumerate(words):
        if word and rng.random() < rate:
            choice = rng.integers(0, 3)
            words[idx] = word.upper() if choice == 0 else word.lower() if choice == 1 else word.title()
    return " ".join(words)


def perturb_whitespace(s: str, rng: np.random.Generator, rate: float) -> str:
    """Double a space, drop one, or add leading/trailing padding."""
    if not s or rate <= 0:
        return s
    out = []
    for ch in s:
        if ch == " " and rng.random() < rate:
            out.append("  " if rng.random() < 0.5 else "")
        else:
            out.append(ch)
    result = "".join(out)
    if rng.random() < rate:
        result = " " * int(rng.integers(1, 4)) + result
    if rng.random() < rate:
        result = result + " " * int(rng.integers(1, 4))
    return result


_UNIT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(V|W|kW|kg|g|mm|cm|m|L|ml|A|Hz|°C|C)\b", re.IGNORECASE)


def vary_units(s: str, rng: np.random.Generator, rate: float) -> str:
    """Vary unit notation: ``230V`` / ``230 V`` / ``230v``; ``1.5kW`` / ``1,5 kW``."""
    if not s or rate <= 0:
        return s

    def repl(match: re.Match) -> str:
        if rng.random() >= rate:
            return match.group(0)
        number, unit = match.group(1), match.group(2)
        if rng.random() < 0.5:
            number = number.replace(".", ",") if "." in number else number.replace(",", ".")
        unit = unit.lower() if rng.random() < 0.5 else unit.upper()
        separator = "" if rng.random() < 0.5 else " "
        return f"{number}{separator}{unit}"

    return _UNIT_RE.sub(repl, s)


def merge_fields(rec: dict, rng: np.random.Generator, rate: float) -> dict:
    """Mudgal-style column misalignment: move the description into the title.

    This reproduces what happens when a spreadsheet's columns drift out of alignment and
    one person's notes end up in the name field.
    """
    out = dict(rec)
    if rng.random() >= rate:
        return out
    description = out.get("description")
    if description:
        out["title"] = f"{out.get('title', '')} {description}".strip()
        out["description"] = None
    return out


def omit_field(rec: dict, rng: np.random.Generator, rate: float) -> dict:
    """Drop an optional field entirely — the row where somebody left a cell blank."""
    out = dict(rec)
    for field_name in _OMITTABLE:
        if out.get(field_name) and rng.random() < rate:
            out[field_name] = None
    return out


# --------------------------------------------------------------------------- record


def degrade_record(
    rec: dict,
    cfg: DegradationConfig,
    rng: np.random.Generator,
    lexicon: dict[str, str] | None = None,
) -> dict:
    """Apply all seven classes to one record.

    Order matters and is fixed: structural classes (merge, omit) run first, then the
    within-text classes. Abbreviation runs before character noise so that abbreviations
    can themselves be corrupted, which is what happens when someone types a shorthand
    badly.
    """
    lexicon = load_lexicon() if lexicon is None else lexicon

    out = merge_fields(rec, rng, cfg.rate("merge"))
    out = omit_field(out, rng, cfg.rate("omit"))

    for field_name in _TEXT_FIELDS:
        value = out.get(field_name)
        if not value:
            continue
        value = abbreviate(value, rng, lexicon, cfg.rate("abbreviate"))
        value = vary_units(value, rng, cfg.rate("units"))
        value = char_noise(value, rng, cfg.rate("charnoise"))
        value = vary_case(value, rng, cfg.rate("case"))
        value = perturb_whitespace(value, rng, cfg.rate("whitespace"))
        out[field_name] = value

    return out


def degrade_frame(
    records: pd.DataFrame,
    cfg: DegradationConfig,
    seed: int,
    lexicon: dict[str, str] | None = None,
    suffix: str = "",
) -> pd.DataFrame:
    """Degrade every record independently. ``suffix`` is appended to each ``record_id``."""
    lexicon = load_lexicon() if lexicon is None else lexicon
    rng = np.random.default_rng(seed)

    rows = []
    for rec in records.to_dict("records"):
        degraded = degrade_record(rec, cfg, rng, lexicon)
        degraded["record_id"] = f"{rec['record_id']}{suffix}"
        rows.append(degraded)
    return pd.DataFrame(rows, columns=records.columns)


# --------------------------------------------------------------------------- pairs


def make_duplicate_pairs(
    records: pd.DataFrame,
    cfg: DegradationConfig,
    seed: int,
    lexicon: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two independent degraded copies of each source record form a positive pair.

    This reproduces the case the paper describes: two members of staff enter the same item
    without reference to one another. The copies are drawn independently, so they differ
    from each other as well as from the source.
    """
    lexicon = load_lexicon() if lexicon is None else lexicon
    left = degrade_frame(records, cfg, seed, lexicon, suffix="::a")
    right = degrade_frame(records, cfg, seed + 1, lexicon, suffix="::b")

    degraded = pd.concat([left, right], ignore_index=True)
    pairs = pd.DataFrame(
        {
            "left_id": left["record_id"].to_numpy(),
            "right_id": right["record_id"].to_numpy(),
            "label": 1,
        }
    )
    return degraded, pairs


def make_distractors(
    records: pd.DataFrame,
    cfg: DegradationConfig,
    seed: int,
    corpus: str,
    sim_threshold: float = 0.75,
    max_per_group: int = 5,
    lexicon: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Near-duplicate NEGATIVES, mined from fields that actually exist.

    ``corpus="cf"``
        Pairs sharing a CPV class with title cosine at or above ``sim_threshold`` and
        distinct ``record_id``.
    ``corpus="abtbuy"``
        Pairs sharing a leading token, with distinct ``record_id``.

    Without these, a detector that cannot separate similar-but-distinct records reports
    high recall for the wrong reason.

    **These are mined, not verified.** Two notices sharing a class and a near-identical
    title may be the same procurement published twice rather than two distinct items, in
    which case the pair is a false negative in the ground truth. The mining rule cannot
    tell the difference and neither can any automatic check, so a sample is audited by
    hand — see ``research/scripts/audit_distractors.py``, whose findings are reported with
    the results.
    """
    if corpus not in ("cf", "abtbuy"):
        raise ValueError(f"corpus must be 'cf' or 'abtbuy', got {corpus!r}")

    rng = np.random.default_rng(seed)
    lexicon = load_lexicon() if lexicon is None else lexicon

    if corpus == "cf":
        pairs = _mine_cf_distractors(records, sim_threshold, max_per_group, rng)
    else:
        pairs = _mine_leading_token_distractors(records, max_per_group, rng)

    pairs["label"] = 0
    return pairs


def _mine_cf_distractors(
    records: pd.DataFrame,
    sim_threshold: float,
    max_per_group: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer

    work = records[records["cpv_code"].notna()].copy()
    work["_class"] = work["cpv_code"].map(cpv_class)

    left, right = [], []
    for _, group in work.groupby("_class"):
        if len(group) < 2:
            continue
        titles = group["title"].fillna("").map(normalise_text).tolist()
        if not any(titles):
            continue
        try:
            matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(titles)
        except ValueError:
            continue

        similarity = (matrix @ matrix.T).toarray()
        np.fill_diagonal(similarity, 0.0)
        ids = group["record_id"].to_numpy()

        rows, cols = np.where(similarity >= sim_threshold)
        keep = rows < cols
        rows, cols = rows[keep], cols[keep]
        if len(rows) > max_per_group:
            pick = rng.choice(len(rows), size=max_per_group, replace=False)
            rows, cols = rows[pick], cols[pick]
        left.extend(ids[rows])
        right.extend(ids[cols])

    return pd.DataFrame({"left_id": left, "right_id": right})


def _mine_leading_token_distractors(
    records: pd.DataFrame, max_per_group: int, rng: np.random.Generator
) -> pd.DataFrame:
    from fcesreg.blocking import block_by_leading_token

    blocks = block_by_leading_token(records)
    left, right = [], []
    for members in blocks.values():
        members = sorted(set(members))
        if len(members) < 2:
            continue
        combos = [(a, b) for i, a in enumerate(members) for b in members[i + 1 :]]
        if len(combos) > max_per_group:
            pick = rng.choice(len(combos), size=max_per_group, replace=False)
            combos = [combos[i] for i in pick]
        for a, b in combos:
            left.append(a)
            right.append(b)

    return pd.DataFrame({"left_id": left, "right_id": right})
