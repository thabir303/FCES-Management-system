"""Text normalisation (§6.2).

The error classes this has to survive are the ones actually present in Contracts Finder:
mojibake from a cp1252/utf-8 round trip, C0 control characters embedded in descriptions,
inconsistent casing and whitespace, and punctuation used inconsistently between two people
recording the same item.

``normalise_key`` is what the exact-match dedup baseline compares.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

__all__ = [
    "fix_mojibake",
    "strip_control",
    "normalise_text",
    "normalise_key",
    "normalise_frame",
]

_REPLACEMENT_CHAR = "�"

# Characters that only appear in text that has been through a cp1252/utf-8 round trip.
# Repair is attempted only when one of these is present, so clean text is never touched.
_MOJIBAKE_SIGNALS = frozenset("ÂÃâ€™“”˜Å¸¢£¤¥¦§¨©ª«¬®¯°±")

_WHITESPACE_RE = re.compile(r"\s+")

# Keep the whitespace characters that carry structure in a description field.
_KEEP_CONTROL = frozenset("\n\r\t")

_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")

# Apostrophes are intra-word punctuation: they are deleted, not spaced, so "buyer's"
# yields "buyers" rather than "buyer s". Every other punctuation mark becomes a space.
_APOSTROPHES = frozenset("'’‘`´")


def fix_mojibake(s: str) -> str:
    """Repair a cp1252/utf-8 round trip and drop replacement characters.

    ``'â€™'`` is the UTF-8 encoding of ``'’'`` read as cp1252. Encoding back to cp1252 and
    decoding as UTF-8 recovers the original. The repair is applied only when it succeeds
    cleanly and actually changes the string, so text that merely contains an accented
    character is left alone.
    """
    if not s:
        return s

    if any(ch in _MOJIBAKE_SIGNALS for ch in s):
        try:
            repaired = _encode_cp1252(s).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        else:
            if repaired != s:
                s = repaired

    return s.replace(_REPLACEMENT_CHAR, "")


def _encode_cp1252(s: str) -> bytes:
    """Encode as cp1252, falling back to latin-1 for the five undefined slots.

    cp1252 leaves 0x81, 0x8D, 0x8F, 0x90 and 0x9D undefined, so a strict encode raises on
    text containing U+0081, U+008D, U+008F, U+0090 or U+009D. Those codepoints appear in
    real mojibake constantly: '”' (U+201D, bytes E2 80 9D) misread as cp1252 produces
    'â€' followed by U+009D. Refusing to encode there would leave the commonest damaged
    character in procurement text — the closing smart quote — unrepaired.
    """
    try:
        return s.encode("cp1252")
    except UnicodeEncodeError:
        return b"".join(
            ch.encode("cp1252") if _cp1252_encodable(ch) else ch.encode("latin-1")
            for ch in s
        )


def _cp1252_encodable(ch: str) -> bool:
    try:
        ch.encode("cp1252")
    except UnicodeEncodeError:
        return False
    return True


def strip_control(s: str) -> str:
    """Drop Unicode category ``Cc`` characters, keeping newline, carriage return and tab."""
    if not s:
        return s
    return "".join(
        ch for ch in s if ch in _KEEP_CONTROL or unicodedata.category(ch) != "Cc"
    )


def _strip_punctuation(s: str) -> str:
    """Replace Unicode punctuation with a space; delete apostrophes.

    Punctuation becomes a space rather than nothing, so ``'pump/valve'`` yields two tokens
    rather than one. Symbol categories (``S*``) are kept: ``°``, ``±`` and ``+`` carry
    meaning in equipment descriptions. Note that ``%`` is Unicode category ``Po``, not a
    symbol, so it is stripped along with the rest.

    A consequence worth stating: decimal separators do not survive, so ``"1.5kW"`` and
    ``"1,5 kW"`` both normalise to ``"1 5kw"``. §6.2 specifies stripping punctuation and
    that is what this does. The loss applies symmetrically to both members of any pair, so
    it cannot manufacture a false match — it makes decimal-separator variation invisible to
    every method equally rather than to one of them.
    """
    out = []
    for ch in s:
        if ch in _APOSTROPHES:
            continue
        out.append(" " if unicodedata.category(ch).startswith("P") else ch)
    return "".join(out)


def normalise_text(s: str | None) -> str:
    """Full normalisation pipeline for comparison and feature extraction.

    Order: repair mojibake, NFKC, strip control characters, casefold, strip punctuation,
    collapse whitespace.

    Mojibake repair runs **before** NFKC, not after. §6.2 lists NFKC first, but NFKC maps
    ``'™'`` to ``'TM'`` and ``'…'`` to ``'...'``, which destroys the very byte sequences the
    repair recognises: ``'â„¢'`` would become ``'â„TM'`` and no longer round trip. Since
    §4.1 records 331 mojibake-carrying rows in the 2025 subset alone, running NFKC first
    would silently fail to repair a documented part of the corpus.
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        if pd.isna(s):
            return ""
        s = str(s)

    s = fix_mojibake(s)
    s = unicodedata.normalize("NFKC", s)
    s = strip_control(s)
    s = s.casefold()
    s = _strip_punctuation(s)
    return _WHITESPACE_RE.sub(" ", s).strip()


def normalise_key(s: str | None) -> str:
    """``normalise_text`` with every non-alphanumeric character removed.

    This is the exact-match baseline's comparison key: it collapses every difference of
    spacing, casing and punctuation, so what survives is a genuine difference in content.
    """
    return _NON_ALNUM_RE.sub("", normalise_text(s))


def normalise_frame(
    df: pd.DataFrame, columns: tuple[str, ...] = ("title", "description")
) -> pd.DataFrame:
    """Add ``<col>_norm`` and ``<col>_key`` columns. Non-destructive: originals survive.

    Missing columns are skipped rather than raising, so this can run on a frame that has
    already been filtered down.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[f"{col}_norm"] = out[col].map(normalise_text)
        out[f"{col}_key"] = out[col].map(normalise_key)
    return out
