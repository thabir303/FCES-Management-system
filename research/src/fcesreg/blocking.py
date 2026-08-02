"""Blocking (§6.8).

Blocking reduces the quadratic comparison space to a tractable candidate set. Recall lost
here cannot be recovered downstream, so it is evaluated separately from matching, by pair
completeness and reduction ratio.

Three keying schemes, and exactly three. Each keys on a column that is actually populated;
there is no ``block_by_manufacturer``, because ``manufacturer`` is null across both
research corpora (see ``schema.NULL_IN_BOTH_CORPORA``) and keying on it yields either one
enormous block or none at all.

**Scheme availability differs between the corpora and that difference is a result.**
``buyer_id`` exists on Corpus B (Contracts Finder) and not on Corpus A (Abt-Buy), so the
buyer scheme can only ever be evaluated on one of them. Reporting a figure averaged across
corpora would hide that. :func:`applicable_schemes` decides availability from column
nullity rather than from a corpus name, so the asymmetry is discovered rather than assumed.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from fcesreg.normalise import normalise_text

__all__ = [
    "SCHEMES",
    "LEADING_STOPWORDS",
    "SchemeUnavailable",
    "BlockingReport",
    "block_by_sorted_ngrams",
    "block_by_leading_token",
    "block_by_buyer",
    "applicable_schemes",
    "candidate_pairs",
    "evaluate_blocking",
]

SCHEMES = ("sorted_ngrams", "leading_token", "buyer")

#: Words that begin a procurement title without identifying the item. The leading-token
#: key skips these to reach the first substantive word. This list is a documented part of
#: the scheme, not a tuning knob: it is fixed before measurement and not revised in
#: response to the result.
LEADING_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "for", "to", "and", "in", "on", "at", "by", "with",
        "supply", "supplies", "provision", "purchase", "procurement", "delivery",
        "installation", "maintenance", "servicing", "service", "hire", "lease",
        "replacement", "upgrade", "framework", "contract", "tender", "notice",
        "invitation", "request", "call", "off", "lot", "re", "new", "used",
    }
)


class SchemeUnavailable(ValueError):
    """A scheme was requested on a corpus whose keying column is not populated."""


@dataclass
class BlockingReport:
    """What blocking cost and what it lost. Every field is a reported number."""

    scheme: str
    n_records: int
    n_blocks: int
    n_candidates: int
    blocks_dropped: int
    records_in_dropped_blocks: int
    n_unblocked_records: int
    largest_block: int
    extras: dict = field(default_factory=dict)


def _norm_titles(df: pd.DataFrame) -> pd.Series:
    if "title_norm" in df.columns:
        return df["title_norm"].fillna("")
    return df["title"].map(normalise_text)


def block_by_sorted_ngrams(
    df: pd.DataFrame, n: int = 3, k: int = 4
) -> dict[str, list[str]]:
    """Key = the first ``k`` sorted character ``n``-grams of the normalised title.

    Sorting makes the key insensitive to where in the title the characters fall, so two
    records that disagree on word order still collide. Applies to both corpora.
    """
    blocks: dict[str, list[str]] = {}
    for record_id, title in zip(df["record_id"], _norm_titles(df), strict=True):
        key = _ngram_key(title, n, k)
        if key is None:
            continue
        blocks.setdefault(key, []).append(record_id)
    return blocks


def _ngram_key(title: str, n: int, k: int) -> str | None:
    """Sorted character n-grams, generated **per token**, not across the whole string.

    Concatenating the title before taking n-grams would produce grams straddling word
    boundaries: "microscope zeiss" yields "eze" where "zeiss microscope" yields "sic".
    Those boundary grams depend on word order, which reintroduces exactly the sensitivity
    that sorting the grams exists to remove — two records differing only in word order
    would land in different blocks. Generating per token keeps the key order-invariant.

    Tokens shorter than ``n`` contribute themselves, so a short but distinctive token
    ("a4", "ph") is not silently dropped.
    """
    grams: set[str] = set()
    for token in title.split():
        if len(token) < n:
            grams.add(token)
            continue
        grams.update(token[i : i + n] for i in range(len(token) - n + 1))
    if not grams:
        return None
    return "|".join(sorted(grams)[:k])


def block_by_leading_token(
    df: pd.DataFrame, stopwords: frozenset[str] = LEADING_STOPWORDS
) -> dict[str, list[str]]:
    """Key = the first substantive token of the normalised title.

    This approximates a brand designation where the title carries one. On a product
    catalogue the leading token usually *is* the brand; on procurement titles it usually is
    not, and how well it performs on each corpus is a measured result reported as-is
    (§6.8). It is a stated approximation, evaluated as one.

    Records whose title yields no substantive token are placed in no block, and that count
    is reported rather than silently absorbed.
    """
    blocks: dict[str, list[str]] = {}
    for record_id, title in zip(df["record_id"], _norm_titles(df), strict=True):
        token = _leading_token(title, stopwords)
        if token is None:
            continue
        blocks.setdefault(token, []).append(record_id)
    return blocks


def _leading_token(title: str, stopwords: frozenset[str]) -> str | None:
    for token in title.split():
        if token in stopwords:
            continue
        if token.isdigit():  # quantities: "2x", "10 microscopes"
            continue
        return token
    return None


def block_by_buyer(df: pd.DataFrame) -> dict[str, list[str]]:
    """Key = ``buyer_id``. **Corpus B only** — Abt-Buy has no publishing authority."""
    if "buyer_id" not in df.columns or df["buyer_id"].isna().all():
        raise SchemeUnavailable(
            "buyer_id is absent or wholly null on this corpus; the buyer scheme cannot "
            "be evaluated here. This asymmetry between the corpora is a reported result, "
            "not a condition to work around."
        )
    blocks: dict[str, list[str]] = {}
    for record_id, buyer in zip(df["record_id"], df["buyer_id"], strict=True):
        if pd.isna(buyer) or buyer == "":
            continue
        blocks.setdefault(str(buyer), []).append(record_id)
    return blocks


_BUILDERS = {
    "sorted_ngrams": block_by_sorted_ngrams,
    "leading_token": block_by_leading_token,
    "buyer": block_by_buyer,
}


def applicable_schemes(df: pd.DataFrame) -> list[str]:
    """Which schemes this frame supports, decided by column nullity, not corpus name."""
    available = ["sorted_ngrams", "leading_token"]
    if "buyer_id" in df.columns and not df["buyer_id"].isna().all():
        available.append("buyer")
    return available


def build_blocks(df: pd.DataFrame, scheme: str, **kwargs) -> dict[str, list[str]]:
    if scheme not in _BUILDERS:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
    return _BUILDERS[scheme](df, **kwargs)


def candidate_pairs(
    df: pd.DataFrame,
    schemes: Iterable[str],
    max_block_size: int = 500,
) -> tuple[pd.DataFrame, list[BlockingReport]]:
    """Union of the blocks produced by each scheme.

    Blocks larger than ``max_block_size`` are dropped and counted. That count is reported:
    it is a source of lost recall, and a scheme whose reduction ratio looks good only
    because its biggest blocks were discarded is not actually doing the work.

    Returns ``(pairs[left_id, right_id], reports)`` with pairs deduplicated across schemes
    and each pair ordered so ``left_id < right_id``.
    """
    all_ids = set(df["record_id"])
    pairs: set[tuple[str, str]] = set()
    reports: list[BlockingReport] = []

    for scheme in schemes:
        blocks = build_blocks(df, scheme)

        kept, dropped, dropped_records = {}, 0, 0
        for key, members in blocks.items():
            if len(members) > max_block_size:
                dropped += 1
                dropped_records += len(members)
                continue
            kept[key] = members

        scheme_pairs: set[tuple[str, str]] = set()
        for members in kept.values():
            for a, b in itertools.combinations(sorted(set(members)), 2):
                scheme_pairs.add((a, b))

        blocked_ids = {m for members in blocks.values() for m in members}
        reports.append(
            BlockingReport(
                scheme=scheme,
                n_records=len(all_ids),
                n_blocks=len(blocks),
                n_candidates=len(scheme_pairs),
                blocks_dropped=dropped,
                records_in_dropped_blocks=dropped_records,
                n_unblocked_records=len(all_ids - blocked_ids),
                largest_block=max((len(m) for m in blocks.values()), default=0),
            )
        )
        pairs |= scheme_pairs

    frame = pd.DataFrame(sorted(pairs), columns=["left_id", "right_id"])
    return frame, reports


def evaluate_blocking(
    candidates: pd.DataFrame, truth: pd.DataFrame, n_records: int
) -> dict:
    """Pair completeness and reduction ratio against a labelled truth set.

    ``pair_completeness`` is the share of true matching pairs the candidate set retains —
    the recall ceiling everything downstream inherits. ``reduction_ratio`` is the share of
    the full comparison space avoided.

    ``truth`` must carry ``left_id``, ``right_id`` and ``label``; only positive pairs
    count towards completeness.
    """
    positives = {
        tuple(sorted((a, b)))
        for a, b, label in zip(
            truth["left_id"], truth["right_id"], truth["label"], strict=True
        )
        if label == 1
    }
    generated = {
        tuple(sorted((a, b)))
        for a, b in zip(candidates["left_id"], candidates["right_id"], strict=True)
    }

    n_possible = n_records * (n_records - 1) // 2
    retained = len(positives & generated)
    return {
        "pair_completeness": retained / len(positives) if positives else float("nan"),
        "reduction_ratio": 1 - (len(generated) / n_possible) if n_possible else float("nan"),
        "n_candidates": len(generated),
        "n_possible": n_possible,
        "n_true_positives": len(positives),
        "n_true_positives_retained": retained,
        "n_true_positives_lost": len(positives) - retained,
    }
