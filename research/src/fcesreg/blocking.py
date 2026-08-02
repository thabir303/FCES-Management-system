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
    df: pd.DataFrame, n: int = 3, k: int = 4, mode: str = "single_key"
) -> dict[str, list[str]]:
    """Character n-gram blocking over the normalised title. Applies to both corpora.

    Two formulations, because they behave very differently and the choice between them is
    a measured decision rather than an assumption:

    ``single_key`` (as §6.8 specifies)
        One composite key per record: the first ``k`` sorted n-grams joined together. Two
        records collide only if they agree exactly on their ``k`` alphabetically-earliest
        n-grams. This is an exact-match key over a derived string, and it is brittle by
        construction — a single differing character early in the alphabet separates two
        records that are otherwise identical.

    ``per_gram`` (q-gram indexing, after Christen)
        One key per n-gram: a record joins a block for every n-gram its title contains, so
        records collide if they share *any* n-gram. This is the standard formulation in
        the blocking literature the paper cites, and it trades reduction ratio for
        completeness.

    Sorting is what makes either key insensitive to where in the title the characters
    fall, so records disagreeing on word order still collide.
    """
    blocks: dict[str, list[str]] = {}
    for record_id, title in zip(df["record_id"], _norm_titles(df), strict=True):
        if mode == "single_key":
            key = _ngram_key(title, n, k)
            if key is not None:
                blocks.setdefault(key, []).append(record_id)
        elif mode == "per_gram":
            for gram in _grams(title, n):
                blocks.setdefault(gram, []).append(record_id)
        else:
            raise ValueError(f"mode must be 'single_key' or 'per_gram', got {mode!r}")
    return blocks


def _grams(title: str, n: int) -> set[str]:
    """Character n-grams generated per token, never across word boundaries."""
    grams: set[str] = set()
    for token in title.split():
        if len(token) < n:
            grams.add(token)
        else:
            grams.update(token[i : i + n] for i in range(len(token) - n + 1))
    return grams


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
    grams = _grams(title, n)
    if not grams:
        return None
    return "|".join(sorted(grams)[:k])


def ngram_overlap_candidates(
    df: pd.DataFrame,
    n: int = 3,
    min_overlap: int = 1,
    max_block_size: int = 500,
    chunk_rows: int = 2000,
) -> tuple[pd.DataFrame, BlockingReport]:
    """Per-gram q-gram indexing with a shared-gram threshold.

    A pair is a candidate when the two records share at least ``min_overlap`` n-grams.
    At ``min_overlap=1`` this is plain q-gram indexing, which retains every true pair on
    Corpus A but only reaches reduction ratio 0.588 — too little pruning to carry a
    four-matcher sweep, let alone a language-model cascade. Raising the threshold trades
    completeness for reduction, and the curve traced by sweeping it is the blocking
    result the paper reports.

    Overlap counts come from a sparse record-by-gram incidence matrix multiplied by its
    own transpose, in row chunks so the intermediate never has to fit in memory at once.
    Grams whose block exceeds ``max_block_size`` are dropped before counting and the loss
    is reported: a gram shared by thousands of records carries no evidence of identity,
    and keeping it would dominate the product.
    """
    import numpy as np
    from scipy import sparse

    ids = list(df["record_id"])
    index = {rid: i for i, rid in enumerate(ids)}

    gram_members: dict[str, list[int]] = {}
    for rid, title in zip(df["record_id"], _norm_titles(df), strict=True):
        for gram in _grams(title, n):
            gram_members.setdefault(gram, []).append(index[rid])

    kept_grams = {g: m for g, m in gram_members.items() if len(m) <= max_block_size}
    dropped = len(gram_members) - len(kept_grams)
    dropped_records = sum(
        len(m) for g, m in gram_members.items() if len(m) > max_block_size
    )

    rows, cols = [], []
    for col, members in enumerate(kept_grams.values()):
        rows.extend(members)
        cols.extend([col] * len(members))
    incidence = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(len(ids), max(len(kept_grams), 1)),
    )

    left, right = [], []
    for start in range(0, len(ids), chunk_rows):
        stop = min(start + chunk_rows, len(ids))
        overlap = (incidence[start:stop] @ incidence.T).tocoo()
        for i, j, count in zip(overlap.row, overlap.col, overlap.data, strict=True):
            gi = start + i
            if gi < j and count >= min_overlap:  # upper triangle only
                left.append(ids[gi])
                right.append(ids[j])

    pairs = pd.DataFrame({"left_id": left, "right_id": right})
    blocked = {i for m in kept_grams.values() for i in m}
    report = BlockingReport(
        scheme="sorted_ngrams",
        n_records=len(ids),
        n_blocks=len(kept_grams),
        n_candidates=len(pairs),
        blocks_dropped=dropped,
        records_in_dropped_blocks=dropped_records,
        n_unblocked_records=len(ids) - len(blocked),
        largest_block=max((len(m) for m in kept_grams.values()), default=0),
        extras={"min_overlap": min_overlap, "n": n, "mode": "per_gram"},
    )
    return pairs, report


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
    scheme_kwargs: dict[str, dict] | None = None,
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

    scheme_kwargs = scheme_kwargs or {}
    for scheme in schemes:
        kwargs = dict(scheme_kwargs.get(scheme, {}))

        # Per-gram indexing with an overlap threshold cannot be expressed as
        # blocks-then-pairs: a pair qualifies on how many blocks it shares, not on
        # sharing any one of them. It has its own counting path.
        if scheme == "sorted_ngrams" and kwargs.get("mode") == "per_gram":
            kwargs.pop("mode", None)
            kwargs.pop("k", None)
            scheme_pairs_df, report = ngram_overlap_candidates(
                df, max_block_size=max_block_size, **kwargs
            )
            reports.append(report)
            pairs |= {
                (a, b)
                for a, b in zip(
                    scheme_pairs_df["left_id"], scheme_pairs_df["right_id"], strict=True
                )
            }
            continue

        blocks = build_blocks(df, scheme, **kwargs)

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
    candidates: pd.DataFrame, truth: pd.DataFrame | None, n_records: int
) -> dict:
    """Pair completeness and reduction ratio.

    ``reduction_ratio`` is the share of the full comparison space avoided. It needs no
    labels and is always computed — a corpus without duplicate ground truth can still
    report what blocking cost.

    ``pair_completeness`` is the share of true matching pairs the candidate set retains,
    the recall ceiling everything downstream inherits. It requires ``truth`` carrying
    ``left_id``, ``right_id`` and ``label``; only positive pairs count. Where no truth
    exists it is reported as ``None`` and left **unmeasured**, never estimated.
    """
    generated = {
        tuple(sorted((a, b)))
        for a, b in zip(candidates["left_id"], candidates["right_id"], strict=True)
    }
    n_possible = n_records * (n_records - 1) // 2

    out = {
        "reduction_ratio": 1 - (len(generated) / n_possible) if n_possible else None,
        "n_candidates": len(generated),
        "n_possible": n_possible,
    }

    if truth is None:
        return out | {
            "pair_completeness": None,
            "pair_completeness_note": "no labelled duplicate pairs for this corpus",
        }

    positives = {
        tuple(sorted((a, b)))
        for a, b, label in zip(
            truth["left_id"], truth["right_id"], truth["label"], strict=True
        )
        if label == 1
    }
    retained = len(positives & generated)
    return out | {
        "pair_completeness": retained / len(positives) if positives else None,
        "n_true_positives": len(positives),
        "n_true_positives_retained": retained,
        "n_true_positives_lost": len(positives) - retained,
    }
