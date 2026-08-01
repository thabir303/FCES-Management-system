"""Sentence embeddings with a disk cache (§6.7).

The cache is what keeps the full experiment sweep tractable: the severity sweep encodes
the same records repeatedly across matchers, severities and seeds, and the classification
runs re-encode the taxonomy on every invocation. It is implemented before anything calls
:func:`embed` for that reason.

Two properties matter and are tested.

**The cache is per text, not per call.** Two runs over overlapping corpora share every
record they have in common, so a sweep that adds one severity level pays only for the
records that severity actually changed.

**A fully cached call never loads the model.** Loading a sentence-transformer touches the
Hugging Face hub, so a cache hit that still constructed the model would neither be fast
nor offline. The model is constructed lazily, only when there is at least one miss.

CPU only (§12.7). Do not call ``.to("cuda")`` or ``.to("mps")`` — MPS is available on the
development machine and must not be used, because a result that depends on the accelerator
is not reproducible on the marker's machine.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

__all__ = ["DEFAULT_MODEL", "DEFAULT_CACHE_DIR", "cache_key", "embed", "cache_stats"]

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CACHE_DIR = Path(".cache/embeddings")

_MODELS: dict[str, object] = {}


def cache_key(model_id: str, text: str) -> str:
    """``sha256(model_id + '\\x00' + text)``.

    The model id is inside the digest, so switching checkpoints cannot silently reuse
    vectors produced by a different model — the commonest way an embedding cache
    invalidates a result without anyone noticing.
    """
    return hashlib.sha256(f"{model_id}\x00{text}".encode()).hexdigest()


def _path_for(cache_dir: Path, key: str) -> Path:
    # Sharded two levels deep: a flat directory of several hundred thousand entries is
    # slow to stat on most filesystems.
    return cache_dir / key[:2] / key[2:4] / f"{key}.npy"


def _load_model(model_id: str):
    """Construct the encoder, once per process. Imported lazily: importing
    sentence_transformers costs seconds, and a fully cached run should not pay it."""
    if model_id not in _MODELS:
        from sentence_transformers import SentenceTransformer

        _MODELS[model_id] = SentenceTransformer(model_id, device="cpu")
    return _MODELS[model_id]


def _read_cached(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(path)
    except (ValueError, OSError):
        # A truncated file from an interrupted write. Treat as a miss and overwrite.
        return None


def _write_cached(path: Path, vector: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename, so an interrupted run leaves no half-written vector that a later
    # run would read back as real. The handle is opened explicitly because np.save appends
    # ".npy" to any path that does not already end in it, which would silently write to a
    # different filename than the one being renamed.
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, vector)
    tmp.replace(path)


def embed(
    texts: Sequence[str],
    model_id: str = DEFAULT_MODEL,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode ``texts`` to L2-normalised float32 of shape ``(len(texts), dim)``.

    Duplicate texts within one call are encoded once. Order is preserved, and the returned
    array is aligned with the input positionally.
    """
    cache_dir = Path(cache_dir)
    if len(texts) == 0:
        return np.empty((0, 0), dtype=np.float32)

    keys = [cache_key(model_id, t) for t in texts]

    # Unique texts only: a corpus with repeated titles should not pay for them twice.
    order: dict[str, str] = {}
    for key, text in zip(keys, texts, strict=True):
        order.setdefault(key, text)

    vectors: dict[str, np.ndarray] = {}
    misses: list[str] = []
    for key in order:
        cached = _read_cached(_path_for(cache_dir, key))
        if cached is None:
            misses.append(key)
        else:
            vectors[key] = cached

    if misses:
        model = _load_model(model_id)
        encoded = model.encode(
            [order[k] for k in misses],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        encoded = np.asarray(encoded, dtype=np.float32)
        for key, vector in zip(misses, encoded, strict=True):
            vectors[key] = vector
            _write_cached(_path_for(cache_dir, key), vector)

    return np.vstack([vectors[k] for k in keys]).astype(np.float32)


def cache_stats(
    texts: Sequence[str], model_id: str = DEFAULT_MODEL, cache_dir: Path = DEFAULT_CACHE_DIR
) -> dict[str, int]:
    """How many of ``texts`` are already cached. Used by the runners to report cost."""
    cache_dir = Path(cache_dir)
    keys = {cache_key(model_id, t) for t in texts}
    hits = sum(1 for k in keys if _path_for(cache_dir, k).exists())
    return {"n_texts": len(texts), "n_unique": len(keys), "n_cached": hits,
            "n_missing": len(keys) - hits}
