"""C1 acceptance: a repeat call is far faster and loads no model at all.

The speed criterion is stated as ">50x faster" in the plan. Wall-clock ratios are a poor
thing to assert in a test suite, so the mechanism behind the speedup is asserted directly
and precisely: a fully cached call must not construct the encoder. That is both the reason
it is fast and the reason it touches no network, and unlike a timing ratio it cannot pass
or fail depending on how loaded the machine is. One timing check is kept as a smoke test
against a deliberately slow stub.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fcesreg import embed as embed_mod
from fcesreg.embed import DEFAULT_MODEL, cache_key, cache_stats, embed

DIM = 8


class StubEncoder:
    """Deterministic stand-in for a sentence-transformer. Counts what it encodes."""

    def __init__(self, delay: float = 0.0):
        self.calls = 0
        self.texts_encoded = 0
        self.delay = delay

    def encode(self, texts, **kwargs):
        self.calls += 1
        self.texts_encoded += len(texts)
        if self.delay:
            time.sleep(self.delay)
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(size=DIM).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out


@pytest.fixture
def stub(monkeypatch):
    enc = StubEncoder()
    loads = {"n": 0}

    def fake_load(model_id):
        loads["n"] += 1
        return enc

    monkeypatch.setattr(embed_mod, "_load_model", fake_load)
    enc.loads = loads
    return enc


class TestCacheKey:
    def test_model_id_is_inside_the_digest(self):
        # Switching checkpoints must not silently reuse the previous model's vectors.
        assert cache_key("model-a", "pump") != cache_key("model-b", "pump")

    def test_stable(self):
        assert cache_key(DEFAULT_MODEL, "pump") == cache_key(DEFAULT_MODEL, "pump")

    def test_distinct_texts_differ(self):
        assert cache_key(DEFAULT_MODEL, "pump") != cache_key(DEFAULT_MODEL, "pumps")


class TestCaching:
    def test_second_call_does_not_construct_the_model(self, stub, tmp_path):
        texts = ["rotary vane pump", "zeiss microscope", "bench centrifuge"]

        embed(texts, cache_dir=tmp_path)
        assert stub.loads["n"] == 1
        assert stub.texts_encoded == 3

        embed(texts, cache_dir=tmp_path)
        # The whole point: no model, therefore no Hugging Face request, therefore fast.
        assert stub.loads["n"] == 1
        assert stub.texts_encoded == 3

    def test_second_call_returns_identical_vectors(self, stub, tmp_path):
        texts = ["rotary vane pump", "zeiss microscope"]
        first = embed(texts, cache_dir=tmp_path)
        second = embed(texts, cache_dir=tmp_path)
        np.testing.assert_array_equal(first, second)

    def test_partial_hit_encodes_only_the_misses(self, stub, tmp_path):
        embed(["a", "b"], cache_dir=tmp_path)
        assert stub.texts_encoded == 2

        embed(["a", "b", "c"], cache_dir=tmp_path)
        assert stub.texts_encoded == 3  # only "c"

    def test_duplicates_within_one_call_are_encoded_once(self, stub, tmp_path):
        out = embed(["pump", "pump", "pump"], cache_dir=tmp_path)
        assert stub.texts_encoded == 1
        assert out.shape == (3, DIM)
        np.testing.assert_array_equal(out[0], out[2])

    def test_a_different_model_id_does_not_reuse_vectors(self, stub, tmp_path):
        embed(["pump"], model_id="model-a", cache_dir=tmp_path)
        embed(["pump"], model_id="model-b", cache_dir=tmp_path)
        assert stub.texts_encoded == 2

    def test_truncated_cache_file_is_treated_as_a_miss(self, stub, tmp_path):
        embed(["pump"], cache_dir=tmp_path)
        path = next(tmp_path.rglob("*.npy"))
        path.write_bytes(b"not a real npy")

        out = embed(["pump"], cache_dir=tmp_path)
        assert stub.texts_encoded == 2  # re-encoded rather than crashing
        assert out.shape == (1, DIM)

    def test_no_temp_files_survive(self, stub, tmp_path):
        embed(["pump", "valve"], cache_dir=tmp_path)
        assert list(tmp_path.rglob("*.tmp")) == []


class TestOutputContract:
    def test_shape_and_dtype(self, stub, tmp_path):
        out = embed(["a", "b", "c"], cache_dir=tmp_path)
        assert out.shape == (3, DIM)
        assert out.dtype == np.float32

    def test_l2_normalised(self, stub, tmp_path):
        out = embed(["a", "b", "c"], cache_dir=tmp_path)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)

    def test_order_is_preserved(self, stub, tmp_path):
        a = embed(["alpha"], cache_dir=tmp_path)[0]
        b = embed(["beta"], cache_dir=tmp_path)[0]
        both = embed(["beta", "alpha"], cache_dir=tmp_path)
        np.testing.assert_array_equal(both[0], b)
        np.testing.assert_array_equal(both[1], a)

    def test_empty_input(self, stub, tmp_path):
        out = embed([], cache_dir=tmp_path)
        assert out.shape == (0, 0)
        assert stub.loads["n"] == 0


class TestCacheStats:
    def test_reports_hits_and_misses(self, stub, tmp_path):
        embed(["a", "b"], cache_dir=tmp_path)
        got = cache_stats(["a", "b", "c", "c"], cache_dir=tmp_path)
        assert got == {"n_texts": 4, "n_unique": 3, "n_cached": 2, "n_missing": 1}


class TestSpeed:
    def test_cached_call_is_much_faster(self, monkeypatch, tmp_path):
        # Smoke test only. The real guarantee is the assertion above that no model is
        # constructed; this shows the consequence against a stub with a 20 ms encode.
        enc = StubEncoder(delay=0.02)
        monkeypatch.setattr(embed_mod, "_load_model", lambda m: enc)
        texts = [f"item {i}" for i in range(5)]

        t0 = time.monotonic()
        embed(texts, cache_dir=tmp_path)
        cold = time.monotonic() - t0

        t0 = time.monotonic()
        embed(texts, cache_dir=tmp_path)
        warm = time.monotonic() - t0

        assert warm < cold
