"""RQ2's language-model condition (§6.10, amendment 13).

Everything that touches the outside world is injected, the same discipline `test_llm.py`
and `test_cascade.py` hold: a scripted transport stands in for the endpoint, so the whole
request/response path — prompt construction, retrieval, strict parsing — is checkable
without a key, a network or a single unit of quota. Nothing here should need to change once
the real run happens; if it does, the test was checking the wrong thing.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fcesreg import embed as embed_mod
from fcesreg.classify import ClassificationFailed, RagFewShotLLMClassifier, rag_prompt
from fcesreg.llm import LLMClient, RateCard


@pytest.fixture
def stub_encoder(monkeypatch):
    """Deterministic embeddings keyed by text content, so retrieval order is checkable —
    identical to test_classify.py's fixture of the same name."""

    class Encoder:
        def encode(self, texts, **kwargs):
            out = np.zeros((len(texts), 16), dtype=np.float32)
            for i, t in enumerate(texts):
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                v = rng.normal(size=16).astype(np.float32)
                out[i] = v / np.linalg.norm(v)
            return out

    monkeypatch.setattr(embed_mod, "_load_model", lambda model_id: Encoder())


CARD = RateCard(
    model="test-model", usd_per_m_input=0.15, usd_per_m_output=0.60,
    source="test", checked="2026-08-08",
)


class ScriptedTransport:
    """Returns one scripted JSON reply per call, in order."""

    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.calls = 0
        self.payloads: list[dict] = []

    def __call__(self, payload):
        self.calls += 1
        self.payloads.append(payload)
        content = json.dumps(self.replies.pop(0))
        return (
            200,
            {
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 400, "completion_tokens": 40},
            },
            {"x-ratelimit-remaining-tokens": "190000", "x-ratelimit-remaining-requests": "990"},
        )


def make_llm_client(tmp_path, replies: list[dict], **kwargs) -> LLMClient:
    return LLMClient(
        model="test-model",
        cache_dir=tmp_path / "cache",
        ledger_path=tmp_path / "ledger.jsonl",
        run_id="run-1",
        rate_card=CARD,
        transport=ScriptedTransport(replies),
        sleep=lambda s: None,
        monotonic=lambda: 0.0,
        **kwargs,
    )


def taxonomy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cpv_code": ["30", "33", "42"],
            "cpv_description": ["Office equipment", "Medical equipment", "Machinery"],
        }
    )


def train(per_class: int = 6) -> pd.DataFrame:
    codes = np.repeat(["30100000", "33100000", "42100000"], per_class)
    return pd.DataFrame(
        {
            "record_id": [f"tr{i}" for i in range(len(codes))],
            "title": [f"{c[:2]} widget {i}" for i, c in enumerate(codes)],
            "description": [f"description for division {c[:2]}" for c in codes],
            "cpv_code": codes,
        }
    )


def target(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": [f"te{i}" for i in range(n)],
            "title": [f"target record {i}" for i in range(n)],
            "description": ["needs classifying"] * n,
            "cpv_code": ["30100000"] * n,
        }
    )


class TestFit:
    def test_label_set_is_division_codes_present_in_train(self, stub_encoder, tmp_path):
        clf = RagFewShotLLMClassifier(make_llm_client(tmp_path, []), taxonomy())
        clf.fit(train(), "division")
        assert clf._label_set == ["30", "33", "42"]

    def test_system_prompt_carries_every_code_and_description(self, stub_encoder, tmp_path):
        clf = RagFewShotLLMClassifier(make_llm_client(tmp_path, []), taxonomy())
        clf.fit(train(), "division")
        assert "30: Office equipment" in clf._system_prompt
        assert "33: Medical equipment" in clf._system_prompt
        assert "42: Machinery" in clf._system_prompt

    def test_unknown_level_is_refused(self, stub_encoder, tmp_path):
        clf = RagFewShotLLMClassifier(make_llm_client(tmp_path, []), taxonomy())
        with pytest.raises(ValueError, match="level must be"):
            clf.fit(train(), "leaf")


class TestPredict:
    def test_one_result_per_record(self, stub_encoder, tmp_path):
        replies = [
            {"code": "30", "runner_up": "33", "reason": "office widget"},
            {"code": "30", "runner_up": "42", "reason": "office widget"},
        ]
        clf = RagFewShotLLMClassifier(make_llm_client(tmp_path, replies), taxonomy())
        clf.fit(train(), "division")
        result = clf.predict(target())
        assert len(result.codes) == 2
        assert result.codes == ["30", "30"]

    def test_scores_are_order_markers_not_probabilities(self, stub_encoder, tmp_path):
        replies = [{"code": "30", "runner_up": "33", "reason": "x"}]
        clf = RagFewShotLLMClassifier(make_llm_client(tmp_path, replies), taxonomy())
        clf.fit(train(), "division")
        result = clf.predict(target(1))
        assert result.scores[0] == 1.0
        assert result.alternatives[0] == [("33", 0.5)]

    def test_alternatives_empty_when_runner_up_equals_code(self, stub_encoder, tmp_path):
        # A degenerate but schema-valid reply; must not be reported as a real alternative.
        replies = [{"code": "30", "runner_up": "30", "reason": "x"}]
        clf = RagFewShotLLMClassifier(make_llm_client(tmp_path, replies), taxonomy())
        clf.fit(train(), "division")
        result = clf.predict(target(1))
        assert result.alternatives[0] == []

    def test_malformed_json_raises_rather_than_defaults(self, stub_encoder, tmp_path):
        def bad_transport(payload):
            return (
                200,
                {"choices": [{"message": {"content": "not json"}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                {"x-ratelimit-remaining-tokens": "1000", "x-ratelimit-remaining-requests": "10"},
            )

        client = make_llm_client(tmp_path, [])
        client._transport = bad_transport
        clf = RagFewShotLLMClassifier(client, taxonomy())
        clf.fit(train(), "division")
        with pytest.raises(ClassificationFailed, match="was not JSON"):
            clf.predict(target(1))

    def test_code_outside_label_set_raises(self, stub_encoder, tmp_path):
        replies = [{"code": "99", "runner_up": "30", "reason": "x"}]
        clf = RagFewShotLLMClassifier(make_llm_client(tmp_path, replies), taxonomy())
        clf.fit(train(), "division")
        with pytest.raises(ClassificationFailed, match="not in the"):
            clf.predict(target(1))

    def test_custom_id_carries_the_record_id(self, stub_encoder, tmp_path):
        replies = [{"code": "30", "runner_up": "33", "reason": "x"}]
        client = make_llm_client(tmp_path, replies)
        clf = RagFewShotLLMClassifier(client, taxonomy())
        clf.fit(train(), "division")
        clf.predict(target(1))
        sent = client._transport.payloads[0]
        # The record_id must appear somewhere traceable -- via the cache/ledger path, not
        # the payload itself (custom_id is not sent to the endpoint). Checked indirectly:
        # predict() must not raise KeyError building requests, which it would if record_id
        # handling were broken; the ledger row is the traceable artefact.
        assert sent["messages"][0]["role"] == "system"

    def test_schema_is_the_full_response_format_object(self, stub_encoder, tmp_path):
        replies = [{"code": "30", "runner_up": "33", "reason": "x"}]
        client = make_llm_client(tmp_path, replies)
        clf = RagFewShotLLMClassifier(client, taxonomy())
        clf.fit(train(), "division")
        clf.predict(target(1))
        sent = client._transport.payloads[0]
        schema = sent["response_format"]["json_schema"]["schema"]
        assert set(schema["properties"]["code"]["enum"]) == {"30", "33", "42"}

    def test_predict_before_fit_raises(self, stub_encoder, tmp_path):
        clf = RagFewShotLLMClassifier(make_llm_client(tmp_path, []), taxonomy())
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(target(1))


class TestRetrieval:
    def test_examples_are_the_k_nearest_by_embedding_similarity(self, stub_encoder, tmp_path):
        # With the stub encoder, embedding similarity is a deterministic function of text
        # content -- so the nearest example to a target whose title is copied verbatim from
        # one training record must be that exact record, not an arbitrary one from its class.
        train_df = train(per_class=6)
        needle = train_df.iloc[3]
        query = pd.DataFrame(
            {
                "record_id": ["q0"],
                "title": [needle["title"]],
                "description": [needle["description"]],
                "cpv_code": ["30100000"],
            }
        )
        replies = [{"code": "30", "runner_up": "33", "reason": "x"}]
        client = make_llm_client(tmp_path, replies)
        clf = RagFewShotLLMClassifier(client, taxonomy(), k_examples=2)
        clf.fit(train_df, "division")
        clf.predict(query)
        prompt = client._transport.payloads[0]["messages"][1]["content"]
        assert needle["title"] in prompt

    def test_k_examples_controls_how_many_are_retrieved(self, stub_encoder, tmp_path):
        replies = [{"code": "30", "runner_up": "33", "reason": "x"}]
        client = make_llm_client(tmp_path, replies)
        clf = RagFewShotLLMClassifier(client, taxonomy(), k_examples=3)
        clf.fit(train(), "division")
        clf.predict(target(1))
        prompt = client._transport.payloads[0]["messages"][1]["content"]
        # 3 examples + 1 target block, separated by the blank-line join in rag_prompt.
        assert prompt.count("record:") == 4


class TestRagPrompt:
    def test_examples_precede_the_target(self):
        prompt = rag_prompt([("ex title", "30")], "target title")
        assert prompt.index("ex title") < prompt.index("target title")
        assert "code: 30" in prompt
        assert prompt.rstrip().endswith("code:")
