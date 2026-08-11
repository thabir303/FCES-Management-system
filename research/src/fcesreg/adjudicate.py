"""The cascade's expensive tier: pairwise duplicate adjudication by a language model (§6.9, C6).

Separate from ``dedup`` on purpose. ``dedup`` is pure array work over DataFrames and must
stay importable and testable without a network, a key or a quota; this module is the one
place the duplicate-detection path is allowed to reach the endpoint, and it satisfies
``dedup.Adjudicator`` so the cascade never learns it exists.

**The prompt is part of the method and is committed here rather than configured.** A cost
figure and an accuracy figure are only comparable across runs if the thing being priced did
not change between them, and a prompt living in a config file is a tuning knob by another
name (§10, runners take no tuning flags).

Verdicts are parsed strictly. A malformed or missing reply is *not* silently read as "not a
duplicate": that would convert an infrastructure failure into a measurement, and it would
bias in a predictable direction, since the pairs hardest to parse are not a random sample.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fcesreg.llm import LLMClient, LLMRequest
from fcesreg.schema import text_of

__all__ = ["SYSTEM_PROMPT", "VERDICT_SCHEMA", "LLMAdjudicator", "AdjudicationFailed"]

#: Committed with the code, not the config — see the module docstring.
SYSTEM_PROMPT = (
    "You decide whether two records describe the same physical product or procurement. "
    "Reply with a JSON object and nothing else: "
    '{"same": true or false, "reason": "<one short sentence>"}. '
    "Records may differ in wording, casing, punctuation, abbreviation or completeness and "
    "still describe the same thing; judge the underlying item, not the text. "
    "Different models, capacities, configurations, lots or phases of one product line or "
    "one programme are NOT the same thing. "
    "If the evidence is genuinely insufficient, answer false."
)

#: Constrains the reply shape. ``reason`` is required so a verdict is never unexplained in
#: the ledger, where it is the only record of why a pair was decided as it was.
VERDICT_SCHEMA: dict = {
    "type": "object",
    "properties": {"same": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["same", "reason"],
    "additionalProperties": False,
}


class AdjudicationFailed(RuntimeError):
    """A banded pair came back missing or unparseable.

    Raised rather than defaulted, because a default is a measurement the model did not
    make, and the pairs that fail to parse are not a random sample of the band.
    """


def _render(record: pd.Series, tag: str) -> str:
    title = record.get("title")
    description = record.get("description")
    title = title if isinstance(title, str) else ""
    description = description if isinstance(description, str) else ""
    return f"[{tag}]\ntitle: {title}\ndescription: {description}".rstrip()


def render_prompt(left: pd.Series, right: pd.Series) -> str:
    """The user turn for one pair. Public because the cost probe prices this exact text."""
    return f"{_render(left, 'A')}\n\n{_render(right, 'B')}\n\nDo A and B describe the same thing?"


class LLMAdjudicator:
    """``dedup.Adjudicator`` backed by :class:`fcesreg.llm.LLMClient`.

    Every call goes through the client's cache, ledger and quota governance, so a resumed
    sweep re-reads what it already paid for and a run that exhausts the daily allowance
    stops rather than degrading quietly.
    """

    def __init__(
        self,
        client: LLMClient,
        condition: str = "cascade",
        max_tokens: int = 96,
    ):
        self.client = client
        self.condition = condition
        self.max_tokens = max_tokens

    def adjudicate(self, pairs: pd.DataFrame, records: pd.DataFrame) -> np.ndarray:
        by_id = records.set_index("record_id")
        # text_of is not used for the prompt itself -- the model is shown the fields
        # separately, because a concatenated blob loses the distinction between a title and
        # a description that the judgement partly turns on.
        requests = []
        for position, (_, pair) in enumerate(pairs.iterrows()):
            left = by_id.loc[pair["left_id"]]
            right = by_id.loc[pair["right_id"]]
            requests.append(
                LLMRequest(
                    custom_id=f"{position}|{pair['left_id']}|{pair['right_id']}",
                    system=SYSTEM_PROMPT,
                    prompt=render_prompt(left, right),
                    max_tokens=self.max_tokens,
                    json_schema=VERDICT_SCHEMA,
                    condition=self.condition,
                )
            )

        results = self.client.complete_many(requests)

        out = np.zeros(len(requests), dtype=np.float64)
        for position, request in enumerate(requests):
            response = results.get(request.custom_id)
            if response is None:
                raise AdjudicationFailed(
                    f"no reply for {request.custom_id}; the band is incomplete and a "
                    f"partial band must not be reported as a whole one"
                )
            out[position] = 1.0 if _verdict(response.text, request.custom_id) else 0.0
        return out


def _verdict(text: str, custom_id: str) -> bool:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AdjudicationFailed(
            f"{custom_id}: reply was not JSON ({exc}); refusing to guess a verdict from "
            f"{text[:120]!r}"
        ) from exc
    if not isinstance(parsed, dict) or "same" not in parsed:
        raise AdjudicationFailed(f"{custom_id}: reply carried no `same` field: {text[:120]!r}")
    same = parsed["same"]
    if not isinstance(same, bool):
        raise AdjudicationFailed(f"{custom_id}: `same` was {same!r}, not a boolean")
    return same
