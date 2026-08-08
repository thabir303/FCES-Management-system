"""C5 acceptance harness: run a small live pilot, then prove the re-run is free.

This is not a results runner and it does not appear in §10's table. It exists because C5's
criterion is about behaviour against the real endpoint — that a repeat of an identical set
issues no HTTP request, consumes no tokens and sums ``usd`` to exactly zero — and that cannot
be closed from the test suite, which necessarily stubs the transport.

Its ledger rows carry ``condition: c5_pilot`` so `run_costs.py` can exclude them. An acceptance
check is not a measurement of the method and must not reach `T8_cost.tex`.

**Prompts carry a per-invocation nonce.** The disk cache is content-addressed and persists
across invocations by design — that is the whole point of it — so a prompt set that read the
same on every run would, on the second run, replay entirely from a previous invocation's
cache. The "live" round would then be zero live calls, proving nothing. The nonce forces a
fresh cache key each time this script runs, while the two rounds within one invocation still
share it, so the in-run replay still proves what it is meant to prove.

    make llm-pilot          # or:
    .venv/bin/python research/scripts/run_llm_pilot.py --config research/configs/llm.yaml
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import yaml
from dotenv import load_dotenv

from fcesreg.llm import (
    DEFAULT_LEDGER_PATH,
    DailyQuotaExhausted,
    LLMClient,
    LLMError,
    LLMRequest,
    Limits,
    RateCard,
    read_ledger,
)
from fcesreg.paths import repo_root
from fcesreg.runs import new_run_id

PROMPTS = [
    "Are these the same product? A: {a}. B: {b}. Answer yes or no.",
]


def build_requests(n: int, condition: str, max_tokens: int, nonce: str) -> list[LLMRequest]:
    """Short and distinct, so the pilot costs as little quota as possible.

    ``nonce`` is fixed for the whole invocation but new on every one, so this invocation's
    live round cannot be served from a previous invocation's cache — see the module
    docstring. ``custom_id`` deliberately excludes it: identity across the live and replay
    rounds within *this* run is what ``complete_many``'s dict keying is tested against.
    """
    return [
        LLMRequest(
            custom_id=f"pilot-{i:03d}",
            system="You adjudicate whether two catalogue entries describe the same item.",
            prompt=PROMPTS[0].format(a=f"widget model {i}", b=f"widget model {i} (boxed)")
            + f" [pilot nonce: {nonce}]",
            max_tokens=max_tokens,
            condition=condition,
        )
        for i in range(n)
    ]


def client_from(config: dict, run_id: str) -> LLMClient:
    card = config["rate_card"]
    limits = config["limits"]
    return LLMClient(
        model=config["model"],
        run_id=run_id,
        cap_usd=float(config["cap_usd"]),
        base_url=config["base_url"],
        rate_card=RateCard(
            model=config["model"],
            usd_per_m_input=float(card["usd_per_m_input"]),
            usd_per_m_output=float(card["usd_per_m_output"]),
            source=card["source"],
            checked=str(card["checked"]),
        ),
        limits=Limits(
            requests_per_minute=int(limits["requests_per_minute"]),
            tokens_per_minute=int(limits["tokens_per_minute"]),
            requests_per_day=int(limits["requests_per_day"]),
            tokens_per_day=int(limits["tokens_per_day"]),
        ),
    )


def main() -> int:
    # .env is never committed and nothing else in the stack loads it into the process
    # environment — llm.py deliberately only reads os.environ, never .env mechanics, so the
    # bootstrap belongs here, in the one script that is meant to be invoked directly. Anchored
    # against the repo root rather than the cwd, for the same reason every path in this
    # package is: `make llm-pilot` and a direct invocation from research/ must behave alike.
    load_dotenv(repo_root() / ".env")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("research/configs/llm.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pilot = config["pilot"]
    nonce = uuid.uuid4().hex[:8]
    requests = build_requests(
        int(pilot["n"]), pilot["condition"], int(pilot["max_tokens"]), nonce
    )

    before = len(read_ledger(DEFAULT_LEDGER_PATH))

    try:
        first = client_from(config, new_run_id("llm_pilot_live", args.config))
        live = first.complete_many(requests)

        second = client_from(config, new_run_id("llm_pilot_replay", args.config))
        replayed = second.complete_many(requests)
    except DailyQuotaExhausted as exhausted:
        print(f"pilot stopped on quota: {exhausted}")
        print("Nothing is lost — the completed calls are cached. Re-run tomorrow to resume.")
        return 2
    except LLMError as error:
        print(f"pilot failed: {error}")
        return 1

    rows = read_ledger(DEFAULT_LEDGER_PATH)[before:]
    live_rows = [r for r in rows if not r["cache_hit"]]
    replay_rows = [r for r in rows if r["cache_hit"]]

    if not live_rows:
        # The nonce above is what should make this impossible — every call in a fresh
        # invocation must miss the cache. Surfacing this plainly beats an IndexError three
        # lines down if that guarantee is ever broken.
        print(
            f"pilot produced zero live calls (nonce {nonce}); the cache should be incapable "
            "of a hit here. Something is wrong with cache keying or the nonce — investigate "
            "before re-running."
        )
        return 1

    live_tokens = sum(r["input_tokens"] + r["output_tokens"] for r in live_rows)
    replay_usd = sum(r["usd"] for r in replay_rows)
    replay_consumed = sum(
        r["input_tokens"] + r["output_tokens"] for r in replay_rows if not r["cache_hit"]
    )

    print(f"live calls      : {len(live_rows)} consuming {live_tokens} tokens")
    print(f"notional cost   : ${sum(r['usd'] for r in live_rows):.6f} at "
          f"{live_rows[0]['rate_usd_per_m_input']}/{live_rows[0]['rate_usd_per_m_output']} per M "
          f"(checked {live_rows[0]['rate_card_checked']})")
    print(f"replayed calls  : {len(replay_rows)}, all cache_hit="
          f"{all(r['cache_hit'] for r in replay_rows)}")
    print(f"replay consumed : {replay_consumed} tokens, ${replay_usd:.2f}")

    checks = {
        "at least 20 live calls": len(live_rows) >= 20,
        "live rows carry non-zero tokens": all(
            r["input_tokens"] > 0 for r in live_rows
        ),
        "live rows carry a rate card": all(r.get("rate_card_checked") for r in live_rows),
        "every replayed row logs cache_hit=true": bool(replay_rows)
        and all(r["cache_hit"] for r in replay_rows),
        "replay consumed zero tokens": replay_consumed == 0,
        "replay cost exactly $0.00": replay_usd == 0.0,
        "every row carries run_id": all(r["run_id"] for r in rows),
        "results keyed by custom_id": set(live) == set(replayed) == {r.custom_id for r in requests},
    }
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
