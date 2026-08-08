"""Measure distractor contamination from a random sample, not a full verification.

Filtering by reference and by buyer removes what it can see. It cannot see everything,
because Contracts Finder carries no reliable process identifier: the published reference
often names the issuing organisation rather than the procurement, and ``buyer_id`` failed
to resolve to one body in nine of fourteen contaminated pairs at the last audit. No stack
of heuristics closes that cleanly, and each rule buys less than the last while being harder
to justify.

**The contamination this measures is a property of the corpus, not of the mining rule.**
Contracts Finder publishes one procurement as several notices, so any high-similarity pair
drawn from it and labelled non-matching may be a hidden duplicate — whether the mining rule
proposed it as a distractor or blocking merely surfaced it as a candidate. A clean,
fully-verified negative set was never achievable by verifying more of *this* rule's output;
every precision figure on Corpus B rests on the same corpus property either way.

So the requirement is not a clean set. It is a measured rate. This tool draws a random
sample from the mined pool — 50 pairs by default, chosen once and fixed for the whole
invocation via ``seed`` — judges it by hand, and reports the contamination rate with a
Wilson score confidence interval. **The full mined pool is used downstream unfiltered.**
Judgements here do not remove pairs from it; they estimate what fraction of it is
contaminated so that a reader can weigh the negative set's known impurity against the
precision figures it produced. Report the rate alongside precision — never use it to
arithmetically correct a precision figure; the two numbers are meant to be read together,
not combined into a third.

Resumable within one ``(pool, sample-size, seed)`` combination: judgements append as they
are made, and a re-run skips what is already judged. Changing the seed draws a different
sample and starts fresh — this file may then hold judgements from more than one sample, so
completion and the reported rate are always scoped to the *current* sample's pairs, not to
everything ever written to this file.

    python annotation/judge_distractors.py
    python annotation/judge_distractors.py --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fcesreg.degrade import DegradationConfig, make_distractors, procurement_ref, title_refs
from fcesreg.paths import annotation_path, data_path
from fcesreg.timing import ItemTiming, summarise, time_item, write_timings

KEYS = {
    "d": ("distinct", "two genuinely different procurements — keep as a negative"),
    "s": ("same_procurement", "one procurement published twice — contaminated"),
    "u": ("unsure", "cannot tell from the evidence shown — counted as contaminated"),
}

#: Verdicts that count against the negative set's purity. "unsure" joins "same_procurement"
#: rather than being excluded from the rate, matching the conservative reading the earlier
#: full-verification tool used ("drop" for both) — an uncertain pair is not evidence of
#: cleanliness.
CONTAMINATED_VERDICTS = ("same_procurement", "unsure")

#: Two-sided 95% quantile of the standard normal. Passed as a literal rather than pulled
#: from scipy so this reporting constant does not need a dependency to exist.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (point_estimate, lower, upper).

    Preferred over the naive normal (Wald) interval, which at n=50 and a low-to-moderate
    rate can push its lower bound below zero or otherwise misstate coverage — the standard
    reason Wilson's interval is used for small-sample proportions. Both bounds are clipped
    into ``[0, 1]``.
    """
    if n <= 0:
        raise ValueError("cannot estimate a rate from zero judged pairs")
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    spread = (z / denom) * ((p_hat * (1 - p_hat) / n + z2 / (4 * n * n)) ** 0.5)
    return p_hat, max(0.0, centre - spread), min(1.0, centre + spread)


def load_pool(corpus_path: Path, divisions: list[str], pool_max_pairs: int | None, seed: int):
    """The mined candidate pool. Not bounded for verification tractability any more — this
    is the negative set C6 will actually use, so ``pool_max_pairs`` (when given) is purely a
    mining-cost cap, not a claim about what fraction of it gets checked by hand."""
    corpus = pd.read_parquet(corpus_path)
    corpus = corpus[corpus["cpv_code"].str[:2].isin(set(divisions))]
    pairs = make_distractors(
        corpus, DegradationConfig(0.0), seed=seed, corpus="cf", max_pairs=pool_max_pairs
    )
    return corpus.set_index("record_id"), pairs


def draw_sample(pairs: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """``n`` pairs drawn uniformly at random from the pool, without replacement.

    Deterministic in ``(pairs, n, seed)``: resuming a partially-judged run against the same
    pool and seed reproduces the identical sample rather than drawing a new one, which is
    what makes "outstanding" mean anything across separate invocations of this script.
    """
    rng = np.random.default_rng(seed)
    take = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
    return pairs.iloc[np.sort(take)].reset_index(drop=True)


def render(index: int, total: int, left, right) -> None:
    print("\n" + "=" * 78)
    print(f"pair {index} of {total}")
    print("=" * 78)
    for tag, rec in (("A", left), ("B", right)):
        print(f"\n[{tag}] {rec['title']}")
        desc = (rec["description"] or "")[:220].replace("\n", " ")
        print(f"    {desc}{'...' if rec['description'] and len(rec['description']) > 220 else ''}")
        print(f"    buyer {rec['buyer_id']}   date {str(rec['release_date'])[:10]}   cpv {rec['cpv_code']}")
        print(f"    ref   {rec['tender_ref']}")
        print(f"    root  {procurement_ref(rec['tender_ref'])}   title-refs {sorted(title_refs(rec['title'])) or '-'}")
    print(f"\n    same buyer: {left['buyer_id'] == right['buyer_id']}")
    print("\n  [d] distinct   [s] same procurement   [u] unsure   [q] save and quit")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path,
                   default=data_path("processed", "corpus_b_contractsfinder.parquet"))
    p.add_argument("--divisions", nargs="+",
                   default=["30", "31", "32", "33", "38", "42", "43", "44"])
    p.add_argument("--pool-max-pairs", type=int, default=2000,
                   help="mining-cost cap on the candidate pool, not a verification bound")
    p.add_argument("--sample-size", type=int, default=50,
                   help="pairs drawn at random from the pool for hand judgement")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=annotation_path("labels", "distractor_judgements.jsonl"))
    p.add_argument("--timings", type=Path,
                   default=annotation_path("labels", "distractor_judgement_timings.jsonl"))
    p.add_argument("--summary", action="store_true", help="report progress and stop")
    args = p.parse_args(argv)

    by_id, pool = load_pool(args.corpus, args.divisions, args.pool_max_pairs, args.seed)
    sample = draw_sample(pool, args.sample_size, args.seed)
    sample_keys = set(zip(sample["left_id"], sample["right_id"], strict=True))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_judged: dict[tuple[str, str], dict] = {}
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                all_judged[(row["left_id"], row["right_id"])] = row
    # Scoped to this seed's sample: a different --seed draws a different sample, and this
    # file may hold judgements from more than one, so completion and the reported rate must
    # not be diluted or corrupted by rows outside the sample currently in play.
    judged = {k: v for k, v in all_judged.items() if k in sample_keys}

    if args.summary:
        return _summary(pool, sample, judged, args)

    outstanding = [
        (left_id, right_id)
        for left_id, right_id in zip(sample["left_id"], sample["right_id"], strict=True)
        if (left_id, right_id) not in judged
    ]
    print(f"pool {len(pool)} mined pairs, sample {len(sample)}, "
          f"{len(judged)} judged, {len(outstanding)} outstanding")
    if not outstanding:
        return _summary(pool, sample, judged, args)

    timings: list[ItemTiming] = []
    with args.out.open("a", encoding="utf-8") as sink:
        for n, (left_id, right_id) in enumerate(outstanding, 1):
            left, right = by_id.loc[left_id], by_id.loc[right_id]
            render(n, len(outstanding), left, right)

            with time_item(f"{left_id}|{right_id}", timings) as abandon:
                while True:
                    choice = input("  > ").strip().lower()
                    if choice == "q":
                        abandon()
                        print(f"\nsaved {n - 1} judgements to {args.out}")
                        write_timings(args.timings, timings)
                        return 0
                    if choice in KEYS:
                        break
                    print(f"  expected one of {sorted(KEYS)} or q")

                reason = input("  reason: ").strip()

            verdict, _ = KEYS[choice]
            row = {
                "left_id": left_id, "right_id": right_id,
                "left_title": left["title"], "right_title": right["title"],
                "verdict": verdict, "reason": reason,
                "seconds": round(timings[-1].seconds, 2),
            }
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            judged[(left_id, right_id)] = row

    write_timings(args.timings, timings)
    return _summary(pool, sample, judged, args)


def _summary(pool: pd.DataFrame, sample: pd.DataFrame, judged: dict, args) -> int:
    counts: dict[str, int] = {}
    for row in judged.values():
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1

    print(f"\npool       {len(pool)} mined pairs — used downstream unfiltered")
    print(f"sample     {len(sample)} drawn at random (seed {args.seed})")
    print(f"judged     {len(judged)}")
    for verdict in ("distinct", "same_procurement", "unsure"):
        print(f"  {verdict:<17} {counts.get(verdict, 0)}")

    if len(judged) == len(sample) and judged:
        contaminated = sum(counts.get(v, 0) for v in CONTAMINATED_VERDICTS)
        rate, lower, upper = wilson_interval(contaminated, len(judged))
        print(
            f"\ncontamination: {rate:.1%} (Wilson 95% CI: {lower:.1%}–{upper:.1%}), "
            f"n={len(judged)}"
        )
        print(
            "\nThis is a measurement of the corpus, not a filter — the pool is used "
            "downstream exactly as mined. Report this rate alongside precision; do not use "
            "it to adjust a precision figure arithmetically."
        )
    else:
        print(f"\nincomplete — {len(sample) - len(judged)} of the sample remain unjudged")

    if args.timings.exists():
        from fcesreg.timing import TooFewTimings, read_timings

        try:
            got = summarise(read_timings(args.timings))
            print(f"\nmedian {got['median_seconds']:.1f}s per judgement "
                  f"over {got['n']} items")
        except TooFewTimings as e:
            print(f"\ntiming: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
