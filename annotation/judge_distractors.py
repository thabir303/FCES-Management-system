"""Judge every mined distractor by hand (C4).

Filtering by reference and by buyer removes what it can see. It cannot see everything,
because Contracts Finder carries no reliable process identifier: the published reference
often names the issuing organisation rather than the procurement, and ``buyer_id`` failed
to resolve to one body in nine of fourteen contaminated pairs at the last audit. No stack
of heuristics closes that cleanly, and each rule buys less than the last while being
harder to justify.

So the surviving set is bounded at a size that admits complete verification, and every
pair in it is judged. This tool presents the evidence; the judgements are the supervisor's.

Resumable: judgements append as they are made, and a re-run skips what is already judged.
Every judgement carries a reason, as the two audits did.

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
    "s": ("same_procurement", "one procurement published twice — drop"),
    "u": ("unsure", "cannot tell from the evidence shown — drop, and say why"),
}


def load_pairs(corpus_path: Path, divisions: list[str], max_pairs: int, seed: int):
    corpus = pd.read_parquet(corpus_path)
    corpus = corpus[corpus["cpv_code"].str[:2].isin(set(divisions))]
    pairs = make_distractors(
        corpus, DegradationConfig(0.0), seed=seed, corpus="cf", max_pairs=max_pairs
    )
    return corpus.set_index("record_id"), pairs


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
    p.add_argument("--max-pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=annotation_path("labels", "distractor_judgements.jsonl"))
    p.add_argument("--timings", type=Path,
                   default=annotation_path("labels", "distractor_judgement_timings.jsonl"))
    p.add_argument("--summary", action="store_true", help="report progress and stop")
    args = p.parse_args(argv)

    by_id, pairs = load_pairs(args.corpus, args.divisions, args.max_pairs, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    judged: dict[tuple[str, str], dict] = {}
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                judged[(row["left_id"], row["right_id"])] = row

    if args.summary:
        return _summary(pairs, judged, args)

    outstanding = [
        (l, r)
        for l, r in zip(pairs["left_id"], pairs["right_id"], strict=True)
        if (l, r) not in judged
    ]
    print(f"{len(pairs)} mined pairs, {len(judged)} judged, {len(outstanding)} outstanding")
    if not outstanding:
        return _summary(pairs, judged, args)

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
            sink.write(json.dumps({
                "left_id": left_id, "right_id": right_id,
                "left_title": left["title"], "right_title": right["title"],
                "verdict": verdict, "reason": reason,
                "seconds": round(timings[-1].seconds, 2),
            }) + "\n")
            sink.flush()
            judged[(left_id, right_id)] = {"verdict": verdict}

    write_timings(args.timings, timings)
    return _summary(pairs, judged, args)


def _summary(pairs: pd.DataFrame, judged: dict, args) -> int:
    counts: dict[str, int] = {}
    for row in judged.values():
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1

    print(f"\nmined      {len(pairs)}")
    print(f"judged     {len(judged)}")
    for verdict in ("distinct", "same_procurement", "unsure"):
        print(f"  {verdict:<17} {counts.get(verdict, 0)}")

    if len(judged) == len(pairs) and judged:
        contaminated = counts.get("same_procurement", 0) + counts.get("unsure", 0)
        print(f"\ncontamination before verification: {contaminated / len(judged):.1%}")
        print(f"after verification:                0.0% by construction "
              f"({counts.get('distinct', 0)} pairs retained)")
    else:
        print("\nincomplete — the set is not usable until every pair is judged")

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
