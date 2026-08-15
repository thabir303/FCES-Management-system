"""CPV label-noise annotation, timed (§6.15, §13.3).

One exercise producing two results, because they come from the same reading:

* the **label-noise rate** — how often the published CPV code fails to describe the
  procurement — which bounds what any classifier trained or evaluated against those codes
  can be said to have achieved;
* the **mean handling time** per judgement, which RQ3 converts into residual hours. There
  is no model substitute for this one: it measures how long a *person* takes.

**Judged by the author, never by a language model, and this matters more here than it did
for the distractor sample.** RQ2 measures how accurately a language model assigns CPV
codes. If a language model also decided which published codes were wrong, then a code the
model would have assigned differently would count as noise *and* as classifier error at
once, and the two quantities would stop being independent. The distractor adjudication was
defensible because the adjudicating model sat outside the comparison it fed; here it would
sit inside it. Nothing in this file consults a model, and nothing pre-fills a judgement.

**The question is whether the published code fairly describes the procurement, not whether
it is the best available code.** The stricter reading would count every defensible-but-
improvable code as noise and inflate the estimate. A code someone could reasonably have
chosen is agreement, even where another would fit better.

The estimate is taken at the level the code is *published* at, which is finer than either
level RQ2 evaluates. A leaf code can be wrong while the class and division it rolls up to
are right, so this rate is an **upper bound** on the noise affecting the reported results —
stated as such in the summary rather than silently equated with them.

    python annotation/annotate.py
    python annotation/annotate.py --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fcesreg.metrics import wilson_interval
from fcesreg.paths import annotation_path, data_path
from fcesreg.schema import present_text
from fcesreg.splits import load as load_splits
from fcesreg.timing import (
    IDLE_CUTOFF_S,
    ItemTiming,
    TooFewTimings,
    read_timings,
    summarise,
    time_item,
    write_timings,
)

KEYS = {
    "a": ("agree", "the published code fairly describes this procurement"),
    "d": ("disagree", "the published code does not describe this procurement"),
    "u": ("unsure", "cannot tell from the notice text"),
}

#: Counted as noise. ``unsure`` is reported separately and is **not** folded in — unlike the
#: distractor tool, where an unresolved pair still contaminated the negative set. Here an
#: item the annotator cannot judge is evidence about the notice text, not about the code.
NOISE_VERDICTS = ("disagree",)


def load_sample(
    corpus_path: Path, taxonomy_path: Path, divisions: list[str], n: int, seed: int
) -> pd.DataFrame:
    """``n`` records drawn from the **test** partition, stratified by CPV division.

    Allocation is proportional to each division's share of the partition, by largest
    remainder. Proportional rather than equal, so the unweighted rate over the sample
    estimates the corpus rate directly; equal allocation would over-represent small
    divisions and need reweighting to mean anything about the corpus.
    """
    corpus = pd.read_parquet(corpus_path)
    corpus = corpus[corpus["cpv_code"].str[:2].isin(set(divisions))]
    test_ids = load_splits().cf_test
    corpus = corpus[corpus["record_id"].isin(test_ids)].reset_index(drop=True)
    if corpus.empty:
        raise SystemExit("no test-partition records in the selected divisions")

    corpus = corpus.assign(_division=corpus["cpv_code"].str[:2])
    shares = corpus["_division"].value_counts().sort_index()
    exact = shares / shares.sum() * n
    take = np.floor(exact).astype(int)
    # Largest remainder, deterministic: ties broken by division code, not by hash order.
    remainder = (exact - take).sort_values(ascending=False, kind="stable")
    for division in remainder.index[: n - int(take.sum())]:
        take[division] += 1

    rng = np.random.default_rng(seed)
    picked = []
    for division, k in take.items():
        if k <= 0:
            continue
        block = corpus[corpus["_division"] == division]
        idx = rng.choice(len(block), size=min(int(k), len(block)), replace=False)
        picked.append(block.iloc[np.sort(idx)])

    sample = pd.concat(picked).reset_index(drop=True)
    # One deterministic shuffle so the annotator does not judge one division in a run and
    # drift into a per-division habit.
    order = rng.permutation(len(sample))
    sample = sample.iloc[order].reset_index(drop=True)

    taxonomy = pd.read_parquet(taxonomy_path)
    lookup = dict(zip(taxonomy["cpv_code"], taxonomy["cpv_description"], strict=True))
    return sample.assign(
        _division_desc=sample["cpv_code"].str[:2].map(lookup),
        _class_desc=sample["cpv_code"].str[:4].map(lookup),
    )


def render(index: int, total: int, record: pd.Series) -> None:
    print("\n" + "=" * 78)
    print(f"item {index} of {total}")
    print("=" * 78)
    print(f"\n  {record['title']}\n")
    description = (present_text(record["description"]) or "").replace("\n", " ")
    print(f"  {description[:600]}{'...' if len(description) > 600 else ''}\n")

    code = record["cpv_code"]
    division = present_text(record["_division_desc"])
    # A code published at division level has no meaningful four-digit class, and saying so
    # is part of the evidence: the annotator is judging a coarser code than usual, not
    # looking at a lookup failure.
    klass = present_text(record["_class_desc"]) or "(published at division level — no class)"
    print(f"  published CPV : {code}")
    print(f"    division {code[:2]}  {division or '-'}")
    print(f"    class    {code[:4]}  {klass}")
    print("\n  Does the published code FAIRLY DESCRIBE this procurement?")
    print("  (not: is it the best possible code — a defensible code is agreement)")
    print("\n  [a] agree   [d] disagree   [u] unsure   [q] save and quit")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path,
                   default=data_path("processed", "corpus_b_contractsfinder.parquet"))
    p.add_argument("--taxonomy", type=Path,
                   default=data_path("processed", "cpv_taxonomy.parquet"))
    p.add_argument("--divisions", nargs="+",
                   default=["30", "31", "32", "33", "38", "42", "43", "44"])
    p.add_argument("--n", type=int, default=40,
                   help="items to judge; 40 gives a usable mean handling time")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=annotation_path("labels", "cpv_label_noise.jsonl"))
    p.add_argument("--timings", type=Path,
                   default=annotation_path("labels", "cpv_label_noise_timings.jsonl"))
    p.add_argument("--summary", action="store_true", help="report and stop")
    args = p.parse_args(argv)

    sample = load_sample(args.corpus, args.taxonomy, args.divisions, args.n, args.seed)
    sample_ids = set(sample["record_id"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_judged: dict[str, dict] = {}
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                all_judged[row["record_id"]] = row
    # Scoped to this seed's sample, for the reason judge_distractors learned: a judgements
    # file accumulates across every seed ever run against it, and a later draw must not
    # have its rate diluted by rows belonging to a different sample.
    judged = {k: v for k, v in all_judged.items() if k in sample_ids}

    if args.summary:
        return _summary(sample, judged, args)

    outstanding = [r for r in sample["record_id"] if r not in judged]
    print(f"sample {len(sample)} items, {len(judged)} judged, {len(outstanding)} outstanding")
    if not outstanding:
        return _summary(sample, judged, args)

    by_id = sample.set_index("record_id")
    timings: list[ItemTiming] = []
    with args.out.open("a", encoding="utf-8") as sink:
        for n, record_id in enumerate(outstanding, 1):
            record = by_id.loc[record_id]
            render(n, len(outstanding), record)

            with time_item(record_id, timings) as abandon:
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

                verdict, _ = KEYS[choice]
                # A reason is required wherever the judgement is not a plain agreement,
                # so every departure from the published code is auditable.
                reason = ""
                if verdict != "agree":
                    while not reason:
                        reason = input("  reason: ").strip()

            row = {
                "record_id": record_id,
                "title": record["title"],
                "cpv_code": record["cpv_code"],
                "verdict": verdict,
                "reason": reason,
                "seconds": round(timings[-1].seconds, 2),
                "abandoned": timings[-1].abandoned,
            }
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            judged[record_id] = row

    write_timings(args.timings, timings)
    return _summary(sample, judged, args)


def _summary(sample: pd.DataFrame, judged: dict, args) -> int:
    counts: dict[str, int] = {}
    for row in judged.values():
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1

    print(f"\nsample     {len(sample)} drawn from the test partition (seed {args.seed})")
    print(f"judged     {len(judged)}")
    for verdict in ("agree", "disagree", "unsure"):
        print(f"  {verdict:<10} {counts.get(verdict, 0)}")

    if len(judged) < len(sample):
        print(f"\nincomplete — {len(sample) - len(judged)} of the sample remain unjudged")
        return 0

    # Unsure is excluded from the denominator, not counted as agreement. Counting it either
    # way would assert something the annotator explicitly declined to assert.
    n_unsure = counts.get("unsure", 0)
    decided = len(judged) - n_unsure
    if decided:
        noise = sum(counts.get(v, 0) for v in NOISE_VERDICTS)
        rate, lower, upper = wilson_interval(noise, decided)
        print(f"\nlabel noise: {rate:.1%} (Wilson 95% CI: {lower:.1%}–{upper:.1%}), "
              f"n={decided} decided")
        print(f"unsure     : {n_unsure} of {len(judged)} "
              f"({n_unsure / len(judged):.1%}), excluded from the rate above")
        print(
            "\nMeasured at the published (8-digit) level, which is finer than either level\n"
            "RQ2 evaluates. A leaf code can be wrong where its class and division are right,\n"
            "so this is an UPPER BOUND on the noise affecting the reported results."
        )
    else:
        print("\nevery judged item was `unsure` — no rate can be reported")

    if args.timings.exists():
        try:
            got = summarise(read_timings(args.timings))
            print(f"\nhandling time: mean {got['mean_seconds']:.1f}s, "
                  f"median {got['median_seconds']:.1f}s, p90 {got['p90_seconds']:.1f}s, "
                  f"n={got['n']}")
            print(f"  excluded {got['n_abandoned']} item(s): quit without judging, or over "
                  f"the {IDLE_CUTOFF_S:.0f}s idle cutoff (an interrupted session, not a "
                  f"slow judgement)")
            print(f"\n  mean_seconds_per_item = {got['mean_seconds']:.2f}  -> operating_point")
        except TooFewTimings as e:
            print(f"\ntiming: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
