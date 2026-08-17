# STATE

## RUN THIS FIRST, EVERY DAY

    .venv/bin/python research/scripts/run_dedup.py --config research/configs/dedup.yaml --corpus abtbuy --cascade

Exit **code 2** = daily quota spent = success. Re-run tomorrow. **Never clear
`.cache/llm`** — it is what makes replay free. A missed day cannot be recovered.

**Missed: 1** (2026-08-16). **605 of ~1,600 banked; 2026-08-17 already spent.**
Report missed days and the projected finish in the first line of each session.

Session handoff. Volatile — update at the end of every §11 task. Hard cap 60 lines.

## Current task

**C7 classical done.** Division macro F1 **0.759** tfidf / 0.709 embed; class **0.560** /
0.410 over 74 genuine classes, **25.8% routed to review**. TF-IDF wins 69 of 74 classes;
embeddings score exactly 0.00 on small lexically-distinctive ones. 4431 collision
propagated — P 0.359 R 0.301, 28 of 78 predictions scored correct on IT procurements.

**`labelled_at` corrects a reported number.** `30000000` truncated to class `"3000"`,
which is not a CPV class; 22.6% dev / 15.3% test publish that way — now routed to review
(macro F1 0.508→0.560, routed 10.7%→25.8%). Superseded figure kept beside it.

C6/C8 closed: Wilson floor one-sided 95%; amendment 8 (**Corpus B carries no precision or
F1**, 42.0% contamination). Cascade sev 0.0 band 6.6% R 0.553 P 0.983; sev 0.15 band
14.7% R 0.417 **P combined 0.851**. Both precisions per severity — the threshold binds
only what it auto-accepts, never the adjudicator.

## Blocked or waiting on the supervisor

- **Shortlist recall gates 2 quota days.** Embedding retrieval @12 = **0.682** on the
  74-code supported pool, 0.260 on the full 1,209-class taxonomy. TF-IDF retrieval is
  *worse* (0.432) — cold matching needs semantics, learned matching needs exact tokens.
  Proposal sent: send all 74 codes, no shortlist (ceiling 1.000, 3.6 days at n=500).
- **Handling time needs the author**: `annotation/annotate.py --timing-only 15`, ~8 min.
  Label noise is done (13.2%, CI 5.8%–27.3%, n=38). A model must not time its own
  reading — that measures LLM latency, and RQ3 derives residual hours from it.
- **Open:** separator-blindness in `normalise_key`; page budget in no amendment.

## Next three tasks in §11 order

1. **System, stopping at the ruled line** — schema incl. six review-queue logging fields,
   asset CRUD + list/detail, bulk import wizard, review queue. Nothing past it before the
   report ships (QR, floor plan, reminders, role UI, audit browser are 14 Sep).
2. **`run_transfer.py`** — recall + pair completeness only (amendment 8).
3. **Amendment 9** — §10 omits the Corpus A sweep and still puts the cascade in the cf
   sweep; the paper now says Corpus A only. Plus the naive-floor write-up.

## Gotchas that are not plan amendments

- Quota: 200k/day binds before 1000 requests/day → **~310 adjudications/day** at 638 tok
  each; RQ2 is 784 tok/record at k=12, ~1,421 with all 74 codes. Cascade sev 0.5
  subsampled to **m=800** stratified by label — full band costs 6.1 days against 2.6.
- TeX at `/Library/TeX/texbin`, not on PATH; `make paper` prepends it. `make experiments`
  names unbuilt runners (build order); **`make data` is a real break** —
  `build_taxonomy.py` does not exist.
- Blocking figures in `results/runs/` are **severity-free**; `F1_severity.pdf` has **no
  Corpus A panel**, as specified. Corpus B pair completeness at `t=8` is far below the
  0.98 floor and non-monotonic — findings, neither refitted.
- Background, dead ends, rejected alternatives: `.claude/docs/session-knowledge.md`.

## Last verified

**2026-08-17** — `make test`: 460 + 4 skipped (research), 28 (annotation), identical from
the repository root, `research/` and `/tmp`. Both runners re-run clean after `18cccb9`.
