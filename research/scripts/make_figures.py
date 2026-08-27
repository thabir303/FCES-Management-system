"""Regenerate results/figures/ from results/runs/ (supervisor request, 2026-08-27).

Same two rules as ``make_tables.py``, and for the same reason:

1. **A run made against a dirty git tree is refused.** An untracked local edit must not be
   able to change a published figure silently.
2. **Every figure's caption in main.tex names the run_id it came from**, so any curve can be
   traced back to the code and environment that produced it. This script prints the run_id
   used for each figure it writes; nothing here computes a statistic that a runner did not
   already measure -- it only plots what is already in ``results/runs/``.

**Design constraints, all deliberate:**

* **Vector PDF, not raster.** ``matplotlib``'s PDF backend is vector by default; nothing here
  calls ``imshow`` or rasterises a line plot.
* **Legible in black and white.** Every figure distinguishes its series by marker shape and
  line style, never by colour alone -- this will be printed and marked on paper.
* **No invented data points.** Where a run reports no value (an undefined threshold, an
  interval no run computed), the figure shows a gap or an explicit annotation, never an
  interpolated or zero-filled substitute.

    python research/scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fcesreg.paths import results_path
from fcesreg.runs import RESULTS_ROOT, load_run

sys.path.insert(0, str(Path(__file__).parent))
from make_tables import DirtyRun, _require_clean, latest_run  # same-directory script import

FIGURES_ROOT = results_path("figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.color": "0.7",
    "pdf.fonttype": 42,  # embed real glyphs rather than a bitmap substitute
})

#: Marker/linestyle pairs cycled across series so no figure relies on colour to distinguish
#: them -- every series here is additionally black or a distinct grey, never colour-coded.
_STYLES = [
    {"marker": "o", "linestyle": "-", "color": "black"},
    {"marker": "s", "linestyle": "--", "color": "0.35"},
    {"marker": "^", "linestyle": "-.", "color": "0.55"},
    {"marker": "D", "linestyle": ":", "color": "0.15"},
]


def _load(script: str) -> dict:
    run = load_run(latest_run(script, RESULTS_ROOT), root=RESULTS_ROOT)
    _require_clean(run)
    return run


def figure_review_vs_lost(op_run: dict) -> str:
    """F1: manual review share against true duplicates lost, across operating points.

    Data: ``run_operating_point``'s ``predictions.parquet`` -- the full precision-target
    curve traced by :func:`operating_point.band_operating_point`/``delivered`` for the tfidf
    matcher at severity 0 (Corpus A). Pair-level, matching the two points the supervisor
    named by number (0.95: 93.4% automated, 84 lost; 0.99: 64.4%, 16 lost) -- the record-level
    correction applies to the two headline points quoted in prose, not to this whole curve,
    which the run this figure reads from does not carry at record granularity.
    """
    curve = pd.read_parquet(op_run["predictions_path"])
    curve = curve[(curve["matcher"] == "tfidf") & (curve["severity"] == 0.0)].sort_values(
        "target"
    )
    review_share = 1.0 - curve["automated_share"]

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(
        review_share, curve["n_duplicates_lost"], marker="o", linestyle="-", color="black",
        markersize=4,
    )
    for target, marker in ((0.95, "s"), (0.99, "D")):
        row = curve[np.isclose(curve["target"], target)].iloc[0]
        x, y = 1.0 - row["automated_share"], row["n_duplicates_lost"]
        ax.plot(x, y, marker=marker, color="black", markersize=8, markerfacecolor="white")
        ax.annotate(
            f"P$\\geq${target:.2f}\n{100 * row['automated_share']:.1f}% automated,\n"
            f"{int(y)} lost",
            (x, y), textcoords="offset points", xytext=(8, 6), fontsize=7,
        )
    ax.set_xlabel("Manual review share (pairs)")
    ax.set_ylabel("True duplicates lost (of 206)")
    ax.set_title("Review avoided vs. duplicates lost, severity 0", fontsize=9)
    fig.tight_layout()
    out = FIGURES_ROOT / "F1_review_vs_lost.pdf"
    fig.savefig(out)
    plt.close(fig)
    return op_run["run_id"]


def figure_blocking_transfer(transfer_run: dict) -> str:
    """F2: pair completeness against candidate volume, cap sweep, both corpora, log volume.

    Data: ``run_transfer_attribution``'s ``rows`` -- the same cap-sweep rows T9_transfer.tex
    is built from, so this figure and that table trace to one run_id.
    """
    rows = pd.DataFrame(transfer_run["metrics"]["rows"])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    styles = {
        ("corpus_a", 0.0): {"marker": "o", "linestyle": "-", "color": "black", "label": "A, sev 0.0"},
        ("corpus_b", 0.0): {"marker": "s", "linestyle": "--", "color": "0.4", "label": "B, sev 0.0"},
        ("corpus_a", 0.25): {"marker": "^", "linestyle": "-.", "color": "black", "label": "A, sev 0.25", "markerfacecolor": "white"},
        ("corpus_b", 0.25): {"marker": "D", "linestyle": ":", "color": "0.4", "label": "B, sev 0.25", "markerfacecolor": "white"},
    }
    for (corpus, severity), style in styles.items():
        sub = rows[(rows["corpus"] == corpus) & (rows["severity"] == severity)].sort_values(
            "n_candidates"
        )
        if sub.empty:
            continue
        ax.plot(sub["n_candidates"], sub["pair_completeness"], markersize=5, **style)
    ax.set_xscale("log")
    ax.set_xlabel("Candidate volume (log scale)")
    ax.set_ylabel("Pair completeness")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("Blocking transfer: completeness vs. cost", fontsize=9)
    fig.tight_layout()
    out = FIGURES_ROOT / "F2_blocking_transfer.pdf"
    fig.savefig(out)
    plt.close(fig)
    return transfer_run["run_id"]


def figure_precision_recall_severity(dedup_run: dict) -> str:
    """F3: precision and recall against degradation severity, Corpus A, one series per
    matcher, cascade separate.

    Data: ``run_dedup``'s ``free_matchers`` (45 rows: 3 matchers x 5 severities x 3 seeds,
    mean over seeds here) and ``cascade`` (3 severities, single repetition). A severity where
    every seed's threshold is undefined (``threshold is None``, i.e. no confident threshold
    exists at :data:`fcesreg.dedup.select_threshold`'s Wilson floor) is a gap in the
    precision line, not a plotted 0 -- precision of an empty accepted set is undefined, not
    a measured failure. Recall is still plotted at those points: a confidence rule that
    declines to accept anything really does recover nothing, and 0 is what was measured.
    """
    free = pd.DataFrame(dedup_run["metrics"]["free_matchers"])
    cascade = pd.DataFrame(dedup_run["metrics"]["cascade"])

    # Several series share a severity of 0 with near-identical precision (tfidf 0.970,
    # cascade 0.983): a small, fixed, and disclosed x-jitter keeps one marker from sitting
    # exactly under another rather than changing what either series measured.
    jitter = {"exact": -0.012, "tfidf": -0.004, "embedding": 0.004, "cascade": 0.012}

    fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(6.8, 2.7), sharex=True)
    matcher_styles = dict(zip(["exact", "tfidf", "embedding"], _STYLES))
    for matcher, style in matcher_styles.items():
        sub = free[free["matcher"] == matcher]
        by_sev = sub.groupby("severity")
        has_threshold = by_sev["threshold"].apply(lambda s: s.notna().any())
        mean_p = by_sev["precision"].mean()
        mean_r = by_sev["recall"].mean()
        mean_p = mean_p.where(has_threshold)  # gap, not 0, where no seed had a threshold
        x = mean_p.index + jitter[matcher]
        ax_p.plot(x, mean_p.values, markersize=5, label=matcher,
                  markerfacecolor="white", **style)
        ax_r.plot(x, mean_r.values, markersize=5, label=matcher,
                  markerfacecolor="white", **style)
    cascade_style = {"marker": "*", "linestyle": "-", "color": "black"}
    ax_p.plot(
        cascade["severity"] + jitter["cascade"], cascade["precision"], markersize=10,
        label="cascade", markerfacecolor="white", **cascade_style,
    )
    ax_r.plot(
        cascade["severity"] + jitter["cascade"], cascade["recall"], markersize=10,
        label="cascade", markerfacecolor="white", **cascade_style,
    )
    ax_p.set_xlabel("Severity"); ax_p.set_ylabel("Precision"); ax_p.set_ylim(-0.05, 1.05)
    ax_r.set_xlabel("Severity"); ax_r.set_ylabel("Recall"); ax_r.set_ylim(-0.05, 1.05)
    ax_p.set_title("Precision", fontsize=9); ax_r.set_title("Recall", fontsize=9)
    ax_r.legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Corpus A: precision and recall vs. severity", fontsize=9, y=1.03)
    fig.tight_layout()
    out = FIGURES_ROOT / "F3_precision_recall_severity.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return dedup_run["run_id"]


def figure_band_size_severity(dedup_run: dict) -> str:
    """F4: cascade review-band size against severity -- 6.6%, 14.7%, 100% (the three
    severities the cascade is evaluated at; see run_dedup's docstring for why only three).
    """
    cascade = pd.DataFrame(dedup_run["metrics"]["cascade"])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot(
        cascade["severity"], 100 * cascade["band_fraction"], marker="o", linestyle="-",
        color="black", markersize=6,
    )
    for _, row in cascade.iterrows():
        ax.annotate(
            f"{100 * row['band_fraction']:.1f}%", (row["severity"], 100 * row["band_fraction"]),
            textcoords="offset points", xytext=(6, 6), fontsize=7,
        )
    ax.set_xlabel("Severity")
    ax.set_ylabel("Review band (% of test pairs)")
    ax.set_ylim(0, 105)
    ax.set_title("Cascade review-band size vs. severity", fontsize=9)
    fig.tight_layout()
    out = FIGURES_ROOT / "F4_band_size_severity.pdf"
    fig.savefig(out)
    plt.close(fig)
    return dedup_run["run_id"]


def figure_taxonomy_f1(classify_run: dict, rag_run: dict) -> str:
    """F5: taxonomy macro F1 across approaches, both levels.

    Data: ``run_classify`` (TF-IDF and embedding, both levels, full test partition) and
    ``run_rag_classify`` (the exploratory language model condition, division level only,
    n=167 of a partial run -- class level was never attempted, so no bar is drawn for it).
    **No error bars**: no run computes a confidence interval on macro F1 itself -- the
    language model condition's own Wilson interval (reported in prose) is on accuracy, a
    different statistic, and inventing one for macro F1 here would not be a measurement.
    The language model's bar is hatched to flag that it alone is a partial-n sample, not a
    full-partition figure like the other five bars.
    """
    levels = classify_run["metrics"]["levels"]
    division = {
        cond: d["macro_f1"] for cond, d in levels["division"]["conditions"].items()
    }
    cls = {cond: d["macro_f1"] for cond, d in levels["class"]["conditions"].items()}
    rag_division = rag_run["metrics"]["conditions_on_sample"]["rag_fewshot_llm"]["macro_f1"]

    labels = ["TF-IDF\n(division)", "Embedding\n(division)", "LM, n=167\n(division)",
              "TF-IDF\n(class)", "Embedding\n(class)"]
    values = [division["tfidf_svm"], division["embedding_logreg"], rag_division,
              cls["tfidf_svm"], cls["embedding_logreg"]]
    hatches = ["", "", "///", "", ""]
    greys = ["0.15", "0.55", "white", "0.15", "0.55"]

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    bars = ax.bar(labels, values, color=greys, edgecolor="black", hatch=hatches)
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.3f}", (bar.get_x() + bar.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.0)
    plt.setp(ax.get_xticklabels(), fontsize=6.5)
    ax.set_title("Taxonomy macro F1 by approach and level", fontsize=9)
    fig.tight_layout()
    out = FIGURES_ROOT / "F5_taxonomy_f1.pdf"
    fig.savefig(out)
    plt.close(fig)
    return f"{classify_run['run_id']} + {rag_run['run_id']}"


def figure_cost_severity(costs_run: dict) -> str:
    """F6: cost against severity, from the ledger, deduplicated by prompt hash.

    Data: ``run_costs``'s ``band`` list -- USD and tokens per 1000 adjudications at each
    cascade severity, already deduplicated by ``prompt_sha256`` (fcesreg.costs.summarise_costs)
    so a resumed sweep's replayed rows cannot inflate the figure.
    """
    band = pd.DataFrame(costs_run["metrics"]["band"])
    fig, ax1 = plt.subplots(figsize=(3.4, 2.6))
    ax1.plot(
        band["severity"], band["usd_per_1000"], marker="o", linestyle="-", color="black",
        markersize=6, label="USD / 1000 adj.",
    )
    ax1.set_xlabel("Severity")
    ax1.set_ylabel("Notional USD per 1000 adjudications")
    ax2 = ax1.twinx()
    ax2.plot(
        band["severity"], band["tokens_per_1000"] / 1000, marker="s", linestyle="--",
        color="0.45", markersize=6, label="k tokens / 1000 adj.",
    )
    ax2.set_ylabel("k tokens per 1000 adjudications")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5, loc="upper left")
    ax1.set_title("Cascade cost vs. severity", fontsize=9)
    fig.tight_layout()
    out = FIGURES_ROOT / "F6_cost_severity.pdf"
    fig.savefig(out)
    plt.close(fig)
    return costs_run["run_id"]


def figure_pipeline_diagram() -> str:
    """F7: the pipeline as a single picture -- ingest, blocking, matching, the cascade's
    three-way accept/band/reject partition, the review queue, the register.

    Schematic, not data-driven: there is nothing in a run record to plot, only the shape of
    the system every other figure and table measures a part of. Built from matplotlib
    primitives (boxes and arrows) rather than a hand-drawn image, so it is regenerated by
    this script like every other exhibit, not maintained as a static asset.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axis("off")
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 4.6)

    def box(x, y, w, h, text, fontsize=7.5):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="black", linewidth=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                 wrap=True)

    def arrow(x0, y0, x1, y1, text=None, ty=0.0):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="black", linewidth=1),
        )
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + ty, text, ha="center", fontsize=6.5)

    box(0.2, 1.8, 1.3, 1.0, "Ingest\n(spreadsheet)")
    box(1.9, 1.8, 1.3, 1.0, "Blocking\n(candidate\npairs)")
    box(3.6, 1.8, 1.3, 1.0, "Matching\n(score)")
    box(5.3, 3.3, 1.6, 0.9, "Accept\n(above upper)")
    box(5.3, 1.85, 1.6, 0.9, "Band\n(adjudicate)")
    box(5.3, 0.4, 1.6, 0.9, "Reject\n(below lower)")
    box(8.0, 1.85, 1.4, 0.9, "Review\nqueue")
    box(9.9, 1.85, 1.1, 0.9, "Register")

    arrow(1.5, 2.3, 1.9, 2.3)
    arrow(3.2, 2.3, 3.6, 2.3)
    arrow(4.9, 2.3, 5.3, 3.75)
    arrow(4.9, 2.3, 5.3, 2.3)
    arrow(4.9, 2.3, 5.3, 0.85)
    arrow(6.9, 3.75, 9.9, 2.5, "merged\n(confident)", ty=0.3)
    arrow(6.9, 2.3, 8.0, 2.3, "uncertain")
    arrow(9.4, 2.3, 9.9, 2.3, "resolved")
    arrow(6.9, 0.85, 9.9, 2.1, "kept apart\n(confident, unmerged)", ty=-0.35)

    ax.set_title("Pipeline: ingest to register", fontsize=9)
    fig.tight_layout()
    out = FIGURES_ROOT / "F7_pipeline_diagram.pdf"
    fig.savefig(out)
    plt.close(fig)
    return "schematic (no run record; the structure every other exhibit measures a part of)"


def figure_architecture_diagram() -> str:
    """A1: appendix-only architecture diagram -- the research library, the API, the
    database, and the one file that crosses between the first two.

    Schematic, like F7, and for the same reason: nothing in a run record to plot, only the
    codebase's own module boundary (``grep -r "import fcesreg" system/`` returns exactly
    ``services/pipeline.py``, the boundary this diagram draws).
    """
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    def box(x, y, w, h, text, fontsize=8, linewidth=1, linestyle="-"):
        ax.add_patch(
            plt.Rectangle(
                (x, y), w, h, fill=False, edgecolor="black", linewidth=linewidth,
                linestyle=linestyle,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x0, y0, x1, y1, text=None, ty=0.0):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="<->", color="black", linewidth=1),
        )
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + ty, text, ha="center", fontsize=6.5)

    box(0.3, 1.3, 2.6, 1.6, "research/\nfcesreg\n\n(pure Python:\npandas/numpy,\nno web, no DB)")
    box(3.7, 1.3, 2.6, 1.6, "system/api\n(FastAPI)\n\nservices/\npipeline.py\nadapts here")
    box(7.1, 1.3, 2.4, 1.6, "Postgres\n(:5433)")

    arrow(2.9, 2.1, 3.7, 2.1, "import fcesreg\n(one file only)", ty=0.35)
    arrow(6.3, 2.1, 7.1, 2.1, "SQLAlchemy")

    ax.text(5.0, 3.35, "The only import boundary crossed anywhere in system/", fontsize=6.5,
            ha="center", style="italic")
    ax.set_title("Architecture: research library, API, database", fontsize=9)
    fig.tight_layout()
    out = FIGURES_ROOT / "A1_architecture.pdf"
    fig.savefig(out)
    plt.close(fig)
    return "schematic (no run record; the module boundary grep -r verifies)"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--allow-dirty", action="store_true",
        help="build anyway; the resulting figures are not reproducible and must not ship",
    )
    args = p.parse_args(argv)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        op_run = _load("run_operating_point")
        transfer_run = _load("run_transfer_attribution")
        dedup_run = _load("run_dedup")
        classify_run = _load("run_classify")
        rag_run = _load("run_rag_classify")
        costs_run = _load("run_costs")
    except DirtyRun as e:
        if not args.allow_dirty:
            print(f"\nREFUSED: {e}\n", file=sys.stderr)
            return 2
        print(f"WARNING: {e}", file=sys.stderr)

    builds = [
        ("F1_review_vs_lost.pdf", figure_review_vs_lost(op_run)),
        ("F2_blocking_transfer.pdf", figure_blocking_transfer(transfer_run)),
        ("F3_precision_recall_severity.pdf", figure_precision_recall_severity(dedup_run)),
        ("F4_band_size_severity.pdf", figure_band_size_severity(dedup_run)),
        ("F5_taxonomy_f1.pdf", figure_taxonomy_f1(classify_run, rag_run)),
        ("F6_cost_severity.pdf", figure_cost_severity(costs_run)),
        ("F7_pipeline_diagram.pdf", figure_pipeline_diagram()),
        ("A1_architecture.pdf", figure_architecture_diagram()),
    ]
    for filename, source in builds:
        print(f"  {FIGURES_ROOT / filename}  <- {source}")
    print(f"built {len(builds)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
