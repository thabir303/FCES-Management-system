"""CPV label noise: disagreement rate between the published code and manual review
(PROJECT_PLAN.md line 1469, §Corpus B).

Reads the completed hand-review sample from ``annotation/labels/cpv_label_noise.jsonl``
(one row per reviewed record, ``verdict`` in ``{agree, disagree, unsure}``) and reports the
disagreement rate with a two-sided 95% Wilson interval, the same instrument the distractor
contamination rate uses. ``unsure`` counts as agreement neither way and is reported
separately -- collapsing it into either side would assert a judgement the review did not
reach.

Zero quota, CPU only.

    python research/scripts/run_label_noise.py --config research/configs/label_noise.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from fcesreg.metrics import wilson_interval
from fcesreg.paths import repo_root
from fcesreg.runs import capture_env, new_run_id, write_run

SCRIPT = "run_label_noise"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_id = new_run_id(SCRIPT, args.config)
    env = capture_env()

    path = repo_root() / cfg["sample_path"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    n = len(rows)
    n_agree = sum(1 for r in rows if r["verdict"] == "agree")
    n_disagree = sum(1 for r in rows if r["verdict"] == "disagree")
    n_unsure = sum(1 for r in rows if r["verdict"] == "unsure")
    if n_agree + n_disagree + n_unsure != n:
        raise ValueError(f"verdicts do not partition the sample: {n_agree}+{n_disagree}+{n_unsure} != {n}")

    rate, lower, upper = wilson_interval(n_disagree, n)

    print(f"n={n}  agree={n_agree}  disagree={n_disagree}  unsure={n_unsure}")
    print(f"disagreement rate {rate:.3f}  95% CI [{lower:.3f}, {upper:.3f}]")

    metrics = {
        "n": n,
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "n_unsure": n_unsure,
        "disagreement_rate": rate,
        "disagreement_rate_ci_lower": lower,
        "disagreement_rate_ci_upper": upper,
    }
    out = write_run(run_id, params=cfg, metrics=metrics, env=env)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
