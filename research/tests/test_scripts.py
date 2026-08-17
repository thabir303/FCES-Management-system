"""Guards on the runner scripts themselves (§10).

Runners are entrypoints, not library code, so they carry a small amount of bootstrapping
that `fcesreg` deliberately refuses to do. Where that bootstrapping is forgotten the failure
is late and confusing — a traceback from deep inside the transport saying a key is unset,
after the expensive setup has already run. These are grep-level guards of the same kind as
`test_paths.py`'s: cheap, and they catch the omission at the point it is made.
"""

from __future__ import annotations

import re

import pytest

from fcesreg.paths import repo_root

SCRIPTS = sorted((repo_root() / "research" / "scripts").glob("run_*.py"))


def test_there_are_runners_to_check():
    # Guards the guard: a glob that silently matches nothing would pass everything below.
    assert SCRIPTS, "no runners found — has research/scripts moved?"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_a_runner_reaching_the_endpoint_loads_dotenv(script):
    """Any runner that can construct an LLMClient must bootstrap `.env` first.

    `llm.py` reads only ``os.environ``: keeping `.env` mechanics out of the library is
    deliberate, so the obligation lands here. This has been forgotten twice — once in
    `run_llm_pilot.py` and once in `run_dedup.py` — and both times it surfaced only when a
    live run died on its first call.
    """
    source = script.read_text(encoding="utf-8")
    if "LLMClient" not in source:
        pytest.skip(f"{script.name} does not reach the endpoint")
    assert "load_dotenv" in source, (
        f"{script.name} constructs an LLMClient but never calls load_dotenv; it will fail "
        f"with GROQ_API_KEY unset however the key is configured in .env"
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_a_runner_takes_a_config_not_tuning_flags(script):
    """Runners are config-driven (§10). A flag that sets a method parameter is a tuning
    knob, and a result produced with one is not reproducible from the committed config."""
    source = script.read_text(encoding="utf-8")
    added = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', source))
    forbidden = {
        "--threshold", "--precision-target", "--lower", "--upper", "--severity",
        "--seed", "--min-overlap", "--max-adjudications", "--model",
    }
    assert not (added & forbidden), (
        f"{script.name} exposes {sorted(added & forbidden)} as a flag; these belong in the "
        f"committed config so a run's parameters are recoverable from the repository"
    )


class TestScoredSetSelection:
    """`run_classify.restrict` decides what a classification figure is measured over.

    It is runner code, but it is the difference between macro F1 0.560 and 0.508, so it
    gets a real test rather than a grep. A division-only code truncates to a four-digit
    string that is not a CPV class, and scoring it as one credits the model for
    reproducing the absence of a label.
    """

    @pytest.fixture
    def restrict(self):
        import sys

        sys.path.insert(0, str(repo_root() / "research" / "scripts"))
        from run_classify import restrict

        return restrict

    def frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "cpv_code": ["30200000", "30000000", "44316400", "99999999"],
                "title": ["laptops", "office stuff", "hardware", "other"],
            }
        )

    def test_a_division_only_code_is_not_scored_at_class_level(self, restrict):
        got = restrict(self.frame(), "class", {"3020", "3000", "4431"})
        assert sorted(got["cpv_code"]) == ["30200000", "44316400"]

    def test_it_is_still_scored_at_division_level(self, restrict):
        got = restrict(self.frame(), "division", {"30", "44"})
        assert "30000000" in set(got["cpv_code"])

    def test_the_superseded_measurement_is_still_reachable(self, restrict):
        # Kept so the figure already reported can be reproduced, not deleted.
        got = restrict(self.frame(), "class", {"3020", "3000", "4431"}, genuine_only=False)
        assert "30000000" in set(got["cpv_code"])

    def test_an_unsupported_label_is_excluded_either_way(self, restrict):
        for genuine_only in (True, False):
            got = restrict(self.frame(), "class", {"3020"}, genuine_only)
            assert "99999999" not in set(got["cpv_code"])
