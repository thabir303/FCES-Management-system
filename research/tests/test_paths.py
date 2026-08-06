"""The regression test for the working-directory defect.

`degrade.DEFAULT_LEXICON_PATH` was `Path("data/lexicon/abbreviations.yaml")`, a bare
relative path resolved against wherever the process happened to start. From the repository
root the suite passed; from `research/` it gave 14 failures that read as broken code.

Every test here changes the working directory first. If they pass from three different
directories, the class of bug is closed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fcesreg import runs, splits
from fcesreg.degrade import DEFAULT_LEXICON_PATH, DegradationConfig, degrade_frame, load_lexicon
from fcesreg.paths import ENV_VAR, ROOT_MARKERS, RootNotFound, data_path, repo_root


@pytest.fixture(params=["root", "research", "tmp"])
def elsewhere(request, tmp_path):
    """Run the test body from three different working directories."""
    target = {
        "root": repo_root(),
        "research": repo_root() / "research",
        "tmp": tmp_path,
    }[request.param]
    previous = Path.cwd()
    os.chdir(target)
    try:
        yield target
    finally:
        os.chdir(previous)


class TestRepoRoot:
    def test_is_absolute(self):
        assert repo_root().is_absolute()

    def test_found_by_marker_not_by_counting_parents(self):
        # Counting parents is shorter and silently wrong the moment the package moves.
        assert any((repo_root() / m).exists() for m in ROOT_MARKERS)

    def test_stable_from_any_directory(self, elsewhere):
        repo_root.cache_clear()
        try:
            assert (repo_root() / "PROJECT_PLAN.md").exists()
        finally:
            repo_root.cache_clear()

    def test_env_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR, str(tmp_path))
        repo_root.cache_clear()
        try:
            assert repo_root() == tmp_path.resolve()
        finally:
            repo_root.cache_clear()

    def test_env_override_pointing_nowhere_fails_loudly(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "/no/such/directory")
        repo_root.cache_clear()
        try:
            with pytest.raises(RootNotFound, match="not a directory"):
                repo_root()
        finally:
            repo_root.cache_clear()

    def test_data_path_composes_under_the_root(self):
        assert data_path("processed", "x.parquet") == repo_root() / "data/processed/x.parquet"


class TestConstantsAreAnchored:
    def test_every_package_constant_is_absolute(self):
        for constant in (DEFAULT_LEXICON_PATH, runs.RESULTS_ROOT, splits.SPLITS_PATH):
            assert constant.is_absolute(), f"{constant} resolves against the cwd"

    def test_lexicon_loads_from_anywhere(self, elsewhere):
        # This is the exact failure: 14 tests died here when run from research/.
        assert load_lexicon()["laboratory"] == "lab"

    def test_default_lexicon_path_exists(self):
        assert DEFAULT_LEXICON_PATH.exists()


class TestFunctionsWorkFromAnywhere:
    def test_degrade_frame_from_anywhere(self, elsewhere):
        frame = pd.DataFrame(
            {
                "record_id": ["r1", "r2"],
                "title": ["Laboratory microscope 230V", "Rotary vane pump"],
                "description": ["Supply of laboratory equipment", "Supply of a pump"],
            }
        )
        out = degrade_frame(frame, DegradationConfig(0.5), seed=1)
        assert len(out) == 2

    def test_write_run_from_anywhere(self, elsewhere, tmp_path):
        config = tmp_path / "c.yaml"
        config.write_text("a: 1\n", encoding="utf-8")
        out = runs.write_run(
            runs.new_run_id("t", config), {"a": 1}, {"f1": 0.5}, root=tmp_path / "runs"
        )
        assert (out / "metrics.json").exists()

    def test_results_root_does_not_move_with_the_cwd(self, elsewhere):
        assert runs.RESULTS_ROOT == repo_root() / "results/runs"

    def test_splits_load_from_anywhere(self, elsewhere):
        if not splits.SPLITS_PATH.exists():
            pytest.skip(f"needs {splits.SPLITS_PATH} — run `make data` first")
        assert splits.load().cf_dev

    def test_seeded_output_is_identical_across_directories(self, tmp_path):
        """The strongest form: the same seed must give the same bytes from anywhere."""
        frame = pd.DataFrame(
            {
                "record_id": ["r1"],
                "title": ["Laboratory microscope 230V"],
                "description": ["Supply of laboratory equipment"],
            }
        )
        outputs = []
        previous = Path.cwd()
        try:
            for target in (repo_root(), repo_root() / "research", tmp_path):
                os.chdir(target)
                outputs.append(degrade_frame(frame, DegradationConfig(0.6), seed=3).to_json())
        finally:
            os.chdir(previous)
        assert len(set(outputs)) == 1, "output depends on the working directory"


def test_no_bare_relative_data_paths_in_the_package():
    """A grep-level guard so the defect cannot be reintroduced by a new module."""
    offenders = []
    for source in (repo_root() / "research/src/fcesreg").glob("*.py"):
        if source.name == "paths.py":  # documents the anti-pattern in its docstring
            continue
        text = source.read_text(encoding="utf-8")
        for prefix in ('Path("data/', 'Path("results/', 'Path("annotation/'):
            if prefix in text:
                offenders.append(f"{source.name}: {prefix}")
    assert not offenders, f"bare cwd-relative paths: {offenders}"


def test_numpy_is_importable():
    """Guards the fixture above, which uses numpy indirectly through degrade_frame."""
    assert np.__version__
