"""The regression test for the working-directory defect (system-side counterpart to
`research/tests/test_paths.py`).

`Settings.storage_root` defaulted to a bare relative `"storage"` string once. Measured
concretely: the fitted-classifier joblib cache written from a manual run at the repo root
landed at `storage/models/classifier.joblib` there, and the identical default under pytest
(run from `system/api/`) landed at `system/api/storage/models/classifier.joblib` instead --
a silent cache miss, refitting from scratch rather than an error. This is the third time
this defect class has appeared in the project (after `fcesreg.paths` and `embed.py`'s cache
directory); a fourth turned up in `seed_categories.py`, which defined its own `_repo_root()`
anchored to `data/processed` existing rather than reusing this module's, so it failed before
`main()`'s "run `make data` first" message could ever fire on a checkout that had not run
`make data` yet.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from fcesapi.config import Settings, _repo_root


@pytest.fixture
def elsewhere(tmp_path):
    """Run the test body from a directory other than the repo root."""
    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(previous)


class TestRepoRoot:
    def test_is_absolute(self):
        assert _repo_root().is_absolute()

    def test_found_by_marker_not_by_counting_parents(self):
        assert (_repo_root() / "docker-compose.yml").exists()

    def test_stable_from_a_different_directory(self, elsewhere):
        assert (_repo_root() / "docker-compose.yml").exists()


class TestSettingsPathsAreAnchored:
    def test_storage_root_is_absolute(self):
        assert Path(Settings().storage_root).is_absolute()

    def test_storage_root_does_not_move_with_the_cwd(self, elsewhere):
        assert Path(Settings().storage_root) == _repo_root() / "storage"

    def test_storage_root_override_still_used_as_given(self, monkeypatch):
        monkeypatch.setenv("STORAGE_ROOT", "/tmp/somewhere-else")
        assert Settings().storage_root == "/tmp/somewhere-else"


def test_no_bare_relative_data_paths_in_the_package():
    """A grep-level guard so the defect cannot be reintroduced by a new module.

    Deliberately a blunt substring match, not an AST walk, mirroring the research-side
    guard in `research/tests/test_paths.py`.
    """
    offenders = []
    package_root = _repo_root() / "system" / "api" / "src" / "fcesapi"
    for source in package_root.rglob("*.py"):
        if source.name == "config.py":  # documents the anti-pattern in its own docstring
            continue
        text = source.read_text(encoding="utf-8")
        for prefix in ('Path("data/', 'Path("storage/', 'Path("results/', 'Path(".cache/'):
            if prefix in text:
                offenders.append(f"{source.relative_to(package_root)}: {prefix}")
    assert not offenders, f"bare cwd-relative paths: {offenders}"


def test_no_second_repo_root_implementation():
    """A fourth `_repo_root`-shaped function is the same defect wearing a new name."""
    offenders = []
    package_root = _repo_root() / "system" / "api" / "src" / "fcesapi"
    for source in package_root.rglob("*.py"):
        if source.name == "config.py":
            continue
        text = source.read_text(encoding="utf-8")
        if "def _repo_root" in text or "def repo_root" in text:
            offenders.append(str(source.relative_to(package_root)))
    assert not offenders, f"duplicate repo-root helpers, should import fcesapi.config._repo_root: {offenders}"
