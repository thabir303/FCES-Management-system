"""A3 acceptance: run ids are stable for the same config, git_dirty reflects reality."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from fcesreg.runs import (
    capture_env,
    config_digest,
    git_info,
    load_run,
    new_run_id,
    write_run,
)


@pytest.fixture
def config_a(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("severity: 0.3\nseeds: [0, 1, 2]\n", encoding="utf-8")
    return p


@pytest.fixture
def config_b(tmp_path):
    p = tmp_path / "b.yaml"
    p.write_text("severity: 0.4\nseeds: [0, 1, 2]\n", encoding="utf-8")
    return p


class TestRunId:
    def test_config_digest_is_stable_across_calls(self, config_a):
        assert config_digest(config_a) == config_digest(config_a)

    def test_config_digest_depends_only_on_bytes(self, tmp_path, config_a):
        copy = tmp_path / "elsewhere.yaml"
        copy.write_text(config_a.read_text(encoding="utf-8"), encoding="utf-8")
        assert config_digest(copy) == config_digest(config_a)

    def test_same_config_gives_same_final_component(self, config_a):
        # The timestamp makes each invocation distinct; the config digest is the part
        # that must be stable, and it is what makes two runs comparable at a glance.
        first = new_run_id("run_dedup", config_a)
        second = new_run_id("run_dedup", config_a)
        assert first.rsplit("-", 1)[-1] == second.rsplit("-", 1)[-1]

    def test_different_config_gives_different_final_component(self, config_a, config_b):
        a = new_run_id("run_dedup", config_a).rsplit("-", 1)[-1]
        b = new_run_id("run_dedup", config_b).rsplit("-", 1)[-1]
        assert a != b

    def test_shape(self, config_a):
        parts = new_run_id("run_dedup", config_a).split("-")
        assert parts[0] == "run_dedup"
        assert len(parts[1]) == len("20260801T120000")
        assert "T" in parts[1]
        assert len(parts[-1]) == 8


class TestCaptureEnv:
    def test_git_dirty_reflects_reality(self):
        env = capture_env()
        assert env["git_dirty"] is git_info().dirty
        assert isinstance(env["git_dirty"], bool)

    def test_records_what_is_needed_to_reconstruct_a_number(self):
        env = capture_env(model_ids={"embed": "BAAI/bge-small-en-v1.5"}, seeds={"rng": 0})
        for key in (
            "git_sha",
            "git_dirty",
            "python_version",
            "packages",
            "model_ids",
            "seeds",
            "hostname",
            "platform",
        ):
            assert key in env
        assert env["packages"]["pandas"] == pd.__version__
        assert env["model_ids"]["embed"] == "BAAI/bge-small-en-v1.5"
        assert env["seeds"]["rng"] == 0

    def test_unknown_package_is_marked_not_installed_not_omitted(self):
        # Silence about a missing package would be worse than the fact itself.
        assert set(capture_env()["packages"]) >= {"numpy", "pandas", "torch"}


class TestWriteRun:
    def test_writes_the_four_artefacts_and_no_ledger(self, tmp_path, config_a):
        run_id = new_run_id("run_dedup", config_a)
        out = write_run(
            run_id,
            params={"severity": 0.3},
            metrics={"f1": 0.61},
            predictions=pd.DataFrame({"left_id": ["A:1"], "score": [0.9]}),
            root=tmp_path,
        )

        assert (out / "params.yaml").exists()
        assert (out / "metrics.json").exists()
        assert (out / "env.json").exists()
        assert (out / "predictions.parquet").exists()
        # Spend goes to the single global results/ledger.jsonl, keyed by run_id.
        assert not (out / "ledger.jsonl").exists()

        assert yaml.safe_load((out / "params.yaml").read_text())["severity"] == 0.3
        assert json.loads((out / "metrics.json").read_text())["f1"] == 0.61

    def test_predictions_are_optional(self, tmp_path, config_a):
        out = write_run(new_run_id("x", config_a), {}, {}, root=tmp_path)
        assert not (out / "predictions.parquet").exists()

    def test_round_trips(self, tmp_path, config_a):
        run_id = new_run_id("run_classify", config_a)
        write_run(run_id, {"level": "class"}, {"macro_f1": 0.42}, root=tmp_path)
        got = load_run(run_id, root=tmp_path)
        assert got["params"]["level"] == "class"
        assert got["metrics"]["macro_f1"] == 0.42
        assert got["predictions_path"] is None

    def test_missing_run_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_run("does-not-exist", root=tmp_path)
