"""Sanity checks for the GitHub Action definition (action.yml)."""

import pathlib
import re

import pytest

yaml = pytest.importorskip("yaml")

ACTION = yaml.safe_load((pathlib.Path(__file__).parents[1] / "action.yml").read_text(encoding="utf-8"))
STEPS = ACTION["runs"]["steps"]


def test_composite_action_shape():
    assert ACTION["runs"]["using"] == "composite"
    assert set(ACTION["outputs"]) == {"drift", "baseline-created"}
    assert all(s.get("shell") == "bash" for s in STEPS if "run" in s)


def test_inputs_reach_scripts_only_through_env():
    # ${{ inputs.* }} inside a run: block would be pasted into the script (script injection)
    for s in STEPS:
        assert "${{" not in s.get("run", ""), s["name"]


def test_every_input_is_used_and_every_reference_exists():
    text = yaml.safe_dump(STEPS)
    used = set(re.findall(r"inputs\.([\w-]+)", text))
    assert used == set(ACTION["inputs"])


def test_descriptions_have_no_expressions():
    for name, spec in ACTION["inputs"].items():
        assert "${{" not in spec["description"], name


def test_run_step_survives_bash_e():
    # GitHub runs `shell: bash` with -e: without `set +e` a drift exit code (1) would end the
    # script before it reads fail-on-drift and writes the outputs.
    run = next(s for s in STEPS if s.get("id") == "run")["run"]
    assert run.lstrip().startswith("set +e")
