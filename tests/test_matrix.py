"""Conformance matrix and drift detection."""

import json

import pytest

pytest.importorskip("torchmetrics")

from metric_diff import MatrixResult, run_matrix  # noqa: E402
from metric_diff.cases import CASES, select  # noqa: E402
from metric_diff.cli import main  # noqa: E402

LIBS = ["sklearn", "torchmetrics", "torcheval"]
SUBSET = ["bin-auroc-large-logits", "mc-accuracy-default", "reg-r2-constant", "seg-dice-default",
          "reg-mse-clean"]


@pytest.fixture(scope="module")
def result():
    return run_matrix(backends=LIBS, cases=SUBSET)


def row(result, case_id):
    return next(r for r in result.rows if r.id == case_id)


def test_case_ids_are_unique():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))


def test_select_by_substring():
    assert {c.id for c in select(["seg-"])} == {c.id for c in CASES if c.id.startswith("seg-")}
    with pytest.raises(ValueError):
        select(["nothing-matches-this"])


def test_cells_and_causes(result):
    r = row(result, "bin-auroc-large-logits")
    assert r.cells["torchmetrics"].status == "differs"
    assert r.cells["torcheval"].status == "agree"
    assert r.causes == ["saturation"] and r.baseline_library == "sklearn"


def test_task_coverage(result):
    # torcheval has no segmentation metrics: it isn't a column for that case at all
    assert "torcheval" not in row(result, "seg-dice-default").cells
    assert row(result, "reg-mse-clean").causes == []


def test_text_and_markdown(result):
    text = result.to_text()
    assert "bin-auroc-large-logits" in text and "0.5000 x" in text
    md = result.to_markdown()
    assert md.count("\n|") >= len(SUBSET) + 1 and ":x:" in md


def test_save_load_roundtrip(result, tmp_path):
    path = tmp_path / "m.json"
    result.save(path)
    loaded = MatrixResult.load(path)
    assert [r.id for r in loaded.rows] == [r.id for r in result.rows]
    assert not result.diff(loaded).found


def test_drift_detected(result, tmp_path):
    data = result.to_dict()
    cell = data["cases"]["bin-auroc-large-logits"]["cells"]["torchmetrics"]
    cell["value"] = 0.9286  # pretend an older torchmetrics got it right
    data["versions"]["torchmetrics"] = "1.8.0"
    data["cases"]["reg-r2-constant"]["cells"]["torcheval"].update(status="error", value=None, error="boom")
    drift = result.diff(MatrixResult.from_dict(data))
    assert drift.found
    kinds = {(c.case, c.library): c.kind for c in drift.changes}
    assert kinds[("bin-auroc-large-logits", "torchmetrics")] == "value changed"
    assert kinds[("reg-r2-constant", "torcheval")] == "support changed" or kinds[("reg-r2-constant", "torcheval")] == "now works"
    assert drift.version_changes["torchmetrics"] == ("1.8.0", result.versions["torchmetrics"])
    assert "DRIFT" in drift.to_text()


def test_new_and_removed_cases(result):
    data = result.to_dict()
    data["cases"]["old-case"] = data["cases"].pop("reg-mse-clean")
    drift = result.diff(MatrixResult.from_dict(data))
    assert drift.new_cases == ["reg-mse-clean"] and drift.removed_cases == ["old-case"]


def test_cli_matrix_save_and_baseline(tmp_path, capsys):
    base = tmp_path / "base.json"
    args = ["matrix", "--backends", "sklearn,torchmetrics", "--case", "reg-mse-clean"]
    assert main(args + ["--save", str(base)]) == 0
    assert main(args + ["--baseline", str(base)]) == 0
    assert "No drift" in capsys.readouterr().out

    data = json.loads(base.read_text())
    data["cases"]["reg-mse-clean"]["cells"]["torchmetrics"]["value"] = 0.5
    base.write_text(json.dumps(data))
    assert main(args + ["--baseline", str(base)]) == 1
    assert "DRIFT" in capsys.readouterr().out


def test_cli_matrix_markdown_output_and_list(tmp_path, capsys):
    out = tmp_path / "matrix.md"
    assert main(["matrix", "--backends", "sklearn,torchmetrics", "--case", "mc-accuracy",
                 "--format", "markdown", "--output", str(out)]) == 0
    assert out.read_text().startswith("**metric-diff conformance matrix**")
    assert main(["matrix", "--list"]) == 0
    assert "bin-auroc-large-logits" in capsys.readouterr().out


def test_cli_matrix_bad_baseline(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": 99}')
    assert main(["matrix", "--case", "reg-mse", "--baseline", str(bad)]) == 2


def test_drift_markdown(result):
    data = result.to_dict()
    data["cases"]["bin-auroc-large-logits"]["cells"]["torchmetrics"]["value"] = 0.9286
    data["versions"]["torchmetrics"] = "1.8.0"
    md = result.diff(MatrixResult.from_dict(data)).to_markdown()
    assert md.startswith("### :x: Metric drift: 1 result(s) changed")
    assert "| `bin-auroc-large-logits` | torchmetrics | 0.9286 | 0.5000 | value changed |" in md
    assert "torchmetrics `1.8.0` →" in md
    assert result.diff(result).to_markdown().startswith("### :white_check_mark: No metric drift")


def test_baseline_from_another_os(result):
    assert result.system  # recorded on every run
    data = result.to_dict()
    data["system"] = "Plan9"
    drift = result.diff(MatrixResult.from_dict(data))
    assert drift.system_change == ("Plan9", result.system) and not drift.found
    assert "one baseline per OS" in drift.to_text() and "one baseline per OS" in drift.to_markdown()
    # baselines saved before the OS was recorded load fine and give no note
    del data["system"]
    old = MatrixResult.from_dict(data)
    assert old.system == "" and result.diff(old).system_change is None


def test_cli_matrix_summary(tmp_path):
    base, summary = tmp_path / "base.json", tmp_path / "summary.md"
    args = ["matrix", "--backends", "sklearn,torchmetrics", "--case", "reg-mse-clean", "--summary", str(summary)]
    assert main(args + ["--save", str(base)]) == 0
    assert summary.read_text(encoding="utf-8").startswith("**metric-diff conformance matrix**")
    summary.write_text("existing\n", encoding="utf-8")
    assert main(args + ["--baseline", str(base)]) == 0
    text = summary.read_text(encoding="utf-8")
    assert text.startswith("existing\n### :white_check_mark: No metric drift")  # appended, not replaced
    assert "<details><summary>Full conformance matrix</summary>" in text
