import inspect
import json

import numpy as np
import pytest

from metric_diff import compare
from metric_diff.backends import DEFAULT_AVERAGE
from metric_diff.cli import main
from metric_diff.inputs import normalise_metric, prepare


@pytest.mark.parametrize("y_pred,task,kind,n", [
    ([0, 1, 1, 0], "binary", "labels", 2),
    ([0.1, 0.9, 0.4, 0.6], "binary", "probs", 2),
    ([-1.0, 2.0, 0.3, 5.0], "binary", "logits", 2),
    ([0, 2, 1, 2], "multiclass", "labels", 3),
    ([[0.7, 0.2, 0.1]] * 4, "multiclass", "probs", 3),
    ([[2.0, -1.0, 0.0]] * 4, "multiclass", "logits", 3),
])
def test_prepare_infers_task_and_kind(y_pred, task, kind, n):
    y_true = [0, 1, 1, 0] if task == "binary" else [0, 1, 2, 2]
    case = prepare(y_true, y_pred)
    assert (case.task, case.pred_kind, case.num_classes) == (task, kind, n)


def test_prepare_accepts_torch_tensors():
    torch = pytest.importorskip("torch")
    case = prepare(torch.tensor([0, 1]), torch.tensor([0.2, 0.7]))
    assert case.pred_kind == "probs"


@pytest.mark.parametrize("y_true,y_pred,msg", [
    ([0, 1], [0, 1, 1], "samples"),
    ([0.5, 1], [0, 1], "integer"),
    ([0, 1, 2], [0.1, 0.5, 0.9], "2-D"),
    ([], [], "empty"),
])
def test_prepare_rejects_bad_inputs(y_true, y_pred, msg):
    with pytest.raises(ValueError, match=msg):
        prepare(y_true, y_pred)


def test_metric_aliases():
    assert normalise_metric("ROC_AUC") == "auroc"
    assert normalise_metric("f1-score") == "f1"
    with pytest.raises(ValueError):
        normalise_metric("bleu")


def test_single_backend_reports_nothing_to_compare():
    r = compare("f1", [0, 1, 1], [0, 1, 0], backends=["sklearn"])
    assert r.status == "single"
    assert "nothing to compare" in str(r)


def test_default_average_table_matches_installed_libraries():
    """Guard: if a library changes its default `average`, this table (used in explanations) is stale."""
    tm = pytest.importorskip("torchmetrics.functional.classification")
    te = pytest.importorskip("torcheval.metrics.functional")
    import sklearn.metrics as skm

    def default(fn):
        return inspect.signature(fn).parameters["average"].default

    assert default(skm.f1_score) == "binary"
    assert default(skm.roc_auc_score) == "macro"
    assert default(tm.multiclass_f1_score) == "macro"
    assert default(tm.multiclass_accuracy) == "macro"
    assert default(te.multiclass_f1_score) == "micro"
    assert default(te.multiclass_accuracy) == "micro"
    assert default(te.multiclass_auroc) == "macro"
    assert default(te.multiclass_auprc) == "macro"
    assert "'micro'" in DEFAULT_AVERAGE["torcheval"]["f1"]


def test_cli_text_and_exit_codes(tmp_path, capsys):
    pytest.importorskip("torchmetrics")
    (tmp_path / "y.csv").write_text("y\n0\n0\n0\n0\n1\n1\n2\n2\n2\n2\n")
    (tmp_path / "p.csv").write_text("0\n0\n1\n0\n1\n2\n2\n2\n0\n2\n")
    code = main(["accuracy", "--true", str(tmp_path / "y.csv"), "--pred", str(tmp_path / "p.csv")])
    out = capsys.readouterr().out
    assert code == 1
    assert "Different default averaging" in out and "confirmed" in out


def test_cli_json_multiclass_scores_npy(tmp_path, capsys):
    rng = np.random.default_rng(0)
    np.save(tmp_path / "y.npy", np.arange(12) % 3)
    p = rng.random((12, 3))
    np.save(tmp_path / "p.npy", p / p.sum(1, keepdims=True))
    code = main(["auroc", "--true", str(tmp_path / "y.npy"), "--pred", str(tmp_path / "p.npy"),
                 "--backends", "sklearn", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["task"] == "multiclass" and data["status"] == "single"


def test_cli_bad_file(capsys):
    assert main(["f1", "--true", "nope.csv", "--pred", "nope.csv"]) == 2
    assert "no such file" in capsys.readouterr().err
