"""Regression cases taken from real upstream issues and from probing the libraries.

Each test asserts (1) the libraries really disagree with default settings and
(2) metric-diff names and *confirms* the right cause. If an upstream library fixes
its behaviour, the first assertion will fail and tell us to update the case.
"""

import numpy as np
import pytest

pytest.importorskip("torchmetrics")
pytest.importorskip("torcheval")

from metric_diff import compare as _compare  # noqa: E402

# These cases pin the three torch/sklearn libraries; Keras has its own tests.
TORCH_AND_SKLEARN = ["sklearn", "torchmetrics", "torcheval"]


def compare(*args, **kwargs):
    return _compare(*args, backends=TORCH_AND_SKLEARN, **kwargs)


def keys(report):
    return [c.key for c in report.causes]


def test_torchmetrics_2819_large_logits_auroc():
    # https://github.com/Lightning-AI/torchmetrics/issues/2819 (still open)
    preds = [98.0950, 98.4612, 98.1145, 98.1506, 97.6037, 98.9425, 99.2644, 99.5014,
             99.7280, 99.6595, 99.6931, 99.4667, 99.9623, 99.8949, 99.8768]
    labels = [0] + [1] * 14
    r = compare("auroc", labels, preds)
    assert r.results["torchmetrics"].value == pytest.approx(0.5)
    assert r.results["sklearn"].value == pytest.approx(0.9285714, abs=1e-6)
    assert r.confirmed and keys(r) == ["saturation"]
    assert r.aligned["torchmetrics"].value == pytest.approx(0.9285714, abs=1e-5)


def test_multiclass_accuracy_default_average():
    # torchmetrics defaults to macro (= balanced accuracy); sklearn and torcheval to micro
    r = compare("accuracy", [0, 0, 0, 0, 1, 1, 2, 2, 2, 2], [0, 0, 1, 0, 1, 2, 2, 2, 0, 2])
    assert r.results["sklearn"].value == pytest.approx(0.7)
    assert r.results["torchmetrics"].value == pytest.approx(2 / 3, abs=1e-6)
    assert r.confirmed and keys(r) == ["average"]
    assert "balanced_accuracy_score" in r.causes[0].fix


def test_multiclass_f1_sklearn_raises_and_defaults_differ():
    # torchmetrics#746 / torcheval#191 family: "same value for everything" / default averaging
    y_true = [0, 1, 2, 2, 1, 0, 2, 2, 1, 1]
    y_pred = [0, 2, 2, 2, 1, 0, 1, 2, 1, 0]
    r = compare("f1", y_true, y_pred)
    assert not r.results["sklearn"].ok and "average" in r.results["sklearn"].error
    assert r.confirmed and keys(r) == ["average"]


def test_multiclass_auroc_logits_needs_softmax_and_average():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(10, 3))
    r = compare("auroc", [0, 0, 0, 0, 1, 1, 2, 2, 2, 2], logits)
    assert r.confirmed and keys(r) == ["softmax", "average"]


def test_binary_logits_thresholded_differently():
    # torchmetrics auto-sigmoids logits, torcheval thresholds raw values at 0.5
    y_true = [0, 1, 1, 0, 1, 1]
    logits = [-2.0, 3.0, -0.5, -3.0, 4.0, 0.3]
    r = compare("f1", y_true, logits)
    assert r.results["torchmetrics"].value == pytest.approx(6 / 7, abs=1e-6)
    assert r.results["torcheval"].value == pytest.approx(2 / 3, abs=1e-6)
    assert r.confirmed and keys(r) == ["logits"]


def test_scores_exactly_on_threshold():
    # torchmetrics uses > 0.5, torcheval uses >= 0.5
    r = compare("precision", [1, 0, 0, 1], [0.5, 0.5, 0.2, 0.8])
    assert r.results["torchmetrics"].value == pytest.approx(1.0)
    assert r.results["torcheval"].value == pytest.approx(2 / 3, abs=1e-6)
    assert r.confirmed and keys(r) == ["threshold_tie"]


def test_labels_one_two_torcheval_f1_above_one():
    # huggingface/evaluate#869 family: binary labels that are not {0, 1}
    r = compare("f1", [1, 2, 1, 1], [1, 2, 2, 1])
    assert r.results["torcheval"].value > 1.0  # impossible F1, silently returned
    assert not r.results["torchmetrics"].ok
    assert r.confirmed and keys(r) == ["pos_label"]


def test_probabilities_not_summing_to_one():
    y_true = [0, 1, 2, 0, 1, 2]
    probs = [[0.9, 0.3, 0.1], [0.2, 0.8, 0.4], [0.1, 0.2, 0.7],
             [0.6, 0.5, 0.2], [0.3, 0.6, 0.5], [0.2, 0.1, 0.9]]
    r = compare("auroc", y_true, probs)
    assert not r.results["sklearn"].ok
    assert r.confirmed and "row_sum" in keys(r)


@pytest.mark.parametrize("metric", ["accuracy", "precision", "recall", "f1", "auroc", "average_precision"])
def test_clean_binary_inputs_agree(metric):
    y_true = [0, 1, 1, 0, 1, 0, 1, 1]
    probs = [0.1, 0.8, 0.65, 0.3, 0.9, 0.45, 0.2, 0.7]
    r = compare(metric, y_true, probs)
    assert r.agree, str(r)
    assert r.causes == []


@pytest.mark.parametrize("metric", ["average_precision", "auroc"])
def test_clean_multiclass_probabilities_ranking_agree_or_explained(metric):
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(30, 4))
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y_true = np.arange(30) % 4
    r = compare(metric, y_true, probs)
    assert r.agree or r.confirmed, str(r)
