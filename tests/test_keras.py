"""Keras 3 behaviour that differs from sklearn / torchmetrics / torcheval."""

import inspect

import numpy as np
import pytest

pytest.importorskip("keras")
pytest.importorskip("torchmetrics")

from metric_diff import compare  # noqa: E402

ALL = ["sklearn", "torchmetrics", "torcheval", "keras"]

Y = [0, 1, 1, 0, 1, 0, 1, 1]
P = [0.1, 0.8, 0.65, 0.3, 0.9, 0.45, 0.2, 0.7]


def keys(r):
    return [c.key for c in r.causes]


def test_keras_binary_f1_without_threshold_predicts_all_positive():
    r = compare("f1", Y, P, backends=ALL)
    assert r.results["keras"].value == pytest.approx(10 / 13, abs=1e-6)  # F1 of "all positive"
    assert r.results["sklearn"].value == pytest.approx(8 / 9, abs=1e-6)
    assert r.confirmed and keys(r) == ["keras_f1_threshold"]


def test_keras_auc_clips_logits():
    logits = [-2.0, 3.0, -0.5, -3.0, 4.0, 0.3, 1.0, -1.0]
    r = compare("auroc", Y, logits, backends=ALL)
    assert r.results["keras"].value < r.results["sklearn"].value - 0.1
    assert r.confirmed and keys(r) == ["keras_logits"]
    assert "from_logits=True" in r.causes[0].fix


def test_keras_pr_auc_is_not_average_precision():
    r = compare("average_precision", Y, P, backends=ALL)
    assert not np.isclose(r.results["keras"].value, r.results["sklearn"].value, atol=1e-3)
    assert r.confirmed and keys(r) == ["keras_pr"]


def test_keras_auc_200_threshold_approximation():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    r = compare("auroc", y, p, backends=ALL)
    gap = abs(r.results["keras"].value - r.results["sklearn"].value)
    assert 1e-5 < gap < 1e-3
    assert r.confirmed and keys(r) == ["keras_bins"]


def test_keras_multiclass_precision_thresholds_each_column():
    rng = np.random.default_rng(0)
    probs = rng.random((10, 3))
    probs /= probs.sum(1, keepdims=True)
    y = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
    r = compare("precision", y, probs, backends=ALL)
    assert r.confirmed and keys(r) == ["average"]
    assert "top_k=1" in r.causes[0].fix


def test_keras_multiclass_f1_returns_per_class_vector():
    rng = np.random.default_rng(0)
    probs = rng.random((10, 3))
    probs /= probs.sum(1, keepdims=True)
    r = compare("f1", [0, 0, 0, 0, 1, 1, 2, 2, 2, 2], probs, backends=ALL)
    assert "one per class" in r.results["keras"].error
    assert r.confirmed and "average" in keys(r)


@pytest.mark.parametrize("metric", ["accuracy", "precision", "recall"])
def test_keras_agrees_on_clean_binary_label_metrics(metric):
    assert compare(metric, Y, P, backends=ALL).agree


def test_keras_defaults_still_match_explanations():
    import keras

    f1 = inspect.signature(keras.metrics.F1Score).parameters
    auc = inspect.signature(keras.metrics.AUC).parameters
    assert f1["average"].default is None and f1["threshold"].default is None
    assert auc["num_thresholds"].default == 200
    assert auc["summation_method"].default == "interpolation"
