"""Hugging Face evaluate backend.

evaluate downloads each metric script from the Hugging Face Hub the first time; these tests
skip when that isn't possible (e.g. offline without a cache).
"""

import numpy as np
import pytest

pytest.importorskip("evaluate")

from metric_diff import compare, compare_batched  # noqa: E402
from metric_diff.backends import _load_evaluate  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _metrics_loadable():
    try:
        _load_evaluate("f1")
    except RuntimeError as exc:
        pytest.skip(str(exc))


SK_EV = ["sklearn", "evaluate"]
Y = [0, 1, 1, 0, 1, 1]
P = [0, 1, 0, 0, 1, 0]


def keys(r):
    return [c.key for c in r.causes]


@pytest.mark.parametrize("metric", ["accuracy", "precision", "recall", "f1"])
def test_binary_label_metrics_match_sklearn(metric):
    assert compare(metric, Y, P, backends=SK_EV).agree


def test_binary_auroc_matches_sklearn():
    assert compare("auroc", Y, [0.1, 0.8, 0.4, 0.2, 0.9, 0.3], backends=SK_EV).agree


def test_multiclass_f1_needs_average_like_sklearn():
    r = compare("f1", [0, 1, 2, 2, 1, 0], [0, 2, 2, 1, 1, 0], backends=SK_EV + ["torchmetrics"])
    assert not r.results["evaluate"].ok
    assert r.confirmed and keys(r) == ["average"]


def test_multiclass_auroc_uses_multiclass_config():
    probs = np.random.default_rng(0).random((12, 3))
    probs /= probs.sum(1, keepdims=True)
    r = compare("auroc", np.arange(12) % 3, probs, average="macro", backends=SK_EV)
    assert r.confirmed and keys(r) == ["sklearn_multi_class"]
    assert r.aligned["evaluate"].value == pytest.approx(r.aligned["sklearn"].value)


def test_average_precision_and_zero_division_are_skipped():
    r = compare("average_precision", Y, [0.1, 0.8, 0.4, 0.2, 0.9, 0.3], backends=SK_EV)
    assert "no average precision" in r.results["evaluate"].skipped
    r = compare("f1", [0, 1, 2, 2], [0, 1, 1, 0], average="macro", zero_division=1, backends=SK_EV)
    assert "always 0" in r.results["evaluate"].skipped


def test_r2_rounded_to_three_decimals():
    r = compare("r2", [3.0, -0.5, 2.0, 7.0, 4.2], [2.5, 0.0, 2.1, 7.8, 3.9], backends=SK_EV)
    assert r.results["evaluate"].value == 0.959
    assert r.confirmed and keys(r) == ["evaluate_r2_rounding"]


def test_r2_constant_target_minus_inf():
    r = compare("r2", [2.0, 2.0, 2.0], [2.0, 2.0, 2.1], backends=SK_EV)
    assert r.results["evaluate"].value == float("-inf")
    assert r.confirmed and keys(r) == ["r2_constant_target"]


@pytest.mark.parametrize("metric", ["mae", "mse", "rmse", "mape"])
def test_regression_matches_sklearn(metric):
    assert compare(metric, [3.0, -0.5, 2.0, 7.0, 4.2], [2.5, 0.0, 2.1, 7.8, 3.9], backends=SK_EV).agree


def test_multi_output_regression_skipped():
    y = np.array([[0.5, 1.0], [-1.0, 1.0], [7.0, -6.0]])
    r = compare("mse", y, y + 0.1, backends=SK_EV)
    assert "multi-output" in r.results["evaluate"].skipped


def test_add_batch_accumulates_exactly():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = rng.integers(0, 2, 300)
    r = compare_batched("f1", [(y[i:i + 64], p[i:i + 64]) for i in range(0, 300, 64)], backends=SK_EV)
    assert r.stateful["evaluate"].value == pytest.approx(r.reference.value)
