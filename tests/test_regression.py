"""Regression metrics: r2, mae, mse, rmse, mape."""

import numpy as np
import pytest

pytest.importorskip("torchmetrics")

from metric_diff import compare, compare_batched  # noqa: E402
from metric_diff.cli import main  # noqa: E402

Y = np.array([3.0, -0.5, 2.0, 7.0, 4.2])
P = np.array([2.5, 0.0, 2.1, 7.8, 3.9])
THREE = ["sklearn", "torchmetrics", "torcheval"]


def keys(r):
    return [c.key for c in r.causes]


@pytest.mark.parametrize("metric", ["r2", "mae", "mse", "rmse"])
def test_clean_regression_agrees(metric):
    # evaluate rounds R2 to 3 decimals; it has its own tests.
    from metric_diff.backends import resolve

    r = compare(metric, Y, P, backends=[b.name for b in resolve(None) if b.name != "evaluate"])
    assert r.agree, str(r)


@pytest.mark.parametrize("metric", ["mae", "rmse", "mape"])
def test_torcheval_lacks_metric_is_skipped(metric):
    r = compare(metric, Y, P, backends=THREE)
    assert "torcheval has no" in r.results["torcheval"].skipped


def test_aliases():
    assert compare("mean_squared_error", Y, P).metric == "mse"
    assert compare("r2_score", Y, P).metric == "r2"


def test_keras_mape_is_a_percentage():
    pytest.importorskip("keras")
    r = compare("mape", Y, P)
    assert r.results["keras"].value == pytest.approx(100 * r.results["sklearn"].value, rel=1e-5)
    assert r.confirmed and keys(r) == ["mape_percent"]


def test_mape_zero_target():
    r = compare("mape", [0.0, 1.0, 2.0], [0.5, 1.0, 2.5], backends=["sklearn", "torchmetrics"])
    assert r.results["sklearn"].value > 1e12 and r.results["torchmetrics"].value < 1e7
    assert r.confirmed and keys(r) == ["mape_zero_target"]


def test_r2_constant_target_torcheval_inf():
    r = compare("r2", [2.0, 2.0, 2.0], [2.0, 2.0, 2.1], backends=THREE)
    assert r.results["torcheval"].value == float("-inf")
    assert r.results["sklearn"].value == 0.0
    assert r.confirmed and keys(r) == ["r2_constant_target"]


def test_float32_rounding_on_large_values():
    # torchmetrics#3077 family: R2 wildly off when values are large relative to their spread
    rng = np.random.default_rng(0)
    y = 1e6 + rng.normal(size=1000)
    p = y + rng.normal(scale=0.1, size=1000)
    r = compare("r2", y, p, backends=["sklearn", "torchmetrics"])
    assert r.results["sklearn"].value == pytest.approx(0.989, abs=1e-3)
    assert r.results["torchmetrics"].value > 0.999
    assert r.confirmed and keys(r) == ["float32_precision"]


def test_multi_output_r2_agrees():
    y = np.array([[0.5, 1.0], [-1.0, 1.0], [7.0, -6.0], [2.0, 3.0]])
    p = np.array([[0.0, 2.0], [-1.0, 2.0], [8.0, -5.0], [2.5, 2.0]])
    r = compare("r2", y, p)
    assert r.agree and r.results["sklearn"].value == pytest.approx(0.9364939, abs=1e-6)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="broadcasts"):
        compare("mse", Y[:, None], np.c_[P, P])


def test_classification_settings_rejected_for_regression():
    with pytest.raises(ValueError, match="classification metrics only"):
        compare("mse", Y, P, average="macro")


def test_batched_r2_average_is_not_dataset_r2():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    p = y + rng.normal(scale=0.5, size=500)
    r = compare_batched("r2", [(y[i:i + 32], p[i:i + 32]) for i in range(0, 500, 32)], backends=THREE)
    assert [f.key for f in r.findings] == ["batch_average"]
    for name in ("torchmetrics", "torcheval"):
        assert r.stateful[name].value == pytest.approx(r.reference.value, abs=1e-4)


def test_cli_regression(tmp_path, capsys):
    np.savetxt(tmp_path / "y.csv", Y)
    np.savetxt(tmp_path / "p.csv", P)
    assert main(["rmse", "--true", str(tmp_path / "y.csv"), "--pred", str(tmp_path / "p.csv")]) == 0
    assert "all libraries agree" in capsys.readouterr().out
