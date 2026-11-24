"""Cases found by running metric-diff on real model predictions (datasets bundled with sklearn)."""

import numpy as np
import pytest

pytest.importorskip("keras")
pytest.importorskip("torchmetrics")

from sklearn.datasets import load_breast_cancer, load_diabetes  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from metric_diff import compare, compare_batched  # noqa: E402

ALL4 = ["sklearn", "torchmetrics", "torcheval", "keras"]


@pytest.fixture(scope="module")
def breast_cancer_logits():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr)
    return yte, m.decision_function(Xte)


@pytest.fixture(scope="module")
def diabetes_predictions():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    return yte, Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)


def test_cause_is_attributed_to_the_library_that_differs(breast_cancer_logits):
    # Only Keras disagrees here (it clips logits). The sigmoid-saturation alignment would also
    # reconcile the numbers, but torchmetrics agrees, so the explanation must name Keras.
    y, logits = breast_cancer_logits
    r = compare("auroc", y, logits, backends=ALL4)
    assert r.results["torchmetrics"].value == pytest.approx(r.results["sklearn"].value, abs=1e-4)
    assert r.results["keras"].value < r.results["sklearn"].value - 0.01
    assert r.confirmed and [c.key for c in r.causes] == ["keras_logits"]


def test_keras_1d_regression_targets_average_batch_means(diabetes_predictions):
    y, p = diabetes_predictions
    batches = [(y[i:i + 32], p[i:i + 32]) for i in range(0, len(y), 32)]  # last batch is smaller
    r = compare_batched("mae", batches, backends=["sklearn", "keras"])
    finding = next(f for f in r.findings if f.key == "stateful_keras")
    assert "(batch, 1)" in finding.fix
    assert r.stateful["keras"].value == pytest.approx(r.batch_mean, rel=1e-5)  # mean of batch means


def test_keras_2d_regression_targets_accumulate_correctly(diabetes_predictions):
    y, p = diabetes_predictions
    y2, p2 = y[:, None], p[:, None]
    batches = [(y2[i:i + 32], p2[i:i + 32]) for i in range(0, len(y), 32)]
    r = compare_batched("mae", batches, backends=["sklearn", "keras"])
    assert not any(f.key == "stateful_keras" for f in r.findings)
    assert r.stateful["keras"].value == pytest.approx(r.reference.value, rel=1e-5)


def test_r2_per_batch_average_is_far_off(diabetes_predictions):
    y, p = diabetes_predictions
    r = compare_batched("r2", [(y[i:i + 32], p[i:i + 32]) for i in range(0, len(y), 32)],
                        backends=["sklearn"])
    assert r.reference.value - r.batch_mean > 0.1
