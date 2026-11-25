"""metric_diff.testing: assertions for users' own test suites."""

import numpy as np
import pytest

pytest.importorskip("torchmetrics")

from metric_diff.testing import (MetricMismatch, assert_batches_consistent,  # noqa: E402
                                 assert_metrics_agree)

THREE = ["sklearn", "torchmetrics", "torcheval"]
Y = [0, 1, 1, 0, 1, 0, 1, 1]
P = [0.1, 0.8, 0.65, 0.3, 0.9, 0.45, 0.2, 0.7]


def test_passes_and_returns_report():
    r = assert_metrics_agree("f1", Y, P, backends=THREE)
    assert r.agree


def test_unexplained_or_unallowed_mismatch_fails_with_full_report():
    with pytest.raises(MetricMismatch) as e:
        assert_metrics_agree("precision", Y, P, threshold=0.3, backends=THREE)
    msg = str(e.value)
    assert "not in allow: threshold_tie" in msg and "[threshold_tie]" in msg
    assert e.value.report.status == "mismatch"
    assert isinstance(e.value, AssertionError)  # works with plain unittest too


def test_allowed_confirmed_cause_passes():
    r = assert_metrics_agree("precision", Y, P, threshold=0.3, backends=THREE, allow=["threshold_tie"])
    assert r.status == "mismatch" and r.confirmed


def test_allow_accepts_a_single_string():
    assert_metrics_agree("precision", Y, P, threshold=0.3, backends=THREE, allow="threshold_tie")


def test_allow_rejects_typos():
    with pytest.raises(ValueError, match="unknown cause key"):
        assert_metrics_agree("f1", Y, P, allow=["threshold-tie"])


def test_expected_value():
    assert_metrics_agree("accuracy", [0, 1, 1, 0], [0, 1, 0, 0], expected=0.75, backends=THREE)
    with pytest.raises(MetricMismatch, match="expected accuracy=0.8"):
        assert_metrics_agree("accuracy", [0, 1, 1, 0], [0, 1, 0, 0], expected=0.8, backends=THREE)


def test_min_libraries():
    with pytest.raises(MetricMismatch, match="only 1 library"):
        assert_metrics_agree("f1", Y, P, backends=["sklearn"])
    assert_metrics_agree("f1", Y, P, backends=["sklearn"], min_libraries=1)


def test_skipped_library_does_not_count():
    # torcheval can't do zero_division=1, so only two libraries really ran.
    with pytest.raises(MetricMismatch, match="only 1 library"):
        assert_metrics_agree("precision", [0, 1, 2, 2, 1, 0], [0, 1, 1, 0, 1, 0],
                             average="macro", zero_division=1, backends=["sklearn", "torcheval"])


def test_batches():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 64)
    p = np.clip(y * 0.4 + rng.random(64) * 0.6, 0, 1)
    batches = [(y[i:i + 16], p[i:i + 16]) for i in range(0, 64, 16)]
    with pytest.raises(MetricMismatch, match="batch_average"):
        assert_batches_consistent("f1", batches, backends=THREE)
    r = assert_batches_consistent("f1", batches, backends=THREE, allow=["batch_average"])
    assert r.findings[0].key == "batch_average"
    with pytest.raises(MetricMismatch, match="duplicates"):
        assert_batches_consistent("accuracy", batches, backends=THREE, expected_n=60,
                                  allow=["batch_average"])
