import numpy as np
import pytest

from metric_diff import compare_batched
from metric_diff.cli import main

rng = np.random.default_rng(0)
Y = rng.integers(0, 2, 1000)
P = np.clip(Y * 0.25 + rng.random(1000) * 0.75, 0, 1)


def batches(size, y=Y, p=P):
    return [(y[i:i + size], p[i:i + size]) for i in range(0, len(y), size)]


def keys(r):
    return [f.key for f in r.findings]


def test_mean_of_batch_f1_is_not_dataset_f1():
    r = compare_batched("f1", batches(64), backends=["sklearn"])
    assert r.reference.value != pytest.approx(r.batch_mean, abs=1e-4)
    assert keys(r) == ["batch_average"]
    assert "metric.update()" in r.findings[0].fix


def test_accuracy_only_needs_batch_size_weighting():
    r = compare_batched("accuracy", batches(64), backends=["sklearn"])
    assert r.weighted_mean == pytest.approx(r.reference.value)
    assert keys(r) == ["batch_average"]
    assert "Weighting by batch size fixes it" in r.findings[0].detail


def test_equal_batches_accuracy_is_ok():
    r = compare_batched("accuracy", batches(100), backends=["sklearn"])
    assert r.ok and "OK" in str(r)


def test_ddp_padding_duplicates_detected():
    padded = batches(100) + [(Y[:24], P[:24])]
    r = compare_batched("accuracy", padded, backends=["sklearn"], expected_n=1000)
    assert "duplicates" in keys(r)


def test_drop_last_detected():
    r = compare_batched("accuracy", batches(100)[:-1], backends=["sklearn"], expected_n=1000)
    assert "missing" in keys(r)


def test_single_class_batches_are_reported_as_skipped():
    y = np.array([0, 0, 1, 1] * 5 + [1, 1, 1, 1] * 5)
    p = np.linspace(0.05, 0.95, len(y))
    r = compare_batched("auroc", batches(4, y, p), backends=["sklearn"])
    assert sum(v is None for v in r.per_batch) == 5
    assert "left out" in r.findings[0].detail


def test_torch_libraries_accumulate_exactly():
    pytest.importorskip("torchmetrics")
    r = compare_batched("auroc", batches(64), backends=["sklearn", "torchmetrics", "torcheval"])
    for name in ("torchmetrics", "torcheval"):
        assert r.stateful[name].value == pytest.approx(r.reference.value, abs=1e-5)
    assert not any(k.startswith("stateful_") for k in keys(r))


def test_stateful_bug_is_reported(monkeypatch):
    pytest.importorskip("torchmetrics")
    from metric_diff import backends as B

    original = B.TorchmetricsBackend.stateful

    def broken(self, *args, **kwargs):  # simulate a library that loses state
        s = original(self, *args, **kwargs)
        real_update = s.update
        calls = {"n": 0}

        def update(y_true, y_pred):
            calls["n"] += 1
            if calls["n"] > 1:
                s.obj.reset()
            real_update(y_true, y_pred)

        s.update = update
        return s

    monkeypatch.setattr(B.TorchmetricsBackend, "stateful", broken)
    r = compare_batched("f1", batches(100), backends=["sklearn", "torchmetrics"])
    assert "stateful_torchmetrics" in keys(r)


def test_multiclass_batches_with_sklearn_needing_average():
    y = rng.integers(0, 3, 300)
    probs = rng.random((300, 3))
    probs /= probs.sum(1, keepdims=True)
    r = compare_batched("f1", batches(32, y, probs), backends=["sklearn"])
    assert r.reference.ok  # metric-diff picked a working average for the reference
    assert keys(r) == ["batch_average"]


def test_cli_batch_mode(tmp_path, capsys):
    np.savetxt(tmp_path / "y.csv", Y, fmt="%d")
    np.savetxt(tmp_path / "p.csv", P)
    code = main(["f1", "--true", str(tmp_path / "y.csv"), "--pred", str(tmp_path / "p.csv"),
                 "--batch-size", "64", "--backends", "sklearn"])
    out = capsys.readouterr().out
    assert code == 1 and "Averaging per-batch values" in out


def test_cli_expected_n_requires_batch_size(capsys):
    assert main(["f1", "--true", "a.csv", "--pred", "b.csv", "--expected-n", "5"]) == 2


def test_bad_batches_rejected():
    with pytest.raises(ValueError, match="no batches"):
        compare_batched("f1", [])
    with pytest.raises(ValueError, match="pair"):
        compare_batched("f1", [np.array([0, 1])])
