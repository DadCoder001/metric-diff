"""User-supplied settings: average, threshold, pos_label, num_classes and per-library overrides."""

import numpy as np
import pytest

pytest.importorskip("torchmetrics")

from metric_diff import compare, compare_batched  # noqa: E402
from metric_diff.backends import SUPPORTED_AVERAGES, resolve  # noqa: E402
from metric_diff.cli import main, parse_settings  # noqa: E402
from metric_diff.core import _final_inputs  # noqa: E402
from metric_diff.inputs import prepare  # noqa: E402

THREE = ["sklearn", "torchmetrics", "torcheval"]

# Imbalanced multiclass data, so macro, micro and weighted give different numbers.
rng = np.random.default_rng(3)
Y = np.array([0] * 20 + [1] * 6 + [2] * 4)
PROBS = rng.random((30, 3)) + np.eye(3)[Y] * 0.6
PROBS /= PROBS.sum(1, keepdims=True)

YB = [0, 1, 1, 0, 1, 0, 1, 1]
PB = [0.1, 0.8, 0.65, 0.3, 0.9, 0.45, 0.2, 0.7]


def keys(r):
    return [c.key for c in r.causes]


def test_common_average_is_used_by_every_library():
    r = compare("f1", Y, PROBS, average="weighted")
    assert r.agree, str(r)
    assert r.results["sklearn"].value != pytest.approx(compare("f1", Y, PROBS, average="macro").results["sklearn"].value)


def test_per_library_average_mismatch_is_explained():
    r = compare("f1", Y, PROBS, backends=THREE,
                settings={"sklearn": {"average": "weighted"}, "torchmetrics": {"average": "macro"}})
    assert r.status == "mismatch"
    assert r.confirmed and keys(r) == ["average"]
    assert "average='weighted' (your setting)" in r.causes[0].detail


def test_unsupported_average_is_skipped_not_a_mismatch():
    r = compare("auroc", Y, PROBS, average="weighted")
    assert r.results["torcheval"].skipped and "supports macro" in r.results["torcheval"].skipped
    assert r.confirmed and keys(r) == ["sklearn_multi_class"]
    assert any("skipped" in n for n in r.notes)


def test_per_library_threshold_mismatch():
    r = compare("precision", YB, PB, settings={"sklearn": {"threshold": 0.3}})
    assert r.results["sklearn"].value != pytest.approx(r.results["torchmetrics"].value)
    assert r.confirmed and keys(r) == ["threshold"]


def test_common_threshold_with_a_score_exactly_on_it():
    r = compare("precision", YB, PB, threshold=0.3, backends=THREE)
    assert r.confirmed and keys(r) == ["threshold_tie"]
    assert "exactly 0.3" in r.causes[0].detail


def test_sklearn_average_on_binary_data():
    r = compare("f1", YB, [0, 1, 0, 0, 1, 1, 1, 0], settings={"sklearn": {"average": "macro"}},
                backends=THREE)
    assert r.confirmed and keys(r) == ["binary_average"]


@pytest.mark.parametrize("pos,expected", [(2, 2 / 3), (1, 0.8)])
def test_pos_label(pos, expected):
    r = compare("f1", [1, 2, 1, 1], [1, 2, 2, 1], pos_label=pos)
    assert r.agree, str(r)
    assert r.results["sklearn"].value == pytest.approx(expected)


def test_pos_label_needs_binary_data():
    with pytest.raises(ValueError, match="binary"):
        compare("f1", [0, 1, 2], [0, 1, 2], pos_label=1)


def test_num_classes_with_class_that_never_occurs():
    r = compare("f1", [0, 1, 2, 2, 1, 0, 2, 2], [0, 2, 2, 2, 1, 0, 1, 2], num_classes=4,
                average="macro", backends=THREE)
    assert r.results["sklearn"].value == pytest.approx(0.5625)
    assert r.results["torchmetrics"].value == pytest.approx(0.75)
    assert r.confirmed and keys(r) == ["absent_classes"]


@pytest.mark.parametrize("kwargs,msg", [
    (dict(num_classes=2), "contains label"),
    (dict(settings={"jax": {"average": "macro"}}), "not being compared"),
    (dict(settings={"sklearn": {"zero_division": 1}}), "unknown setting"),
    (dict(average="samples"), "average must be"),
    (dict(threshold=1.5), "between 0 and 1"),
])
def test_invalid_settings(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        compare("f1", [0, 1, 2, 2], [0, 1, 2, 1], backends=THREE, **kwargs)


def test_batched_uses_settings():
    batches = [(Y[i:i + 8], PROBS[i:i + 8]) for i in range(0, 30, 8)]
    r = compare_batched("f1", batches, average="weighted", backends=THREE)
    for name in ("torchmetrics", "torcheval"):
        assert r.stateful[name].value == pytest.approx(r.reference.value, abs=1e-5)


def test_cli_per_library_setting(tmp_path, capsys):
    np.savetxt(tmp_path / "y.csv", YB, fmt="%d")
    np.savetxt(tmp_path / "p.csv", PB)
    code = main(["precision", "--true", str(tmp_path / "y.csv"), "--pred", str(tmp_path / "p.csv"),
                 "--set", "sklearn.threshold=0.3", "--backends", "sklearn,torchmetrics"])
    out = capsys.readouterr().out
    assert code == 1
    assert "your settings: sklearn: threshold=0.3" in out and "Different decision thresholds" in out


def test_parse_settings():
    assert parse_settings(["sklearn.average=weighted", "keras.threshold=0.3"]) == {
        "sklearn": {"average": "weighted"}, "keras": {"threshold": 0.3}}
    with pytest.raises(ValueError):
        parse_settings(["average=macro"])


def test_supported_averages_table_matches_installed_libraries():
    """Guard: the table decides when a library is skipped, so it must match reality."""
    case = prepare(Y, PROBS)
    for b in resolve(None):
        for metric, expected in SUPPORTED_AVERAGES[b.name].items():
            works = set()
            for avg in ("micro", "macro", "weighted"):
                yt, yp = _final_inputs(metric, case, case.y_true, case.y_pred)
                try:
                    b.compute(metric, "multiclass", yt, yp, 3, {"average": avg, "multi_class": "ovr"})
                    works.add(avg)
                except Exception:
                    pass
            assert works == expected, (b.name, metric, works)
