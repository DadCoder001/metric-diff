"""Segmentation metrics (Dice, IoU) on label maps."""

import numpy as np
import pytest

pytest.importorskip("torchmetrics")

from metric_diff import compare, compare_batched  # noqa: E402
from metric_diff.backends import resolve  # noqa: E402
from metric_diff.cli import main  # noqa: E402

# 3 images, 4x4, 3 classes (0 = background):
#  image 0: classes 1 and 2 present; image 1: class 2 only predicted (false positive);
#  image 2: class 2 absent from both.
T = np.zeros((3, 4, 4), int)
P = np.zeros((3, 4, 4), int)
T[0, :2, :2] = 1; P[0, :2, :3] = 1; T[0, 2:, 2:] = 2; P[0, 3:, 2:] = 2  # noqa: E702
T[1, :2, :] = 1; P[1, :2, :] = 1; P[1, 3, 3] = 2  # noqa: E702
T[2, 1:3, 1:3] = 1; P[2, 1:3, 1:2] = 1  # noqa: E702

SK_TM = ["sklearn", "torchmetrics"]
installed = {b.name for b in resolve(None)}


def keys(r):
    return [c.key for c in r.causes]


def test_dataset_vs_per_image_dice():
    r = compare("dice", T, P, backends=SK_TM)
    assert r.results["sklearn"].value == pytest.approx(0.7745, abs=1e-4)      # pooled
    assert r.results["torchmetrics"].value == pytest.approx(0.7261, abs=1e-4)  # per image
    assert r.confirmed and keys(r) == ["seg_aggregation"]


def test_same_aggregation_agrees():
    for agg in ("dataset", "per_image"):
        assert compare("dice", T, P, aggregation=agg, backends=SK_TM).agree


def test_torchmetrics_iou_has_no_dataset_level():
    r = compare("iou", T, P, aggregation="dataset", backends=SK_TM)
    assert "no dataset-level" in r.results["torchmetrics"].skipped


def test_background_difference():
    r = compare("dice", T, P, aggregation="dataset", backends=SK_TM,
                settings={"sklearn": {"include_background": False}})
    assert r.confirmed and keys(r) == ["seg_background"]


def test_torchmetrics_input_format_note():
    r = compare("dice", T, P, backends=SK_TM)
    assert any("input_format='index'" in n for n in r.notes)


@pytest.mark.skipif("monai" not in installed, reason="monai not installed")
def test_monai_ignore_empty_hides_false_positives():
    r = compare("dice", T, P, aggregation="per_image", backends=["sklearn", "monai"])
    assert r.results["monai"].value == pytest.approx(0.8335, abs=1e-4)
    assert r.confirmed and keys(r) == ["monai_ignore_empty"]
    assert r.aligned["monai"].value == pytest.approx(0.7261, abs=1e-4)


@pytest.mark.skipif("monai" not in installed, reason="monai not installed")
def test_monai_has_no_dataset_level():
    r = compare("iou", T, P, aggregation="dataset", backends=["sklearn", "monai"])
    assert r.results["monai"].skipped


@pytest.mark.skipif("keras" not in installed, reason="keras not installed")
def test_keras_mean_iou_is_dataset_level():
    r = compare("iou", T, P, backends=["sklearn", "keras"])
    assert r.agree and r.results["keras"].value == pytest.approx(0.653, abs=1e-3)
    assert compare("dice", T, P, backends=["sklearn", "keras"]).results["keras"].skipped


def test_all_libraries_agree_after_alignment():
    r = compare("iou", T, P)
    assert r.confirmed, str(r)
    vals = [x.value for x in r.aligned.values() if x.ok]
    assert len(vals) >= 2 and max(vals) - min(vals) < 1e-4


def test_monai_excluded_from_classification_by_default():
    r = compare("f1", [0, 1, 1, 0], [0, 1, 0, 0])
    assert "monai" not in r.results and "torcheval" in r.results


def test_torcheval_excluded_from_segmentation_by_default():
    assert "torcheval" not in compare("dice", T, P).results


def test_3d_volumes():
    t3 = np.stack([T, T])          # (2, 3, 4, 4)
    p3 = np.stack([P, P])
    r = compare("dice", t3, p3, aggregation="dataset", backends=SK_TM)
    assert r.agree


@pytest.mark.parametrize("kwargs,msg", [
    (dict(average="macro"), "segmentation metrics take"),
    (dict(aggregation="global"), "aggregation must be"),
])
def test_invalid_segmentation_settings(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        compare("dice", T, P, backends=SK_TM, **kwargs)


def test_segmentation_settings_rejected_for_classification():
    with pytest.raises(ValueError, match="segmentation metrics"):
        compare("f1", [0, 1], [0, 1], include_background=False)


def test_input_validation():
    with pytest.raises(ValueError, match="must match"):
        compare("dice", T, P[:2])
    with pytest.raises(ValueError, match="argmax"):
        compare("dice", T.astype(float) / 3, P)


def test_batched_segmentation():
    tt = np.concatenate([T] * 4)
    pp = np.concatenate([P] * 4)
    r = compare_batched("dice", [(tt[i:i + 4], pp[i:i + 4]) for i in range(0, 12, 4)], backends=SK_TM)
    assert r.stateful["torchmetrics"].value == pytest.approx(r.dataset.results["torchmetrics"].value, abs=1e-5)


def test_cli_segmentation(tmp_path, capsys):
    np.save(tmp_path / "t.npy", T)
    np.save(tmp_path / "p.npy", P)
    code = main(["iou", "--true", str(tmp_path / "t.npy"), "--pred", str(tmp_path / "p.npy"),
                 "--aggregation", "per_image", "--backends", "sklearn,torchmetrics"])
    assert code == 0 and "all libraries agree" in capsys.readouterr().out
