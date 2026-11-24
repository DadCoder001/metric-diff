"""Segmentation adapters: Dice and IoU on label maps of shape (n_images, *spatial).

Two conventions decide most of the differences between libraries:

* aggregation: "dataset" pools the pixels of all images and computes one score per class
  (sklearn on flattened masks, Keras MeanIoU, evaluate mean_iou); "per_image" computes a
  score per image and averages the images (torchmetrics, MONAI).
* classes: scores are averaged over the classes that occur (in y_true or y_pred) in the
  image / dataset; a class absent from both is left out. MONAI additionally leaves out a
  class whose *target* is empty even if it was predicted (ignore_empty=True).

``opts`` keys: aggregation, include_background, monai_count_empty_target.
"""

from __future__ import annotations

import numpy as np

NATIVE_AGGREGATION = {
    "sklearn": "dataset",
    "torchmetrics": "per_image",
    "monai": "per_image",
    "keras": "dataset",
    "evaluate": "dataset",
}

DEFAULT_TEXT = {
    "sklearn": "pools all pixels of the dataset (f1_score / jaccard_score on flattened masks)",
    "torchmetrics": "per image, then averaged (aggregation_level='samplewise')",
    "monai": "per image, then averaged, and ignores classes whose target is empty (ignore_empty=True)",
    "keras": "pools all pixels of the dataset (one confusion matrix)",
    "evaluate": "pools all pixels of the dataset (total areas per class)",
}


def _agg(name, opts):
    return opts.get("aggregation", NATIVE_AGGREGATION[name])


def unsupported(name: str, metric: str, opts: dict, ndim: int) -> str | None:
    agg = _agg(name, opts)
    bg = opts.get("include_background", True)
    if name == "torcheval":
        return "torcheval has no segmentation metrics"
    if name == "torchmetrics" and metric == "iou" and agg == "dataset":
        return "torchmetrics mean_iou has no dataset-level aggregation"
    if name == "monai" and agg == "dataset":
        return "MONAI averages per image; it has no dataset-level aggregation"
    if name in ("keras", "evaluate") and metric == "dice":
        return f"{name} has no Dice metric"
    if name == "keras" and not bg:
        return ("Keras MeanIoU can't leave a class out of the average (ignore_class=0 drops the "
                "background pixels instead, which changes the result)")
    if name == "evaluate" and not bg:
        return "evaluate mean_iou can't leave a class out of the average (reduce_labels shifts labels instead)"
    if name == "evaluate" and ndim != 3:
        return "evaluate mean_iou only takes 2-D images (N, H, W)"
    return None


def compute(name: str, metric: str, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int,
            opts: dict) -> float:
    return {"sklearn": _sklearn, "torchmetrics": _torchmetrics, "monai": _monai,
            "keras": _keras, "evaluate": _evaluate}[name](metric, y_true, y_pred, num_classes, opts)


def _per_image(fn, y_true, y_pred) -> float:
    values = [fn(t, p) for t, p in zip(y_true, y_pred)]
    values = [v for v in values if v == v]
    if not values:
        raise ValueError("no image contains any of the scored classes")
    return float(np.mean(values))


def _sklearn(metric, y_true, y_pred, num_classes, opts):
    import sklearn.metrics as skm

    fn = skm.f1_score if metric == "dice" else skm.jaccard_score
    bg = opts.get("include_background", True)

    def one(t, p):
        t, p = t.ravel(), p.ravel()
        labels = np.union1d(t, p)
        if not bg:
            labels = labels[labels != 0]
        if len(labels) == 0:
            return float("nan")
        return float(fn(t, p, labels=list(labels), average="macro", zero_division=0))

    if _agg("sklearn", opts) == "dataset":
        return one(y_true, y_pred)
    return _per_image(one, y_true, y_pred)


def _torchmetrics(metric, y_true, y_pred, num_classes, opts):
    import torch
    import torchmetrics.functional.segmentation as S

    preds, target = torch.as_tensor(y_pred), torch.as_tensor(y_true)
    bg = opts.get("include_background", True)
    if metric == "dice":
        level = "global" if _agg("torchmetrics", opts) == "dataset" else "samplewise"
        v = S.dice_score(preds, target, num_classes=num_classes, include_background=bg,
                         input_format="index", aggregation_level=level)
    else:
        v = S.mean_iou(preds, target, num_classes=num_classes, include_background=bg, input_format="index")
    return float(torch.nanmean(v.float()))


def torchmetrics_default_format(metric, y_true, y_pred, num_classes) -> float | None:
    """What torchmetrics returns when label maps are passed without input_format='index'."""
    import torch
    import torchmetrics.functional.segmentation as S

    try:
        fn = S.dice_score if metric == "dice" else S.mean_iou
        v = fn(torch.as_tensor(y_pred), torch.as_tensor(y_true), num_classes=num_classes)
        return float(torch.nanmean(v.float()))
    except Exception:
        return None


def _one_hot(y, num_classes):
    import torch

    oh = np.moveaxis(np.eye(num_classes, dtype=np.float32)[y], -1, 1)
    return torch.as_tensor(oh)


def _monai_raw(metric, y_true, y_pred, num_classes, opts):
    import monai.metrics as M
    import torch

    bg = opts.get("include_background", True)
    count_empty = opts.get("monai_count_empty_target", False)
    cls = M.DiceMetric if metric == "dice" else M.MeanIoU
    m = cls(include_background=bg, reduction="none", ignore_empty=not count_empty)
    raw = m(_one_hot(y_pred, num_classes), _one_hot(y_true, num_classes)).float()
    if count_empty:
        # ignore_empty=False scores a class absent from both target and prediction as 1.0;
        # leave those out, like the other libraries do.
        axes = tuple(range(1, y_true.ndim))
        present = np.stack([((y_true == c) | (y_pred == c)).any(axis=axes)
                            for c in range(0 if bg else 1, num_classes)], axis=1)
        raw = torch.where(torch.as_tensor(present), raw, torch.tensor(float("nan")))
    return raw


def _monai(metric, y_true, y_pred, num_classes, opts):
    import torch

    raw = _monai_raw(metric, y_true, y_pred, num_classes, opts)
    per_image = torch.nanmean(raw, dim=1)
    return float(torch.nanmean(per_image))


def _keras(metric, y_true, y_pred, num_classes, opts):
    from .backends import _ensure_keras_backend

    _ensure_keras_backend()
    import keras

    def one(t, p):
        m = keras.metrics.MeanIoU(num_classes=num_classes)
        m.update_state(t.ravel(), p.ravel())
        return float(keras.ops.convert_to_numpy(m.result()))

    if _agg("keras", opts) == "dataset":
        return one(y_true, y_pred)
    return _per_image(one, y_true, y_pred)


def _evaluate(metric, y_true, y_pred, num_classes, opts):
    from .backends import _load_evaluate

    ev = _load_evaluate("mean_iou")

    def one(t, p):
        t, p = (t, p) if t.ndim == 3 else (t[None], p[None])
        out = ev.compute(predictions=list(p), references=list(t), num_labels=num_classes, ignore_index=-1)
        return float(out["mean_iou"])

    if _agg("evaluate", opts) == "dataset":
        return one(y_true, y_pred)
    return _per_image(one, y_true, y_pred)


class _Stateful:
    def __init__(self, update, compute):
        self.update, self.compute = update, compute


def stateful(name: str, metric: str, num_classes: int, opts: dict):
    """Accumulating objects, used with each library's own default aggregation."""
    if name in NATIVE_AGGREGATION and _agg(name, opts) != NATIVE_AGGREGATION[name]:
        return None  # the object can only accumulate the library's own way
    bg = opts.get("include_background", True)
    if name == "torchmetrics":
        import torch
        import torchmetrics.segmentation as S

        obj = (S.DiceScore(num_classes=num_classes, include_background=bg, input_format="index")
               if metric == "dice" else
               S.MeanIoU(num_classes=num_classes, include_background=bg, input_format="index"))
        return _Stateful(lambda t, p: obj.update(torch.as_tensor(p), torch.as_tensor(t)),
                         lambda: float(torch.nanmean(obj.compute().float())))
    if name == "monai":
        import monai.metrics as M

        cls = M.DiceMetric if metric == "dice" else M.MeanIoU
        obj = cls(include_background=bg)
        return _Stateful(lambda t, p: obj(_one_hot(p, num_classes), _one_hot(t, num_classes)),
                         lambda: float(obj.aggregate()))
    if name == "keras":
        from .backends import _ensure_keras_backend

        _ensure_keras_backend()
        import keras

        obj = keras.metrics.MeanIoU(num_classes=num_classes)
        return _Stateful(lambda t, p: obj.update_state(t.ravel(), p.ravel()),
                         lambda: float(keras.ops.convert_to_numpy(obj.result())))
    return None
