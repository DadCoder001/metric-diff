"""Normalise user inputs (lists, numpy arrays, torch tensors) and describe them."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

METRICS = ("accuracy", "precision", "recall", "f1", "auroc", "average_precision")
LABEL_METRICS = ("accuracy", "precision", "recall", "f1")
RANKING_METRICS = ("auroc", "average_precision")

_ALIASES = {
    "acc": "accuracy",
    "f1_score": "f1",
    "f1score": "f1",
    "roc_auc": "auroc",
    "auc": "auroc",
    "ap": "average_precision",
    "auprc": "average_precision",
}


def normalise_metric(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    key = _ALIASES.get(key, key)
    if key not in METRICS:
        raise ValueError(f"unknown metric {name!r}; choose one of {', '.join(METRICS)}")
    return key


def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch.Tensor without importing torch
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _is_integer_valued(a: np.ndarray) -> bool:
    if a.dtype.kind in "iub":
        return True
    if a.dtype.kind != "f":
        return False
    return bool(np.all(np.isfinite(a)) and np.all(np.mod(a, 1) == 0))


@dataclass(frozen=True)
class Case:
    """The inputs, plus what metric-diff inferred about them."""

    y_true: np.ndarray
    y_pred: np.ndarray
    task: str  # "binary" | "multiclass"
    num_classes: int
    pred_kind: str  # "labels" | "probs" | "logits"

    @property
    def n_samples(self) -> int:
        return int(self.y_true.shape[0])

    def describe(self) -> str:
        kinds = {
            "labels": "predicted labels",
            "probs": "probabilities",
            "logits": "logits (values outside [0, 1])",
        }
        task = self.task if self.task == "binary" else f"multiclass, {self.num_classes} classes"
        return f"{task}, {self.n_samples} samples, predictions look like {kinds[self.pred_kind]}"


def prepare(y_true, y_pred) -> Case:
    yt = to_numpy(y_true)
    yp = to_numpy(y_pred)

    if yt.ndim == 2 and yt.shape[1] == 1:
        yt = yt[:, 0]
    if yp.ndim == 2 and yp.shape[1] == 1:
        yp = yp[:, 0]
    if yt.ndim != 1:
        raise ValueError("y_true must be 1-D class labels (one-hot / multilabel is not supported yet)")
    if yp.ndim not in (1, 2):
        raise ValueError("y_pred must be 1-D (labels or binary scores) or 2-D (n_samples, n_classes)")
    if yp.shape[0] != yt.shape[0]:
        raise ValueError(f"y_true has {yt.shape[0]} samples but y_pred has {yp.shape[0]}")
    if yt.shape[0] == 0:
        raise ValueError("inputs are empty")
    if not _is_integer_valued(yt):
        raise ValueError("y_true must contain integer class labels")
    yt = yt.astype(np.int64)
    if yt.min() < 0:
        raise ValueError("y_true contains negative labels")

    if yp.ndim == 2:
        yp = yp.astype(np.float64)
        n_classes = yp.shape[1]
        if yt.max() >= n_classes:
            raise ValueError(f"y_true has label {yt.max()} but y_pred only has {n_classes} columns")
        kind = "probs" if (yp.min() >= 0 and yp.max() <= 1) else "logits"
        return Case(yt, yp, "multiclass", n_classes, kind)

    if _is_integer_valued(yp):
        yp = yp.astype(np.int64)
        labels = np.union1d(yt, yp)
        if len(labels) <= 2:
            return Case(yt, yp, "binary", 2, "labels")
        if yp.min() < 0:
            raise ValueError("y_pred contains negative labels")
        return Case(yt, yp, "multiclass", int(labels.max()) + 1, "labels")

    yp = yp.astype(np.float64)
    if len(np.unique(yt)) > 2:
        raise ValueError(
            "y_true has more than 2 classes but y_pred is 1-D scores; "
            "pass multiclass scores as a 2-D array (n_samples, n_classes)"
        )
    kind = "probs" if (yp.min() >= 0 and yp.max() <= 1) else "logits"
    return Case(yt, yp, "binary", 2, kind)
