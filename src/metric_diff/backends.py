"""Adapters that call each metric library the way a user would.

Every backend receives already-prepared numpy arrays:

* ``y_true``: int64 labels
* ``y_pred``: int64 labels (1-D) or float scores (1-D binary, 2-D multiclass)

``opts`` holds settings that metric-diff aligns while looking for a cause.
An empty ``opts`` means "call the library with its own defaults".
Known keys: ``average``, ``threshold``, ``multi_class`` (sklearn only).
"""

from __future__ import annotations

import importlib.util
import warnings
from typing import Callable

import numpy as np

from .inputs import LABEL_METRICS

# What each library does when you don't pass `average` (multiclass).
# Kept next to the code that calls the libraries; tests check it against the installed versions.
DEFAULT_AVERAGE: dict[str, dict[str, str]] = {
    "sklearn": {
        "accuracy": "micro (accuracy_score has no `average`)",
        "precision": "'binary' (raises an error on multiclass data)",
        "recall": "'binary' (raises an error on multiclass data)",
        "f1": "'binary' (raises an error on multiclass data)",
        "auroc": "'macro', but multi_class='raise' (raises an error on multiclass data)",
        "average_precision": "'macro'",
    },
    "torchmetrics": {
        "accuracy": "'macro' (macro accuracy = sklearn balanced_accuracy_score)",
        "precision": "'macro'",
        "recall": "'macro'",
        "f1": "'macro'",
        "auroc": "'macro'",
        "average_precision": "'macro'",
    },
    "torcheval": {
        "accuracy": "'micro'",
        "precision": "'micro'",
        "recall": "'micro'",
        "f1": "'micro'",
        "auroc": "'macro'",
        "average_precision": "'macro'",
    },
}


class Backend:
    name: str = ""
    module: str = ""

    def available(self) -> bool:
        return importlib.util.find_spec(self.module) is not None

    def compute(self, metric: str, task: str, y_true: np.ndarray, y_pred: np.ndarray,
                num_classes: int, opts: dict) -> float:
        raise NotImplementedError

    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version({"sklearn": "scikit-learn"}.get(self.module, self.module))
        except Exception:  # pragma: no cover
            return "?"


class SklearnBackend(Backend):
    """scikit-learn, the reference implementation.

    sklearn's label metrics need hard labels, so scores are turned into labels the way most
    users do it: ``y_pred >= threshold`` (default 0.5) for binary, ``argmax`` for multiclass.
    """

    name = "sklearn"
    module = "sklearn"

    def compute(self, metric, task, y_true, y_pred, num_classes, opts):
        import sklearn.metrics as skm

        avg = opts.get("average")
        if metric in LABEL_METRICS:
            if y_pred.dtype.kind == "f":
                if y_pred.ndim == 1:
                    y_pred = (y_pred >= opts.get("threshold", 0.5)).astype(np.int64)
                else:
                    y_pred = y_pred.argmax(axis=1)
            if metric == "accuracy":
                if avg in (None, "micro"):
                    return float(skm.accuracy_score(y_true, y_pred))
                if avg == "macro":
                    return float(skm.balanced_accuracy_score(y_true, y_pred))
                raise ValueError(f"sklearn has no accuracy with average={avg!r}")
            fn = {"precision": skm.precision_score, "recall": skm.recall_score,
                  "f1": skm.f1_score}[metric]
            kw = {"average": avg} if avg else {}
            return float(fn(y_true, y_pred, zero_division=0, **kw))

        if metric == "auroc":
            if task == "binary":
                return float(skm.roc_auc_score(y_true, y_pred))
            kw = {"average": avg} if avg else {}
            return float(skm.roc_auc_score(
                y_true, y_pred, multi_class=opts.get("multi_class", "raise"),
                labels=list(range(num_classes)), **kw))

        if metric == "average_precision":
            if task == "binary":
                return float(skm.average_precision_score(y_true, y_pred))
            kw = {"average": avg} if avg else {}
            onehot = np.eye(num_classes, dtype=np.int64)[y_true]
            return float(skm.average_precision_score(onehot, y_pred, **kw))

        raise ValueError(metric)  # pragma: no cover


def _tensors(y_true, y_pred):
    import torch

    target = torch.as_tensor(y_true, dtype=torch.long)
    if y_pred.dtype.kind == "f":
        preds = torch.as_tensor(y_pred, dtype=torch.float32)
    else:
        preds = torch.as_tensor(y_pred, dtype=torch.long)
    return target, preds


class TorchmetricsBackend(Backend):
    name = "torchmetrics"
    module = "torchmetrics"

    def compute(self, metric, task, y_true, y_pred, num_classes, opts):
        import torchmetrics.functional.classification as F

        target, preds = _tensors(y_true, y_pred)
        if task == "binary":
            fn = {"accuracy": F.binary_accuracy, "precision": F.binary_precision,
                  "recall": F.binary_recall, "f1": F.binary_f1_score,
                  "auroc": F.binary_auroc, "average_precision": F.binary_average_precision}[metric]
            kw = {}
            if metric in LABEL_METRICS and "threshold" in opts:
                kw["threshold"] = opts["threshold"]
            if metric not in LABEL_METRICS:
                preds = preds.float()
            return float(fn(preds, target, **kw))

        fn = {"accuracy": F.multiclass_accuracy, "precision": F.multiclass_precision,
              "recall": F.multiclass_recall, "f1": F.multiclass_f1_score,
              "auroc": F.multiclass_auroc,
              "average_precision": F.multiclass_average_precision}[metric]
        kw = {"average": opts["average"]} if "average" in opts else {}
        return float(fn(preds, target, num_classes=num_classes, **kw))


class TorchevalBackend(Backend):
    name = "torcheval"
    module = "torcheval"

    def compute(self, metric, task, y_true, y_pred, num_classes, opts):
        import torcheval.metrics.functional as F

        target, preds = _tensors(y_true, y_pred)
        if task == "binary":
            fn = {"accuracy": F.binary_accuracy, "precision": F.binary_precision,
                  "recall": F.binary_recall, "f1": F.binary_f1_score,
                  "auroc": F.binary_auroc, "average_precision": F.binary_auprc}[metric]
            kw = {}
            if metric in LABEL_METRICS and "threshold" in opts:
                kw["threshold"] = opts["threshold"]
            if metric not in LABEL_METRICS:
                preds = preds.float()
            return float(fn(preds, target, **kw))

        fn = {"accuracy": F.multiclass_accuracy, "precision": F.multiclass_precision,
              "recall": F.multiclass_recall, "f1": F.multiclass_f1_score,
              "auroc": F.multiclass_auroc, "average_precision": F.multiclass_auprc}[metric]
        kw = {"average": opts["average"]} if "average" in opts else {}
        return float(fn(preds, target, num_classes=num_classes, **kw))


ALL_BACKENDS: dict[str, Callable[[], Backend]] = {
    "sklearn": SklearnBackend,
    "torchmetrics": TorchmetricsBackend,
    "torcheval": TorchevalBackend,
}


def resolve(names: list[str] | None) -> list[Backend]:
    if names is None:
        backends = [cls() for cls in ALL_BACKENDS.values()]
        return [b for b in backends if b.available()]
    out = []
    for n in names:
        if n not in ALL_BACKENDS:
            raise ValueError(f"unknown backend {n!r}; choose from {', '.join(ALL_BACKENDS)}")
        b = ALL_BACKENDS[n]()
        if not b.available():
            raise ImportError(f"backend {n!r} is not installed (pip install 'metric-diff[torch]')")
        out.append(b)
    return out


def safe_compute(backend: Backend, *args) -> tuple[float | None, str | None]:
    """Run one backend; return (value, None) or (None, error message)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            value = backend.compute(*args)
        except Exception as exc:  # libraries raise many different types
            msg = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            return None, f"{type(exc).__name__}: {msg[:160]}"
    if value != value:  # NaN
        return None, "returned NaN"
    return value, None
