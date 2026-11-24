"""Adapters that call each metric library the way a user would.

Every backend receives already-prepared numpy arrays:

* ``y_true``: int64 labels
* ``y_pred``: int64 labels (1-D) or float scores (1-D binary, 2-D multiclass)

``opts`` holds settings that metric-diff aligns while looking for a cause.
An empty ``opts`` means "call the library with its own defaults". Known keys:

* ``average``: averaging for multiclass metrics
* ``threshold``: decision threshold for binary label metrics
* ``multi_class``: sklearn multiclass AUROC strategy
* ``exact_thresholds``: Keras AUC uses every distinct score as a threshold (not 200 bins)
* ``pr_summation``: Keras PR-AUC summation method ("majoring" equals average precision)
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from typing import Callable

import numpy as np

from .inputs import LABEL_METRICS

# What each library does when you don't pass `average` (multiclass).
# Used in explanations; tests check it against the installed versions.
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
    "keras": {
        "accuracy": "micro (CategoricalAccuracy)",
        "precision": "Precision() thresholds every class column at 0.5 like multi-label data, not argmax",
        "recall": "Recall() thresholds every class column at 0.5 like multi-label data, not argmax",
        "f1": "F1Score(average=None) returns one score per class",
        "auroc": "AUC() flattens all classes into one binary problem (micro), using 200 thresholds",
        "average_precision": "AUC(curve='PR') flattens all classes into one binary problem",
    },
}

_TORCH_NAMES = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1Score", "auroc": "AUROC",
}


class Backend:
    name: str = ""
    module: str = ""

    def available(self) -> bool:
        return importlib.util.find_spec(self.module) is not None

    def compute(self, metric: str, task: str, y_true: np.ndarray, y_pred: np.ndarray,
                num_classes: int, opts: dict) -> float:
        raise NotImplementedError

    def stateful(self, metric: str, task: str, num_classes: int, opts: dict):
        """Return an object with ``update(y_true, y_pred)`` and ``compute() -> float``,
        or None if the library has no accumulating metric objects."""
        return None

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


def _torch_kwargs(metric, task, num_classes, opts):
    kw = {}
    if task == "binary":
        if metric in LABEL_METRICS and "threshold" in opts:
            kw["threshold"] = opts["threshold"]
    else:
        kw["num_classes"] = num_classes
        if "average" in opts:
            kw["average"] = opts["average"]
    return kw


class _TorchStateful:
    """Wraps a torchmetrics / torcheval metric object."""

    def __init__(self, obj, metric):
        self.obj, self.metric = obj, metric

    def update(self, y_true, y_pred):
        target, preds = _tensors(y_true, y_pred)
        if self.metric not in LABEL_METRICS:
            preds = preds.float()
        self.obj.update(preds, target)  # both libraries take (preds/input, target)

    def compute(self) -> float:
        return float(self.obj.compute())


class TorchmetricsBackend(Backend):
    name = "torchmetrics"
    module = "torchmetrics"

    def compute(self, metric, task, y_true, y_pred, num_classes, opts):
        import torchmetrics.functional.classification as F

        target, preds = _tensors(y_true, y_pred)
        if metric not in LABEL_METRICS:
            preds = preds.float()
        name = {"f1": "f1_score", "auroc": "auroc"}.get(metric, metric)
        fn = getattr(F, f"{task}_{name}")
        return float(fn(preds, target, **_torch_kwargs(metric, task, num_classes, opts)))

    def stateful(self, metric, task, num_classes, opts):
        import torchmetrics.classification as C

        cls_name = _TORCH_NAMES.get(metric, "AveragePrecision")
        cls = getattr(C, f"{task.capitalize()}{cls_name}")
        return _TorchStateful(cls(**_torch_kwargs(metric, task, num_classes, opts)), metric)


class TorchevalBackend(Backend):
    name = "torcheval"
    module = "torcheval"

    def compute(self, metric, task, y_true, y_pred, num_classes, opts):
        import torcheval.metrics.functional as F

        target, preds = _tensors(y_true, y_pred)
        if metric not in LABEL_METRICS:
            preds = preds.float()
        name = {"f1": "f1_score", "average_precision": "auprc"}.get(metric, metric)
        fn = getattr(F, f"{task}_{name}")
        return float(fn(preds, target, **_torch_kwargs(metric, task, num_classes, opts)))

    def stateful(self, metric, task, num_classes, opts):
        import torcheval.metrics as M

        cls_name = {"average_precision": "AUPRC"}.get(metric, _TORCH_NAMES.get(metric))
        cls = getattr(M, f"{task.capitalize()}{cls_name}")
        return _TorchStateful(cls(**_torch_kwargs(metric, task, num_classes, opts)), metric)


_MAX_KERAS_THRESHOLDS = 5000
_KERAS_CHUNK = 4096


def _ensure_keras_backend() -> None:
    """Use Keras' torch backend when TensorFlow isn't installed, without overriding a user choice."""
    if "keras" in sys.modules or "KERAS_BACKEND" in os.environ:
        return
    if importlib.util.find_spec("tensorflow") is None and importlib.util.find_spec("torch") is not None:
        os.environ["KERAS_BACKEND"] = "torch"


class _KerasStateful:
    def __init__(self, objs, metric, task, num_classes, opts):
        self.objs, self.metric, self.task = objs, metric, task
        self.num_classes, self.opts = num_classes, opts

    def _arrays(self, y_true, y_pred):
        if self.task == "binary":
            yt = y_true.astype(np.float32)
            yp = y_pred.astype(np.float32)
            if self.metric == "f1":  # F1Score needs (n, 1)
                yt, yp = yt[:, None], yp[:, None]
            return yt, yp
        yt = np.eye(self.num_classes, dtype=np.float32)[y_true]
        yp = (np.eye(self.num_classes, dtype=np.float32)[y_pred] if y_pred.ndim == 1
              else y_pred.astype(np.float32))
        return yt, yp

    def update(self, y_true, y_pred):
        yt, yp = self._arrays(y_true, y_pred)
        for start in range(0, len(yt), _KERAS_CHUNK):  # AUC memory is thresholds x samples
            sl = slice(start, start + _KERAS_CHUNK)
            for o in self.objs:
                o.update_state(yt[sl], yp[sl])

    def compute(self) -> float:
        import keras

        values = [np.asarray(keras.ops.convert_to_numpy(o.result()), dtype=np.float64) for o in self.objs]
        if len(values) > 1:  # per-class objects for macro precision/recall
            return float(np.mean([v.item() for v in values]))
        v = values[0]
        if v.size != 1:
            raise ValueError(f"returned {v.size} values, one per class (average=None by default)")
        return float(v.item())


class KerasBackend(Backend):
    """Keras 3 metrics (any Keras backend; torch is used when TensorFlow isn't installed).

    Keras metric objects are stateful, so ``compute`` is a single ``update_state`` + ``result``.
    Multiclass y_true is one-hot encoded, as Keras expects.
    """

    name = "keras"
    module = "keras"

    def available(self) -> bool:
        if importlib.util.find_spec("keras") is None:
            return False
        return any(importlib.util.find_spec(m) is not None for m in ("torch", "tensorflow", "jax"))

    def compute(self, metric, task, y_true, y_pred, num_classes, opts):
        s = self.stateful(metric, task, num_classes, opts, scores=y_pred)
        s.update(y_true, y_pred)
        return s.compute()

    def stateful(self, metric, task, num_classes, opts, scores=None):
        _ensure_keras_backend()
        import keras

        M = keras.metrics
        avg = opts.get("average")
        thr = opts.get("threshold")

        if metric in ("auroc", "average_precision"):
            kw = {}
            if metric == "average_precision":
                kw["curve"] = "PR"
                if "pr_summation" in opts:
                    kw["summation_method"] = opts["pr_summation"]
            if opts.get("exact_thresholds") and scores is not None:
                t = np.unique(np.asarray(scores, dtype=np.float32))
                if len(t) > _MAX_KERAS_THRESHOLDS:  # keep memory bounded on big data
                    t = np.unique(np.quantile(t, np.linspace(0, 1, _MAX_KERAS_THRESHOLDS)).astype(np.float32))
                t = t[(t > 0) & (t < 1)]
                kw["thresholds"] = t.tolist()
            if task == "multiclass" and avg is not None:
                if avg != "macro":
                    raise ValueError(f"Keras AUC has no average={avg!r}")
                kw.update(multi_label=True, num_labels=num_classes)
            objs = [M.AUC(**kw)]
        elif metric == "accuracy":
            if task == "binary":
                objs = [M.BinaryAccuracy(threshold=0.5 if thr is None else thr)]
            elif avg in (None, "micro"):
                objs = [M.CategoricalAccuracy()]
            else:
                raise ValueError(f"Keras has no accuracy with average={avg!r}")
        elif metric in ("precision", "recall"):
            cls = M.Precision if metric == "precision" else M.Recall
            if task == "binary":
                objs = [cls(thresholds=thr)]
            elif avg is None:
                objs = [cls()]
            elif avg == "micro":
                objs = [cls(top_k=1)]
            elif avg == "macro":
                objs = [cls(top_k=1, class_id=k) for k in range(num_classes)]
            else:
                raise ValueError(f"metric-diff does not map Keras {metric} average={avg!r}")
        else:  # f1
            if task == "binary":
                objs = [M.F1Score(threshold=thr)]
            else:
                objs = [M.F1Score(average=avg)]
        return _KerasStateful(objs, metric, task, num_classes, opts)


ALL_BACKENDS: dict[str, Callable[[], Backend]] = {
    "sklearn": SklearnBackend,
    "torchmetrics": TorchmetricsBackend,
    "torcheval": TorchevalBackend,
    "keras": KerasBackend,
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
            extra = "keras" if n == "keras" else "torch"
            raise ImportError(f"backend {n!r} is not installed (pip install 'metric-diff[{extra}]')")
        out.append(b)
    return out


def _short_error(exc: Exception) -> str:
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return f"{type(exc).__name__}: {msg[:160]}"


def safe_call(fn, *args) -> tuple[float | None, str | None]:
    """Run fn(*args); return (value, None) or (None, short error message)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            value = fn(*args)
        except Exception as exc:  # libraries raise many different types
            return None, _short_error(exc)
    if value != value:  # NaN
        return None, "returned NaN"
    return value, None


def safe_compute(backend: Backend, *args) -> tuple[float | None, str | None]:
    return safe_call(backend.compute, *args)
