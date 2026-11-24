"""Normalise user inputs (lists, numpy arrays, torch tensors) and describe them."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SETTING_KEYS = ("average", "threshold", "zero_division", "top_k", "include_background", "aggregation")
AVERAGES = ("micro", "macro", "weighted")


@dataclass(frozen=True)
class Settings:
    """Settings the user called the libraries with.

    ``common`` applies to every library; ``per_library`` overrides it for one library,
    e.g. ``{"sklearn": {"average": "weighted"}}``. Keys: ``average``, ``threshold``.
    ``explicit_classes`` is set when the user passed ``num_classes``.
    """

    common: dict = field(default_factory=dict)
    per_library: dict = field(default_factory=dict)
    explicit_classes: bool = False

    def for_backend(self, name: str) -> dict:
        opts = {**self.common, **self.per_library.get(name, {})}
        if self.explicit_classes:
            opts["explicit_classes"] = True
        return opts

    def value(self, name: str, key: str, default=None):
        return self.for_backend(name).get(key, default)

    @property
    def empty(self) -> bool:
        return not (self.common or self.per_library or self.explicit_classes)

    def values(self, key: str) -> set:
        """Every value of ``key`` in play (common or per library)."""
        out = {self.common[key]} if key in self.common else set()
        out |= {s[key] for s in self.per_library.values() if key in s}
        return out

    def describe(self) -> str:
        parts = [", ".join(f"{k}={v!r}" for k, v in self.common.items())] if self.common else []
        for lib, s in self.per_library.items():
            parts.append(f"{lib}: " + ", ".join(f"{k}={v!r}" for k, v in s.items()))
        return "; ".join(parts)


def make_settings(average=None, threshold=None, settings=None, explicit_classes=False,
                  zero_division=None, top_k=None, include_background=None,
                  aggregation=None) -> Settings:
    common = _check_all({"average": average, "threshold": threshold,
                         "zero_division": zero_division, "top_k": top_k,
                         "include_background": include_background, "aggregation": aggregation})
    per = {}
    for lib, kw in (settings or {}).items():
        unknown = set(kw) - set(SETTING_KEYS)
        if unknown:
            raise ValueError(f"unknown setting(s) {sorted(unknown)} for {lib}; supported: {', '.join(SETTING_KEYS)}")
        per[lib] = _check_all(kw)
    return Settings(common, per, explicit_classes)


def _check_all(kw: dict) -> dict:
    checks = {"average": _check_average, "threshold": _check_threshold,
              "zero_division": _check_zero_division, "top_k": _check_top_k,
              "include_background": _check_bool, "aggregation": _check_aggregation}
    return {k: checks[k](v) for k, v in kw.items() if v is not None}


def _check_bool(v):
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return v.lower() == "true"
    if not isinstance(v, (bool, np.bool_)):
        raise ValueError(f"include_background must be True or False (got {v!r})")
    return bool(v)


def _check_aggregation(a):
    if a not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {', '.join(AGGREGATIONS)} (got {a!r})")
    return a


def _check_zero_division(z):
    if isinstance(z, str) and z.lower() == "nan" or (isinstance(z, float) and z != z):
        return "nan"
    if z in (0, 1):
        return int(z)
    raise ValueError(f"zero_division must be 0, 1 or 'nan' (got {z!r})")


def _check_top_k(k):
    if isinstance(k, bool) or int(k) != k or int(k) < 1:
        raise ValueError(f"top_k must be a positive integer (got {k!r})")
    return int(k)


def _check_average(a):
    if a not in AVERAGES:
        raise ValueError(f"average must be one of {', '.join(AVERAGES)} (got {a!r})")
    return a


def _check_threshold(t):
    t = float(t)
    if not 0.0 < t < 1.0:
        raise ValueError(f"threshold must be between 0 and 1 (got {t})")
    return t

CLASSIFICATION_METRICS = ("accuracy", "precision", "recall", "f1", "auroc", "average_precision")
REGRESSION_METRICS = ("r2", "mae", "mse", "rmse", "mape")
SEGMENTATION_METRICS = ("dice", "iou")
METRICS = CLASSIFICATION_METRICS + REGRESSION_METRICS + SEGMENTATION_METRICS
SEGMENTATION_KEYS = ("include_background", "aggregation")
AGGREGATIONS = ("dataset", "per_image")
LABEL_METRICS = ("accuracy", "precision", "recall", "f1")
RANKING_METRICS = ("auroc", "average_precision")

_ALIASES = {
    "r2_score": "r2",
    "r_squared": "r2",
    "mean_absolute_error": "mae",
    "mean_squared_error": "mse",
    "root_mean_squared_error": "rmse",
    "mean_absolute_percentage_error": "mape",
    "dice_score": "dice",
    "jaccard": "iou",
    "mean_iou": "iou",
    "miou": "iou",
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
    settings: Settings = field(default_factory=Settings)
    backends: tuple = ()  # names of the libraries being compared

    @property
    def n_samples(self) -> int:
        return int(self.y_true.shape[0])

    def describe(self) -> str:
        if self.task == "segmentation":
            shape = "x".join(str(s) for s in self.y_true.shape[1:])
            return f"segmentation, {self.n_samples} images of {shape}, {self.num_classes} classes"
        if self.task == "regression":
            outputs = "" if self.y_true.ndim == 1 else f", {self.y_true.shape[1]} outputs"
            return f"regression, {self.n_samples} samples{outputs}"
        kinds = {
            "labels": "predicted labels",
            "probs": "probabilities",
            "logits": "logits (values outside [0, 1])",
        }
        task = self.task if self.task == "binary" else f"multiclass, {self.num_classes} classes"
        return f"{task}, {self.n_samples} samples, predictions look like {kinds[self.pred_kind]}"


def apply_pos_label(y_true: np.ndarray, y_pred: np.ndarray, pos_label):
    """Map a binary problem to {0, 1} with ``pos_label`` as the positive class."""
    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(f"pos_label is for binary data, but y_true has {len(labels)} classes")
    y_true = (y_true == pos_label).astype(np.int64)
    if y_pred.ndim == 1 and _is_integer_valued(y_pred):
        y_pred = (y_pred == pos_label).astype(np.int64)
    return y_true, y_pred


def prepare(y_true, y_pred, *, pos_label=None, num_classes=None) -> Case:
    yt = to_numpy(y_true)
    yp = to_numpy(y_pred)
    if pos_label is not None:
        if yt.ndim == 2 and yt.shape[1] == 1:
            yt = yt[:, 0]
        if yp.ndim == 2 and yp.shape[1] == 1:
            yp = yp[:, 0]
        yt, yp = apply_pos_label(yt, yp, pos_label)
    case = _prepare(yt, yp)
    if num_classes is not None:
        num_classes = int(num_classes)
        if case.task == "binary" and num_classes != 2:
            if num_classes < 2 or case.pred_kind != "labels":
                raise ValueError(f"num_classes={num_classes} does not fit binary inputs")
            case = Case(case.y_true, case.y_pred, "multiclass", num_classes, "labels")
        elif case.task == "multiclass":
            if case.y_pred.ndim == 2 and num_classes != case.num_classes:
                raise ValueError(f"num_classes={num_classes} but y_pred has {case.num_classes} columns")
            if num_classes < case.num_classes:
                raise ValueError(f"num_classes={num_classes} but the data contains label {case.num_classes - 1}")
            case = Case(case.y_true, case.y_pred, "multiclass", num_classes, case.pred_kind)
    return case


def prepare_segmentation(y_true, y_pred, num_classes=None) -> Case:
    """Label maps: integer arrays of shape (n_images, *spatial), e.g. (N, H, W) or (N, D, H, W)."""
    yt, yp = to_numpy(y_true), to_numpy(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true has shape {yt.shape} but y_pred has shape {yp.shape}; they must match")
    if yt.ndim < 2:
        raise ValueError("segmentation inputs must be label maps of shape (n_images, *spatial), "
                         "e.g. (N, H, W); add a leading axis for a single image")
    if yt.size == 0:
        raise ValueError("inputs are empty")
    if not (_is_integer_valued(yt) and _is_integer_valued(yp)):
        raise ValueError("segmentation inputs must be integer class labels per pixel (argmax your "
                         "probabilities first)")
    yt, yp = yt.astype(np.int64), yp.astype(np.int64)
    if min(yt.min(), yp.min()) < 0:
        raise ValueError("label maps contain negative labels")
    inferred = int(max(yt.max(), yp.max())) + 1
    n = inferred if num_classes is None else int(num_classes)
    if n < inferred:
        raise ValueError(f"num_classes={n} but the label maps contain label {inferred - 1}")
    return Case(yt, yp, "segmentation", max(n, 2), "labels")


def prepare_regression(y_true, y_pred) -> Case:
    yt = to_numpy(y_true).astype(np.float64)
    yp = to_numpy(y_pred).astype(np.float64)
    if yt.ndim == 2 and yt.shape[1] == 1:
        yt = yt[:, 0]
    if yp.ndim == 2 and yp.shape[1] == 1:
        yp = yp[:, 0]
    if yt.ndim not in (1, 2):
        raise ValueError("y_true must be 1-D, or 2-D (n_samples, n_outputs) for multi-output regression")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true has shape {yt.shape} but y_pred has shape {yp.shape}; "
                         "they must match (a (n, 1) vs (n,) mix silently broadcasts in plain torch/numpy)")
    if yt.shape[0] == 0:
        raise ValueError("inputs are empty")
    if not (np.all(np.isfinite(yt)) and np.all(np.isfinite(yp))):
        raise ValueError("inputs contain NaN or infinite values")
    return Case(yt, yp, "regression", 0, "values")


def _prepare(yt: np.ndarray, yp: np.ndarray) -> Case:

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
