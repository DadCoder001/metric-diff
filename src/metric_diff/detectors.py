"""Known causes of metric disagreement.

Each detector knows how to *align* one setting across all libraries (by transforming the
inputs and/or passing an explicit option). metric-diff re-runs every library with the
alignment applied: a cause is only reported as **confirmed** when the libraries then agree.
"""

from __future__ import annotations

import numpy as np

from .backends import DEFAULT_AVERAGE
from .inputs import LABEL_METRICS, RANKING_METRICS, Case


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64) - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fmt_set(values) -> str:
    return "{" + ", ".join(str(int(v)) for v in values) + "}"


class Detector:
    key = ""
    title = ""
    reference: str | None = None
    opts: dict = {}
    requires: str | None = None  # only relevant when this backend is being compared

    def relevant(self, case: Case, metric: str) -> bool:
        raise NotImplementedError

    def options(self, case: Case, metric: str) -> dict:
        return dict(self.opts)

    def transform(self, y_true, y_pred, case: Case):
        return y_true, y_pred

    def explain(self, case: Case, metric: str, backends: list[str]) -> tuple[str, str]:
        raise NotImplementedError


class PositiveLabel(Detector):
    key = "pos_label"
    title = "Binary labels are not {0, 1}"
    reference = "huggingface/evaluate#869"

    def _labels(self, case):
        labels = np.unique(case.y_true)
        if case.pred_kind == "labels":
            labels = np.union1d(labels, case.y_pred)
        return labels

    def relevant(self, case, metric):
        return case.task == "binary" and not set(self._labels(case).tolist()) <= {0, 1}

    def transform(self, y_true, y_pred, case):
        pos = int(self._labels(case).max())
        y_true = (y_true == pos).astype(np.int64)
        if y_pred.dtype.kind in "iu":
            y_pred = (y_pred == pos).astype(np.int64)
        return y_true, y_pred

    def explain(self, case, metric, backends):
        labels = self._labels(case)
        pos = int(labels.max())
        detail = (
            f"The labels are {_fmt_set(labels)}. sklearn treats 1 as the positive class by "
            "default (pos_label=1), torchmetrics only accepts {0, 1} and raises, and torcheval"
            + (" and Keras do" if "keras" in backends else " does")
            + " not validate the labels and can return impossible values such as F1 > 1. "
            f"To compare like with like, metric-diff mapped {pos} -> 1 and everything else -> 0."
        )
        fix = (
            f"Map the labels to {{0, 1}} before computing any metric: y = (y == {pos}).astype(int). "
            f"In sklearn you can also pass pos_label={pos}; in metric-diff, compare(..., pos_label={pos})."
        )
        return detail, fix


class RowNormalisation(Detector):
    key = "row_sum"
    title = "Class probabilities do not sum to 1"

    def relevant(self, case, metric):
        if not (case.task == "multiclass" and case.pred_kind == "probs" and metric in RANKING_METRICS):
            return False
        return not np.allclose(case.y_pred.sum(axis=1), 1.0, atol=1e-6)

    def transform(self, y_true, y_pred, case):
        s = y_pred.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return y_true, y_pred / s

    def explain(self, case, metric, backends):
        detail = (
            "The rows of y_pred are in [0, 1] but do not sum to 1. sklearn requires probabilities "
            "that sum to 1 for multiclass AUROC and raises; torch libraries accept them as-is."
        )
        fix = "Normalise each row (p / p.sum(axis=1, keepdims=True)) or use softmax outputs."
        return detail, fix


class Softmax(Detector):
    key = "softmax"
    title = "Multiclass logits vs probabilities"

    def relevant(self, case, metric):
        return case.task == "multiclass" and case.pred_kind == "logits" and metric in RANKING_METRICS

    def transform(self, y_true, y_pred, case):
        return y_true, _softmax(y_pred)

    def explain(self, case, metric, backends):
        detail = (
            "The scores are logits (values outside [0, 1]). torchmetrics applies softmax "
            "automatically, torcheval ranks the raw logits per class (which changes one-vs-rest "
            "rankings), and sklearn requires probabilities that sum to 1 and raises."
        )
        fix = "Apply softmax yourself before every library: torch.softmax(logits, dim=1)."
        return detail, fix


class BinaryLogits(Detector):
    key = "logits"
    title = "Logits vs probabilities"
    reference = "Lightning-AI/torchmetrics#2329"

    def relevant(self, case, metric):
        return case.task == "binary" and case.pred_kind == "logits" and metric in LABEL_METRICS

    def transform(self, y_true, y_pred, case):
        return y_true, _sigmoid(y_pred)

    def explain(self, case, metric, backends):
        detail = (
            "Predictions contain values outside [0, 1], so they look like logits. torchmetrics "
            "detects this and applies sigmoid before thresholding at 0.5. torcheval"
            + (", Keras" if "keras" in backends else "")
            + " and the usual `y_pred >= 0.5` for sklearn threshold the raw logits at 0.5, which "
            "is a probability of about 0.62."
        )
        fix = (
            "Convert to probabilities yourself before every library (torch.sigmoid(logits)), "
            "or threshold logits at 0: y_hat = (logits > 0)."
        )
        return detail, fix


class SigmoidSaturation(Detector):
    key = "saturation"
    title = "Large logits saturate when squashed to probabilities"
    reference = "Lightning-AI/torchmetrics#2819"

    def _collapsed(self, case) -> tuple[int, int]:
        raw = len(np.unique(case.y_pred))
        with np.errstate(over="ignore"):
            squashed = 1.0 / (1.0 + np.exp(-case.y_pred.astype(np.float32)))
        return raw, len(np.unique(squashed.astype(np.float32)))

    def relevant(self, case, metric):
        if not (case.task == "binary" and case.pred_kind == "logits" and metric in RANKING_METRICS):
            return False
        raw, squashed = self._collapsed(case)
        return squashed < raw

    def transform(self, y_true, y_pred, case):
        from scipy.stats import rankdata

        # Monotone rescale into (0, 1]: ranking metrics are unchanged, nothing saturates.
        return y_true, rankdata(y_pred, method="average") / len(y_pred)

    def explain(self, case, metric, backends):
        raw, squashed = self._collapsed(case)
        detail = (
            f"torchmetrics applies sigmoid to logits in float32: the {raw} distinct scores collapse "
            f"to {squashed} distinct value{'s' if squashed != 1 else ''} (large logits all become 1.0), "
            "so their ranking is lost "
            f"and {metric} drifts toward 0.5. sklearn and torcheval rank the raw scores."
        )
        if "keras" in backends:
            detail += (" Keras AUC also assumes probabilities and clips values outside [0, 1] "
                       "(use from_logits=True).")
        fix = (
            "For ranking metrics pass the raw scores to sklearn, or rescale monotonically before "
            "torchmetrics, e.g. torch.sigmoid((logits - logits.mean()).double())."
        )
        return detail, fix


def _threshold(case: Case) -> float:
    return case.settings.common.get("threshold", 0.5)


def _thresholds_by_backend(case: Case) -> dict[str, float]:
    return {b: case.settings.value(b, "threshold", 0.5) for b in case.backends}


class ThresholdMismatch(Detector):
    key = "threshold"
    title = "Different decision thresholds"

    def relevant(self, case, metric):
        if not (case.task == "binary" and case.pred_kind != "labels" and metric in LABEL_METRICS):
            return False
        return len(set(_thresholds_by_backend(case).values())) > 1

    def options(self, case, metric):
        return {"threshold": _threshold(case)}

    def explain(self, case, metric, backends):
        per = "; ".join(f"{b}: {t}" for b, t in _thresholds_by_backend(case).items())
        detail = f"The libraries turn scores into labels at different thresholds ({per})."
        fix = ("Use the same threshold everywhere (threshold= in torchmetrics/torcheval/Keras, "
               "y_pred >= t for sklearn).")
        return detail, fix


class ThresholdTie(Detector):
    key = "threshold_tie"
    title = "Scores exactly on the decision threshold"

    def _probs(self, case):
        return case.y_pred if case.pred_kind == "probs" else _sigmoid(case.y_pred)

    def relevant(self, case, metric):
        if not (case.task == "binary" and case.pred_kind != "labels" and metric in LABEL_METRICS):
            return False
        return bool(np.any(self._probs(case) == _threshold(case)))

    def transform(self, y_true, y_pred, case):
        return y_true, (self._probs(case) >= _threshold(case)).astype(np.int64)

    def explain(self, case, metric, backends):
        t = _threshold(case)
        k = int(np.sum(self._probs(case) == t))
        detail = (
            f"{k} prediction(s) are exactly {t}. torchmetrics"
            + (" and Keras count" if "keras" in backends else " counts")
            + f" a sample as positive when score > {t}, torcheval when score >= {t} "
            f"(and so does `y_pred >= {t}` for sklearn)."
        )
        fix = f"Threshold explicitly yourself, y_hat = (scores >= {t}), and pass labels to every library."
        return detail, fix


class BinaryAverage(Detector):
    key = "binary_average"
    title = "sklearn `average` on binary data"
    requires = "sklearn"

    def relevant(self, case, metric):
        return (case.task == "binary" and metric in ("precision", "recall", "f1")
                and case.settings.value("sklearn", "average") is not None)

    def options(self, case, metric):
        return {"average": None}

    def explain(self, case, metric, backends):
        avg = case.settings.value("sklearn", "average")
        extra = " (micro = plain accuracy)" if avg == "micro" else ""
        detail = (f"With average={avg!r} on binary data, sklearn scores both classes (0 and 1) and "
                  f"averages them{extra}. torchmetrics, torcheval and Keras binary metrics score "
                  "only the positive class.")
        fix = ("For binary data leave sklearn's average at its default 'binary' (choose the positive "
               "class with pos_label), or use the multiclass metrics in every library if you want "
               "both classes averaged.")
        return detail, fix


class AbsentClasses(Detector):
    key = "absent_classes"
    title = "Classes that never occur"

    def _present(self, case):
        pred = case.y_pred if case.y_pred.ndim == 1 else case.y_pred.argmax(axis=1)
        return np.union1d(case.y_true, pred)

    def relevant(self, case, metric):
        return (case.task == "multiclass" and metric in LABEL_METRICS and metric != "accuracy"
                and case.settings.explicit_classes and len(self._present(case)) < case.num_classes)

    def transform(self, y_true, y_pred, case):
        present = self._present(case)
        index = np.full(case.num_classes, -1, dtype=np.int64)
        index[present] = np.arange(len(present))
        y_true = index[y_true]
        y_pred = index[y_pred] if y_pred.ndim == 1 else y_pred[:, present]
        return y_true, y_pred

    def options(self, case, metric):
        return {"num_classes": len(self._present(case))}

    def explain(self, case, metric, backends):
        absent = sorted(set(range(case.num_classes)) - set(self._present(case).tolist()))
        counted = ["sklearn (labels=range(num_classes))"] + (["Keras"] if "keras" in backends else [])
        detail = (f"num_classes={case.num_classes}, but class(es) {absent} appear in neither y_true "
                  f"nor y_pred. {' and '.join(counted)} count them as 0 in the average; torchmetrics "
                  "and torcheval leave them out.")
        fix = ("Decide whether unseen classes should count. To leave them out, pass only the classes "
               "that occur (sklearn labels=[...]); to count them, compute per-class scores "
               "(average=None) and average them yourself.")
        return detail, fix


class KerasLogits(Detector):
    key = "keras_logits"
    title = "Keras AUC expects probabilities"
    requires = "keras"

    def relevant(self, case, metric):
        return (case.task == "binary" and case.pred_kind == "logits" and metric in RANKING_METRICS
                and not SigmoidSaturation().relevant(case, metric))

    def transform(self, y_true, y_pred, case):
        from scipy.stats import rankdata

        return y_true, rankdata(y_pred, method="average") / len(y_pred)

    def explain(self, case, metric, backends):
        detail = ("The scores are logits (values outside [0, 1]). keras.metrics.AUC treats its "
                  "input as probabilities, so values outside [0, 1] are clipped and their ranking "
                  "is lost. sklearn, torchmetrics and torcheval handle raw scores.")
        fix = "Pass from_logits=True: keras.metrics.AUC(from_logits=True), or apply sigmoid first."
        return detail, fix


class KerasThresholdBins(Detector):
    key = "keras_bins"
    title = "Keras approximates AUC with 200 thresholds"
    requires = "keras"
    opts = {"exact_thresholds": True}

    def relevant(self, case, metric):
        return metric in RANKING_METRICS

    def explain(self, case, metric, backends):
        detail = ("keras.metrics.AUC estimates the curve from num_thresholds=200 evenly spaced "
                  "thresholds; the other libraries use every distinct score, so Keras is off by "
                  "a small, data-dependent amount.")
        fix = ("Raise num_thresholds (e.g. AUC(num_thresholds=2000)), pass the sorted unique scores "
               "as thresholds=..., or report the final number with sklearn.")
        return detail, fix


class KerasPRSummation(Detector):
    key = "keras_pr"
    title = "Keras PR-AUC is not average precision"
    requires = "keras"
    opts = {"exact_thresholds": True, "pr_summation": "majoring"}

    def relevant(self, case, metric):
        return metric == "average_precision"

    def explain(self, case, metric, backends):
        detail = ("keras.metrics.AUC(curve='PR') integrates an interpolated precision-recall curve "
                  "(summation_method='interpolation', 200 thresholds). sklearn, torchmetrics and "
                  "torcheval compute average precision, a step-wise sum, which is a different number.")
        fix = ("Use AUC(curve='PR', summation_method='majoring', thresholds=<sorted unique scores>) "
               "to match average precision, or compute it with sklearn.")
        return detail, fix


class KerasF1Threshold(Detector):
    key = "keras_f1_threshold"
    title = "Keras F1Score has no threshold by default"
    requires = "keras"
    opts = {"threshold": 0.5}

    def relevant(self, case, metric):
        return case.task == "binary" and metric == "f1"

    def explain(self, case, metric, backends):
        detail = ("keras.metrics.F1Score(threshold=None) assumes one column per class and marks "
                  "the highest column of each sample as positive. With a single binary column, "
                  "every sample counts as positive.")
        fix = "Pass a threshold: keras.metrics.F1Score(threshold=0.5)."
        return detail, fix


class MapePercent(Detector):
    key = "mape_percent"
    title = "Keras MAPE is a percentage"
    requires = "keras"
    opts = {"mape_fraction": True}

    def relevant(self, case, metric):
        return metric == "mape"

    def explain(self, case, metric, backends):
        detail = ("keras.metrics.MeanAbsolutePercentageError returns a percentage (e.g. 28.0); "
                  "sklearn and torchmetrics return a fraction (0.28).")
        fix = "Divide the Keras value by 100 (or multiply the others by 100) before comparing."
        return detail, fix


class MapeZeroTarget(Detector):
    key = "mape_zero_target"
    title = "MAPE with zero targets"

    def _zeros(self, case):
        yt = case.y_true if case.y_true.ndim == 1 else case.y_true.reshape(len(case.y_true), -1)
        return (yt == 0) if yt.ndim == 1 else (yt == 0).any(axis=1)

    def relevant(self, case, metric):
        return metric == "mape" and bool(self._zeros(case).any())

    def transform(self, y_true, y_pred, case):
        keep = ~self._zeros(case)
        return y_true[keep], y_pred[keep]

    def explain(self, case, metric, backends):
        n = int(self._zeros(case).sum())
        detail = (f"{n} target(s) are exactly 0, so the percentage error divides by zero. Each library "
                  "divides by max(|y|, eps) with a different eps (sklearn about 2.2e-16, torchmetrics "
                  "1.17e-6, Keras 1e-7), so the result is huge and different in every library.")
        fix = ("MAPE is undefined when targets can be 0. Drop those samples, or use a metric that "
               "handles zeros, such as WAPE (sum|y - p| / sum|y|) or symmetric MAPE.")
        return detail, fix


class R2ConstantTarget(Detector):
    key = "r2_constant_target"
    title = "R2 with a constant target"
    opts = {"r2_force_finite": True}

    def relevant(self, case, metric):
        if metric != "r2" or not ({"torcheval", "evaluate"} & set(case.backends)):
            return False
        yt = case.y_true if case.y_true.ndim == 2 else case.y_true[:, None]
        return bool(np.any(yt.max(axis=0) == yt.min(axis=0)))

    def explain(self, case, metric, backends):
        inf_libs = " and ".join(b for b in ("torcheval", "evaluate") if b in backends)
        detail = (f"y_true has zero variance, so R2 = 1 - SSE / 0. {inf_libs} return(s) -inf (torcheval "
                  "returns NaN for a perfect prediction); sklearn (force_finite=True), torchmetrics "
                  "and Keras return 0.0 (1.0 if perfect).")
        fix = ("R2 is not meaningful for a constant target: report MSE/MAE instead, or evaluate on a "
               "split where the target varies.")
        return detail, fix


class EvaluateR2Rounding(Detector):
    key = "evaluate_r2_rounding"
    title = "evaluate rounds R2 to 3 decimals"
    requires = "evaluate"
    opts = {"round_digits": 3}

    def relevant(self, case, metric):
        return metric == "r2"

    def explain(self, case, metric, backends):
        detail = ("evaluate's r_squared metric rounds its result to 3 decimal places (e.g. 0.959 "
                  "instead of 0.9595); the other libraries return full precision.")
        fix = "Use sklearn.metrics.r2_score when you need more than 3 decimals."
        return detail, fix


class Float32Precision(Detector):
    key = "float32_precision"
    title = "Float32 rounding on large values"
    reference = "Lightning-AI/torchmetrics#3077"

    def _ratio(self, case):
        yt = case.y_true
        spread = max(float(np.std(yt)), float(np.mean(np.abs(case.y_pred - yt))), 1e-300)
        return float(np.max(np.abs(yt))) / spread

    def relevant(self, case, metric):
        return case.task == "regression" and metric != "mape" and self._ratio(case) > 100

    def transform(self, y_true, y_pred, case):
        # Shifting both by the same constant leaves R2, MAE, MSE and RMSE unchanged mathematically.
        c = y_true.mean(axis=0)
        return y_true - c, y_pred - c

    def explain(self, case, metric, backends):
        f32 = [b for b in backends if b != "sklearn"]
        detail = (f"The values are large compared with their spread (max |y| is about {self._ratio(case):.0f}x "
                  f"the spread). {', '.join(f32)} compute in float32, which keeps about "
                  "7 significant digits, so the small differences that decide the metric are lost to "
                  "rounding. sklearn computes in float64.")
        fix = ("Compute in float64 (tensor.double() / dtype='float64'), or subtract a constant such as "
               "the mean of y_true from both y_true and y_pred first; R2, MAE, MSE and RMSE are unchanged.")
        return detail, fix


class ZeroDivisionMismatch(Detector):
    key = "zero_division"
    title = "Different zero_division"

    def _per_backend(self, case):
        return {b: case.settings.value(b, "zero_division", 0) for b in case.backends}

    def relevant(self, case, metric):
        return metric in ("precision", "recall", "f1") and len(set(self._per_backend(case).values())) > 1

    def options(self, case, metric):
        return {"zero_division": case.settings.common.get("zero_division", 0)}

    def explain(self, case, metric, backends):
        per = "; ".join(f"{b}: {v}" for b, v in self._per_backend(case).items())
        detail = (f"When a class has no predicted samples (precision) or no true samples (recall), "
                  f"{metric} is 0/0, and each library fills it with its zero_division value ({per}).")
        fix = ("Use the same zero_division everywhere. torcheval and Keras have no option and always "
               "use 0.")
        return detail, fix


class TopKTies(Detector):
    key = "top_k_ties"
    title = "Tied scores at the top-k cut-off"

    def _k(self, case):
        return case.settings.common.get("top_k") or max(case.settings.values("top_k"))

    def _tied_rows(self, case):
        k = self._k(case)
        s = -np.sort(-case.y_pred, axis=1)  # descending
        return s[:, k - 1] == s[:, k] if k < case.num_classes else np.zeros(len(s), bool)

    def relevant(self, case, metric):
        return (metric == "accuracy" and case.task == "multiclass" and case.y_pred.ndim == 2
                and bool(case.settings.values("top_k")) and bool(self._tied_rows(case).any()))

    def transform(self, y_true, y_pred, case):
        # Break ties the way torchmetrics does (lower class index first), so all libraries agree.
        values = np.unique(y_pred)
        gap = np.diff(values).min() if len(values) > 1 else 1.0
        eps = min(1e-6, gap / (case.num_classes + 1))
        return y_true, y_pred - eps * np.arange(case.num_classes)

    def explain(self, case, metric, backends):
        k = self._k(case)
        n = int(self._tied_rows(case).sum())
        others = [b for b in backends if b != "torchmetrics"]
        detail = (f"{n} sample(s) have tied scores at the top-{k} cut-off. torchmetrics keeps only k "
                  f"classes (the lowest class indices win the tie), while {' and '.join(others)} count the "
                  "sample as correct if the true class is among the tied scores.")
        fix = ("Top-k accuracy is ambiguous for tied scores: break ties before computing (for example "
               "add tiny noise or use higher-precision scores) so every library ranks the same way.")
        return detail, fix


class SklearnMultiClass(Detector):
    key = "sklearn_multi_class"
    title = "sklearn multiclass AUROC needs multi_class"
    requires = "sklearn"
    opts = {"multi_class": "ovr"}

    def relevant(self, case, metric):
        # When averaging differs too, the Averaging detector already sets multi_class.
        return (case.task == "multiclass" and metric == "auroc"
                and not Averaging().relevant(case, metric))

    def explain(self, case, metric, backends):
        detail = ("sklearn's roc_auc_score raises on multiclass data unless you choose "
                  "multi_class='ovr' (one-vs-rest, what torchmetrics, torcheval and Keras do) or 'ovo'.")
        fix = "Pass multi_class='ovr' to roc_auc_score."
        return detail, fix


class Averaging(Detector):
    key = "average"
    title = "Different averaging"
    reference = "Lightning-AI/torchmetrics#746, meta-pytorch/torcheval#191"

    def _effective(self, case):
        return {b: case.settings.value(b, "average") for b in case.backends}

    def relevant(self, case, metric):
        if case.task != "multiclass":
            return False
        values = set(self._effective(case).values())
        return len(values) > 1 or values == {None}

    def options(self, case, metric):
        common = case.settings.common.get("average")
        if common is not None:
            target = common
        else:  # every library supports micro accuracy, and macro for the rest
            target = "micro" if metric == "accuracy" else "macro"
        return {"average": target, "multi_class": "ovr"}

    def explain(self, case, metric, backends):
        lines = []
        for b, v in self._effective(case).items():
            if v is not None:
                lines.append(f"{b}: average={v!r} (your setting)")
            elif b in DEFAULT_AVERAGE:
                lines.append(f"{b}: {DEFAULT_AVERAGE[b][metric]}")
        detail = "The libraries average per-class scores differently: " + "; ".join(lines) + "."
        fix = 'Pass the same averaging explicitly to every library, e.g. average="macro" (or "micro").'
        if metric == "accuracy":
            fix += " Macro accuracy in sklearn is balanced_accuracy_score."
        if "keras" in backends and metric in ("precision", "recall"):
            fix += " In Keras use top_k=1 so the predicted class is the argmax."
        if metric == "auroc":
            fix += ' sklearn also needs multi_class="ovr".'
        return detail, fix


# Order matters: transforms are applied in this order when combined.
DETECTORS: list[Detector] = [
    PositiveLabel(),
    RowNormalisation(),
    Softmax(),
    SigmoidSaturation(),
    KerasLogits(),
    BinaryLogits(),
    ThresholdMismatch(),
    ThresholdTie(),
    KerasF1Threshold(),
    KerasPRSummation(),
    KerasThresholdBins(),
    BinaryAverage(),
    AbsentClasses(),
    MapeZeroTarget(),
    MapePercent(),
    R2ConstantTarget(),
    EvaluateR2Rounding(),
    Float32Precision(),
    ZeroDivisionMismatch(),
    TopKTies(),
    SklearnMultiClass(),
    Averaging(),
]


def notes_for(case: Case, metric: str) -> list[str]:
    """Observations worth showing even when they don't cause a mismatch."""
    notes = []
    if case.task == "regression":
        if metric == "r2" and case.n_samples < 2:
            notes.append("R2 needs at least 2 samples.")
        return notes
    if metric in LABEL_METRICS and case.pred_kind != "labels" and not case.settings.values("top_k"):
        notes.append("sklearn has no threshold argument: metric-diff used `y_pred >= 0.5` (binary) "
                     "or argmax (multiclass), as most users do.")
    if case.task == "multiclass":
        missing = sorted(set(range(case.num_classes)) - set(np.unique(case.y_true).tolist()))
        if missing:
            notes.append(f"Class(es) {missing} never appear in y_true; macro averages depend on "
                         "whether a library counts absent classes.")
    if metric in RANKING_METRICS and len(np.unique(case.y_true)) < 2:
        notes.append(f"y_true contains a single class, so {metric} is undefined.")
    return notes
