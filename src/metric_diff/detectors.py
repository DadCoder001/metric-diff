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
            f"In sklearn you can also pass pos_label={pos}."
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


class ThresholdTie(Detector):
    key = "threshold_tie"
    title = "Scores exactly on the 0.5 threshold"

    def _probs(self, case):
        return case.y_pred if case.pred_kind == "probs" else _sigmoid(case.y_pred)

    def relevant(self, case, metric):
        if not (case.task == "binary" and case.pred_kind != "labels" and metric in LABEL_METRICS):
            return False
        return bool(np.any(self._probs(case) == 0.5))

    def transform(self, y_true, y_pred, case):
        return y_true, (self._probs(case) >= 0.5).astype(np.int64)

    def explain(self, case, metric, backends):
        k = int(np.sum(self._probs(case) == 0.5))
        detail = (
            f"{k} prediction(s) are exactly 0.5. torchmetrics"
            + (" and Keras count" if "keras" in backends else " counts")
            + " a sample as positive when score > 0.5, torcheval when score >= 0.5 "
            "(and so does `y_pred >= 0.5` for sklearn)."
        )
        fix = "Threshold explicitly yourself, y_hat = (scores >= 0.5), and pass labels to every library."
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


class Averaging(Detector):
    key = "average"
    title = "Different default averaging"
    reference = "Lightning-AI/torchmetrics#746, meta-pytorch/torcheval#191"

    def relevant(self, case, metric):
        return case.task == "multiclass"

    def options(self, case, metric):
        if metric == "accuracy":  # every library supports micro (plain) accuracy
            return {"average": "micro"}
        return {"average": "macro", "multi_class": "ovr"}

    def explain(self, case, metric, backends):
        lines = [f"{b}: {DEFAULT_AVERAGE[b][metric]}" for b in backends if b in DEFAULT_AVERAGE]
        detail = "Without `average`, each library averages per-class scores differently: " + "; ".join(lines) + "."
        fix = 'Pass the averaging explicitly to every library, e.g. average="macro" (or "micro").'
        if metric == "accuracy":
            fix += " Macro accuracy in sklearn is balanced_accuracy_score."
        if "keras" in backends and metric in ("precision", "recall"):
            fix += " In Keras use top_k=1 so the predicted class is the argmax."
        if metric == "auroc":
            fix += " sklearn also needs multi_class=\"ovr\"."
        return detail, fix


# Order matters: transforms are applied in this order when combined.
DETECTORS: list[Detector] = [
    PositiveLabel(),
    RowNormalisation(),
    Softmax(),
    SigmoidSaturation(),
    KerasLogits(),
    BinaryLogits(),
    ThresholdTie(),
    KerasF1Threshold(),
    KerasPRSummation(),
    KerasThresholdBins(),
    Averaging(),
]


def notes_for(case: Case, metric: str) -> list[str]:
    """Observations worth showing even when they don't cause a mismatch."""
    notes = []
    if metric in LABEL_METRICS and case.pred_kind != "labels":
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
