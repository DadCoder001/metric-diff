"""compare(): run every library, and when they disagree, find and confirm the cause."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from . import backends as _backends
from .detectors import DETECTORS, Detector, notes_for
from .inputs import LABEL_METRICS, Case, normalise_metric, prepare


@dataclass
class Result:
    value: float | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class Cause:
    key: str
    title: str
    detail: str
    fix: str
    reference: str | None = None


@dataclass
class Report:
    metric: str
    case: Case
    versions: dict[str, str]
    results: dict[str, Result]
    status: str  # "agree" | "mismatch" | "single"
    causes: list[Cause] = field(default_factory=list)
    confirmed: bool = False
    aligned: dict[str, Result] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def agree(self) -> bool:
        return self.status == "agree"

    def to_dict(self) -> dict:
        def res(d):
            return {k: {"value": r.value, "error": r.error} for k, r in d.items()}

        return {
            "metric": self.metric,
            "task": self.case.task,
            "num_classes": self.case.num_classes,
            "n_samples": self.case.n_samples,
            "pred_kind": self.case.pred_kind,
            "versions": self.versions,
            "status": self.status,
            "results": res(self.results),
            "confirmed": self.confirmed,
            "causes": [c.__dict__ for c in self.causes],
            "aligned": res(self.aligned),
            "notes": self.notes,
        }

    def __str__(self) -> str:
        from .report import render

        return render(self)


def _close(values: list[float], rtol: float, atol: float) -> bool:
    ref = values[0]
    return all(np.isclose(v, ref, rtol=rtol, atol=atol) for v in values[1:])


def _status(results: dict[str, Result], rtol: float, atol: float) -> str:
    if len(results) < 2:
        return "single"
    if not all(r.ok for r in results.values()):
        return "mismatch"
    return "agree" if _close([r.value for r in results.values()], rtol, atol) else "mismatch"


def _final_inputs(metric: str, case: Case, y_true, y_pred):
    """Shape conversions every library needs (not a 'cause', just plumbing)."""
    if metric not in LABEL_METRICS:
        if case.task == "multiclass" and y_pred.ndim == 1:
            y_pred = np.eye(case.num_classes, dtype=np.float64)[y_pred]
        else:
            y_pred = y_pred.astype(np.float64)
    return y_true, y_pred


def _run(bks, metric, case, detectors: tuple[Detector, ...] = ()) -> dict[str, Result]:
    y_true, y_pred = case.y_true, case.y_pred
    opts: dict = {}
    for d in detectors:
        y_true, y_pred = d.transform(y_true, y_pred, case)
        opts.update(d.options(case, metric))
    y_true, y_pred = _final_inputs(metric, case, y_true, y_pred)
    out = {}
    for b in bks:
        value, err = _backends.safe_compute(b, metric, case.task, y_true, y_pred, case.num_classes, opts)
        out[b.name] = Result(value, err)
    return out


def compare(metric: str, y_true, y_pred, *, backends: list[str] | None = None,
            rtol: float = 1e-4, atol: float = 1e-6) -> Report:
    """Compute ``metric`` with every installed library and explain any disagreement.

    Parameters
    ----------
    metric: one of accuracy, precision, recall, f1, auroc, average_precision (aliases accepted).
    y_true: 1-D integer labels.
    y_pred: 1-D labels or binary scores, or 2-D (n_samples, n_classes) multiclass scores.
        Lists, numpy arrays and torch tensors are accepted.
    backends: subset of ["sklearn", "torchmetrics", "torcheval"]; default = all installed.
    """
    metric = normalise_metric(metric)
    case = prepare(y_true, y_pred)
    bks = _backends.resolve(backends)
    names = [b.name for b in bks]

    results = _run(bks, metric, case)
    status = _status(results, rtol, atol)
    report = Report(metric, case, {b.name: b.version() for b in bks}, results, status,
                    notes=notes_for(case, metric))
    if status != "mismatch":
        return report

    relevant = [d for d in DETECTORS
                if (d.requires is None or d.requires in names) and d.relevant(case, metric)]
    for size in range(1, len(relevant) + 1):
        for combo in combinations(relevant, size):
            aligned = _run(bks, metric, case, combo)
            if _status(aligned, rtol, atol) == "agree":
                report.confirmed = True
                report.aligned = aligned
                report.causes = [_cause(d, case, metric, names) for d in combo]
                return report

    # Nothing reproduced agreement: list what looks suspicious, clearly marked as unconfirmed.
    report.causes = [_cause(d, case, metric, names) for d in relevant]
    return report


def _cause(d: Detector, case, metric, names) -> Cause:
    detail, fix = d.explain(case, metric, names)
    return Cause(d.key, d.title, detail, fix, d.reference)
