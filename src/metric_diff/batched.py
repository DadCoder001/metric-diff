"""compare_batched(): is the number you logged per batch the number for your dataset?

Three ways people end up with an "epoch" metric:

1. Compute it once on the whole dataset (the reference).
2. Accumulate with a metric object: ``update()`` per batch, ``compute()`` at the end.
3. Compute it per batch and average the batch values (what naive loops, Keras 2 and
   ``self.log(value, on_epoch=True)`` in Lightning do).

(1) and (2) should match exactly; (3) generally does not for non-linear metrics such as
F1, precision, recall or AUROC. This module measures all three and explains the gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import backends as _backends
from .core import Report, Result, _close, _final_inputs, compare
from .detectors import Averaging
from .inputs import Case, normalise_metric, prepare, to_numpy


@dataclass
class Finding:
    key: str
    title: str
    detail: str
    fix: str
    reference: str | None = None


@dataclass
class BatchReport:
    metric: str
    case: Case
    batch_sizes: list[int]
    reference_name: str
    reference: Result
    per_batch: list[float | None]
    batch_mean: float | None
    weighted_mean: float | None
    stateful: dict[str, Result]
    dataset: Report
    expected_n: int | None = None
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def n_samples(self) -> int:
        return int(sum(self.batch_sizes))

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "n_batches": len(self.batch_sizes),
            "n_samples": self.n_samples,
            "expected_n": self.expected_n,
            "reference": {"library": self.reference_name, "value": self.reference.value,
                          "error": self.reference.error},
            "per_batch": self.per_batch,
            "batch_mean": self.batch_mean,
            "weighted_mean": self.weighted_mean,
            "stateful": {k: {"value": r.value, "error": r.error} for k, r in self.stateful.items()},
            "ok": self.ok,
            "findings": [f.__dict__ for f in self.findings],
            "dataset": self.dataset.to_dict(),
        }

    def __str__(self) -> str:
        from .report import render_batched

        return render_batched(self)


def _batches_to_arrays(batches) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for i, b in enumerate(batches):
        try:
            yt, yp = b
        except (TypeError, ValueError):
            raise ValueError(f"batch {i} must be a (y_true, y_pred) pair") from None
        yt, yp = to_numpy(yt), to_numpy(yp)
        if yt.ndim == 0 or yp.ndim == 0:
            raise ValueError(f"batch {i} must be a (y_true, y_pred) pair of arrays, not scalars")
        out.append((yt, yp))
    if not out:
        raise ValueError("no batches given")
    return out


def _cast(case: Case, y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred[:, 0]
    y_pred = y_pred.astype(np.int64 if case.pred_kind == "labels" else np.float64)
    return y_true, y_pred


def compare_batched(metric: str, batches, *, backends: list[str] | None = None,
                    expected_n: int | None = None, rtol: float = 1e-4,
                    atol: float = 1e-6) -> BatchReport:
    """Compare per-batch, accumulated and whole-dataset values of ``metric``.

    Parameters
    ----------
    batches: iterable of ``(y_true, y_pred)`` pairs, in the shapes ``compare()`` accepts.
    expected_n: number of samples in the dataset, if known. Used to detect samples counted
        twice (e.g. DistributedSampler padding) or dropped (drop_last=True).
    """
    metric = normalise_metric(metric)
    raw = _batches_to_arrays(batches)
    case = prepare(np.concatenate([b[0] for b in raw]), np.concatenate([b[1] for b in raw]))
    parts = [_cast(case, yt, yp) for yt, yp in raw]
    sizes = [len(p[0]) for p in parts]

    bks = _backends.resolve(backends)
    dataset = compare(metric, case.y_true, case.y_pred, backends=[b.name for b in bks],
                      rtol=rtol, atol=atol)

    # Reference library: sklearn if present. If its defaults raise (e.g. multiclass F1
    # without `average`), use settings that work; batching is a separate question.
    ref = next((b for b in bks if b.name == "sklearn"), bks[0])
    ref_opts: dict = {}
    if not dataset.results[ref.name].ok and case.task == "multiclass":
        ref_opts = Averaging().options(case, metric)
    yt_all, yp_all = _final_inputs(metric, case, case.y_true, case.y_pred)
    value, err = _backends.safe_compute(ref, metric, case.task, yt_all, yp_all, case.num_classes, ref_opts)
    reference = Result(value, err)

    per_batch: list[float | None] = []
    for yt, yp in parts:
        yt, yp = _final_inputs(metric, case, yt, yp)
        v, _ = _backends.safe_compute(ref, metric, case.task, yt, yp, case.num_classes, ref_opts)
        per_batch.append(v)
    used = [(v, n) for v, n in zip(per_batch, sizes) if v is not None]
    batch_mean = float(np.mean([v for v, _ in used])) if used else None
    weighted = float(np.average([v for v, _ in used], weights=[n for _, n in used])) if used else None

    stateful: dict[str, Result] = {}
    for b in bks:
        def run(b=b):
            s = b.stateful(metric, case.task, case.num_classes, {})
            if s is None:
                return None
            for yt, yp in parts:
                s.update(*_final_inputs(metric, case, yt, yp))
            return s.compute()

        v, e = _backends.safe_call(run)
        if v is None and e is None:
            continue  # library has no metric objects (sklearn)
        stateful[b.name] = Result(v, e)

    report = BatchReport(metric, case, sizes, ref.name, reference, per_batch, batch_mean,
                         weighted, stateful, dataset, expected_n)
    _find(report, rtol, atol)
    return report


def _find(r: BatchReport, rtol: float, atol: float) -> None:
    m = r.metric
    skipped = sum(v is None for v in r.per_batch)
    if r.reference.ok and r.batch_mean is not None and not _close([r.batch_mean, r.reference.value], rtol, atol):
        weighted_ok = _close([r.weighted_mean, r.reference.value], rtol, atol)
        detail = (f"The mean of the {len(r.per_batch) - skipped} per-batch values is {r.batch_mean:.4f}, "
                  f"but {m} on the whole dataset is {r.reference.value:.4f} "
                  f"(off by {r.batch_mean - r.reference.value:+.4f}). ")
        if weighted_ok:
            detail += ("Weighting by batch size fixes it here, so the gap comes from batches of "
                       "different sizes (often a smaller last batch).")
        else:
            detail += (f"{m} is not an average of per-sample values, so averaging batch scores gives a "
                       f"different number (weighted by batch size: {r.weighted_mean:.4f}).")
        if skipped:
            detail += (f" {skipped} batch(es) were left out because {m} is undefined for them "
                       "(for example only one class present), which biases the mean further.")
        fix = ("Accumulate over the whole epoch instead of averaging batch values: call "
               "metric.update() every batch and metric.compute() once (torchmetrics, torcheval, "
               "Keras update_state()/result()). In Lightning, log the Metric object itself "
               "(self.log('val_f1', self.val_f1)), not a value computed per batch. Or collect all "
               "predictions and compute once.")
        r.findings.append(Finding("batch_average", "Averaging per-batch values is not the dataset metric",
                                  detail, fix,
                                  "keras-team/keras#8657, Lightning-AI/pytorch-lightning discussion #8083"))

    for name, res in r.stateful.items():
        own = r.dataset.results.get(name)
        if own is None:
            continue
        if res.ok and own.ok and _close([res.value, own.value], rtol, atol):
            continue
        if not res.ok and not own.ok:
            continue  # same default-settings failure either way; compare() covers it
        got = f"{res.value:.4f}" if res.ok else res.error
        once = f"{own.value:.4f}" if own.ok else own.error
        detail = (f"{name} gives {got} when updated batch by batch, but {once} when computed once "
                  "on the same data. Accumulating should not change the result.")
        fix = (f"Check that the metric is reset() between epochs and not shared between train and "
               f"validation. If it persists, report it to {name} with this output.")
        ref = "Lightning-AI/torchmetrics#3573" if name == "torchmetrics" else None
        r.findings.append(Finding(f"stateful_{name}", f"{name}: update()/compute() differs from one-shot",
                                  detail, fix, ref))

    if r.expected_n is not None and r.n_samples != r.expected_n:
        diff = r.n_samples - r.expected_n
        if diff > 0:
            r.findings.append(Finding(
                "duplicates", "More samples than the dataset has",
                f"The batches contain {r.n_samples} samples but the dataset has {r.expected_n}: "
                f"{diff} sample(s) were counted more than once. torch's DistributedSampler pads the "
                "dataset so every process gets the same number of samples, and the padded samples "
                "are evaluated twice.",
                "Evaluate on one process, or gather predictions with their sample ids and drop "
                "duplicates before computing the metric.",
                "Project-MONAI/MONAI#5357"))
        else:
            r.findings.append(Finding(
                "missing", "Fewer samples than the dataset has",
                f"The batches contain {r.n_samples} samples but the dataset has {r.expected_n}: "
                f"{-diff} sample(s) were never evaluated, often because of drop_last=True.",
                "Use drop_last=False for evaluation DataLoaders.", None))
