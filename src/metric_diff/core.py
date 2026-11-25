"""compare(): run every library, and when they disagree, find and confirm the cause."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from . import backends as _backends
from .detectors import DETECTORS, Detector, notes_for
from .inputs import (LABEL_METRICS, REGRESSION_METRICS, SEGMENTATION_KEYS, SEGMENTATION_METRICS,
                     RETRIEVAL_KEYS, RETRIEVAL_METRICS, SETTING_KEYS, TEXT_KEYS, TEXT_METRICS, Case,
                     make_settings, normalise_metric, prepare, prepare_regression, prepare_segmentation)


@dataclass
class Result:
    value: float | None
    error: str | None = None
    skipped: str | None = None  # the library can't do what was asked (e.g. an unsupported average)

    @property
    def ok(self) -> bool:
        return self.error is None and self.skipped is None


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
            return {k: {"value": r.value, "error": r.error, "skipped": r.skipped} for k, r in d.items()}

        s = self.case.settings
        return {
            "metric": self.metric,
            "task": self.case.task,
            "num_classes": self.case.num_classes,
            "n_samples": self.case.n_samples,
            "pred_kind": self.case.pred_kind,
            "settings": {"common": dict(s.common),
                         "per_library": {k: dict(v) for k, v in s.per_library.items()},
                         "explicit_num_classes": s.explicit_classes},
            "versions": dict(self.versions),
            "status": self.status,
            "results": res(self.results),
            "confirmed": self.confirmed,
            "causes": [dict(c.__dict__) for c in self.causes],
            "aligned": res(self.aligned),
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        from .report import render

        return render(self)


def _close(values: list[float], rtol: float, atol: float) -> bool:
    ref = values[0]
    return all(np.isclose(v, ref, rtol=rtol, atol=atol) for v in values[1:])


def _status(results: dict[str, Result], rtol: float, atol: float) -> str:
    active = [r for r in results.values() if r.skipped is None]
    if len(active) < 2:
        return "single"
    if not all(r.ok for r in active):
        return "mismatch"
    return "agree" if _close([r.value for r in active], rtol, atol) else "mismatch"


def _final_inputs(metric: str, case: Case, y_true, y_pred, num_classes: int | None = None):
    """Shape conversions every library needs (not a 'cause', just plumbing)."""
    if case.task in ("segmentation", "text", "ranking"):
        return y_true, y_pred
    if metric not in LABEL_METRICS:
        if case.task == "multiclass" and y_pred.ndim == 1:
            y_pred = np.eye(num_classes or case.num_classes, dtype=np.float64)[y_pred]
        else:
            y_pred = y_pred.astype(np.float64)
    return y_true, y_pred


def backend_opts(case: Case, name: str, detector_opts: dict | None = None) -> dict:
    """User settings for this library, then detector alignment on top (None removes a key)."""
    opts = case.settings.for_backend(name)
    if case.task == "regression" and case.y_true.ndim == 2:
        opts["multi_output"] = True
    if case.column_vector:
        opts["column_vector"] = True  # Keras accumulates differently for (n, 1) and (n,)
    for k, v in (detector_opts or {}).items():
        if v is None:
            opts.pop(k, None)
        else:
            opts[k] = v
    return opts


def unsupported(case: Case, metric: str, name: str, opts: dict) -> str | None:
    """Reason this library can't honour the user's settings, or None."""
    if case.task == "segmentation":
        from . import segmentation

        return segmentation.unsupported(name, metric, opts, case.y_true.ndim)
    if case.task == "text":
        from . import text

        return text.unsupported(name, metric, opts)
    if case.task == "ranking":
        from . import ranking

        return ranking.unsupported(name, metric, opts)
    backend = _backends.ALL_BACKENDS.get(name)
    reason = backend().unsupported(metric, case.task, opts) if backend else None
    if reason:
        return reason
    avg = opts.get("average")
    if case.task != "multiclass" or avg is None or avg != case.settings.value(name, "average"):
        return None
    supported = _backends.SUPPORTED_AVERAGES.get(name, {}).get(metric)
    if supported is not None and avg not in supported:
        return f"no average={avg!r} for {metric} (supports {', '.join(sorted(supported))})"
    return None


def _run(bks, metric, case, detectors: tuple[Detector, ...] = ()) -> dict[str, Result]:
    y_true, y_pred = case.y_true, case.y_pred
    det_opts: dict = {}
    for d in detectors:
        y_true, y_pred = d.transform(y_true, y_pred, case)
        det_opts.update(d.options(case, metric))
    out = {}
    for b in bks:
        opts = backend_opts(case, b.name, det_opts)
        reason = unsupported(case, metric, b.name, opts)
        if reason:
            out[b.name] = Result(None, skipped=reason)
            continue
        n_classes = opts.pop("num_classes", case.num_classes)
        digits = opts.pop("round_digits", None)
        yt, yp = _final_inputs(metric, case, y_true, y_pred, n_classes)
        value, err = _backends.safe_compute(b, metric, case.task, yt, yp, n_classes, opts)
        if value is not None and digits is not None:
            value = round(value, digits)
        out[b.name] = Result(value, err)
    return out


def task_of(metric: str) -> str:
    if metric in REGRESSION_METRICS:
        return "regression"
    if metric in SEGMENTATION_METRICS:
        return "segmentation"
    if metric in TEXT_METRICS:
        return "text"
    if metric in RETRIEVAL_METRICS:
        return "ranking"
    return "classification"


def resolve_for(metric: str, names: list[str] | None):
    """Backends to compare: the ones named, or every installed one that covers this task."""
    bks = _backends.resolve(names)
    if names is None:
        bks = [b for b in bks if task_of(metric) in b.tasks]
        if task_of(metric) in ("text", "ranking"):
            from .ranking import SUPPORTS as RANKING
            from .text import SUPPORTS as TEXT

            table = TEXT if task_of(metric) == "text" else RANKING
            bks = [b for b in bks if metric in table.get(b.name, ())]
        if not bks:
            extra = {"text": "text", "ranking": "ranking"}.get(task_of(metric), "all")
            raise ImportError(f"no installed library computes {metric}; "
                              f"pip install 'metric-diff[{extra}]'")
    return bks


def build_case(metric, y_true, y_pred, bks, *, average=None, threshold=None, pos_label=None,
               num_classes=None, zero_division=None, top_k=None, include_background=None,
               aggregation=None, settings=None, **text_settings) -> Case:
    names = tuple(b.name for b in bks)
    for lib in settings or {}:
        if lib not in names:
            raise ValueError(f"settings given for {lib!r}, which is not being compared ({', '.join(names)})")
    s = make_settings(average, threshold, settings, explicit_classes=num_classes is not None,
                      zero_division=zero_division, top_k=top_k,
                      include_background=include_background, aggregation=aggregation,
                      **text_settings)
    if metric in TEXT_METRICS:
        from .text import METRIC_KEYS, prepare_text

        other = {k for k in SETTING_KEYS if s.values(k)} - set(METRIC_KEYS[metric])
        if other or pos_label is not None or num_classes is not None:
            raise ValueError(f"{metric} takes {', '.join(METRIC_KEYS[metric])} only "
                             f"(got {sorted(other) or ['pos_label/num_classes']})")
        if s.values("aggregation") - {"corpus", "sentence"}:
            raise ValueError("aggregation for text metrics is 'corpus' or 'sentence'")
        return dataclasses.replace(prepare_text(y_true, y_pred), settings=s, backends=names)
    if metric in RETRIEVAL_METRICS:
        from .ranking import NEEDS_K, prepare_ranking

        other = {k for k in SETTING_KEYS if s.values(k)} - set(RETRIEVAL_KEYS)
        if other or pos_label is not None or num_classes is not None:
            raise ValueError(f"{metric} takes top_k, empty_queries and adaptive_k only "
                             f"(got {sorted(other) or ['pos_label/num_classes']})")
        if metric in NEEDS_K and not s.values("top_k"):
            raise ValueError(f"{metric} needs top_k (the cut-off k)")
        if s.values("adaptive_k") and metric != "precision_at_k":
            raise ValueError("adaptive_k applies to precision_at_k only")
        return dataclasses.replace(prepare_ranking(y_true, y_pred), settings=s, backends=names)
    ir_keys = {k for k in ("empty_queries", "adaptive_k") if s.values(k)}
    if ir_keys:
        raise ValueError(f"{sorted(ir_keys)} apply to ranking metrics only")
    text_keys = {k for k in TEXT_KEYS if s.values(k)}
    if text_keys:
        raise ValueError(f"{sorted(text_keys)} apply to text metrics (bleu, chrf, rouge) only")
    if s.values("aggregation") & {"corpus", "sentence"}:
        raise ValueError("aggregation 'corpus'/'sentence' applies to text metrics (bleu, chrf) only")
    seg_keys = {k for k in SEGMENTATION_KEYS if s.values(k)}
    if metric in SEGMENTATION_METRICS:
        other = {k for k in SETTING_KEYS if s.values(k)} - set(SEGMENTATION_KEYS)
        if other or pos_label is not None:
            raise ValueError("segmentation metrics take include_background, aggregation and "
                             f"num_classes only (got {sorted(other) or ['pos_label']})")
        case = prepare_segmentation(y_true, y_pred, num_classes)
        return dataclasses.replace(case, settings=s, backends=names)
    if seg_keys:
        raise ValueError(f"{sorted(seg_keys)} apply to segmentation metrics (dice, iou) only")
    if metric in REGRESSION_METRICS:
        if not s.empty or pos_label is not None:
            raise ValueError("average, threshold, pos_label, num_classes, zero_division and top_k "
                             "apply to classification metrics only")
        return dataclasses.replace(prepare_regression(y_true, y_pred), backends=names)
    case = prepare(y_true, y_pred, pos_label=pos_label, num_classes=num_classes)
    if s.values("top_k"):
        if metric != "accuracy":
            raise ValueError("top_k is supported for accuracy only")
        if case.task != "multiclass" or case.y_pred.ndim != 2:
            raise ValueError("top_k needs multiclass scores: y_pred of shape (n_samples, n_classes)")
        if max(s.values("top_k")) > case.num_classes:
            raise ValueError(f"top_k={max(s.values('top_k'))} is larger than num_classes={case.num_classes}")
    return dataclasses.replace(case, settings=s, backends=names)


def compare(metric: str, y_true, y_pred, *, average: str | None = None,
            threshold: float | None = None, pos_label=None, num_classes: int | None = None,
            zero_division=None, top_k: int | None = None, include_background: bool | None = None,
            aggregation: str | None = None, tokenize: str | None = None,
            smooth: bool | None = None, lowercase: bool | None = None,
            word_order: int | None = None, use_stemmer: bool | None = None,
            empty_queries: str | None = None, adaptive_k: bool | None = None,
            settings: dict[str, dict] | None = None, backends: list[str] | None = None,
            rtol: float = 1e-4, atol: float = 1e-6) -> Report:
    """Compute ``metric`` with every installed library and explain any disagreement.

    Parameters
    ----------
    metric: classification: accuracy, precision, recall, f1, auroc, average_precision;
        regression: r2, mae, mse, rmse, mape; segmentation: dice, iou; text generation: bleu,
        chrf, rouge1, rouge2, rougeL, rougeLsum; ranking: ndcg, map, mrr, precision_at_k,
        recall_at_k, hit_rate (aliases accepted).
        Segmentation inputs are integer label maps of shape (n_images, *spatial).
        Text inputs: y_true = references (a string, or a list of strings, per sample),
        y_pred = generated texts (one string per sample).
        Ranking inputs: one row per query, y_true = relevance of each candidate (0 = not
        relevant), y_pred = scores; a 2-D array, or a list of 1-D arrays of different lengths.
    y_true: classification: 1-D integer labels. Regression: 1-D values, or 2-D for multi-output.
    y_pred: classification: 1-D labels or binary scores, or 2-D (n_samples, n_classes)
        multiclass scores. Regression: same shape as y_true.
        Lists, numpy arrays and torch tensors are accepted.
    average: "micro", "macro" or "weighted", passed to every library (multiclass).
    threshold: decision threshold for binary label metrics, passed to every library.
    pos_label: which label is the positive class (binary). Labels are mapped to {0, 1}.
    num_classes: total number of classes, if some never occur in the data.
    zero_division: 0, 1 or "nan": the value of precision/recall/F1 when it is 0/0.
    top_k: top-k accuracy (multiclass scores); ranking: the cut-off k (ndcg@k, map@k, ...).
    empty_queries: ranking: score a query without relevant documents as "zero", "one", or "skip" it.
    adaptive_k: precision_at_k: divide by the list length when a query has fewer than k candidates.
    include_background: segmentation: count class 0 in the average (default: each library's default, True).
    aggregation: segmentation: "dataset" (pool all pixels) or "per_image" (score each image, then average);
        bleu/chrf: "corpus" (pool n-gram counts) or "sentence" (mean of sentence scores).
    tokenize: bleu: "13a", "whitespace", "intl", "char" or "zh" (sacrebleu's tokenizers).
    smooth: bleu: smoothing on or off (each library's own method when on).
    lowercase: bleu/chrf: compare lowercased text.
    word_order: chrf: 0 for chrF, 2 for chrF++.
    use_stemmer: rouge: Porter stemming.
    settings: per-library overrides, e.g. ``{"sklearn": {"average": "weighted"}}``, to
        reproduce code that calls each library differently.
    backends: subset of the library names (sklearn, torchmetrics, torcheval, keras, evaluate, monai,
        sacrebleu, nltk, rouge_score, pytrec_eval); default = every installed library that covers the metric.
    """
    metric = normalise_metric(metric)
    bks = resolve_for(metric, backends)
    names = [b.name for b in bks]
    case = build_case(metric, y_true, y_pred, bks, average=average, threshold=threshold,
                      pos_label=pos_label, num_classes=num_classes, zero_division=zero_division,
                      top_k=top_k, include_background=include_background, aggregation=aggregation,
                      settings=settings, tokenize=tokenize, smooth=smooth, lowercase=lowercase,
                      word_order=word_order, use_stemmer=use_stemmer, empty_queries=empty_queries,
                      adaptive_k=adaptive_k)

    results = _run(bks, metric, case)
    status = _status(results, rtol, atol)
    report = Report(metric, case, {b.name: b.version() for b in bks}, results, status,
                    notes=notes_for(case, metric) + [f"{n}: skipped, {r.skipped}"
                                                     for n, r in results.items() if r.skipped])
    if status != "mismatch":
        return report

    relevant = [d for d in all_detectors()
                if (d.requires is None or d.requires in names) and d.relevant(case, metric)]
    differing = _differing(results, rtol, atol)
    for size in range(1, len(relevant) + 1):
        fixes = []
        for combo in combinations(relevant, size):
            aligned = _run(bks, metric, case, combo)
            if _status(aligned, rtol, atol) == "agree":
                fixes.append((combo, aligned))
        if fixes:
            # Several alignments can reproduce agreement (e.g. two causes that share a transform);
            # explain with the one whose causes best match the libraries that actually disagree.
            combo, aligned = max(fixes, key=lambda f: _fit(f[0], differing, names, case, metric))
            report.confirmed = True
            report.aligned = aligned
            report.causes = [_cause(d, case, metric, names) for d in combo]
            return report

    # Nothing reproduced agreement: list what looks suspicious, clearly marked as unconfirmed.
    report.causes = [_cause(d, case, metric, names) for d in relevant]
    return report


def all_detectors() -> list[Detector]:
    from .ranking import RANKING_DETECTORS
    from .text import TEXT_DETECTORS

    return DETECTORS + TEXT_DETECTORS + RANKING_DETECTORS


def _differing(results: dict[str, Result], rtol: float, atol: float) -> set[str]:
    """Libraries that fail or differ from sklearn (or from the majority value)."""
    ref = results.get("sklearn")
    if ref is not None and ref.ok:
        base = ref.value
    else:
        values = [round(r.value, 6) for r in results.values() if r.ok]
        base = max(set(values), key=values.count) if values else None
    out = set()
    for name, r in results.items():
        if r.skipped:
            continue
        if not r.ok or (base is not None and not _close([r.value, base], rtol, atol)):
            out.add(name)
    return out


def _fit(combo, differing: set[str], names, case=None, metric=None) -> int:
    score = 0
    for d in combo:
        affected = d.affected(case, metric, names)
        score += len(affected & differing) - len(affected - differing)
    return score


def _cause(d: Detector, case, metric, names) -> Cause:
    detail, fix = d.explain(case, metric, names)
    return Cause(d.key, d.title, detail, fix, d.reference)
