"""Assertions for your own test suite: lock in that your metrics mean what you think they mean.

Works with pytest, unittest or plain ``assert``-style scripts (failures raise ``AssertionError``)::

    from metric_diff.testing import assert_metrics_agree

    def test_val_f1_is_library_independent():
        assert_metrics_agree("f1", y_val, model_scores, average="macro")

Typical uses:

* Pin behaviour across a dependency bump (torchmetrics, Keras, ...): the test fails, with the
  full metric-diff explanation, when a library starts to disagree.
* Accept a difference you know about, and nothing else: ``allow=["threshold_tie"]``.
* Check a hand-written metric against the libraries: ``expected=my_f1(y, p)``.
* Check your training loop's epoch metric: ``assert_batches_consistent("f1", batches)``.
"""

from __future__ import annotations

from collections.abc import Iterable

from .batched import BatchReport, compare_batched
from .core import Report, compare


class MetricMismatch(AssertionError):
    """Raised by the assertions below. ``.report`` holds the full Report / BatchReport."""

    def __init__(self, message: str, report: Report | BatchReport):
        super().__init__(message)
        self.report = report


BATCH_FINDINGS = {"batch_average", "duplicates", "missing"}  # plus "stateful_<library>"


def _known(key: str) -> bool:
    from .core import all_detectors

    return (key in {d.key for d in all_detectors()} or key in BATCH_FINDINGS
            or key.startswith("stateful_"))


def _check_allow(allow: Iterable[str]) -> set[str]:
    allow = {allow} if isinstance(allow, str) else set(allow)
    unknown = {k for k in allow if not _known(k)}
    if unknown:
        raise ValueError(f"unknown cause key(s) in allow: {sorted(unknown)}. Cause keys are shown "
                         f"in brackets in a metric-diff report, e.g. [threshold_tie].")
    return allow


def assert_metrics_agree(metric: str, y_true, y_pred, *, expected: float | None = None,
                         allow: Iterable[str] = (), min_libraries: int = 2,
                         backends: list[str] | None = None, rtol: float = 1e-4, atol: float = 1e-6,
                         **settings) -> Report:
    """Fail unless every library gives the same value for ``metric``.

    Parameters
    ----------
    metric, y_true, y_pred, backends, rtol, atol: as in :func:`metric_diff.compare`.
    **settings: any ``compare()`` setting (average, threshold, pos_label, num_classes, ...).
    expected: also require every library's value to be close to this number (e.g. the value
        your own code computes, or a number from a paper).
    allow: cause keys you accept, e.g. ``["threshold_tie"]``. A disagreement passes only when
        metric-diff confirms it is fully explained by these causes.
    min_libraries: fail if fewer libraries could compute the metric (so a test can't pass
        silently because only one library is installed). Use 1 together with ``expected``.

    Returns the Report, for further checks.
    """
    __tracebackhide__ = True  # pytest: point the failure at the caller's line
    allow = _check_allow(allow)
    report = compare(metric, y_true, y_pred, backends=backends, rtol=rtol, atol=atol, **settings)
    ran = {n: r for n, r in report.results.items() if r.skipped is None}

    if len(ran) < min_libraries:
        names = ", ".join(ran) or "none"
        raise MetricMismatch(
            f"metric-diff: only {len(ran)} library could compute {report.metric!r} ({names}); "
            f"min_libraries={min_libraries}. Install more (pip install 'metric-diff[all]') or "
            f"lower min_libraries.\n\n{report}", report)

    if report.status == "mismatch":
        keys = {c.key for c in report.causes}
        if not (report.confirmed and keys and keys <= allow):
            why = ("cause(s) not in allow: " + ", ".join(sorted(keys - allow)) if report.confirmed
                   else "no confirmed explanation")
            raise MetricMismatch(f"metric-diff: libraries disagree on {report.metric!r} ({why}).\n\n"
                                 f"{report}", report)

    if expected is not None:
        import numpy as np

        off = {n: r.value for n, r in ran.items()
               if r.value is None or not np.isclose(r.value, expected, rtol=rtol, atol=atol)}
        if off:
            got = ", ".join(f"{n}={v}" for n, v in off.items())
            raise MetricMismatch(f"metric-diff: expected {report.metric}={expected}, got {got}.\n\n"
                                 f"{report}", report)
    return report


def assert_batches_consistent(metric: str, batches, *, allow: Iterable[str] = (),
                              expected_n: int | None = None, backends: list[str] | None = None,
                              rtol: float = 1e-4, atol: float = 1e-6, **settings) -> BatchReport:
    """Fail if batching changes ``metric``: per-batch averaging, duplicated or missing samples.

    ``batches`` is what your loop sees: an iterable of ``(y_true, y_pred)`` pairs.
    ``allow`` takes finding keys you accept: "batch_average" (you log the mean of per-batch
    values on purpose), "duplicates", "missing", or library cause keys. Returns the BatchReport.
    """
    __tracebackhide__ = True
    allow = _check_allow(allow)
    report = compare_batched(metric, batches, backends=backends, expected_n=expected_n,
                             rtol=rtol, atol=atol, **settings)
    bad = [f for f in report.findings if f.key not in allow]
    if bad:
        raise MetricMismatch(f"metric-diff: batching changes {report.metric!r} "
                             f"({', '.join(f.key for f in bad)}).\n\n{report}", report)
    return report


__all__ = ["assert_metrics_agree", "assert_batches_consistent", "MetricMismatch"]
