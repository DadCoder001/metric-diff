"""Plain-text rendering of a Report."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .core import Report, Result


def _val(r: "Result") -> str:
    if r.skipped:
        return "skipped"
    return f"{r.value:.4f}" if r.ok else f"error: {r.error}"


def _wrap(text: str, indent: str) -> str:
    return textwrap.fill(text, width=92, initial_indent=indent, subsequent_indent=indent)


def render(report: "Report") -> str:
    lines = [f"metric-diff: {report.metric} ({report.case.describe()})"]
    if not report.case.settings.empty:
        s = report.case.settings
        desc = s.describe()
        if s.explicit_classes:
            desc = (desc + "; " if desc else "") + f"num_classes={report.case.num_classes}"
        lines.append(f"your settings: {desc}")
    lines.append("")
    width = max(len(n) for n in report.results) + 2
    values = [r.value for r in report.results.values() if r.ok]
    for name, r in report.results.items():
        mark = ""
        if report.status == "mismatch" and r.skipped is None and (not r.ok or _is_outlier(r.value, values)):
            mark = "  <-"
        version = report.versions.get(name, "")
        lines.append(f"  {name:<{width}} {_val(r):<10} ({version}){mark}")
    lines.append("")

    if report.status == "single":
        lines.append("Only one library is installed, so there is nothing to compare.")
        lines.append("Install the others with: pip install 'metric-diff[torch]'")
    elif report.agree:
        lines.append("OK: all libraries agree.")
    elif report.confirmed:
        n = len(report.causes)
        lines.append(f"MISMATCH: explained by {n} cause{'s' if n > 1 else ''} (confirmed)")
    else:
        lines.append("MISMATCH: no known cause reproduces agreement (unconfirmed).")
        if report.causes:
            lines.append("Suspicious settings found:")

    for i, c in enumerate(report.causes, 1):
        lines.append("")
        lines.append(f"  {i}. {c.title}")
        lines.append(_wrap(c.detail, "     "))
        lines.append(_wrap("fix: " + c.fix, "     "))
        if c.reference:
            lines.append(f"     see: {c.reference}")

    if report.confirmed:
        vals = sorted({f"{r.value:.4f}" for r in report.aligned.values() if r.ok})
        lines.append("")
        lines.append(f"  confirmed: after aligning these settings, every library reports {vals[0]}")
    elif report.status == "mismatch":
        lines.append("")
        lines.append("  This may be a new bug. Please report it, with this output (--json for the")
        lines.append("  machine-readable form): https://github.com/DadCoder001/metric-diff/issues")

    if report.notes:
        lines.append("")
        lines.append("notes:")
        for note in report.notes:
            lines.append(_wrap("- " + note, "  "))
    return "\n".join(lines)


def render_batched(report) -> str:
    m = report.metric
    n_b = len(report.batch_sizes)
    sizes = sorted(set(report.batch_sizes))
    size_txt = f"batch size {sizes[0]}" if len(sizes) == 1 else f"batch sizes {sizes[0]}-{sizes[-1]}"
    lines = [f"metric-diff batched: {m} ({report.case.task}, {report.n_samples} samples in {n_b} "
             f"batches, {size_txt})", ""]

    def fmt(v):
        return f"{v:.4f}" if v is not None else "n/a"

    rows = [(f"whole dataset ({report.reference_name})", fmt(report.reference.value)
             if report.reference.ok else f"error: {report.reference.error}", "reference")]
    rows.append(("mean of per-batch values", fmt(report.batch_mean), "naive average of batch scores"))
    rows.append(("batch-size-weighted mean", fmt(report.weighted_mean),
                 "Lightning self.log(value, on_epoch=True)"))
    for name, r in report.stateful.items():
        rows.append((f"{name} update()/compute()", fmt(r.value) if r.ok else f"error: {r.error}",
                     "accumulated"))
    width = max(len(a) for a, _, _ in rows) + 2
    for label, val, note in rows:
        lines.append(f"  {label:<{width}} {val:<10} {note}")
    lines.append("")

    if report.ok:
        lines.append("OK: batching does not change the result.")
    else:
        n = len(report.findings)
        lines.append(f"PROBLEM: {n} issue{'s' if n > 1 else ''} with how the metric is aggregated")
    for i, f in enumerate(report.findings, 1):
        lines.append("")
        lines.append(f"  {i}. {f.title}")
        lines.append(_wrap(f.detail, "     "))
        lines.append(_wrap("fix: " + f.fix, "     "))
        if f.reference:
            lines.append(f"     see: {f.reference}")

    if report.dataset.status == "mismatch":
        lines.append("")
        lines.append(_wrap("note: the libraries also disagree with each other on the whole dataset. "
                           f"Run metric-diff without --batch-size (or compare('{m}', ...)) for the cause.",
                           "  "))
    return "\n".join(lines)


def _is_outlier(value: float, values: list[float]) -> bool:
    """Mark values that differ from the most common value."""
    rounded = [round(v, 6) for v in values]
    most_common = max(set(rounded), key=rounded.count)
    return round(value, 6) != most_common
