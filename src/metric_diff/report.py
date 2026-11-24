"""Plain-text rendering of a Report."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .core import Report, Result


def _val(r: "Result") -> str:
    return f"{r.value:.4f}" if r.ok else f"error: {r.error}"


def _wrap(text: str, indent: str) -> str:
    return textwrap.fill(text, width=92, initial_indent=indent, subsequent_indent=indent)


def render(report: "Report") -> str:
    lines = [f"metric-diff: {report.metric} ({report.case.describe()})", ""]
    width = max(len(n) for n in report.results) + 2
    values = [r.value for r in report.results.values() if r.ok]
    for name, r in report.results.items():
        mark = ""
        if report.status == "mismatch" and (not r.ok or _is_outlier(r.value, values)):
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
        vals = sorted({f"{r.value:.4f}" for r in report.aligned.values()})
        lines.append("")
        lines.append(f"  confirmed: after aligning these settings, every library reports {vals[0]}")
    elif report.status == "mismatch":
        lines.append("")
        lines.append("  This may be a new bug: please report it to the metric-diff issue tracker,")
        lines.append("  including this output (run with --json for the machine-readable form).")

    if report.notes:
        lines.append("")
        lines.append("notes:")
        for note in report.notes:
            lines.append(_wrap("- " + note, "  "))
    return "\n".join(lines)


def _is_outlier(value: float, values: list[float]) -> bool:
    """Mark values that differ from the most common value."""
    rounded = [round(v, 6) for v in values]
    most_common = max(set(rounded), key=rounded.count)
    return round(value, 6) != most_common
