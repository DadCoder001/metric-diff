"""Conformance matrix: every edge case x every installed library, plus drift between runs.

    result = run_matrix()
    print(result.to_text())
    result.save("metric-matrix.json")
    ...
    drift = run_matrix().diff(MatrixResult.load("metric-matrix.json"))
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from . import backends as _backends
from .cases import EdgeCase, select
from .core import _close, compare, task_of

FORMAT_VERSION = 1


@dataclass
class Cell:
    value: float | None = None
    error: str | None = None
    skipped: str | None = None
    status: str = "n/a"  # agree | differs | error | skipped | value | n/a

    def short(self) -> str:
        if self.status == "n/a":
            return ""
        if self.status == "skipped":
            return "-"
        if self.status == "error":
            return "error"
        mark = {"agree": "", "differs": " x", "value": ""}[self.status]
        return _fmt(self.value) + mark


@dataclass
class Row:
    id: str
    title: str
    metric: str
    reference: str | None
    baseline_library: str | None  # whose value the others are compared with
    cells: dict[str, Cell]
    causes: list[str]
    confirmed: bool
    error: str | None = None


@dataclass
class MatrixResult:
    versions: dict[str, str]
    rows: list[Row]
    created: str = ""
    python: str = ""
    metric_diff: str = __version__
    libraries: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ summaries
    def counts(self) -> dict[str, int]:
        out = {"cases": len(self.rows), "agree": 0, "differ": 0}
        for r in self.rows:
            states = {c.status for c in r.cells.values()}
            out["differ" if ("differs" in states or "error" in states) else "agree"] += 1
        return out

    # ------------------------------------------------------------ rendering
    def to_text(self) -> str:
        libs = self.libraries
        head = ["case"] + libs + ["confirmed cause"]
        body = [[r.id] + [r.cells.get(lib, Cell()).short() for lib in libs]
                + [", ".join(r.causes) if r.causes else ("error: " + r.error if r.error else "")]
                for r in self.rows]
        widths = [max(len(str(x)) for x in col) for col in zip(head, *body)]
        widths[-1] = min(widths[-1], 48)
        line = lambda cells: "  ".join(str(c)[:w].ljust(w) for c, w in zip(cells, widths)).rstrip()
        c = self.counts()
        out = [f"metric-diff conformance matrix: {c['cases']} cases, {c['differ']} with differences",
               "versions: " + ", ".join(f"{k} {v}" for k, v in self.versions.items()), "",
               line(head), line(["-" * w for w in widths])]
        out += [line(b) for b in body]
        out += ["", "x = differs from sklearn (or from the majority when sklearn can't compute it)",
                "- = library can't do this (not supported), blank = library doesn't cover this task"]
        return "\n".join(out)

    def to_markdown(self) -> str:
        libs = self.libraries
        out = [f"**metric-diff conformance matrix** ({self.created[:10]}): "
               + ", ".join(f"{k} {v}" for k, v in self.versions.items()), "",
               "| Case | " + " | ".join(libs) + " | Confirmed cause |",
               "|---|" + "---|" * len(libs) + "---|"]
        for r in self.rows:
            cells = []
            for lib in libs:
                c = r.cells.get(lib, Cell())
                text = c.short().replace(" x", "")
                cells.append(f"**{text}** :x:" if c.status == "differs" else ("error :x:" if c.status == "error" else text))
            title = f"{r.title}" + (f" ({r.reference})" if r.reference else "")
            out.append(f"| {title} | " + " | ".join(cells) + " | " + ", ".join(f"`{k}`" for k in r.causes) + " |")
        out += ["", ":x: differs from sklearn (or the majority); `-` not supported by that library."]
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "format": FORMAT_VERSION,
            "metric_diff": self.metric_diff,
            "created": self.created,
            "python": self.python,
            "versions": dict(self.versions),
            "libraries": list(self.libraries),
            "cases": {
                r.id: {"title": r.title, "metric": r.metric, "reference": r.reference,
                       "baseline_library": r.baseline_library, "causes": list(r.causes),
                       "confirmed": r.confirmed, "error": r.error,
                       "cells": {k: dict(c.__dict__) for k, c in r.cells.items()}}
                for r in self.rows
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "MatrixResult":
        if d.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported matrix file format {d.get('format')!r}")
        rows = [Row(cid, c["title"], c["metric"], c.get("reference"), c.get("baseline_library"),
                    {k: Cell(**v) for k, v in c["cells"].items()}, c.get("causes", []),
                    c.get("confirmed", False), c.get("error"))
                for cid, c in d["cases"].items()]
        return cls(d["versions"], rows, d.get("created", ""), d.get("python", ""),
                   d.get("metric_diff", ""), d.get("libraries", []))

    @classmethod
    def load(cls, path: str | Path) -> "MatrixResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------ drift
    def diff(self, baseline: "MatrixResult", rtol: float = 1e-4, atol: float = 1e-6) -> "Drift":
        return diff_matrices(baseline, self, rtol, atol)


@dataclass
class Change:
    case: str
    library: str
    before: str
    after: str
    kind: str  # "value changed" | "now fails" | "now works" | "support changed"


@dataclass
class Drift:
    version_changes: dict[str, tuple[str | None, str | None]]
    changes: list[Change]
    new_cases: list[str]
    removed_cases: list[str]

    @property
    def found(self) -> bool:
        return bool(self.changes)

    def to_text(self) -> str:
        out = []
        if self.version_changes:
            out.append("library versions: " + ", ".join(
                f"{k} {a or '(none)'} -> {b or '(none)'}" for k, (a, b) in self.version_changes.items()))
        if not self.changes:
            out.append("No drift: every result matches the baseline.")
        else:
            out.append(f"DRIFT: {len(self.changes)} result(s) changed since the baseline")
            for c in self.changes:
                out.append(f"  {c.case:28s} {c.library:13s} {c.before} -> {c.after}   ({c.kind})")
        if self.new_cases:
            out.append("new cases (not in the baseline): " + ", ".join(self.new_cases))
        if self.removed_cases:
            out.append("cases missing from this run: " + ", ".join(self.removed_cases))
        return "\n".join(out)


def _fmt(v: float | None) -> str:
    if v is None:
        return ""
    if v != v:
        return "nan"
    if v in (float("inf"), float("-inf")):
        return "-inf" if v < 0 else "inf"
    return f"{v:.3g}" if abs(v) >= 1e4 else f"{v:.4f}"


def _describe(c: Cell) -> str:
    if c.status == "skipped":
        return "skipped"
    if c.status == "error":
        return "error"
    if c.status == "n/a":
        return "n/a"
    return _fmt(c.value)


def diff_matrices(old: MatrixResult, new: MatrixResult, rtol: float = 1e-4, atol: float = 1e-6) -> Drift:
    versions = {}
    for lib in sorted(set(old.versions) | set(new.versions)):
        a, b = old.versions.get(lib), new.versions.get(lib)
        if a != b:
            versions[lib] = (a, b)
    old_rows = {r.id: r for r in old.rows}
    new_rows = {r.id: r for r in new.rows}
    changes = []
    for cid, nr in new_rows.items():
        orow = old_rows.get(cid)
        if orow is None:
            continue
        for lib in sorted(set(orow.cells) & set(nr.cells)):
            a, b = orow.cells[lib], nr.cells[lib]
            if a.status == "n/a" or b.status == "n/a":
                continue
            a_val = a.value is not None and a.status not in ("error", "skipped")
            b_val = b.value is not None and b.status not in ("error", "skipped")
            if a_val and b_val:
                same = (a.value == b.value) or _close([a.value, b.value], rtol, atol)
                if not same:
                    changes.append(Change(cid, lib, _describe(a), _describe(b), "value changed"))
            elif a.status != b.status:
                kind = ("now fails" if b.status == "error" else "now works" if b_val and a.status == "error"
                        else "support changed")
                changes.append(Change(cid, lib, _describe(a), _describe(b), kind))
    return Drift(versions, changes,
                 [c for c in new_rows if c not in old_rows], [c for c in old_rows if c not in new_rows])


def _baseline_value(results) -> tuple[str | None, float | None]:
    sk = results.get("sklearn")
    if sk is not None and sk.ok:
        return "sklearn", sk.value
    values = [round(r.value, 6) for r in results.values() if r.ok]
    if len(values) < 2:
        return None, None
    common = max(set(values), key=values.count)
    if values.count(common) < 2:
        return None, None
    return "majority", common


def run_matrix(*, backends: list[str] | None = None, cases: list[str] | None = None,
               rtol: float = 1e-4, atol: float = 1e-6) -> MatrixResult:
    """Run every edge case (or those whose id contains one of ``cases``) on every installed
    library (or ``backends``)."""
    selected: list[EdgeCase] = select(cases)
    available = [b.name for b in _backends.resolve(backends)]
    task_of_lib = {n: _backends.ALL_BACKENDS[n]().tasks for n in available}
    versions = {b.name: b.version() for b in _backends.resolve(available)}
    rows = []
    for case in selected:
        names = [n for n in available if task_of(case.metric) in task_of_lib[n]]
        y_true, y_pred = case.inputs()
        try:
            report = compare(case.metric, y_true, y_pred, backends=names, rtol=rtol, atol=atol, **case.kwargs)
        except Exception as exc:  # the case itself can't run here (e.g. too few libraries)
            rows.append(Row(case.id, case.title, case.metric, case.reference, None, {}, [], False,
                            f"{type(exc).__name__}: {exc}"))
            continue
        base_lib, base = _baseline_value(report.results)
        cells = {}
        for lib, res in report.results.items():
            if res.skipped:
                status = "skipped"
            elif res.error:
                status = "error"
            elif base is None:
                status = "value"
            else:
                status = "agree" if _close([res.value, base], rtol, atol) else "differs"
            cells[lib] = Cell(res.value, res.error, res.skipped, status)
        rows.append(Row(case.id, case.title, case.metric, case.reference, base_lib, cells,
                        [c.key for c in report.causes] if report.confirmed else [], report.confirmed))
    return MatrixResult(versions, rows, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        platform.python_version(), __version__, available)
