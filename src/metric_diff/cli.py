"""Command line: metric-diff <metric> --true FILE --pred FILE [--json]."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .core import compare
from .inputs import METRICS


def load_array(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path}: no such file")
    if p.suffix == ".npy":
        return np.load(p, allow_pickle=False)
    text = p.read_text(encoding="utf-8-sig")
    first = text.splitlines()[0] if text.strip() else ""
    delimiter = "," if "," in first else None
    skip = 0
    try:
        [float(v) for v in first.replace(",", " ").split()]
    except ValueError:  # header row
        skip = 1
    return np.loadtxt(p, delimiter=delimiter, skiprows=skip, ndmin=1)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="metric-diff",
        description="Compute a metric with sklearn, torchmetrics and torcheval; explain any disagreement.",
    )
    ap.add_argument("metric", help=f"one of: {', '.join(METRICS)}")
    ap.add_argument("--true", required=True, dest="y_true", help="file with true labels (.csv, .txt, .npy)")
    ap.add_argument("--pred", required=True, dest="y_pred",
                    help="file with predictions: labels, binary scores, or one column per class")
    ap.add_argument("--backends", help="comma-separated subset, e.g. sklearn,torchmetrics")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--version", action="version", version=f"metric-diff {__version__}")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        y_true = load_array(args.y_true)
        y_pred = load_array(args.y_pred)
        backends = [b.strip() for b in args.backends.split(",")] if args.backends else None
        report = compare(args.metric, y_true, y_pred, backends=backends)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"metric-diff: error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report)
    # exit code: 0 agree / single backend, 1 mismatch (handy in CI)
    return 1 if report.status == "mismatch" else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
