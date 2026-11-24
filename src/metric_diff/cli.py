"""Command line: metric-diff <metric> --true FILE --pred FILE [--json]."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .batched import compare_batched
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


def parse_settings(items: list[str]) -> dict[str, dict]:
    """['sklearn.average=weighted', 'keras.threshold=0.3'] -> {'sklearn': {...}, 'keras': {...}}"""
    out: dict[str, dict] = {}
    for item in items:
        key, sep, value = item.partition("=")
        lib, dot, name = key.partition(".")
        if not (sep and dot and lib and name and value):
            raise ValueError(f"--set expects LIB.KEY=VALUE, got {item!r}")
        name, value = name.strip(), value.strip()
        if name == "threshold":
            parsed = float(value)
        elif name == "top_k" or (name == "zero_division" and value in ("0", "1")):
            parsed = int(value)
        else:
            parsed = value
        out.setdefault(lib.strip(), {})[name] = parsed
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="metric-diff",
        description="Compute a metric with sklearn, torchmetrics and torcheval; explain any disagreement.",
    )
    ap.add_argument("metric", help=f"one of: {', '.join(METRICS)}")
    ap.add_argument("--true", required=True, dest="y_true", help="file with true labels (.csv, .txt, .npy)")
    ap.add_argument("--pred", required=True, dest="y_pred",
                    help="file with predictions: labels, binary scores, or one column per class")
    ap.add_argument("--backends", help="comma-separated subset, e.g. sklearn,torchmetrics,keras")
    ap.add_argument("--average", choices=["micro", "macro", "weighted"],
                    help="averaging passed to every library (multiclass)")
    ap.add_argument("--threshold", type=float, help="decision threshold passed to every library (binary)")
    ap.add_argument("--pos-label", type=int, help="label of the positive class (binary)")
    ap.add_argument("--num-classes", type=int, help="total number of classes, if some never occur")
    ap.add_argument("--zero-division", choices=["0", "1", "nan"],
                    help="value of precision/recall/F1 when it is 0/0, passed to every library")
    ap.add_argument("--top-k", type=int, help="top-k accuracy (multiclass scores)")
    ap.add_argument("--set", action="append", default=[], metavar="LIB.KEY=VALUE",
                    help="per-library setting, e.g. --set sklearn.average=weighted "
                         "--set torchmetrics.threshold=0.3 (repeatable)")
    ap.add_argument("--batch-size", type=int,
                    help="split the rows into batches of this size and check per-batch vs "
                         "accumulated vs whole-dataset values")
    ap.add_argument("--expected-n", type=int,
                    help="with --batch-size: number of samples the dataset really has "
                         "(detects duplicated or dropped samples)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--version", action="version", version=f"metric-diff {__version__}")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        y_true = load_array(args.y_true)
        y_pred = load_array(args.y_pred)
        backends = [b.strip() for b in args.backends.split(",")] if args.backends else None
        zd = args.zero_division
        kwargs = dict(average=args.average, threshold=args.threshold, pos_label=args.pos_label,
                      num_classes=args.num_classes, zero_division=int(zd) if zd in ("0", "1") else zd,
                      top_k=args.top_k, settings=parse_settings(args.set) or None,
                      backends=backends)
        if args.expected_n is not None and args.batch_size is None:
            raise ValueError("--expected-n needs --batch-size")
        if args.batch_size is not None:
            if args.batch_size < 1:
                raise ValueError("--batch-size must be at least 1")
            n = len(y_true)
            batches = [(y_true[i:i + args.batch_size], y_pred[i:i + args.batch_size])
                       for i in range(0, n, args.batch_size)]
            report = compare_batched(args.metric, batches, expected_n=args.expected_n, **kwargs)
            failed = not report.ok
        else:
            report = compare(args.metric, y_true, y_pred, **kwargs)
            failed = report.status == "mismatch"
    except (ValueError, FileNotFoundError, ImportError) as exc:
        print(f"metric-diff: error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report)
    # exit code: 0 ok, 1 mismatch / aggregation problem (handy in CI)
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
