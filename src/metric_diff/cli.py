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
from .inputs import EMPTY_QUERIES, METRICS, RETRIEVAL_METRICS, TEXT_METRICS, TOKENIZERS, normalise_metric


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


def load_lines(path: str) -> list[str]:
    """One text per line (text-generation metrics)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path}: no such file")
    return p.read_text(encoding="utf-8-sig").splitlines()


def load_texts(true_paths: list[str], pred_path: str):
    """References: one file per reference stream (as sacrebleu's CLI), one text per line."""
    preds = load_lines(pred_path)
    streams = [load_lines(f) for f in true_paths]
    for f, s in zip(true_paths, streams):
        if len(s) != len(preds):
            raise ValueError(f"{f} has {len(s)} lines but {pred_path} has {len(preds)}")
    return [list(group) for group in zip(*streams)], preds


def load_rows(path: str) -> list[np.ndarray]:
    """Ranking: one query per line, numbers separated by commas or spaces (lines may differ in length)."""
    rows = []
    for i, line in enumerate(load_lines(path), 1):
        if not line.strip():
            continue
        try:
            rows.append(np.array([float(v) for v in line.replace(",", " ").split()]))
        except ValueError:
            if i == 1:  # header
                continue
            raise ValueError(f"{path}, line {i}: expected numbers") from None
    return rows


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
        elif name in ("top_k", "word_order") or (name == "zero_division" and value in ("0", "1")):
            parsed = int(value)
        elif name == "include_background":
            parsed = value.lower() == "true"
        else:
            parsed = value
        out.setdefault(lib.strip(), {})[name] = parsed
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="metric-diff",
        description="Compute a metric with every installed library and explain any disagreement. "
                    "Run `metric-diff matrix` for the conformance matrix of known edge cases.",
    )
    ap.add_argument("metric", help=f"one of: {', '.join(METRICS)}")
    ap.add_argument("--true", required=True, dest="y_true", action="append",
                    help="file with true labels (.csv, .txt, .npy); for text metrics a .txt file of "
                         "references, one per line (repeat --true for several references)")
    ap.add_argument("--pred", required=True, dest="y_pred",
                    help="file with predictions: labels, binary scores, or one column per class; "
                         "for text metrics generated texts, one per line")
    ap.add_argument("--backends", help="comma-separated subset, e.g. sklearn,torchmetrics,keras")
    ap.add_argument("--average", choices=["micro", "macro", "weighted"],
                    help="averaging passed to every library (multiclass)")
    ap.add_argument("--threshold", type=float, help="decision threshold passed to every library (binary)")
    ap.add_argument("--pos-label", type=int, help="label of the positive class (binary)")
    ap.add_argument("--num-classes", type=int, help="total number of classes, if some never occur")
    ap.add_argument("--zero-division", choices=["0", "1", "nan"],
                    help="value of precision/recall/F1 when it is 0/0, passed to every library")
    ap.add_argument("--top-k", type=int, help="top-k accuracy (multiclass scores)")
    ap.add_argument("--include-background", choices=["true", "false"],
                    help="segmentation: count class 0 in the average")
    ap.add_argument("--aggregation", choices=["dataset", "per_image", "corpus", "sentence"],
                    help="segmentation: pool all pixels (dataset) or score each image and average "
                         "(per_image); bleu/chrf: corpus score or mean of sentence scores")
    ap.add_argument("--tokenize", choices=list(TOKENIZERS), help="bleu: tokenizer for every library")
    ap.add_argument("--smooth", choices=["true", "false"], help="bleu: smoothing on or off")
    ap.add_argument("--lowercase", action="store_true", default=None, help="bleu/chrf: lowercase first")
    ap.add_argument("--word-order", type=int, choices=[0, 1, 2], help="chrf: 0 = chrF, 2 = chrF++")
    ap.add_argument("--use-stemmer", choices=["true", "false"], help="rouge: Porter stemming")
    ap.add_argument("--empty-queries", choices=list(EMPTY_QUERIES),
                    help="ranking: score queries without relevant documents as zero or one, or skip them")
    ap.add_argument("--adaptive-k", action="store_true", default=None,
                    help="precision_at_k: divide by the list length when a query has fewer than k candidates")
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


def build_matrix_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="metric-diff matrix",
        description="Run every known edge case on every installed library. Save the result and "
                    "compare later runs with it to catch drift when a library is upgraded.",
    )
    ap.add_argument("--backends", help="comma-separated subset of libraries")
    ap.add_argument("--case", action="append", default=[], metavar="TEXT",
                    help="only cases whose id contains TEXT (repeatable), e.g. --case seg- --case auroc")
    ap.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    ap.add_argument("--output", help="write the table to this file instead of the terminal")
    ap.add_argument("--save", metavar="FILE", help="save this run as JSON (a baseline for later)")
    ap.add_argument("--baseline", metavar="FILE",
                    help="compare with a saved run; exit code 1 if any result changed")
    ap.add_argument("--summary", metavar="FILE",
                    help="append a Markdown summary (drift and the matrix) to FILE, "
                         "e.g. $GITHUB_STEP_SUMMARY")
    ap.add_argument("--list", action="store_true", help="list the edge cases and exit")
    return ap


def matrix_main(argv: list[str]) -> int:
    from .cases import CASES
    from .matrix import MatrixResult, run_matrix

    args = build_matrix_parser().parse_args(argv)
    if args.list:
        for c in CASES:
            ref = f"  [{c.reference}]" if c.reference else ""
            print(f"{c.id:28s} {c.metric:18s} {c.title}{ref}")
        return 0
    try:
        baseline = MatrixResult.load(args.baseline) if args.baseline else None
        backends = [b.strip() for b in args.backends.split(",")] if args.backends else None
        result = run_matrix(backends=backends, cases=args.case or None)
    except (ValueError, FileNotFoundError, ImportError, json.JSONDecodeError) as exc:
        print(f"metric-diff: error: {exc}", file=sys.stderr)
        return 2

    text = {"text": result.to_text, "markdown": result.to_markdown,
            "json": lambda: json.dumps(result.to_dict(), indent=2)}[args.format]()
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.save:
        result.save(args.save)
    drift = result.diff(baseline) if baseline is not None else None
    if args.summary:
        parts = [drift.to_markdown()] if drift is not None else []
        table = result.to_markdown()
        parts.append(table if drift is None else
                     f"<details><summary>Full conformance matrix</summary>\n\n{table}\n\n</details>")
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write("\n\n".join(parts) + "\n\n")
    if drift is not None:
        print()
        print(drift.to_text())
        return 1 if drift.found else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "matrix":
        return matrix_main(argv[1:])
    args = build_parser().parse_args(argv)
    try:
        metric = normalise_metric(args.metric)
        is_text, is_ranking = metric in TEXT_METRICS, metric in RETRIEVAL_METRICS
        if is_text:
            y_true, y_pred = load_texts(args.y_true, args.y_pred)
        elif is_ranking:
            if len(args.y_true) > 1:
                raise ValueError("--true can be repeated for text metrics only")
            y_true, y_pred = load_rows(args.y_true[0]), load_rows(args.y_pred)
        else:
            if len(args.y_true) > 1:
                raise ValueError("--true can be repeated for text metrics only")
            y_true = load_array(args.y_true[0])
            y_pred = load_array(args.y_pred)
        backends = [b.strip() for b in args.backends.split(",")] if args.backends else None
        zd = args.zero_division
        kwargs = dict(average=args.average, threshold=args.threshold, pos_label=args.pos_label,
                      num_classes=args.num_classes, zero_division=int(zd) if zd in ("0", "1") else zd,
                      top_k=args.top_k,
                      include_background=None if args.include_background is None
                      else args.include_background == "true",
                      aggregation=args.aggregation, settings=parse_settings(args.set) or None,
                      backends=backends)
        if is_text:
            flag = {"true": True, "false": False, None: None}
            kwargs.update(tokenize=args.tokenize, smooth=flag[args.smooth], lowercase=args.lowercase,
                          word_order=args.word_order, use_stemmer=flag[args.use_stemmer])
        if is_ranking:
            kwargs.update(empty_queries=args.empty_queries, adaptive_k=args.adaptive_k)
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
