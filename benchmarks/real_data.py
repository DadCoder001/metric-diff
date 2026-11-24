"""Run metric-diff on real model predictions from public datasets bundled with scikit-learn.

No downloads: breast_cancer (binary), digits (10 classes), wine (3 classes, small) and
diabetes (regression) ship with scikit-learn. A model is trained on 70% of each dataset and
its predictions on the other 30% (labels, probabilities and raw scores / logits) are fed to
compare() and compare_batched().

    python benchmarks/real_data.py                 # prints a Markdown report
    python benchmarks/real_data.py --output benchmarks/RESULTS.md
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_wine  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from metric_diff import compare, compare_batched  # noqa: E402
from metric_diff.backends import resolve  # noqa: E402

SEED = 0


def _split(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y if y.dtype.kind in "iu" else None)


def binary():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = _split(X, y)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr)
    return yte, {"labels": m.predict(Xte), "probabilities": m.predict_proba(Xte)[:, 1],
                 "logits": m.decision_function(Xte)}


def multiclass(loader):
    X, y = loader(return_X_y=True)
    Xtr, Xte, ytr, yte = _split(X, y)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr)
    return yte, {"labels": m.predict(Xte), "probabilities": m.predict_proba(Xte),
                 "logits": m.decision_function(Xte)}


def regression():
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = _split(X, y)
    m = Ridge(alpha=1.0).fit(Xtr, ytr)
    return yte, {"predictions": m.predict(Xte)}


def _fmt(r) -> str:
    if r.skipped:
        return "skipped"
    if r.error:
        return "error"
    return f"{r.value:.4f}"


def run(libs: list[str]) -> tuple[list[dict], list[dict]]:
    rows, batched = [], []

    def add(dataset, n, metric, kind, y, p, **kw):
        t = time.perf_counter()
        r = compare(metric, y, p, **kw)
        dt = time.perf_counter() - t
        setting = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        rows.append({"dataset": dataset, "n": n, "metric": metric, "input": kind, "settings": setting,
                     "values": {lib: _fmt(r.results[lib]) if lib in r.results else "" for lib in libs},
                     "status": r.status, "causes": [c.key for c in r.causes] if r.confirmed else
                     ([f"unconfirmed: {c.key}" for c in r.causes] or ([] if r.status != "mismatch" else ["unexplained"])),
                     "seconds": dt})

    y, preds = binary()
    n = len(y)
    for metric in ("accuracy", "precision", "recall", "f1"):
        add("breast_cancer", n, metric, "probabilities", y, preds["probabilities"])
    add("breast_cancer", n, "f1", "logits", y, preds["logits"])
    add("breast_cancer", n, "auroc", "probabilities", y, preds["probabilities"])
    add("breast_cancer", n, "auroc", "logits", y, preds["logits"])
    add("breast_cancer", n, "average_precision", "probabilities", y, preds["probabilities"])

    for name, loader in (("digits", load_digits), ("wine", load_wine)):
        y, preds = multiclass(loader)
        n = len(y)
        add(name, n, "accuracy", "labels", y, preds["labels"])
        add(name, n, "f1", "labels", y, preds["labels"])
        add(name, n, "f1", "labels", y, preds["labels"], average="macro")
        add(name, n, "precision", "probabilities", y, preds["probabilities"], average="macro")
        add(name, n, "auroc", "probabilities", y, preds["probabilities"])
        add(name, n, "auroc", "logits", y, preds["logits"])
        add(name, n, "average_precision", "probabilities", y, preds["probabilities"], average="macro")
        if name == "digits":
            add(name, n, "accuracy", "probabilities", y, preds["probabilities"], top_k=3, average="micro")

    y, preds = regression()
    n = len(y)
    for metric in ("r2", "mae", "mse", "rmse", "mape"):
        add("diabetes", n, metric, "predictions", y, preds["predictions"])

    # Per-batch vs whole-dataset on the real predictions (batch size 32, as in a training loop).
    def add_batched(dataset, metric, y, p, bs=32, **kw):
        batches = [(y[i:i + bs], p[i:i + bs]) for i in range(0, len(y), bs)]
        r = compare_batched(metric, batches, **kw)
        batched.append({"dataset": dataset, "metric": metric, "batch_size": bs, "n": len(y),
                        "whole": r.reference.value, "mean": r.batch_mean, "weighted": r.weighted_mean,
                        "skipped": sum(v is None for v in r.per_batch), "n_batches": len(r.per_batch),
                        "findings": [f.key for f in r.findings]})

    y, preds = binary()
    add_batched("breast_cancer", "f1", y, preds["probabilities"])
    add_batched("breast_cancer", "auroc", y, preds["probabilities"])
    y, preds = multiclass(load_digits)
    add_batched("digits", "f1", y, preds["labels"], average="macro")
    add_batched("digits", "auroc", y, preds["probabilities"], average="macro")
    y, preds = regression()
    add_batched("diabetes", "r2", y, preds["predictions"])
    add_batched("diabetes", "mape", y, preds["predictions"])
    return rows, batched


def markdown(libs, rows, batched, versions) -> str:
    out = ["# metric-diff on real model predictions", "",
           "Models trained on 70% of datasets bundled with scikit-learn; metrics computed on the other 30%. "
           "Reproduce with `python benchmarks/real_data.py`.", "",
           "Versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()), "",
           "## Libraries compared on the same predictions", "",
           "| Dataset (n) | Metric | Input | Settings | " + " | ".join(libs) + " | Result | Time |",
           "|---|---|---|---|" + "---|" * len(libs) + "---|---|"]
    for r in rows:
        res = "agree" if r["status"] == "agree" else ", ".join(f"`{c}`" for c in r["causes"]) or r["status"]
        out.append(f"| {r['dataset']} ({r['n']}) | {r['metric']} | {r['input']} | {r['settings']} | "
                   + " | ".join(r["values"][lib] for lib in libs) + f" | {res} | {r['seconds']:.1f}s |")
    out += ["", "## Per-batch averaging vs the whole test set (batch size 32)", "",
            "| Dataset (n) | Metric | Whole test set | Mean of batches | Batch-size weighted | Batches left out | Findings |",
            "|---|---|---|---|---|---|---|"]
    for b in batched:
        out.append(f"| {b['dataset']} ({b['n']}) | {b['metric']} | {b['whole']:.4f} | {b['mean']:.4f} | "
                   f"{b['weighted']:.4f} | {b['skipped']} of {b['n_batches']} | "
                   + ", ".join(f"`{f}`" for f in b["findings"]) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output")
    args = ap.parse_args()
    bks = resolve(None)
    libs = [b.name for b in bks if "classification" in b.tasks]
    versions = {b.name: b.version() for b in bks if b.name in libs}
    t = time.perf_counter()
    rows, batched = run(libs)
    text = markdown(libs, rows, batched, versions)
    text += f"\n\nTotal run time: {time.perf_counter() - t:.0f}s.\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
