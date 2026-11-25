"""Run metric-diff on real predictions from pretrained Hugging Face models on public datasets.

* Text classification: distilbert-base-uncased-finetuned-sst-2-english on the SST-2
  validation split (872 sentences).
* Semantic segmentation: nvidia/segformer-b0-finetuned-ade-512-512 on ADE20K validation
  images (first N, default 200), scored at the model's 128x128 output resolution.

Needs `pip install transformers` and these files in the Hugging Face cache (about 720 MB):
  stanfordnlp/sst2                  data/validation-00000-of-00001.parquet
  distilbert/...sst-2-english       config.json, tokenizer_config.json, vocab.txt, model.safetensors
  nvidia/segformer-b0-...ade-512    config.json, preprocessor_config.json, model.safetensors
  1aurent/ADE20K                    data/validation-00000-of-00001.parquet
Run benchmarks/download_hf.py once to fetch exactly those. This script then works offline.

Datasets and models are downloaded from the Hugging Face Hub, not redistributed here; their own
licences and terms of use apply (ADE20K, for example, has terms of use for research purposes).

About the ADE20K labels: 1aurent/ADE20K is the full ADE20K release (~3,600 object classes),
not the 150-class SceneParsing subset the model predicts. Classes are mapped to the model's
150 labels by name; unmapped and unlabelled pixels are ignored (class 0 in both masks, scored
with include_background=False). That is fine here: metric-diff checks whether libraries agree
on the same masks, not how good the model is.

    python benchmarks/hf_data.py --output benchmarks/RESULTS_HF.md
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # use only what is already downloaded
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from metric_diff import compare, compare_batched  # noqa: E402
from metric_diff.backends import resolve  # noqa: E402

CACHE = Path(__file__).parent / ".cache"
SST2_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
SEG_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"


def _hub(repo, name, kind="model"):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, name, repo_type=kind)


# ------------------------------------------------------------------ text: SST-2

def sst2_predictions():
    out = CACHE / "sst2_logits.npz"
    if out.exists():
        d = np.load(out)
        return d["y"], d["logits"]
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    table = pq.read_table(_hub("stanfordnlp/sst2", "data/validation-00000-of-00001.parquet", "dataset"))
    sentences, y = table.column("sentence").to_pylist(), np.array(table.column("label").to_pylist())
    tok = AutoTokenizer.from_pretrained(SST2_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SST2_MODEL).eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(sentences), 32):
            enc = tok(sentences[i:i + 32], padding=True, truncation=True, return_tensors="pt")
            logits.append(model(**enc).logits.numpy())
    logits = np.concatenate(logits)
    CACHE.mkdir(exist_ok=True)
    np.savez(out, y=y, logits=logits)
    return y, logits


# ------------------------------------------------------------------ segmentation: ADE20K

def _label_lookup():
    cfg = json.loads(Path(_hub(SEG_MODEL, "config.json")).read_text())
    lookup = {}
    for k, v in cfg["id2label"].items():
        for syn in v.replace(";", ",").split(","):
            lookup.setdefault(syn.strip(), int(k))
    return lookup


def _decode(row, offset):
    from PIL import Image

    seg = np.array(Image.open(io.BytesIO(row["segmentations"][0]["bytes"])).convert("RGB")).astype(np.int64)
    cls = seg[..., 0] // 10 * 256 + seg[..., 1]
    names = {o["name_ndx"] + offset: o["name"] for o in row["objects"]}
    return cls, names


def _map(cls, names, lookup):
    """Full-ADE class map -> 0 (ignored) or 1..150 (model label + 1)."""
    out = np.zeros(cls.shape, np.int64)
    for c in np.unique(cls):
        name = names.get(int(c))
        if c == 0 or name is None:
            continue
        label = lookup.get(name.split(",")[0].strip())
        if label is not None:
            out[cls == c] = label + 1
    return out


def ade_predictions(n_images):
    out = CACHE / f"ade20k_{n_images}.npz"
    if out.exists():
        d = np.load(out)
        return d["y"], d["p"], float(d["coverage"]), int(d["offset"])
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    f = pq.ParquetFile(_hub("1aurent/ADE20K", "data/validation-00000-of-00001.parquet", "dataset"))
    rows = []
    for g in range(f.num_row_groups):
        rows += f.read_row_group(g, columns=["image", "segmentations", "objects"]).to_pylist()
        if len(rows) >= n_images:
            break
    rows = rows[:n_images]
    lookup = _label_lookup()

    # Which index convention do the masks use? Pick the one whose names map best.
    def coverage(offset, sample):
        cov = []
        for r in sample:
            cls, names = _decode(r, offset)
            cov.append((_map(cls, names, lookup) > 0).sum() / max((cls > 0).sum(), 1))
        return float(np.mean(cov))

    offset = max((0, 1), key=lambda o: coverage(o, rows[:20]))

    proc = AutoImageProcessor.from_pretrained(SEG_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL).eval()
    ys, ps, cov = [], [], []
    with torch.no_grad():
        for r in rows:
            image = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            logits = model(**proc(images=image, return_tensors="pt")).logits  # (1, 150, 128, 128)
            pred = logits.argmax(1)[0].numpy() + 1
            cls, names = _decode(r, offset)
            gt = _map(cls, names, lookup)
            cov.append((gt > 0).sum() / max((cls > 0).sum(), 1))
            gt = np.array(Image.fromarray(gt.astype(np.uint8)).resize(pred.shape[::-1], Image.NEAREST))
            pred[gt == 0] = 0  # ignored pixels: same value in both masks, left out of the average
            ys.append(gt)
            ps.append(pred)
    y, p = np.stack(ys).astype(np.uint8), np.stack(ps).astype(np.uint8)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(out, y=y, p=p, coverage=np.mean(cov), offset=offset)
    return y, p, float(np.mean(cov)), offset


# ------------------------------------------------------------------ report

def _fmt(r):
    return "skipped" if r.skipped else ("error" if r.error else f"{r.value:.4f}")


def run(n_images):
    libs_cls = [b.name for b in resolve(None) if "classification" in b.tasks]
    libs_seg = [b.name for b in resolve(None) if "segmentation" in b.tasks]
    rows, batched = [], []

    def add(group, libs, metric, inputs, y, p, **kw):
        t = time.perf_counter()
        r = compare(metric, y, p, **kw)
        causes = ([c.key for c in r.causes] if r.confirmed
                  else [f"unconfirmed: {c.key}" for c in r.causes] or (["unexplained"] if r.status == "mismatch" else []))
        rows.append({"group": group, "libs": libs, "metric": metric, "input": inputs,
                     "settings": ", ".join(f"{k}={v!r}" for k, v in kw.items()),
                     "values": [_fmt(r.results[x]) if x in r.results else "" for x in libs],
                     "result": "agree" if r.status == "agree" else ", ".join(f"`{c}`" for c in causes),
                     "seconds": time.perf_counter() - t})

    def add_batched(group, metric, y, p, bs, **kw):
        r = compare_batched(metric, [(y[i:i + bs], p[i:i + bs]) for i in range(0, len(y), bs)], **kw)
        batched.append({"group": group, "metric": metric, "bs": bs, "whole": r.reference.value,
                        "mean": r.batch_mean, "weighted": r.weighted_mean,
                        "left_out": f"{sum(v is None for v in r.per_batch)} of {len(r.per_batch)}",
                        "findings": ", ".join(f"`{f.key}`" for f in r.findings)})

    t0 = time.perf_counter()
    y, logits = sst2_predictions()
    t_text = time.perf_counter() - t0
    e = np.exp(logits - logits.max(1, keepdims=True))
    probs2 = e / e.sum(1, keepdims=True)
    p1, margin = probs2[:, 1], logits[:, 1] - logits[:, 0]
    g = f"SST-2 ({len(y)})"
    for metric in ("accuracy", "precision", "recall", "f1"):
        add(g, libs_cls, metric, "probabilities", y, p1)
    add(g, libs_cls, "f1", "logit margin", y, margin)
    add(g, libs_cls, "auroc", "probabilities", y, p1)
    add(g, libs_cls, "auroc", "logit margin", y, margin)
    add(g, libs_cls, "average_precision", "probabilities", y, p1)
    add(g, libs_cls, "f1", "2-column logits", y, logits)
    add(g, libs_cls, "auroc", "2-column logits", y, logits)
    add_batched(g, "f1", y, p1, 32)
    add_batched(g, "auroc", y, p1, 32)

    t0 = time.perf_counter()
    ys, ps, coverage, offset = ade_predictions(n_images)
    t_seg = time.perf_counter() - t0
    g = f"ADE20K ({len(ys)} images)"
    for metric in ("dice", "iou"):
        add(g, libs_seg, metric, "label maps", ys, ps, include_background=False)
        add(g, libs_seg, metric, "label maps", ys, ps, include_background=False, aggregation="per_image")
    add(g, libs_seg, "iou", "label maps", ys, ps, include_background=False, aggregation="dataset")
    add_batched(g, "iou", ys, ps, 16, include_background=False, backends=["sklearn", "torchmetrics"])
    return rows, batched, libs_cls, libs_seg, {"text_model_s": t_text, "seg_model_s": t_seg,
                                               "coverage": coverage, "offset": offset}


def markdown(rows, batched, libs_cls, libs_seg, info, versions) -> str:
    out = ["# metric-diff on pretrained Hugging Face models", "",
           f"- Text: `{SST2_MODEL}` on SST-2 validation.",
           f"- Segmentation: `{SEG_MODEL}` on ADE20K validation images, 128x128 model output. "
           f"Full-ADE classes mapped to the model's 150 by name cover {info['coverage']:.0%} of labelled "
           "pixels; the rest are ignored (class 0, `include_background=False`).",
           "- Reproduce: `python benchmarks/download_hf.py` (about 720 MB, once), then "
           "`python benchmarks/hf_data.py`.",
           "- Datasets and models are downloaded from the Hugging Face Hub, not redistributed; their own "
           "licences and terms of use apply.", "",
           "Versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()), ""]
    for group_libs, title in ((libs_cls, "Text classification"), (libs_seg, "Segmentation")):
        sel = [r for r in rows if r["libs"] == group_libs]
        if not sel:
            continue
        out += [f"## {title}", "",
                "| Data | Metric | Input | Settings | " + " | ".join(group_libs) + " | Result | Time |",
                "|---|---|---|---|" + "---|" * len(group_libs) + "---|---|"]
        for r in sel:
            out.append(f"| {r['group']} | {r['metric']} | {r['input']} | {r['settings']} | "
                       + " | ".join(r["values"]) + f" | {r['result']} | {r['seconds']:.1f}s |")
        out.append("")
    out += ["## Per-batch averaging vs the whole set", "",
            "| Data | Metric | Batch size | Whole set | Mean of batches | Batch-size weighted | Batches left out | Findings |",
            "|---|---|---|---|---|---|---|---|"]
    for b in batched:
        out.append(f"| {b['group']} | {b['metric']} | {b['bs']} | {b['whole']:.4f} | {b['mean']:.4f} | "
                   f"{b['weighted']:.4f} | {b['left_out']} | {b['findings']} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=200)
    ap.add_argument("--output")
    args = ap.parse_args()
    t = time.perf_counter()
    rows, batched, lc, ls, info = run(args.images)
    versions = {b.name: b.version() for b in resolve(None)}
    import transformers

    versions["transformers"] = transformers.__version__
    text = markdown(rows, batched, lc, ls, info, versions)
    text += (f"\n\nModel inference: text {info['text_model_s']:.0f}s, segmentation {info['seg_model_s']:.0f}s "
             f"(cached after the first run). Total run time: {time.perf_counter() - t:.0f}s.\n")
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
