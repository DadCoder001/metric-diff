# metric-diff

**Your F1 score is 0.75 in torchmetrics, 0.70 in torcheval, a vector in Keras and an error in sklearn. Which one is right, and why?**

`metric-diff` computes the same metric with **scikit-learn**, **torchmetrics**, **torcheval**, **Keras 3** and **Hugging Face evaluate**. When they disagree, it finds the setting that causes it, **re-runs every library with that setting aligned to confirm**, and tells you how to fix your code.

It also answers the other classic question: **is the number I logged per batch the number for my dataset?**

```text
$ metric-diff auroc --true labels.csv --pred logits.csv

metric-diff: auroc (binary, 15 samples, predictions look like logits (values outside [0, 1]))

  sklearn        0.9286     (1.9.1)
  torchmetrics   0.5000     (1.9.0)  <-
  torcheval      0.9286     (0.0.7)
  keras          0.5000     (3.15.1)  <-

MISMATCH: explained by 1 cause (confirmed)

  1. Large logits saturate when squashed to probabilities
     torchmetrics applies sigmoid to logits in float32: the 15 distinct scores collapse to 1
     distinct value (large logits all become 1.0), so their ranking is lost and auroc drifts
     toward 0.5. sklearn and torcheval rank the raw scores. Keras AUC also assumes
     probabilities and clips values outside [0, 1] (use from_logits=True).
     fix: For ranking metrics pass the raw scores to sklearn, or rescale monotonically
     before torchmetrics, e.g. torch.sigmoid((logits - logits.mean()).double()).
     see: Lightning-AI/torchmetrics#2819

  confirmed: after aligning these settings, every library reports 0.9286
```

## Why

Metric libraries look interchangeable but make different silent choices. Developers keep filing "wrong result" bugs that turn out to be defaults ([torchmetrics#746](https://github.com/Lightning-AI/torchmetrics/issues/746), [torcheval#191](https://github.com/meta-pytorch/torcheval/issues/191), [keras#5400](https://github.com/keras-team/keras/issues/5400)), and some are real bugs ([torchmetrics#2819](https://github.com/Lightning-AI/torchmetrics/issues/2819), [evaluate#869](https://github.com/huggingface/evaluate/issues/869)). metric-diff tells the two apart in seconds.

## Install

```bash
pip install metric-diff            # sklearn only (reference)
pip install "metric-diff[torch]"   # + torchmetrics and torcheval
pip install "metric-diff[keras]"   # + Keras 3
pip install "metric-diff[evaluate]"  # + Hugging Face evaluate
pip install "metric-diff[all]"     # everything
```

Hugging Face `evaluate` downloads each metric script from the Hugging Face Hub the first time it is used (cached afterwards), so that backend needs internet access once.

Keras runs on whichever backend you have. If TensorFlow isn't installed and you haven't set `KERAS_BACKEND`, metric-diff uses the torch backend.

## Use

### Why do libraries disagree?

```python
from metric_diff import compare

report = compare("f1", y_true, y_pred)   # lists, numpy arrays or torch tensors
print(report)          # human-readable explanation
report.agree           # True / False
report.causes          # list of Cause(key, title, detail, fix, reference)
report.to_dict()       # JSON-serialisable
```

### Check your own settings

By default each library is called with its own defaults. Pass the settings your code uses:

```python
compare("f1", y, probs, average="weighted")            # every library, same setting
compare("precision", y, scores, threshold=0.3)
compare("f1", y, p, pos_label=2)                        # binary labels {1, 2}
compare("f1", y, p, num_classes=10, average="macro")   # some classes never occur
compare("precision", y, p, average="macro", zero_division=1)   # value used for 0/0
compare("accuracy", y, probs, top_k=5)                  # top-k accuracy

# Reproduce code that calls each library differently:
compare("f1", y, probs, settings={
    "sklearn": {"average": "weighted"},
    "torchmetrics": {"average": "macro"},
})
```

If a library can't do what you asked (torcheval has no weighted AUROC, for example), it is shown as `skipped` with the reason, not counted as a mismatch.

As a test guard:

```python
assert compare("auroc", y, scores).agree, compare("auroc", y, scores)
```

### Is my per-batch metric the dataset metric?

```python
from metric_diff import compare_batched

report = compare_batched("f1", [(y_batch, p_batch) for y_batch, p_batch in val_batches],
                         expected_n=len(val_dataset))
print(report)
```

```text
metric-diff batched: f1 (binary, 1000 samples in 16 batches, batch sizes 40-64)

  whole dataset (sklearn)           0.6558     reference
  mean of per-batch values          0.6537     naive average of batch scores
  batch-size-weighted mean          0.6530     Lightning self.log(value, on_epoch=True)
  torchmetrics update()/compute()   0.6558     accumulated
  torcheval update()/compute()      0.6558     accumulated

PROBLEM: 1 issue with how the metric is aggregated

  1. Averaging per-batch values is not the dataset metric
     ...
```

It reports:

- averaging per-batch values (plain or batch-size-weighted) vs. computing on the whole dataset
- batches left out because the metric is undefined for them (e.g. one class only)
- a library whose `update()`/`compute()` doesn't match its own one-shot result
- with `expected_n`: samples counted twice (DistributedSampler padding) or dropped (`drop_last=True`)

### Command line

`.csv`, `.txt` or `.npy`, with one column per class for multiclass scores:

```bash
metric-diff f1 --true y_true.csv --pred y_pred.csv
metric-diff auroc --true y.npy --pred scores.npy --json
metric-diff f1 --true y.csv --pred p.csv --batch-size 64 --expected-n 1000
metric-diff f1 --true y.csv --pred p.csv --backends sklearn,keras
metric-diff f1 --true y.csv --pred p.csv --average weighted
metric-diff precision --true y.csv --pred p.csv --set sklearn.threshold=0.3 --set torchmetrics.threshold=0.5
```

Settings flags: `--average`, `--threshold`, `--pos-label`, `--num-classes`, `--zero-division`, `--top-k`, and `--set LIB.KEY=VALUE` (repeatable) for per-library overrides.

Exit code: `0` = OK, `1` = mismatch or aggregation problem, `2` = bad input, so it can fail a CI job.

## What it detects

| Cause | Example |
|---|---|
| Different default averaging | torchmetrics `macro`, torcheval `micro`, sklearn `binary` (error on multiclass); torchmetrics macro accuracy = sklearn balanced accuracy |
| Binary logits vs probabilities | torchmetrics applies sigmoid automatically; torcheval and Keras threshold raw logits at 0.5 |
| Sigmoid saturation of large logits | torchmetrics AUROC = 0.5 instead of 0.93 (torchmetrics#2819) |
| Multiclass logits vs probabilities | torchmetrics softmaxes, torcheval ranks raw logits, sklearn raises |
| Scores exactly on the threshold | torchmetrics and Keras use `> 0.5`, torcheval uses `>= 0.5` |
| Binary labels not `{0, 1}` | sklearn `pos_label=1`, torchmetrics raises, torcheval and Keras return F1 > 1 |
| Probabilities that don't sum to 1 | sklearn multiclass AUROC raises |
| **Keras** `F1Score()` without `threshold` | binary F1 scored as if every sample were predicted positive |
| **Keras** multiclass `Precision()` / `Recall()` | each class column thresholded at 0.5 like multi-label data, not argmax (use `top_k=1`) |
| **Keras** `AUC()` with logits | values clipped to [0, 1] unless `from_logits=True` |
| **Keras** `AUC()` 200 thresholds | AUROC is approximated; small but real differences |
| **Keras** `AUC(curve="PR")` | interpolated PR-AUC, not average precision |
| Different `average` / `threshold` per library (your settings) | e.g. sklearn `weighted` vs torchmetrics `macro` |
| sklearn `average=` on binary data | averages both classes; torch/Keras binary metrics score the positive class only |
| `num_classes` with classes that never occur | sklearn (`labels=`) and Keras count them as 0; torchmetrics and torcheval leave them out |
| sklearn multiclass AUROC without `multi_class` | raises unless `multi_class="ovr"` |
| Different `zero_division` | what 0/0 becomes; torcheval and Keras always use 0 |
| Tied scores at the top-k cut-off | torchmetrics keeps the lowest class indices; sklearn, torcheval and Keras count any tied class as a hit |
| Per-batch averaging, padding, dropped samples | see `compare_batched` above |
| **Regression:** Keras MAPE is a percentage | 28.0 in Keras vs 0.28 in sklearn and torchmetrics |
| **Regression:** MAPE with zero targets | each library divides by a different tiny epsilon: 7.5e14 vs 142,450 vs 1.7e8 |
| **Regression:** R2 with a constant target | torcheval and evaluate return -inf (torcheval NaN if perfect); others return 0.0 (1.0) |
| **evaluate** `r_squared` rounds to 3 decimals | 0.959 vs 0.9595 |
| **Regression:** float32 rounding on large values | R2 of 1.0000001 instead of 0.989 when values are about 1e6 (torchmetrics#3077) |

Metrics:

- **Classification** (binary and multiclass): `accuracy`, `precision`, `recall`, `f1`, `auroc`, `average_precision`
- **Regression** (single or multi-output): `r2`, `mae`, `mse`, `rmse`, `mape` (torcheval only has `r2` and `mse`; evaluate is single-output only)

evaluate has no average precision, `zero_division` or top-k; those cases show it as skipped.

**How a cause is confirmed:** each detector knows how to align one setting (for example pass `average="macro"` everywhere, or apply softmax first). metric-diff re-runs all libraries with the alignment and only says *confirmed* if they then agree. If nothing reproduces agreement, it says so: that is often a new bug worth reporting.

## Limitations

- Settings supported: `average` (micro / macro / weighted), `threshold`, `pos_label`, `num_classes`, `zero_division`, `top_k` (accuracy). Not yet: `ignore_index` (masking the samples gives the same result in every library), sample weights, per-class output (`average=None`).
- No multilabel, sample weights or per-class output yet. Regression has no settings yet (e.g. `multioutput`).
- Explanations describe the library versions tested (sklearn 1.9, torchmetrics 1.9, torcheval 0.0.7, Keras 3.15, evaluate 0.4.6). Tests check the key defaults so they fail loudly when a library changes.

## Roadmap

- More settings: sample weights, per-class output
- Multilabel classification; regression settings (`multioutput`, sample weights)
- Segmentation (Dice / IoU) and detection mAP
- pytest plugin

## Contributing

Every known mismatch lives in `tests/` as a regression case linked to its upstream issue. Found a new one? Open an issue with the `--json` output.

## License

MIT
