# metric-diff

**Your F1 score is 0.75 in torchmetrics, 0.70 in torcheval and an error in sklearn. Which one is right, and why?**

`metric-diff` computes the same metric with **scikit-learn**, **torchmetrics** and **torcheval**. When they disagree, it finds the setting that causes it, **re-runs every library with that setting aligned to confirm**, and tells you how to fix your code.

```text
$ metric-diff auroc --true labels.csv --pred logits.csv

metric-diff: auroc (binary, 15 samples, predictions look like logits (values outside [0, 1]))

  sklearn        0.9286     (1.9.1)
  torchmetrics   0.5000     (1.9.0)  <-
  torcheval      0.9286     (0.0.7)

MISMATCH: explained by 1 cause (confirmed)

  1. Large logits saturate when squashed to probabilities
     torchmetrics applies sigmoid to logits in float32: the 15 distinct scores collapse to 1
     distinct value (large logits all become 1.0), so their ranking is lost and auroc drifts
     toward 0.5. sklearn and torcheval rank the raw scores.
     fix: For ranking metrics pass the raw scores to sklearn, or rescale monotonically
     before torchmetrics, e.g. torch.sigmoid((logits - logits.mean()).double()).
     see: Lightning-AI/torchmetrics#2819

  confirmed: after aligning these settings, every library reports 0.9286
```

## Why

Metric libraries look interchangeable but make different silent choices. Developers keep filing "wrong result" bugs that turn out to be defaults ([torchmetrics#746](https://github.com/Lightning-AI/torchmetrics/issues/746), [#298](https://github.com/Lightning-AI/torchmetrics/issues/298), [torcheval#191](https://github.com/meta-pytorch/torcheval/issues/191)), and some are real bugs ([torchmetrics#2819](https://github.com/Lightning-AI/torchmetrics/issues/2819), [evaluate#869](https://github.com/huggingface/evaluate/issues/869)). metric-diff tells the two apart in seconds.

## Install

```bash
pip install metric-diff            # sklearn only (reference)
pip install "metric-diff[torch]"   # + torchmetrics and torcheval
```

## Use

```python
from metric_diff import compare

report = compare("f1", y_true, y_pred)   # lists, numpy arrays or torch tensors
print(report)          # human-readable explanation
report.agree           # True / False
report.causes          # list of Cause(key, title, detail, fix, reference)
report.to_dict()       # JSON-serialisable
```

As a test guard:

```python
assert compare("auroc", y, scores).agree, compare("auroc", y, scores)
```

Command line (`.csv`, `.txt` or `.npy`; one column per class for multiclass scores):

```bash
metric-diff f1 --true y_true.csv --pred y_pred.csv
metric-diff auroc --true y.npy --pred scores.npy --json
```

Exit code: `0` = libraries agree, `1` = mismatch, `2` = bad input, so it can fail a CI job.

## What it detects (v0.1)

| Cause | Example |
|---|---|
| Different default averaging | torchmetrics `macro`, torcheval `micro`, sklearn `binary` (error on multiclass); torchmetrics macro accuracy = sklearn balanced accuracy |
| Binary logits vs probabilities | torchmetrics applies sigmoid automatically, torcheval thresholds raw logits at 0.5 |
| Sigmoid saturation of large logits | torchmetrics AUROC = 0.5 instead of 0.93 (torchmetrics#2819) |
| Multiclass logits vs probabilities | torchmetrics softmaxes, torcheval ranks raw logits, sklearn raises |
| Scores exactly on the threshold | torchmetrics uses `> 0.5`, torcheval uses `>= 0.5` |
| Binary labels not `{0, 1}` | sklearn `pos_label=1`, torchmetrics raises, torcheval returns F1 > 1 |
| Probabilities that don't sum to 1 | sklearn multiclass AUROC raises |

Metrics: `accuracy`, `precision`, `recall`, `f1`, `auroc`, `average_precision`. Tasks: binary and multiclass.

**How a cause is confirmed:** each detector knows how to align one setting (for example pass `average="macro"` everywhere, or apply softmax first). metric-diff re-runs all libraries with the alignment and only says *confirmed* if they then agree. If nothing reproduces agreement, it says so: that is often a new bug worth reporting.

## Roadmap

- Regression metrics (R², MAE, MAPE), multilabel, top-k
- Stateful / batch-accumulated metrics (`update()` + `compute()` vs one shot)
- Hugging Face `evaluate` backend
- pytest plugin

## Contributing

Every known mismatch lives in `tests/` as a regression case linked to its upstream issue. Found a new one? Open an issue with the `--json` output.

## License

MIT
