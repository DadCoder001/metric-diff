# metric-diff user guide

metric-diff answers two questions:

1. **"Why do these libraries give me different numbers for the same metric?"** (`compare`)
2. **"Is the number I logged during training the number for my dataset?"** (`compare_batched`)

It also keeps a **conformance matrix** of known edge cases (`metric-diff matrix`), which you can run in CI to catch metrics that change when a library is upgraded.

This guide is organised by **what you're seeing**. Find your scenario, run the snippet with your own data, and read the report.

- [Install](#install)
- [Reading a report](#reading-a-report)
- Scenarios
  1. [My training-loop metric doesn't match my sklearn evaluation](#1-my-training-loop-metric-doesnt-match-my-sklearn-evaluation)
  2. [Precision, recall, F1 and accuracy are all the same number](#2-precision-recall-f1-and-accuracy-are-all-the-same-number)
  3. [My AUROC is 0.5 (or suspiciously low) on raw model scores](#3-my-auroc-is-05-or-suspiciously-low-on-raw-model-scores)
  4. [I switched libraries and my numbers changed](#4-i-switched-libraries-and-my-numbers-changed)
  5. [My per-epoch metric doesn't match a final evaluation](#5-my-per-epoch-metric-doesnt-match-a-final-evaluation)
  6. [Multi-GPU evaluation gives slightly different numbers](#6-multi-gpu-evaluation-gives-slightly-different-numbers)
  7. [My Dice / IoU differs between MONAI, torchmetrics and a paper](#7-my-dice--iou-differs-between-monai-torchmetrics-and-a-paper)
  8. [My MAPE or R2 looks absurd](#8-my-mape-or-r2-looks-absurd)
  9. [I want to check the exact settings my code uses](#9-i-want-to-check-the-exact-settings-my-code-uses)
  10. [I'm upgrading a metric library: will any numbers change?](#10-im-upgrading-a-metric-library-will-any-numbers-change)
  11. [I'm writing a paper or report and need to document my metrics](#11-im-writing-a-paper-or-report-and-need-to-document-my-metrics)
  12. [metric-diff says "no known cause"](#12-metric-diff-says-no-known-cause)
  13. [My BLEU / ROUGE doesn't match the paper (or another library)](#13-my-bleu--rouge-doesnt-match-the-paper-or-another-library)
  14. [My retrieval metrics (NDCG, MAP, MRR) don't match trec_eval or a leaderboard](#14-my-retrieval-metrics-ndcg-map-mrr-dont-match-trec_eval-or-a-leaderboard)
  15. [I want my test suite to fail when a metric changes meaning](#15-i-want-my-test-suite-to-fail-when-a-metric-changes-meaning)
- [Quick reference](#quick-reference)
- [FAQ](#faq)

---

## Install

```bash
pip install metric-diff               # scikit-learn only (the reference)
pip install "metric-diff[torch]"      # + torchmetrics, torcheval
pip install "metric-diff[text]"       # + sacrebleu, nltk, rouge-score
pip install "metric-diff[ranking]"    # + pytrec_eval (trec_eval)
pip install "metric-diff[all]"        # + Keras 3, Hugging Face evaluate, MONAI, text, ranking
```

metric-diff compares whichever of these are installed. Install the libraries **you** use: the point is to compare your library against the reference.

## Reading a report

```text
metric-diff: auroc (binary, 15 samples, predictions look like logits (values outside [0, 1]))   <- what metric-diff inferred
your settings: average='macro'                                                                   <- only if you passed settings

  sklearn        0.9286     (1.9.1)          <- each library, its result, its version
  torchmetrics   0.5000     (1.9.0)  <-      <- "<-" marks the odd one out

MISMATCH: explained by 1 cause (confirmed)   <- the verdict

  1. Large logits saturate when squashed to probabilities  [saturation]  <- the cause (and its key)
     torchmetrics applies sigmoid to logits in float32: ...     <- why the numbers differ
     fix: ... torch.sigmoid((logits - logits.mean()).double()).  <- what to change in your code
     see: Lightning-AI/torchmetrics#2819                        <- upstream issue, if there is one

  confirmed: after aligning these settings, every library reports 0.9286

notes:                                        <- things worth knowing that didn't cause a difference
  - ...
```

| Verdict | Meaning |
|---|---|
| `OK: all libraries agree.` | Same number everywhere, within a relative tolerance of 1e-4. |
| `MISMATCH: explained by N cause(s) (confirmed)` | metric-diff aligned those settings, **re-ran every library**, and they then agreed. The explanation is proven on your data, not guessed. |
| `MISMATCH: no known cause reproduces agreement (unconfirmed)` | Suspicious settings are listed, but aligning them didn't reconcile the numbers. Possibly a new bug: see [scenario 12](#12-metric-diff-says-no-known-cause). |
| `skipped` (per library) | That library can't do what you asked (e.g. torcheval has no weighted AUROC). It isn't counted as a mismatch; the reason is in the notes. |
| `error: ...` (per library) | The library raised an error with its defaults, often itself the answer (e.g. sklearn needs `average=` for multiclass F1). |

In Python, everything in the report is available as data:

```python
from metric_diff import compare

r = compare("f1", [0, 1, 1, 0], [0, 1, 0, 0])
r.agree                      # True / False
r.status                     # "agree" | "mismatch" | "single"
r.results["sklearn"].value   # per-library value (or .error / .skipped)
[c.key for c in r.causes]    # machine-readable cause ids
r.to_dict()                  # JSON-serialisable
```

---

## Scenarios

### 1. My training-loop metric doesn't match my sklearn evaluation

**You see:** torchmetrics (or Keras, or torcheval) in your training loop reports one number; your evaluation script with scikit-learn reports another.

**Run:**

```python
from metric_diff import compare

y_true = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
y_pred = [0, 0, 1, 0, 1, 2, 2, 2, 0, 2]
print(compare("accuracy", y_true, y_pred, backends=["sklearn", "torchmetrics"]))
```

```text
  sklearn        0.7000     (1.9.1)
  torchmetrics   0.6667     (1.9.0)  <-

MISMATCH: explained by 1 cause (confirmed)

  1. Different averaging  [average]
     ... sklearn: micro (accuracy_score has no `average`); torchmetrics: 'macro'
     (macro accuracy = sklearn balanced_accuracy_score).
```

**What to do:** pass the averaging explicitly everywhere. The most common causes in this scenario are default averaging, logits vs probabilities, and the threshold; metric-diff checks all of them, so run it on *your* `y_true` / `y_pred` (lists, numpy arrays or torch tensors).

### 2. Precision, recall, F1 and accuracy are all the same number

**You see:** every metric shows exactly the same value. This is one of the most-reported "bugs" (torchmetrics#746, #298, torcheval#191).

**Why:** micro averaging. For single-label multiclass data, micro precision = micro recall = micro F1 = accuracy, by definition. torcheval defaults to micro; older torchmetrics did too.

```python
from metric_diff import compare

y_true = [0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
y_pred = [0, 0, 1, 0, 1, 2, 2, 2, 0, 2]
for metric in ("accuracy", "precision", "recall", "f1"):
    print(metric, compare(metric, y_true, y_pred, backends=["torcheval"]).results["torcheval"].value)
# accuracy 0.7, precision 0.7, recall 0.7, f1 0.7
```

**What to do:** `average="macro"` (each class counts equally) or `"weighted"` (by class frequency) is usually what you want to report. Check with `compare("f1", y_true, y_pred, average="macro")`.

### 3. My AUROC is 0.5 (or suspiciously low) on raw model scores

**You see:** AUROC near 0.5 although the model clearly works, or Keras reporting a much lower AUROC than everyone else.

```python
from metric_diff import compare

logits = [98.0950, 98.4612, 98.1145, 98.1506, 97.6037, 98.9425, 99.2644, 99.5014,
          99.7280, 99.6595, 99.6931, 99.4667, 99.9623, 99.8949, 99.8768]
labels = [0] + [1] * 14
print(compare("auroc", labels, logits, backends=["sklearn", "torchmetrics"]))
```

```text
  sklearn        0.9286     (1.9.1)
  torchmetrics   0.5000     (1.9.0)  <-

  1. Large logits saturate when squashed to probabilities  [saturation]
     torchmetrics applies sigmoid to logits in float32: the 15 distinct scores collapse to 1
     distinct value ...
     see: Lightning-AI/torchmetrics#2819
```

Related causes metric-diff detects here:

- **Keras clips raw scores** to [0, 1] in `AUC()`: on a real DistilBERT model, 0.911 instead of 0.972. Fix: `keras.metrics.AUC(from_logits=True)`.
- **Keras uses 200 fixed thresholds** in `AUC()`: confident predictions bunched near 0 and 1 lose resolution (0.959 vs 0.972 on SST-2).
- **Two score columns** from a binary model's softmax: sklearn wants `scores[:, 1]`.

### 4. I switched libraries and my numbers changed

**You see:** the same model and data, but the numbers moved after porting from Keras to PyTorch, or from sklearn to torchmetrics.

**Run the matrix** to see every known difference between your installed libraries at once:

```bash
metric-diff matrix
metric-diff matrix --case bin- --case mc-     # only classification cases
```

Then run `compare()` on your own predictions for the metrics you report. Keras is the library that differs most often with its defaults: `F1Score()` has no threshold (binary F1 of 0.675 instead of 0.914 on SST-2), multiclass `Precision()`/`Recall()` threshold each class column instead of taking the argmax, MAPE is a percentage, and `AUC(curve="PR")` is not average precision.

### 5. My per-epoch metric doesn't match a final evaluation

**You see:** the validation F1 / AUROC / R2 logged at the end of an epoch differs from evaluating the same model on the same validation set in one go.

**Why:** most metrics are not averages of per-sample values, so the mean of per-batch scores is a different number. Lightning's `self.log(value, on_epoch=True)` averages per-batch values (weighted by batch size), and so does a hand-written loop that appends batch scores.

```python
import numpy as np
from metric_diff import compare_batched

rng = np.random.default_rng(0)
y = rng.integers(0, 2, 1000)
scores = np.clip(y * 0.25 + rng.random(1000) * 0.75, 0, 1)
batches = [(y[i:i + 64], scores[i:i + 64]) for i in range(0, 1000, 64)]   # what your loop sees
print(compare_batched("f1", batches, backends=["sklearn", "torchmetrics"]))
```

```text
  whole dataset (sklearn)           0.6558     reference
  mean of per-batch values          0.6537     naive average of batch scores
  batch-size-weighted mean          0.6530     Lightning self.log(value, on_epoch=True)
  torchmetrics update()/compute()   0.6558     accumulated

PROBLEM: 1 issue with how the metric is aggregated
  1. Averaging per-batch values is not the dataset metric  [batch_average]
```

**What to do:** accumulate, don't average: `metric.update()` each batch and `metric.compute()` once, or log the torchmetrics `Metric` object itself in Lightning. On real data the gap can be large: per-batch R2 on the diabetes dataset averaged **0.223** against **0.365** for the whole test set.

The batch check also catches:

- batches left out because the metric is undefined for them (e.g. AUROC on a batch with one class);
- a library whose `update()`/`compute()` disagrees with its own one-shot result. Two real ones: **Keras MAE / MSE / MAPE with 1-D targets** average batch means (reshape targets to `(batch, 1)`), and **torchmetrics' `MeanIoU` class** pools (image, class) pairs while its `mean_iou` function averages per image (0.304 vs 0.335 on ADE20K).

### 6. Multi-GPU evaluation gives slightly different numbers

**You see:** evaluating on 4 GPUs gives a slightly different number from 1 GPU.

**Why:** `DistributedSampler` pads the dataset so every process gets the same number of samples, and the padded samples are evaluated twice. `drop_last=True` does the opposite. Pass the true dataset size:

```python
import numpy as np
from metric_diff import compare_batched

rng = np.random.default_rng(0)
y = rng.integers(0, 2, 1000)
scores = np.clip(y * 0.25 + rng.random(1000) * 0.75, 0, 1)
batches = [(y[i:i + 64], scores[i:i + 64]) for i in range(0, 1000, 64)]
batches.append((y[:24], scores[:24]))          # 24 padded samples, evaluated twice
report = compare_batched("accuracy", batches, expected_n=1000, backends=["sklearn"])
print([f.key for f in report.findings])        # ['batch_average', 'duplicates']
```

**What to do:** evaluate on one process, or gather predictions with sample ids and drop duplicates before computing; use `drop_last=False` for evaluation loaders.

### 7. My Dice / IoU differs between MONAI, torchmetrics and a paper

**You see:** segmentation scores that don't match between libraries or with published numbers. On a real SegFormer model on ADE20K, the *same* predictions gave Dice **0.284**, **0.388** or **0.538** depending on the library's defaults.

Pass integer label maps of shape `(n_images, *spatial)`, e.g. `(N, H, W)` or `(N, D, H, W)`:

```python
import numpy as np
from metric_diff import compare

t = np.zeros((3, 4, 4), int); p = np.zeros((3, 4, 4), int)
t[0, :2, :2] = 1; p[0, :2, :3] = 1; t[0, 2:, 2:] = 2; p[0, 3:, 2:] = 2
t[1, :2, :] = 1; p[1, :2, :] = 1; p[1, 3, 3] = 2           # class 2 predicted but absent
t[2, 1:3, 1:3] = 1; p[2, 1:3, 1:2] = 1
print(compare("dice", t, p, backends=["sklearn", "torchmetrics"]))
```

The three conventions behind almost every difference:

| Convention | Options | Who does what |
|---|---|---|
| Aggregation | pool all pixels of the dataset vs score each image then average | sklearn, Keras, evaluate pool; torchmetrics, MONAI per image |
| Empty targets | count false positives on images where the class is absent, or ignore them | MONAI ignores them by default (`ignore_empty=True`), which raises the score |
| Background | include class 0 in the average or not | `include_background` |

Set them explicitly and report them: `compare("iou", t, p, aggregation="per_image", include_background=False)`. The [Metrics Reloaded](https://doi.org/10.1038/s41592-023-02151-z) recommendations (Nature Methods 2024) are the reference for choosing.

metric-diff also warns when torchmetrics would silently misread label maps passed without `input_format="index"`.

### 8. My MAPE or R2 looks absurd

```python
from metric_diff import compare

print(compare("mape", [0.0, 1.0, 2.0], [0.5, 1.0, 2.5], backends=["sklearn", "torchmetrics"]))
# sklearn 7.5e14, torchmetrics 142450: a zero target divides by each library's own epsilon
```

Regression causes metric-diff detects:

| You see | Cause | Fix |
|---|---|---|
| MAPE of 1e14 or 1e5 | a target is exactly 0 | use WAPE or sMAPE, or drop zero targets |
| Keras MAPE 28.0, others 0.28 | Keras returns a percentage | divide by 100 |
| R2 of 1.0000001, or billions | float32 rounding on large values (torchmetrics#3077) | compute in float64, or subtract the mean of `y_true` first |
| R2 of -inf or NaN | constant target (torcheval, evaluate) | report MSE/MAE instead |
| R2 0.959 vs 0.9595 | evaluate rounds R2 to 3 decimals | use sklearn for more precision |

### 9. I want to check the exact settings my code uses

By default every library runs with its own defaults. To check *your* call, pass your settings, either for every library or per library:

```python
from metric_diff import compare

y = [0, 1, 1, 0, 1, 0, 1, 1]
scores = [0.1, 0.8, 0.65, 0.3, 0.9, 0.45, 0.2, 0.7]

compare("f1", y, scores, threshold=0.3)                               # every library
compare("precision", y, scores, settings={"sklearn": {"threshold": 0.3}},
        backends=["sklearn", "torchmetrics"])                           # reproduce two different calls
```

```text
your settings: sklearn: threshold=0.3

  sklearn        0.6667     (1.9.1)
  torchmetrics   1.0000     (1.9.0)  <-

  1. Different decision thresholds  [threshold]
     The libraries turn scores into labels at different thresholds (sklearn: 0.3; torchmetrics: 0.5).
```

Available settings are in the [quick reference](#quick-reference). A library that can't honour a setting is shown as `skipped` with the reason, not as a mismatch.

### 10. I'm upgrading a metric library: will any numbers change?

Save a baseline once, commit it, and compare in CI:

```bash
metric-diff matrix --save metric-baseline.json       # once, on the versions you trust
metric-diff matrix --baseline metric-baseline.json   # in CI: exit code 1 if any result changed
```

On GitHub, the metric-diff Action does this for you. Install your own pinned requirements first, so the matrix runs on the versions you actually use:

```yaml
# .github/workflows/metric-drift.yml
name: metric drift
on: [pull_request]   # e.g. a Dependabot or Renovate PR that bumps torchmetrics
jobs:
  metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt   # your pinned library versions
      - uses: DadCoder001/metric-diff@v0.1.0a3
        with:
          baseline: metric-baseline.json
```

The first run has no baseline yet: it saves one and uploads it as the `metric-baseline` artifact for you to commit. After that, every run compares with it and writes the drift report to the job summary. The report lists library version changes and each cell that changed value, started failing or started working. After an intended upgrade, review it, run once with `update-baseline: true` and commit the new baseline.

Keep one baseline per operating system (`metric-baseline-${{ runner.os }}.json` in a matrix job): some libraries order tied scores differently on Linux, macOS and Windows, and the report warns when the baseline came from another OS.

### 11. I'm writing a paper or report and need to document my metrics

```bash
metric-diff matrix --format markdown --output metrics-matrix.md
```

Attach the table, and state in your methods section the library and version, the averaging (`macro` / `micro` / `weighted`), the threshold, and for segmentation the aggregation and whether background is included. metric-diff's report on your own predictions tells you which of these actually changes your number.

### 12. metric-diff says "no known cause"

```text
MISMATCH: no known cause reproduces agreement (unconfirmed).
```

The libraries disagree and none of the known causes explains it. That is often a new bug, either in a library or in metric-diff. Please report it with the machine-readable output:

```bash
metric-diff f1 --true y_true.csv --pred y_pred.csv --json > report.json
```

Every confirmed report becomes a new detector and a new case in the matrix.

### 13. My BLEU / ROUGE doesn't match the paper (or another library)

BLEU depends on tokenisation, smoothing, the scale and whether you average sentences; ROUGE on stemming and sentence splitting. Pass references and generated texts (one string per sample, or a list of references per sample):

```python
from metric_diff import compare

references = ["The cat is sitting on the mat.", "Hello, there."]
generated = ["The cat sat on the mat.", "Hello there!"]

print(compare("bleu", references, generated))
```

Typical causes it confirms:

- **0-100 vs 0-1** `[text_scale]`: sacrebleu reports 39.2 where torchmetrics reports 0.392.
- **Tokenisation** `[bleu_tokenize]`: sacrebleu and evaluate split off punctuation (13a); torchmetrics `BLEUScore` and nltk split on spaces, so "mat." never matches "mat".
- **Smoothing** `[bleu_smoothing]`: sacrebleu smooths by default; averaging sentence BLEU without smoothing gives 0 for every sentence with no 4-gram match.
- **Corpus vs sentence average** `[text_aggregation]`: the mean of sentence BLEU is not corpus BLEU.
- **chrF vs chrF++** `[chrf_word_order]`: torchmetrics `CHRFScore` is chrF++ by default.
- **ROUGE-Lsum** `[rougelsum_sentences]`: torchmetrics splits sentences itself; rouge-score and evaluate need one sentence per line.

Reproduce your own call: `compare("bleu", refs, hyps, tokenize="13a", smooth=False)`, or per library with `settings={"sacrebleu": {"tokenize": "intl"}}`. To compare with a paper, use sacrebleu and report its signature (tokenizer, smoothing, version). For Chinese, Japanese or Thai, BLEU needs `tokenize="zh"` or `"char"`, and ROUGE drops non-Latin characters unless you pass your own tokenizer: metric-diff adds a note when it sees such text.

From the command line, one text per line and one `--true` file per reference set:

```bash
metric-diff bleu --true refs.txt --pred generated.txt
metric-diff chrf --true refs_a.txt --true refs_b.txt --pred generated.txt --word-order 2
```

### 14. My retrieval metrics (NDCG, MAP, MRR) don't match trec_eval or a leaderboard

Give one row per query: the relevance of each candidate (0 = not relevant, graded values for NDCG) and your model's scores. Rows can have different lengths:

```python
from metric_diff import compare

relevance = [[3, 2, 0, 1, 0], [0, 1, 0, 0, 1], [2, 0, 0, 1]]
scores = [[0.5, 0.5, 0.5, 0.2, 0.1], [0.5, 0.5, 0.9, 0.5, 0.2], [0.3, 0.3, 0.3, 0.2]]

print(compare("map", relevance, scores))
print(compare("ndcg", relevance, scores, top_k=3))
```

Typical causes:

- **Tied scores** `[ranking_ties]`: sklearn averages over tie orders, torchmetrics keeps its sort order, trec_eval sorts ties by document id. Common with rounded, quantised or saturated scores. Break ties yourself before evaluating.
- **Scores of 0 or below** `[torchmetrics_nonpositive_scores]`: torchmetrics' MAP, MRR, precision and recall treat such candidates as not retrieved ([torchmetrics#3518](https://github.com/Lightning-AI/torchmetrics/issues/3518)). Pass sigmoid(logits) or shifted scores.
- **MAP@k** `[map_cutoff]`: trec_eval divides by all relevant documents; torchmetrics by those found in the top k.
- **Queries without relevant documents** `[empty_queries]`: counted as 0 by default everywhere; `empty_queries="skip"` reproduces torchmetrics' `empty_target_action="skip"`.
- **precision@k on short lists** `[adaptive_k]`: divide by k or by the list length.

For TREC, MS MARCO or BEIR comparisons, trec_eval (pytrec_eval) is the reference.

### 15. I want my test suite to fail when a metric changes meaning

`metric_diff.testing` has assertions that print the full report when they fail:

```python
import numpy as np
from metric_diff.testing import assert_metrics_agree, assert_batches_consistent

y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
scores = np.array([0.1, 0.8, 0.65, 0.3, 0.9, 0.45, 0.2, 0.7])

assert_metrics_agree("auroc", y, scores)                      # every installed library agrees
assert_metrics_agree("f1", y, scores, threshold=0.5,
                     expected=0.8889)                         # ...and equals the value you expect
assert_metrics_agree("precision", y, scores, threshold=0.3,
                     allow=["threshold_tie"])                 # a difference you know about, nothing else
assert_batches_consistent("accuracy", [(y[:4], scores[:4]), (y[4:], scores[4:])])
```

- `allow` takes cause keys (the names in brackets in a report). A typo raises an error, so a test can't silently allow nothing.
- `min_libraries=2` (default) fails when fewer than two libraries could compute the metric, so a test can't pass just because a library isn't installed in CI. Use `min_libraries=1` with `expected=` to check one library against a known value.
- The assertions raise `MetricMismatch`, an `AssertionError` whose `.report` holds the full report.

---

## Quick reference

### Metrics

| Task | Metrics | Input |
|---|---|---|
| Classification (binary, multiclass) | `accuracy`, `precision`, `recall`, `f1`, `auroc`, `average_precision` | `y_true`: 1-D integer labels. `y_pred`: labels, binary scores (probabilities or logits), or `(n, n_classes)` scores |
| Regression | `r2`, `mae`, `mse`, `rmse`, `mape` | `y_true`, `y_pred`: same shape, 1-D or `(n, n_outputs)` |
| Segmentation | `dice`, `iou` | integer label maps `(n_images, *spatial)` |
| Text generation | `bleu`, `chrf`, `rouge1`, `rouge2`, `rougeL`, `rougeLsum` | `y_true`: a reference string (or list of references) per sample. `y_pred`: one generated string per sample |
| Ranking / retrieval | `ndcg`, `map`, `mrr`, `precision_at_k`, `recall_at_k`, `hit_rate` | one row per query: `y_true` relevance (0 = not relevant), `y_pred` scores; rows may differ in length |

Aliases work too: `roc_auc`, `f1_score`, `r2_score`, `mean_squared_error`, `jaccard`, `mean_iou`, `sacrebleu`, `rouge_l`, `ndcg@k`, `p@k`, `recall@k`, …

### Settings

| Setting | Values | Applies to |
|---|---|---|
| `average` | `"micro"`, `"macro"`, `"weighted"` | multiclass classification |
| `threshold` | 0 < t < 1 | binary label metrics |
| `pos_label` | a label | binary data with labels other than {0, 1} |
| `num_classes` | int | classes that never occur in the data |
| `zero_division` | `0`, `1`, `"nan"` | precision / recall / F1 when 0/0 |
| `top_k` | int | multiclass accuracy; ranking cut-off k |
| `include_background` | `True` / `False` | segmentation |
| `aggregation` | `"dataset"`, `"per_image"` / `"corpus"`, `"sentence"` | segmentation / BLEU and chrF |
| `tokenize` | `"13a"`, `"whitespace"`, `"intl"`, `"char"`, `"zh"` | BLEU |
| `smooth` | `True` / `False` | BLEU |
| `lowercase` | `True` / `False` | BLEU, chrF |
| `word_order` | `0` (chrF), `2` (chrF++) | chrF |
| `use_stemmer` | `True` / `False` | ROUGE |
| `empty_queries` | `"zero"`, `"skip"`, `"one"` | ranking: queries without relevant documents |
| `adaptive_k` | `True` / `False` | precision@k when a query has fewer than k candidates |
| `settings={lib: {...}}` | any of the above, per library | reproducing code that calls libraries differently |
| `backends=[...]` | `sklearn`, `torchmetrics`, `torcheval`, `keras`, `evaluate`, `monai`, `sacrebleu`, `nltk`, `rouge_score`, `pytrec_eval` | which libraries to compare |

### Command line

```bash
metric-diff <metric> --true FILE --pred FILE [settings flags] [--backends a,b] [--json]
metric-diff <metric> --true FILE --pred FILE --batch-size N [--expected-n N]
metric-diff matrix [--case TEXT] [--format text|markdown|json] [--output FILE] [--save FILE] [--baseline FILE] [--summary FILE] [--list]
```

Files: `.csv`, `.txt` or `.npy` (`.npy` for 2-D/3-D masks). Text metrics: one text per line, `--true` repeated for several references. Ranking metrics: one query per line, values separated by commas or spaces. Settings flags: `--average`, `--threshold`, `--pos-label`, `--num-classes`, `--zero-division`, `--top-k`, `--include-background`, `--aggregation`, `--tokenize`, `--smooth`, `--lowercase`, `--word-order`, `--use-stemmer`, `--empty-queries`, `--adaptive-k`, `--set LIB.KEY=VALUE`.

Exit codes: `0` OK, `1` mismatch / aggregation problem / drift, `2` bad input.

---

## FAQ

**Which number is right?**
Usually none is wrong: the libraries answer slightly different questions (macro vs micro, per image vs pooled). Choose the convention your field reports, set it explicitly, and state it. metric-diff uses scikit-learn as the reference because it is the de facto standard; for segmentation, follow Metrics Reloaded.

**Does metric-diff change my data or send it anywhere?**
No. Everything runs locally on your arrays. The only network access is Hugging Face `evaluate` downloading its metric scripts the first time it is used, and torchmetrics' ROUGE-Lsum downloading nltk's `punkt_tab` sentence splitter (both cached afterwards).

**Why is a library `skipped`?**
It can't do what was asked: torcheval has no weighted AUROC, no MAE and no segmentation; Keras and evaluate have no Dice; MONAI only averages per image. The note gives the exact reason.

**Why is it slow?**
To confirm a cause, metric-diff re-runs every library with candidate settings aligned. Importing torch, Keras and evaluate takes several seconds the first time in a process; after that most comparisons take well under a second. Hugging Face `evaluate` adds about 2 s per metric the first time it loads it, and large segmentation sets (hundreds of images, 150 classes) take 15 to 35 s. Pass `backends=[...]` to compare only the libraries you use.

**Can I use it in tests?**
Yes, see [scenario 15](#15-i-want-my-test-suite-to-fail-when-a-metric-changes-meaning):

```python
from metric_diff.testing import assert_metrics_agree

assert_metrics_agree("auroc", [0, 1, 1, 0], [0.2, 0.9, 0.6, 0.4], backends=["sklearn", "torchmetrics"])
```

**Which library versions was this written against?**
scikit-learn 1.9, torchmetrics 1.9, torcheval 0.0.7, Keras 3.15, evaluate 0.4.6, MONAI 1.6, sacrebleu 2.6, nltk 3.10, rouge-score 0.1.2, pytrec_eval-terrier 0.5. Library behaviour changes over time; run `metric-diff matrix` to see your versions.
