# metric-diff

[![tests](https://github.com/DadCoder001/metric-diff/actions/workflows/tests.yml/badge.svg)](https://github.com/DadCoder001/metric-diff/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/DadCoder001/metric-diff/blob/main/LICENSE)
![status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

**Your F1 score is 0.75 in torchmetrics, 0.70 in torcheval, a vector in Keras and an error in sklearn. Which one is right, and why?**

`metric-diff` computes the same metric with **scikit-learn**, **torchmetrics**, **torcheval**, **Keras 3**, **Hugging Face evaluate** and **MONAI** (segmentation), and for text generation and retrieval with **sacrebleu**, **nltk**, **rouge-score** and **pytrec_eval** (trec_eval). When they disagree, it finds the setting that causes it, **re-runs every library with that setting aligned to confirm**, and tells you how to fix your code.

It also answers the other classic question: **is the number I logged per batch the number for my dataset?**

**New here?** The [user guide](https://github.com/DadCoder001/metric-diff/blob/main/docs/USER_GUIDE.md) is organised by what you're seeing: *"my training-loop F1 doesn't match sklearn"*, *"all my metrics are the same number"*, *"my Dice differs from MONAI"*, *"will upgrading torchmetrics change my numbers?"* and eight more scenarios, each with a snippet to run.

```text
$ metric-diff auroc --true labels.csv --pred logits.csv

metric-diff: auroc (binary, 15 samples, predictions look like logits (values outside [0, 1]))

  sklearn        0.9286     (1.9.1)
  torchmetrics   0.5000     (1.9.0)  <-
  torcheval      0.9286     (0.0.7)
  keras          0.5000     (3.15.1)  <-

MISMATCH: explained by 1 cause (confirmed)

  1. Large logits saturate when squashed to probabilities  [saturation]
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

Metric libraries look interchangeable but make different silent choices. Developers keep filing "wrong result" bugs that turn out to be defaults ([torchmetrics#746](https://github.com/Lightning-AI/torchmetrics/issues/746), [torcheval#191](https://github.com/meta-pytorch/torcheval/issues/191), [keras#5400](https://github.com/keras-team/keras/issues/5400)), and some are real bugs ([torchmetrics#2819](https://github.com/Lightning-AI/torchmetrics/issues/2819), [torchmetrics#3518](https://github.com/Lightning-AI/torchmetrics/issues/3518), [evaluate#869](https://github.com/huggingface/evaluate/issues/869)). In text generation, the same BLEU score is famously not comparable between papers because of tokenisation and smoothing choices ([Post 2018](https://aclanthology.org/W18-6319/)). metric-diff tells defaults and bugs apart in seconds.

## Install

```bash
pip install metric-diff            # sklearn only (reference)
pip install "metric-diff[torch]"   # + torchmetrics and torcheval
pip install "metric-diff[keras]"   # + Keras 3
pip install "metric-diff[evaluate]"  # + Hugging Face evaluate
pip install "metric-diff[monai]"   # + MONAI (segmentation)
pip install "metric-diff[text]"    # + sacrebleu, nltk, rouge-score (BLEU, chrF, ROUGE)
pip install "metric-diff[ranking]" # + pytrec_eval (trec_eval: NDCG, MAP, MRR, @k)
pip install "metric-diff[all]"     # everything
```

Hugging Face `evaluate` downloads each metric script from the Hugging Face Hub the first time it is used (cached afterwards), so that backend needs internet access once. torchmetrics' ROUGE-Lsum downloads nltk's small `punkt_tab` sentence splitter the first time, the same way.

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

### In your test suite

`metric_diff.testing` turns any comparison into an assertion, for pytest, unittest or plain scripts. A failing test prints the full explanation:

```python
from metric_diff.testing import assert_metrics_agree, assert_batches_consistent

def test_val_f1_is_library_independent():
    assert_metrics_agree("f1", y_val, scores, average="macro")

def test_known_difference_only():
    # passes only if the libraries agree, or differ *only* for a reason you accept
    assert_metrics_agree("precision", y, scores, threshold=0.3, allow=["threshold_tie"])

def test_my_own_metric():
    assert_metrics_agree("f1", y, p, expected=my_f1(y, p))     # your code vs every library

def test_epoch_metric():
    assert_batches_consistent("f1", val_batches, expected_n=len(val_dataset))
```

`allow` takes the keys shown in brackets in a report (`[threshold_tie]`); a typo raises an error instead of silently allowing nothing. By default at least two libraries must compute the metric (`min_libraries=2`), so a test can't pass just because only one library is installed.

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

  1. Averaging per-batch values is not the dataset metric  [batch_average]
     ...
```

It reports:

- averaging per-batch values (plain or batch-size-weighted) vs. computing on the whole dataset
- batches left out because the metric is undefined for them (e.g. one class only)
- a library whose `update()`/`compute()` doesn't match its own one-shot result
- with `expected_n`: samples counted twice (DistributedSampler padding) or dropped (`drop_last=True`)

### Segmentation (Dice, IoU)

Pass integer label maps of shape `(n_images, *spatial)`, e.g. `(N, H, W)` or `(N, D, H, W)`:

```python
compare("dice", true_masks, pred_masks)                        # each library's defaults
compare("iou", true_masks, pred_masks, aggregation="per_image", include_background=False)
```

On the same 3 images, "mean IoU" is 0.653 in sklearn, Keras and evaluate (all pixels pooled), 0.631 in torchmetrics (per image) and 0.735 in MONAI (per image, and false positives on images without the class are ignored). metric-diff names and confirms both causes.

### Text generation (BLEU, chrF, ROUGE)

`y_true` = references (a string per sample, or a list of strings for several references), `y_pred` = generated texts:

```python
compare("bleu", references, generated)
compare("bleu", references, generated, tokenize="13a", smooth=False)   # settings for every library
compare("chrf", references, generated, word_order=2)                  # chrF++
compare("rougeLsum", references, generated, use_stemmer=True)
```

```text
metric-diff: bleu (text, 3 generated texts, 1 reference each)

  torchmetrics   0.3289     (1.9.0)
  evaluate       0.3916     (0.4.6)  <-
  sacrebleu      39.1597    (2.6.0)  <-
  nltk           0.3206     (3.10.4)  <-

MISMATCH: explained by 3 causes (confirmed)

  1. Different scales: 0-100 vs 0-1  [text_scale]
  2. Different tokenisation  [bleu_tokenize]
     ... The 13a tokenizer (sacrebleu, evaluate's bleu) splits off punctuation: 'mat.' becomes
     'mat .'; torchmetrics BLEUScore and nltk (called on text.split()) split on spaces only ...
  3. nltk counts texts shorter than 4 words differently  [nltk_short_hypotheses]
```

Settings: `tokenize` (`13a`, `whitespace`, `intl`, `char`, `zh`), `smooth`, `lowercase`, `aggregation` (`corpus` or `sentence` mean), `word_order` (chrF), `use_stemmer` (ROUGE).

### Ranking and retrieval (NDCG, MAP, MRR, precision@k, recall@k, hit rate)

One row per query: `y_true` = relevance of each candidate (0 = not relevant, graded values for NDCG), `y_pred` = the model's scores. A 2-D array, or a list of rows of different lengths:

```python
compare("ndcg", relevance, scores, top_k=10)
compare("map", relevance, scores)
compare("precision_at_k", relevance, scores, top_k=5)
compare("mrr", relevance, scores, empty_queries="skip")   # queries without a relevant document
```

On three queries with tied scores, MAP is 0.506 in sklearn, 0.678 in torchmetrics and 0.585 in trec_eval: each orders ties differently. With raw logits as scores, torchmetrics' MAP, MRR, precision and recall drop every candidate scored 0 or below ([torchmetrics#3518](https://github.com/Lightning-AI/torchmetrics/issues/3518)). metric-diff confirms both.

### Command line

`.csv`, `.txt` or `.npy`, with one column per class for multiclass scores. Text metrics read one text per line (repeat `--true` for several references, as sacrebleu does); ranking metrics read one query per line:

```bash
metric-diff f1 --true y_true.csv --pred y_pred.csv
metric-diff auroc --true y.npy --pred scores.npy --json
metric-diff f1 --true y.csv --pred p.csv --batch-size 64 --expected-n 1000
metric-diff f1 --true y.csv --pred p.csv --backends sklearn,keras
metric-diff f1 --true y.csv --pred p.csv --average weighted
metric-diff precision --true y.csv --pred p.csv --set sklearn.threshold=0.3 --set torchmetrics.threshold=0.5
metric-diff bleu --true refs.txt --true refs2.txt --pred generated.txt --tokenize 13a
metric-diff ndcg --true relevance.csv --pred scores.csv --top-k 10
```

Settings flags: `--average`, `--threshold`, `--pos-label`, `--num-classes`, `--zero-division`, `--top-k`, `--include-background`, `--aggregation`, `--tokenize`, `--smooth`, `--lowercase`, `--word-order`, `--use-stemmer`, `--empty-queries`, `--adaptive-k`, and `--set LIB.KEY=VALUE` (repeatable) for per-library overrides.

Exit code: `0` = OK, `1` = mismatch or aggregation problem, `2` = bad input, so it can fail a CI job.

## Conformance matrix: every library, every known edge case

`compare()` answers *"why do my numbers disagree?"* for your data. The conformance matrix asks the question in advance: it runs a library of known edge cases (45 today, covering classification, regression, segmentation, text generation and ranking, each linked to the upstream issue where there is one) on every installed library, and shows who agrees and who doesn't.

```bash
metric-diff matrix                                   # table in the terminal
metric-diff matrix --format markdown --output metrics-matrix.md
metric-diff matrix --save baseline.json              # remember today's results
metric-diff matrix --baseline baseline.json          # what changed since? exit 1 on drift
metric-diff matrix --case seg- --case auroc          # only some cases
metric-diff matrix --list                            # the edge cases
```

An excerpt (sklearn 1.9.1, torchmetrics 1.9.0, torcheval 0.0.7, Keras 3.15.1, evaluate 0.4.6, MONAI 1.6.2):

```text
case                        sklearn   torchmetrics  torcheval  keras       evaluate  monai     confirmed cause
bin-auroc-large-logits      0.9286    0.5000 x      0.9286     0.5000 x    0.9286              saturation
bin-f1-labels-1-2           0.8000    error         1.1111 x   1.1111 x    0.8000              pos_label
mc-accuracy-default         0.7000    0.6667 x      0.7000     0.7000      0.7000              average
mc-top2-ties                1.0000    0.3333 x      1.0000     1.0000      -                   top_k_ties
reg-mape                    0.2805    0.2805        -          28.0476 x   0.2805              mape_percent
reg-r2-large-values         0.9890    1.0000 x      1.0000 x   1.0000 x    0.9890              float32_precision
seg-iou-default             0.6530    0.6308 x                 0.6530      0.6530    0.7350 x  monai_ignore_empty, seg_aggregation
```

`x` = differs from sklearn (or from the majority when sklearn can't compute it), `-` = the library can't do this.

### Why it matters

- **Upgrade safety.** Save a baseline, then re-run after upgrading a library. If torchmetrics 1.9 -> 2.0 changes AUROC on saturated logits, or macro F1 with an unseen class, you see that exact cell change, the day you upgrade instead of months later when results quietly shift between experiments.
- **Choosing or switching libraries.** Moving a pipeline from Keras to PyTorch, or from sklearn to torchmetrics? See every known difference before you migrate, not one bug report at a time.
- **Reproducible reporting.** Attach the Markdown matrix to a paper, model card or internal report: "computed with torchmetrics 1.9, which differs from sklearn in these documented ways". Reviewers and other teams can compare numbers fairly.
- **Spotting likely bugs.** Red cells on control cases point at library behaviour worth a closer look: the open torchmetrics#2819 and #3518, and F1 values above 1 from torcheval and Keras on labels {1, 2} (reported as [torcheval#241](https://github.com/meta-pytorch/torcheval/issues/241) and [keras#24360](https://github.com/keras-team/keras/issues/24360)), show up exactly like that.
- **Shared knowledge.** Every reported mismatch becomes a new edge case, so the matrix keeps growing and everyone's CI checks it.

### Who it's for

| You are | Use | How |
|---|---|---|
| A developer debugging a number today | `compare()` | on your own data |
| An ML platform / MLOps team | `matrix --baseline` | in CI, on every dependency upgrade |
| A researcher reporting results | both | attach `matrix --format markdown` to the paper |
| A metric library maintainer | `matrix` | a cross-library test suite that points at regressions |

### In CI

```yaml
# .github/workflows/metric-drift.yml
- run: pip install "metric-diff[all]"
- run: metric-diff matrix --baseline metric-baseline.json   # fails the job if any result changed
```

Commit `metric-baseline.json` (created once with `--save`); after an intended upgrade, review the drift and save a new baseline.

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
| Tied scores at the top-k cut-off | torchmetrics keeps exactly k classes, and which tied one depends on the platform (0.33 on Windows/macOS, 0.67 on Linux for the same data); sklearn, torcheval and Keras count any tied class as a hit |
| Per-batch averaging, padding, dropped samples | see `compare_batched` above |
| **Regression:** Keras MAPE is a percentage | 28.0 in Keras vs 0.28 in sklearn and torchmetrics |
| **Regression:** MAPE with zero targets | each library divides by a different tiny epsilon: 7.5e14 vs 142,450 vs 1.7e8 |
| **Regression:** R2 with a constant target | torcheval and evaluate return -inf (torcheval NaN if perfect); others return 0.0 (1.0) |
| **Segmentation:** per-image vs whole-dataset averaging | torchmetrics and MONAI average per image; sklearn, Keras and evaluate pool all pixels |
| **Segmentation:** MONAI `ignore_empty=True` | false positives on images where the class is absent are ignored, so the score looks better |
| **Segmentation:** background counted or not | `include_background` differs between your calls |
| **Segmentation:** torchmetrics `input_format` | label maps without `input_format="index"` silently give a wrong number (shown as a note) |
| **evaluate** `r_squared` rounds to 3 decimals | 0.959 vs 0.9595 |
| **Regression:** float32 rounding on large values | R2 of 1.0000001 instead of 0.989 when values are about 1e6 (torchmetrics#3077) |
| **Text:** 0-100 vs 0-1 | sacrebleu (BLEU, chrF) and evaluate (chrF) report 0-100 |
| **Text:** BLEU tokenisation | 13a (sacrebleu, evaluate) vs whitespace (torchmetrics `BLEUScore`, nltk): 39.2 vs 32.9 on the same text |
| **Text:** BLEU smoothing | sacrebleu smooths by default; the others don't, and each smooths differently when asked |
| **Text:** corpus vs mean of sentence BLEU/chrF | 0.11 vs 0.32 (nltk) on the same three sentences |
| **Text:** nltk and short texts | nltk counts a text shorter than 4 words as having unmatched n-grams, lowering corpus BLEU |
| **Text:** chrF vs chrF++ | torchmetrics `CHRFScore` defaults to chrF++ (`n_word_order=2`) |
| **Text:** chrF averaging | sacrebleu averages precision/recall over n-gram orders, torchmetrics averages F-scores (`eps_smoothing`) |
| **Text:** ROUGE-Lsum sentence splitting | torchmetrics splits sentences with nltk (0.87); rouge-score and evaluate only on newlines (0.43) |
| **Text:** stemming, lowercasing per library | your settings differ between calls |
| **Text:** non-Latin text in ROUGE | every library drops the characters by default and scores 0 (shown as a note) |
| **Ranking:** tied scores | sklearn averages over tie orders, torchmetrics keeps its sort order, trec_eval sorts ties by document id |
| **Ranking:** scores of 0 or below | torchmetrics MAP/MRR/precision/recall treat them as not retrieved (torchmetrics#3518) |
| **Ranking:** MAP@k denominator | trec_eval divides by all relevant documents, torchmetrics by those found in the top k |
| **Ranking:** queries without relevant documents | counted as 0, skipped or counted as 1 |
| **Ranking:** precision@k on short lists | divide by k, or by the list length (`adaptive_k` / `limit_k_to_size`) |

Metrics:

- **Classification** (binary and multiclass): `accuracy`, `precision`, `recall`, `f1`, `auroc`, `average_precision`
- **Segmentation** (label maps, 2-D or 3-D): `dice`, `iou` (Keras and evaluate have IoU only; torcheval has none)
- **Regression** (single or multi-output): `r2`, `mae`, `mse`, `rmse`, `mape` (torcheval only has `r2` and `mse`; evaluate is single-output only)
- **Text generation**: `bleu`, `chrf` (sacrebleu, nltk for BLEU, torchmetrics, evaluate), `rouge1`, `rouge2`, `rougeL`, `rougeLsum` (rouge-score, torchmetrics, evaluate)
- **Ranking / retrieval**: `ndcg`, `map` (sklearn, torchmetrics, pytrec_eval), `mrr`, `precision_at_k`, `recall_at_k`, `hit_rate` (torchmetrics, pytrec_eval; torcheval for precision@k)

evaluate has no average precision, `zero_division` or top-k; those cases show it as skipped.

**How a cause is confirmed:** each detector knows how to align one setting (for example pass `average="macro"` everywhere, or apply softmax first). metric-diff re-runs all libraries with the alignment and only says *confirmed* if they then agree. If nothing reproduces agreement, it says so: that is often a new bug worth reporting.

## Tested on real model predictions

Beyond the hand-made edge cases, metric-diff runs on real model predictions: logistic regression and ridge models trained on public datasets bundled with scikit-learn (breast_cancer, digits, wine, diabetes), evaluated on a 30% test split. Full tables: [benchmarks/RESULTS.md](https://github.com/DadCoder001/metric-diff/blob/main/benchmarks/RESULTS.md); reproduce with `python benchmarks/real_data.py` (no downloads, about a minute).

What showed up on real data:

- **Keras F1 on real probabilities: 0.787 instead of 0.967** (breast_cancer), because `F1Score()` has no threshold by default.
- **Keras AUROC on real logits: 0.965 instead of 0.996** (breast_cancer) and **0.848 instead of 0.999** (digits, 10 classes), because Keras clips scores to [0, 1].
- **Averaging per-batch R2 (batch size 32): 0.223 instead of 0.365** (diabetes), and per-batch F1 slightly off on every dataset.
- **Keras MAE / MSE / MAPE accumulated over batches with 1-D targets average the batch means**, not the samples, so the result is wrong whenever the last batch is smaller; targets shaped `(batch, 1)` are fine. Found on the diabetes predictions.
- torchmetrics' macro default for multiclass accuracy: 0.9721 vs 0.9722 on digits. Small, but a different number in a results table.
- When the predictions are clean probabilities and settings are explicit, all five libraries agree to 4 decimals.

And on pretrained Hugging Face models ([benchmarks/RESULTS_HF.md](https://github.com/DadCoder001/metric-diff/blob/main/benchmarks/RESULTS_HF.md); `python benchmarks/download_hf.py` fetches about 720 MB once, then `python benchmarks/hf_data.py` runs offline):

- **DistilBERT on SST-2 (872 sentences):** Keras F1 **0.675 instead of 0.914**; Keras AUROC **0.959** on probabilities (its 200 fixed thresholds can't resolve confident predictions bunched near 0 and 1) and **0.911** on logits, vs **0.972** everywhere else; Keras PR-AUC 0.943 vs average precision 0.973. On the model's 2-column output, sklearn raises and wants `scores[:, 1]`.
- **SegFormer on ADE20K (200 validation images, 150 classes):** the same predictions give **Dice 0.284** (all pixels pooled: sklearn), **0.388** (per image: torchmetrics) and **0.538** (MONAI's `ignore_empty=True` default); **IoU 0.209 / 0.335 / 0.464**. MONAI's default alone adds about 0.15 by ignoring false positives on images without the class.
- **torchmetrics' own two IoU APIs disagree:** the `MeanIoU` class gives 0.304 and the `mean_iou` function 0.335 on the same masks, because the class pools all (image, class) pairs while the function averages per image. Found by the batch check; not caused by batching. Reported as [torchmetrics#3660](https://github.com/Lightning-AI/torchmetrics/issues/3660).
- Running on real data also made metric-diff faster: segmentation now processes images in chunks (one-hot masks over 150 classes would need gigabytes) and uses MONAI's label-map Dice path (3x faster, identical values).

## Which number is right?

Usually none is "wrong": the libraries answer slightly different questions. Pick the convention your field reports and state it:

- **Classification and regression:** scikit-learn is the de facto reference; metric-diff uses it as the baseline.
- **Text generation:** report BLEU and chrF from sacrebleu with its signature (tokenizer, smoothing, version), as recommended by [Post 2018](https://aclanthology.org/W18-6319/), and say whether ROUGE used stemming and one sentence per line.
- **Ranking:** trec_eval (pytrec_eval) is the reference behind TREC, MS MARCO and BEIR results; break score ties yourself and state how queries without relevant documents are handled.
- **Segmentation:** follow [Metrics Reloaded](https://doi.org/10.1038/s41592-023-02151-z) (Maier-Hein et al., *Nature Methods* 2024), the consortium recommendations for choosing and aggregating image-analysis metrics, and report whether Dice/IoU is per image or pooled and whether background is included.

## Limitations

- Settings supported: `average` (micro / macro / weighted), `threshold`, `pos_label`, `num_classes`, `zero_division`, `top_k` (accuracy). Not yet: `ignore_index` (masking the samples gives the same result in every library), sample weights, per-class output (`average=None`).
- Segmentation has no `ignore_index` yet. The usual workaround (ignored pixels set to class 0 in both masks, with `include_background=False`) is exact for sklearn, torchmetrics and MONAI, but Keras and evaluate can't leave class 0 out, so they are skipped in that setup.
- No multilabel, sample weights or per-class output yet. Regression has no settings yet (e.g. `multioutput`).
- Text: BLEU, chrF and ROUGE only (no METEOR, BERTScore, WER yet); nltk is compared on BLEU only (its chrF uses other definitions). `compare_batched` doesn't cover text or ranking metrics.
- Ranking: linear-gain NDCG only (as in sklearn, torchmetrics and trec_eval); `ranx` isn't compared yet (it pulls in a large dependency tree).
- Explanations describe the library versions tested (sklearn 1.9, torchmetrics 1.9, torcheval 0.0.7, Keras 3.15, evaluate 0.4.6, MONAI 1.6). Tests check the key defaults so they fail loudly when a library changes.

## Roadmap

- More settings: sample weights, per-class output
- Multilabel classification; regression settings (`multioutput`, sample weights)
- Detection mAP (pycocotools / COCO conventions)
- More edge cases in the matrix (send yours as an issue with `--json` output)
- Lightning and Keras callbacks that check the logged epoch metric
- WER/CER, calibration (ECE), multilabel

## Contributing

Every known mismatch lives in `src/metric_diff/cases.py` (the matrix) and in `tests/` as a regression case linked to its upstream issue. Found a new one? [Open an issue](https://github.com/DadCoder001/metric-diff/issues/new/choose) with the `--json` output. Questions and ideas go to [Discussions](https://github.com/DadCoder001/metric-diff/discussions).

See [CONTRIBUTING.md](https://github.com/DadCoder001/metric-diff/blob/main/CONTRIBUTING.md) for how to add a detector or an edge case, and the [Code of Conduct](https://github.com/DadCoder001/metric-diff/blob/main/CODE_OF_CONDUCT.md). Changes are listed in the [changelog](https://github.com/DadCoder001/metric-diff/blob/main/CHANGELOG.md).

## License

[MIT](https://github.com/DadCoder001/metric-diff/blob/main/LICENSE)
