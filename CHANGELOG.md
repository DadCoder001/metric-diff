# Changelog

All notable changes are listed here. This project follows [semantic versioning](https://semver.org/); while the version starts with 0, minor releases may change behaviour.

## Unreleased

### GitHub Action
- `uses: DadCoder001/metric-diff@<version>` runs the conformance matrix on the library versions your workflow installs and fails the job when a result changed since the committed baseline. On the first run it saves the baseline and uploads it as an artifact; `update-baseline: true` refreshes it. Inputs `baseline`, `extras`, `cases`, `backends`, `fail-on-drift`, `artifact-name`; outputs `drift`, `baseline-created`.
- `metric-diff matrix --summary FILE` appends a Markdown drift report and the matrix to a file such as `$GITHUB_STEP_SUMMARY`.
- Matrix runs record the operating system, and the drift report warns when the baseline came from a different OS (tie order can differ by platform). Older baseline files still load.

## 0.1.0a2 (2026-11-25)

### Test-suite assertions
- `metric_diff.testing.assert_metrics_agree()`: fails unless every installed library agrees (or differs only for causes you `allow`), optionally checks an `expected` value, and requires at least `min_libraries` libraries so a test can't pass because one isn't installed. Failures raise `MetricMismatch` (an `AssertionError`) with the full report.
- `assert_batches_consistent()`: fails on per-batch averaging, duplicated or missing samples.
- Reports now show each cause's key in brackets (`[threshold_tie]`), the name used by `allow`.

### Text generation: BLEU, chrF, ROUGE
- New libraries: sacrebleu, nltk (BLEU), rouge-score; torchmetrics and Hugging Face evaluate also compute text metrics. `pip install "metric-diff[text]"`.
- Metrics `bleu`, `chrf`, `rouge1`, `rouge2`, `rougeL`, `rougeLsum`; one or several references per sample.
- Settings `tokenize`, `smooth`, `lowercase`, `aggregation` ("corpus" / "sentence"), `word_order`, `use_stemmer`.
- 10 detectors: 0-100 vs 0-1 scale, BLEU tokenisation (13a vs whitespace), smoothing (on by default in sacrebleu, a different method in each library), corpus vs sentence averaging, nltk's handling of texts shorter than 4 words, chrF vs chrF++ (torchmetrics defaults to chrF++), chrF averaging over n-gram orders (sacrebleu `eps_smoothing`), lowercasing, ROUGE stemming, ROUGE-Lsum sentence splitting (torchmetrics splits sentences, rouge-score and evaluate only on newlines). A note warns when non-Latin text would be dropped or mis-tokenised.
- CLI: one text per line; repeat `--true` for several references.

### Ranking and retrieval: NDCG, MAP, MRR, precision@k, recall@k, hit rate
- New library: pytrec_eval (trec_eval); sklearn, torchmetrics and torcheval also compute ranking metrics. `pip install "metric-diff[ranking]"`.
- Inputs: one row per query (relevance and scores), rows may differ in length; graded relevance for NDCG.
- Settings `top_k`, `empty_queries` ("zero" / "skip" / "one"), `adaptive_k`.
- 5 detectors: tied scores (each library orders them differently), torchmetrics ignoring candidates scored 0 or below in MAP/MRR/precision/recall (Lightning-AI/torchmetrics#3518), MAP@k denominators (trec_eval vs torchmetrics), queries without relevant documents, precision@k on short lists.
- CLI: one query per line.

### Conformance matrix
- 15 new edge cases (45 in total): 7 text generation, 8 ranking.

### Fixes
- Attribution of causes now uses the libraries each setting actually changes, so the right cause is named when several alignments reproduce agreement.

## 0.1.0a1 (first public alpha)

### Compare libraries
- `compare()` / `metric-diff <metric>`: computes a metric with every installed library (scikit-learn, torchmetrics, torcheval, Keras 3, Hugging Face evaluate, MONAI) and, when they disagree, finds the cause and **confirms** it by re-running every library with that setting aligned.
- Classification: accuracy, precision, recall, F1, AUROC, average precision (binary and multiclass; labels, probabilities or logits).
- Regression: R2, MAE, MSE, RMSE, MAPE (single and multi-output).
- Segmentation: Dice and IoU on label maps (2-D and 3-D).
- Your own settings, for every library or per library: `average`, `threshold`, `pos_label`, `num_classes`, `zero_division`, `top_k`, `include_background`, `aggregation`, `settings={lib: {...}}`. Libraries that can't honour a setting are reported as skipped, with the reason.
- 25 detectors, including default averaging, logits vs probabilities, sigmoid saturation of large logits (torchmetrics#2819), threshold ties, binary labels other than {0, 1}, Keras defaults (F1 without threshold, per-column precision/recall, AUC clipping and 200-threshold binning, PR-AUC vs average precision, MAPE as a percentage), MAPE with zero targets, constant-target R2, float32 rounding on large values, per-image vs whole-dataset segmentation, MONAI `ignore_empty`, and background handling.

### Batches
- `compare_batched()` / `--batch-size`: per-batch averages vs accumulated `update()`/`compute()` vs the whole dataset; flags batches left out, duplicated samples (DistributedSampler padding, `--expected-n`) and dropped samples. Explains Keras' 1-D target accumulation and torchmetrics' `MeanIoU` class vs `mean_iou` function.

### Conformance matrix
- `metric-diff matrix` / `run_matrix()`: 30 edge cases on every installed library, as a terminal table, Markdown or JSON; `--save` and `--baseline` detect drift between library versions (exit code 1, for CI).

### Documentation and benchmarks
- Scenario-based [user guide](docs/USER_GUIDE.md); every Python example in it is tested.
- Benchmarks on real model predictions: scikit-learn datasets (`benchmarks/real_data.py`) and pretrained Hugging Face models on SST-2 and ADE20K (`benchmarks/hf_data.py`).
