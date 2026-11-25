# Changelog

All notable changes are listed here. This project follows [semantic versioning](https://semver.org/); while the version starts with 0, minor releases may change behaviour.

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
