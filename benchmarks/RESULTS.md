# metric-diff on real model predictions

Models trained on 70% of datasets bundled with scikit-learn; metrics computed on the other 30%. Reproduce with `python benchmarks/real_data.py`.

Versions: sklearn 1.9.1, torchmetrics 1.9.0, torcheval 0.0.7, keras 3.15.1, evaluate 0.4.6

## Libraries compared on the same predictions

| Dataset (n) | Metric | Input | Settings | sklearn | torchmetrics | torcheval | keras | evaluate | Result | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| breast_cancer (171) | accuracy | probabilities |  | 0.9591 | 0.9591 | 0.9591 | 0.9591 | 0.9591 | agree | 21.6s |
| breast_cancer (171) | precision | probabilities |  | 0.9717 | 0.9717 | 0.9717 | 0.9717 | 0.9717 | agree | 2.1s |
| breast_cancer (171) | recall | probabilities |  | 0.9626 | 0.9626 | 0.9626 | 0.9626 | 0.9626 | agree | 2.1s |
| breast_cancer (171) | f1 | probabilities |  | 0.9671 | 0.9671 | 0.9671 | 0.7868 | 0.9671 | `keras_f1_threshold` | 2.2s |
| breast_cancer (171) | f1 | logits |  | 0.9619 | 0.9671 | 0.9619 | 0.7698 | 0.9619 | `logits`, `keras_f1_threshold` | 0.2s |
| breast_cancer (171) | auroc | probabilities |  | 0.9956 | 0.9956 | 0.9956 | 0.9956 | 0.9956 | agree | 2.0s |
| breast_cancer (171) | auroc | logits |  | 0.9956 | 0.9956 | 0.9956 | 0.9650 | 0.9956 | `keras_logits` | 0.2s |
| breast_cancer (171) | average_precision | probabilities |  | 0.9974 | 0.9974 | 0.9974 | 0.9974 | skipped | agree | 0.0s |
| digits (540) | accuracy | labels |  | 0.9722 | 0.9721 | 0.9722 | 0.9722 | 0.9722 | `average` | 0.1s |
| digits (540) | f1 | labels |  | error | 0.9725 | 0.9722 | error | error | `average` | 1.1s |
| digits (540) | f1 | labels | average='macro' | 0.9725 | 0.9725 | 0.9725 | 0.9725 | 0.9725 | agree | 0.1s |
| digits (540) | precision | probabilities | average='macro' | 0.9736 | 0.9736 | 0.9736 | 0.9736 | 0.9736 | agree | 0.1s |
| digits (540) | auroc | probabilities |  | error | 0.9992 | 0.9992 | 0.9993 | error | `average` | 3.9s |
| digits (540) | auroc | logits |  | error | 0.9992 | 0.9933 | 0.8480 | error | `softmax`, `average` | 4.5s |
| digits (540) | average_precision | probabilities | average='macro' | 0.9925 | 0.9925 | 0.9925 | 0.9923 | skipped | `keras_pr` | 1.0s |
| digits (540) | accuracy | probabilities | top_k=3, average='micro' | 1.0000 | 1.0000 | 1.0000 | 1.0000 | skipped | agree | 0.0s |
| wine (54) | accuracy | labels |  | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | agree | 0.0s |
| wine (54) | f1 | labels |  | error | 1.0000 | 1.0000 | error | error | `average` | 1.1s |
| wine (54) | f1 | labels | average='macro' | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | agree | 0.0s |
| wine (54) | precision | probabilities | average='macro' | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | agree | 0.1s |
| wine (54) | auroc | probabilities |  | error | 1.0000 | 1.0000 | 1.0000 | error | `average` | 2.1s |
| wine (54) | auroc | logits |  | error | 1.0000 | 1.0000 | 0.9815 | error | `softmax`, `average` | 3.3s |
| wine (54) | average_precision | probabilities | average='macro' | 1.0000 | 1.0000 | 1.0000 | 1.0000 | skipped | agree | 0.1s |
| diabetes (133) | r2 | predictions |  | 0.3652 | 0.3652 | 0.3652 | 0.3652 | 0.3650 | `evaluate_r2_rounding` | 2.3s |
| diabetes (133) | mae | predictions |  | 45.8088 | 45.8088 | skipped | 45.8088 | 45.8088 | agree | 2.0s |
| diabetes (133) | mse | predictions |  | 3238.4805 | 3238.4805 | 3238.4805 | 3238.4805 | 3238.4805 | agree | 2.2s |
| diabetes (133) | rmse | predictions |  | 56.9076 | 56.9076 | skipped | 56.9076 | 56.9076 | agree | 0.0s |
| diabetes (133) | mape | predictions |  | 0.4183 | 0.4183 | skipped | 41.8331 | 0.4183 | `mape_percent` | 2.0s |

## Per-batch averaging vs the whole test set (batch size 32)

| Dataset (n) | Metric | Whole test set | Mean of batches | Batch-size weighted | Batches left out | Findings |
|---|---|---|---|---|---|---|
| breast_cancer (171) | f1 | 0.9671 | 0.9613 | 0.9638 | 0 of 6 | `batch_average` |
| breast_cancer (171) | auroc | 0.9956 | 0.9971 | 0.9967 | 0 of 6 | `batch_average` |
| digits (540) | f1 | 0.9725 | 0.9678 | 0.9676 | 0 of 17 | `batch_average` |
| digits (540) | auroc | 0.9992 | 0.9992 | 0.9992 | 6 of 17 |  |
| diabetes (133) | r2 | 0.3652 | 0.2231 | 0.3282 | 0 of 5 | `batch_average` |
| diabetes (133) | mape | 0.4183 | 0.4144 | 0.4183 | 0 of 5 | `batch_average`, `stateful_keras` |

Total run time: 61s.
