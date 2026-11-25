# metric-diff on pretrained Hugging Face models

- Text: `distilbert/distilbert-base-uncased-finetuned-sst-2-english` on SST-2 validation.
- Segmentation: `nvidia/segformer-b0-finetuned-ade-512-512` on ADE20K validation images, 128x128 model output. Full-ADE classes mapped to the model's 150 by name cover 91% of labelled pixels; the rest are ignored (class 0, `include_background=False`).
- Reproduce: `python benchmarks/download_hf.py` (about 720 MB, once), then `python benchmarks/hf_data.py`.
- Datasets and models are downloaded from the Hugging Face Hub, not redistributed; their own licences and terms of use apply.

Versions: sklearn 1.9.1, torchmetrics 1.9.0, torcheval 0.0.7, keras 3.15.1, evaluate 0.4.6, monai 1.6.2, transformers 5.24.0

## Text classification

| Data | Metric | Input | Settings | sklearn | torchmetrics | torcheval | keras | evaluate | Result | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| SST-2 (872) | accuracy | probabilities |  | 0.9106 | 0.9106 | 0.9106 | 0.9106 | 0.9106 | agree | 22.5s |
| SST-2 (872) | precision | probabilities |  | 0.8978 | 0.8978 | 0.8978 | 0.8978 | 0.8978 | agree | 1.1s |
| SST-2 (872) | recall | probabilities |  | 0.9302 | 0.9302 | 0.9302 | 0.9302 | 0.9302 | agree | 1.1s |
| SST-2 (872) | f1 | probabilities |  | 0.9137 | 0.9137 | 0.9137 | 0.6748 | 0.9137 | `keras_f1_threshold` | 1.1s |
| SST-2 (872) | f1 | logit margin |  | 0.9113 | 0.9137 | 0.9113 | 0.6748 | 0.9113 | `logits`, `keras_f1_threshold` | 0.1s |
| SST-2 (872) | auroc | probabilities |  | 0.9717 | 0.9717 | 0.9717 | 0.9591 | 0.9717 | `keras_bins` | 1.2s |
| SST-2 (872) | auroc | logit margin |  | 0.9717 | 0.9717 | 0.9717 | 0.9105 | 0.9717 | `keras_logits` | 0.1s |
| SST-2 (872) | average_precision | probabilities |  | 0.9730 | 0.9730 | 0.9730 | 0.9432 | skipped | `keras_pr` | 0.1s |
| SST-2 (872) | f1 | 2-column logits |  | 0.9137 | 0.9104 | 0.9106 | error | 0.9137 | `average` | 0.1s |
| SST-2 (872) | auroc | 2-column logits |  | error | 0.9717 | 0.9717 | 0.9159 | error | `softmax`, `keras_bins`, `average` | 3.4s |

## Segmentation

| Data | Metric | Input | Settings | sklearn | torchmetrics | keras | evaluate | monai | Result | Time |
|---|---|---|---|---|---|---|---|---|---|---|
| ADE20K (200 images) | dice | label maps | include_background=False | 0.2844 | 0.3882 | skipped | skipped | 0.5383 | `monai_ignore_empty`, `seg_aggregation` | 34.1s |
| ADE20K (200 images) | dice | label maps | include_background=False, aggregation='per_image' | 0.3882 | 0.3882 | skipped | skipped | 0.5383 | `monai_ignore_empty` | 18.7s |
| ADE20K (200 images) | iou | label maps | include_background=False | 0.2092 | 0.3350 | skipped | skipped | 0.4639 | `monai_ignore_empty`, `seg_aggregation` | 29.0s |
| ADE20K (200 images) | iou | label maps | include_background=False, aggregation='per_image' | 0.3350 | 0.3350 | skipped | skipped | 0.4639 | `monai_ignore_empty` | 17.0s |
| ADE20K (200 images) | iou | label maps | include_background=False, aggregation='dataset' | 0.2092 | skipped | skipped | skipped | skipped |  | 3.4s |

## Per-batch averaging vs the whole set

| Data | Metric | Batch size | Whole set | Mean of batches | Batch-size weighted | Batches left out | Findings |
|---|---|---|---|---|---|---|---|
| SST-2 (872) | f1 | 32 | 0.9137 | 0.9091 | 0.9105 | 0 of 28 | `batch_average` |
| SST-2 (872) | auroc | 32 | 0.9717 | 0.9769 | 0.9763 | 0 of 28 | `batch_average` |
| ADE20K (200 images) | iou | 16 | 0.2092 | 0.2130 | 0.2101 | 0 of 13 | `batch_average`, `stateful_torchmetrics` |

Model inference: text 0s, segmentation 0s (cached after the first run). Total run time: 147s.
