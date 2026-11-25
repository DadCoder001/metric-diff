"""The edge-case library behind the conformance matrix.

Each case is a small, deterministic input known to separate (or, for the "clean" controls,
not separate) metric libraries. Data is built lazily so importing metric-diff stays cheap.
Add a case here whenever a new mismatch is reported: it then runs in every matrix and is
watched for drift across library versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class EdgeCase:
    id: str
    title: str
    metric: str
    data: Callable[[], tuple]
    kwargs: dict = field(default_factory=dict)
    reference: str | None = None  # upstream issue or source

    def inputs(self):
        return self.data()


# ---------------------------------------------------------------- data builders

_HYP = ["The cat sat on the mat.", "It is a guide to action, which ensures that the military always "
        "obeys the commands of the party.", "Hello there!"]
_REF = ["The cat is sitting on the mat.", "It is a guide to action that ensures that the military will "
        "forever heed Party commands.", "Hello, there."]


def _text():
    return _REF, _HYP


def _text_two_sentences():
    return (["The dog barked at night. The cat sat on a mat."],
            ["The cat sat on the mat. The dog barked loudly at night."])


_REL = [[3, 2, 0, 1, 0], [0, 1, 0, 0, 1], [2, 0, 0, 1, 0]]


def _ranking():
    return _REL, [[0.9, 0.8, 0.7, 0.2, 0.1], [0.5, 0.4, 0.9, 0.3, 0.2], [0.3, 0.6, 0.1, 0.8, 0.2]]


def _ranking_ties():
    return _REL, [[0.5, 0.5, 0.5, 0.2, 0.1], [0.5, 0.5, 0.9, 0.5, 0.2], [0.3, 0.3, 0.3, 0.3, 0.2]]


def _ranking_logits():
    rel, sc = _ranking()
    return rel, [[x * 4 - 2 for x in s] for s in sc]


def _ranking_empty():
    rel, sc = _ranking()
    return rel + [[0, 0, 0, 0, 0]], sc + [[0.1, 0.2, 0.3, 0.4, 0.5]]

_YB = [0, 1, 1, 0, 1, 0, 1, 1]
_PB = [0.1, 0.8, 0.65, 0.3, 0.9, 0.45, 0.2, 0.7]


def _rng(seed=0):
    return np.random.default_rng(seed)


def _bin_probs():
    return _YB, _PB


def _bin_logits():
    return [0, 1, 1, 0, 1, 1], [-2.0, 3.0, -0.5, -3.0, 4.0, 0.3]


def _bin_logits_small():
    return _YB, [-2.0, 3.0, -0.5, -3.0, 4.0, 0.3, 1.0, -1.0]


def _large_logits():
    preds = [98.0950, 98.4612, 98.1145, 98.1506, 97.6037, 98.9425, 99.2644, 99.5014,
             99.7280, 99.6595, 99.6931, 99.4667, 99.9623, 99.8949, 99.8768]
    return [0] + [1] * 14, preds


def _ties():
    return [1, 0, 0, 1], [0.5, 0.5, 0.2, 0.8]


def _labels_12():
    return [1, 2, 1, 1], [1, 2, 2, 1]


def _bin_500():
    r = _rng(0)
    y = r.integers(0, 2, 500)
    return y, np.clip(y * 0.3 + r.random(500) * 0.7, 0, 1)


def _mc_labels():
    return [0, 0, 0, 0, 1, 1, 2, 2, 2, 2], [0, 0, 1, 0, 1, 2, 2, 2, 0, 2]


def _mc_probs():
    r = _rng(3)
    y = np.array([0] * 20 + [1] * 6 + [2] * 4)
    p = r.random((30, 3)) + np.eye(3)[y] * 0.6
    return y, p / p.sum(1, keepdims=True)


def _mc_logits():
    return [0, 0, 0, 0, 1, 1, 2, 2, 2, 2], _rng(0).normal(size=(10, 3))


def _mc_unnormalised():
    y = [0, 1, 2, 0, 1, 2]
    p = [[0.9, 0.3, 0.1], [0.2, 0.8, 0.4], [0.1, 0.2, 0.7],
         [0.6, 0.5, 0.2], [0.3, 0.6, 0.5], [0.2, 0.1, 0.9]]
    return y, p


def _absent_class():
    return [0, 1, 2, 2, 1, 0, 2, 2], [0, 2, 2, 2, 1, 0, 1, 2]


def _zero_division():
    return [0, 1, 2, 2, 1, 0], [0, 1, 1, 0, 1, 0]


def _topk_ties():
    return [2, 3, 2], [[0.4, 0.3, 0.3, 0.0], [0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.6, 0.1]]


def _reg():
    return [3.0, -0.5, 2.0, 7.0, 4.2], [2.5, 0.0, 2.1, 7.8, 3.9]


def _reg_zero():
    return [0.0, 1.0, 2.0], [0.5, 1.0, 2.5]


def _reg_constant():
    return [2.0, 2.0, 2.0], [2.0, 2.0, 2.1]


def _reg_large():
    r = _rng(0)
    y = 1e6 + r.normal(size=1000)
    return y, y + r.normal(scale=0.1, size=1000)


def _reg_multi():
    y = np.array([[0.5, 1.0], [-1.0, 1.0], [7.0, -6.0], [2.0, 3.0]])
    p = np.array([[0.0, 2.0], [-1.0, 2.0], [8.0, -5.0], [2.5, 2.0]])
    return y, p


def _masks():
    t = np.zeros((3, 4, 4), int)
    p = np.zeros((3, 4, 4), int)
    t[0, :2, :2] = 1; p[0, :2, :3] = 1; t[0, 2:, 2:] = 2; p[0, 3:, 2:] = 2  # noqa: E702
    t[1, :2, :] = 1; p[1, :2, :] = 1; p[1, 3, 3] = 2  # noqa: E702
    t[2, 1:3, 1:3] = 1; p[2, 1:3, 1:2] = 1  # noqa: E702
    return t, p


# ---------------------------------------------------------------- the cases

CASES: list[EdgeCase] = [
    # classification, binary
    EdgeCase("bin-f1-probs", "Binary F1, probabilities", "f1", _bin_probs),
    EdgeCase("bin-auroc-clean", "Binary AUROC, probabilities (control)", "auroc", _bin_probs),
    EdgeCase("bin-f1-logits", "Binary F1 on logits", "f1", _bin_logits, reference="Lightning-AI/torchmetrics#2329"),
    EdgeCase("bin-auroc-large-logits", "Binary AUROC, large logits", "auroc", _large_logits,
             reference="Lightning-AI/torchmetrics#2819"),
    EdgeCase("bin-auroc-logits", "Binary AUROC on logits", "auroc", _bin_logits_small),
    EdgeCase("bin-precision-tie", "Precision, scores exactly 0.5", "precision", _ties),
    EdgeCase("bin-f1-labels-1-2", "Binary F1, labels {1, 2}", "f1", _labels_12,
             reference="huggingface/evaluate#869"),
    EdgeCase("bin-ap", "Average precision", "average_precision", _bin_probs),
    EdgeCase("bin-auroc-500", "AUROC, 500 samples (threshold binning)", "auroc", _bin_500),
    # classification, multiclass
    EdgeCase("mc-accuracy-default", "Multiclass accuracy, defaults", "accuracy", _mc_labels,
             reference="Lightning-AI/torchmetrics#746"),
    EdgeCase("mc-f1-default", "Multiclass F1, defaults", "f1", _mc_labels,
             reference="meta-pytorch/torcheval#191"),
    EdgeCase("mc-f1-weighted", "Multiclass F1, average='weighted'", "f1", _mc_probs, {"average": "weighted"}),
    EdgeCase("mc-precision-probs", "Multiclass precision from probabilities", "precision", _mc_probs,
             {"average": "micro"}),
    EdgeCase("mc-auroc-logits", "Multiclass AUROC on logits", "auroc", _mc_logits),
    EdgeCase("mc-auroc-unnormalised", "Multiclass AUROC, rows not summing to 1", "auroc", _mc_unnormalised,
             {"average": "macro"}),
    EdgeCase("mc-f1-absent-class", "Macro F1, num_classes=4, class 3 unseen", "f1", _absent_class,
             {"num_classes": 4, "average": "macro"}),
    EdgeCase("mc-precision-zero-division", "Macro precision, zero_division=1", "precision", _zero_division,
             {"average": "macro", "zero_division": 1}),
    EdgeCase("mc-top2-ties", "Top-2 accuracy with tied scores", "accuracy", _topk_ties,
             {"top_k": 2, "average": "micro"}),
    # regression
    EdgeCase("reg-r2-clean", "R2 (control)", "r2", _reg),
    EdgeCase("reg-mse-clean", "MSE (control)", "mse", _reg),
    EdgeCase("reg-mape", "MAPE", "mape", _reg),
    EdgeCase("reg-mape-zero-target", "MAPE with a zero target", "mape", _reg_zero),
    EdgeCase("reg-r2-constant", "R2, constant target", "r2", _reg_constant),
    EdgeCase("reg-r2-large-values", "R2, values around 1e6", "r2", _reg_large,
             reference="Lightning-AI/torchmetrics#3077"),
    EdgeCase("reg-r2-multioutput", "R2, two outputs", "r2", _reg_multi),
    # segmentation
    EdgeCase("seg-dice-default", "Dice, defaults", "dice", _masks),
    EdgeCase("seg-iou-default", "IoU, defaults", "iou", _masks),
    EdgeCase("seg-dice-per-image", "Dice, aggregation='per_image'", "dice", _masks, {"aggregation": "per_image"}),
    EdgeCase("seg-iou-dataset", "IoU, aggregation='dataset'", "iou", _masks, {"aggregation": "dataset"}),
    EdgeCase("seg-iou-no-background", "IoU per image, no background", "iou", _masks,
             {"aggregation": "per_image", "include_background": False}),
    # text generation
    EdgeCase("text-bleu-default", "BLEU, defaults", "bleu", _text),
    EdgeCase("text-bleu-13a", "BLEU, tokenize='13a'", "bleu", _text, {"tokenize": "13a"}),
    EdgeCase("text-bleu-sentence", "BLEU, mean of sentence scores", "bleu", _text,
             {"aggregation": "sentence", "tokenize": "13a"}),
    EdgeCase("text-chrf-default", "chrF, defaults", "chrf", _text),
    EdgeCase("text-chrf-plusplus", "chrF++, word_order=2", "chrf", _text, {"word_order": 2}),
    EdgeCase("text-rougeL", "ROUGE-L (control)", "rougeL", _text),
    EdgeCase("text-rougeLsum-one-line", "ROUGE-Lsum, sentences on one line", "rougeLsum", _text_two_sentences),
    # ranking / retrieval
    EdgeCase("rank-ndcg-clean", "NDCG (control)", "ndcg", _ranking),
    EdgeCase("rank-map-clean", "MAP (control)", "map", _ranking),
    EdgeCase("rank-map-ties", "MAP, tied scores", "map", _ranking_ties),
    EdgeCase("rank-mrr-ties", "MRR, tied scores", "mrr", _ranking_ties),
    EdgeCase("rank-map-at-3", "MAP@3", "map", _ranking, {"top_k": 3}),
    EdgeCase("rank-map-logits", "MAP, negative scores (logits)", "map", _ranking_logits,
             reference="Lightning-AI/torchmetrics#3518"),
    EdgeCase("rank-p-at-4-logits", "precision@4, negative scores", "precision_at_k", _ranking_logits,
             {"top_k": 4}, reference="Lightning-AI/torchmetrics#3518"),
    EdgeCase("rank-ndcg-empty-query", "NDCG, a query without relevant documents", "ndcg", _ranking_empty),
]


def select(patterns: list[str] | None) -> list[EdgeCase]:
    if not patterns:
        return list(CASES)
    out = [c for c in CASES if any(p in c.id for p in patterns)]
    if not out:
        raise ValueError(f"no edge case matches {patterns}; ids look like {CASES[0].id!r}")
    return out
