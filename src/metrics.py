"""二分类指标。"""

from __future__ import annotations

from typing import Iterable


def compute_binary_metrics(
    labels: Iterable[int],
    scores: Iterable[float],
    threshold: float = 0.0,
) -> dict[str, float]:
    label_list = [int(x) for x in labels]
    pred_list = [1 if float(score) > threshold else 0 for score in scores]

    if len(label_list) != len(pred_list):
        raise ValueError("labels and scores must have the same length")
    if not label_list:
        return {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "positive_F1": 0.0,
            "macro_F1": 0.0,
        }

    tp = sum(1 for y, p in zip(label_list, pred_list) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(label_list, pred_list) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(label_list, pred_list) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(label_list, pred_list) if y == 1 and p == 0)

    accuracy = (tp + tn) / len(label_list)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    negative_precision = tn / (tn + fn) if (tn + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    negative_f1 = (
        2.0 * negative_precision * specificity / (negative_precision + specificity)
        if (negative_precision + specificity)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_F1": f1,
        "macro_F1": (f1 + negative_f1) / 2.0,
    }
