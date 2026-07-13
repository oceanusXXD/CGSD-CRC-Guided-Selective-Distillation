from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


def compare_scored_models(
    *,
    task: str,
    base_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    uncertainty_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    aligned = _align_rows(base_rows, random_rows, uncertainty_rows)
    if task == "classification":
        metric_function = _classification_metrics
    elif task == "preference":
        metric_function = _preference_metrics
    else:
        raise ValueError(f"unsupported task: {task}")
    metrics = {
        name: metric_function(rows)
        for name, rows in zip(("base", "random", "uncertainty"), aligned, strict=True)
    }
    return {
        "task": task,
        "size": len(base_rows),
        "models": metrics,
        "deltas_vs_base": {
            "random": _numeric_deltas(metrics["random"], metrics["base"]),
            "uncertainty": _numeric_deltas(metrics["uncertainty"], metrics["base"]),
        },
        "uncertainty_minus_random": _numeric_deltas(
            metrics["uncertainty"], metrics["random"]
        ),
    }


def _align_rows(
    base_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    uncertainty_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    base_ids = [str(row["id"]) for row in base_rows]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError("base rows contain duplicate ids")
    aligned = [base_rows]
    for name, rows in (("random", random_rows), ("uncertainty", uncertainty_rows)):
        by_id = {str(row["id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"{name} rows contain duplicate ids")
        if set(by_id) != set(base_ids):
            raise ValueError(f"{name} rows do not match base ids")
        aligned.append([by_id[sample_id] for sample_id in base_ids])
    _validate_groundtruth(aligned)
    return aligned[0], aligned[1], aligned[2]


def _validate_groundtruth(aligned: list[list[dict[str, Any]]]) -> None:
    fields = ("label", "preferred_response")
    for row_index in range(len(aligned[0])):
        for field in fields:
            values = [rows[row_index].get(field) for rows in aligned]
            present = [value for value in values if value is not None]
            if present and any(value != present[0] for value in present):
                raise ValueError(f"groundtruth field {field!r} differs across scored files")


def _classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groundtruth = [int(row["label"]) for row in rows]
    predictions = [int(row["predicted_label"]) for row in rows]
    labels = sorted(set(groundtruth) | set(predictions))
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in labels:
        true_positive = sum(
            truth == label and prediction == label
            for truth, prediction in zip(groundtruth, predictions, strict=True)
        )
        false_positive = sum(
            truth != label and prediction == label
            for truth, prediction in zip(groundtruth, predictions, strict=True)
        )
        false_negative = sum(
            truth == label and prediction != label
            for truth, prediction in zip(groundtruth, predictions, strict=True)
        )
        support = sum(truth == label for truth in groundtruth)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[str(label)] = {
            "support": support,
            "accuracy": true_positive / support if support else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    correct = sum(truth == prediction for truth, prediction in zip(groundtruth, predictions, strict=True))
    return {
        "evaluated_size": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "macro_f1": mean(f1_values) if f1_values else 0.0,
        "prediction_counts": dict(sorted(Counter(str(value) for value in predictions).items())),
        "per_class": per_class,
    }


def _preference_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in rows if int(row.get("preferred_response", 0)) in (1, 2)]
    correct = sum(
        int(row["predicted_response"]) == int(row["preferred_response"]) for row in evaluated
    )
    metrics: dict[str, Any] = {
        "evaluated_size": len(evaluated),
        "tie_size": len(rows) - len(evaluated),
        "accuracy_excluding_ties": correct / len(evaluated) if evaluated else None,
        "predicted_response_1_rate": (
            sum(int(row["predicted_response"]) == 1 for row in rows) / len(rows) if rows else None
        ),
    }
    for field in ("order_disagreement", "entropy", "margin", "probability_response_1"):
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        if values:
            metrics[f"mean_{field}"] = mean(values)
    return metrics


def _numeric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in sorted(set(left) & set(right)):
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, bool) or isinstance(right_value, bool):
            continue
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            deltas[key] = float(left_value) - float(right_value)
    return deltas
