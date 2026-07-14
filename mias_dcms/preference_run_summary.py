from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from mias_dcms.records import RunRecord


def estimate_preference_train_tokens(rows: Iterable[Mapping[str, Any]]) -> int:
    total = 0
    for row in rows:
        if row.get("train_tokens") is not None:
            total += int(row["train_tokens"])
            continue
        total += (
            _word_count(row.get("prompt", ""))
            + _word_count(row.get("response_1", ""))
            + _word_count(row.get("response_2", ""))
            + 4
        )
    return total


def build_preference_run_record(
    *,
    dataset: str,
    model: str,
    method: str,
    budget: int,
    seed: int,
    config_hash: str,
    selection_summary: Mapping[str, Any],
    reveal_summary: Mapping[str, Any],
    training_rows: Iterable[Mapping[str, Any]],
    training_metrics: Mapping[str, Any] | None = None,
    evaluation_metrics: Mapping[str, Any] | None = None,
    seed_label_count: int = 0,
    evaluation_label_count: int = 0,
    judge_calls: int = 0,
    selector_compute_seconds: float = 0.0,
) -> RunRecord:
    selected_count = int(selection_summary.get("selected_count", budget))
    if selected_count != int(budget):
        raise ValueError("selection selected_count must equal budget")
    revealed_count = int(reveal_summary.get("revealed_count", selected_count))
    if revealed_count != int(budget):
        raise ValueError("reveal revealed_count must equal budget")

    train_rows = [dict(row) for row in training_rows]
    selection_metrics = dict(selection_summary.get("selection_metrics") or {})
    for field in ("pool_size", "selected_score_min", "selected_score_max", "utility_retained", "max_constraint_violation"):
        if field in selection_summary and selection_summary[field] is not None:
            selection_metrics[field] = selection_summary[field]

    merged_training_metrics = dict(training_metrics or {})
    merged_training_metrics.setdefault("dpo_train_row_count", int(reveal_summary.get("dpo_train_row_count", len(train_rows))))
    merged_training_metrics.setdefault("revealed_count", revealed_count)
    merged_training_metrics.setdefault("unrevealed_count", int(reveal_summary.get("unrevealed_count", 0)))

    cost_metrics = {
        "seed_label_count": int(seed_label_count),
        "active_label_count": revealed_count,
        "evaluation_label_count": int(evaluation_label_count),
        "judge_calls": int(judge_calls),
        "train_tokens": estimate_preference_train_tokens(train_rows),
        "selector_compute_seconds": float(selector_compute_seconds),
        "oracle_label_calls": revealed_count,
    }

    return RunRecord(
        dataset=str(dataset),
        model=str(model),
        method=str(method),
        budget=int(budget),
        seed=int(seed),
        selected_count=selected_count,
        config_hash=str(config_hash),
        selection_metrics=selection_metrics,
        training_metrics=merged_training_metrics,
        evaluation_metrics=dict(evaluation_metrics or {}),
        cost_metrics=cost_metrics,
        continuous_moments=_numeric_mapping(selection_summary.get("continuous_moments")),
        rounded_moments=_numeric_mapping(selection_summary.get("rounded_moments")),
        robust_lower_moments=_numeric_mapping(selection_summary.get("robust_lower_moments")),
        robust_upper_moments=_numeric_mapping(selection_summary.get("robust_upper_moments")),
        utility_retained=float(selection_summary.get("utility_retained", selection_metrics.get("utility_retained", 0.0))),
        max_constraint_violation=float(
            selection_summary.get(
                "max_constraint_violation", selection_metrics.get("max_constraint_violation", 0.0)
            )
        ),
        solver_status=str(selection_summary.get("solver_status", "")),
        selected_slack=(
            float(selection_summary["selected_slack"])
            if selection_summary.get("selected_slack") is not None
            else None
        ),
        rounding_seed=(
            int(selection_summary["rounding_seed"])
            if selection_summary.get("rounding_seed") is not None
            else None
        ),
    )


def _word_count(value: Any) -> int:
    return len(str(value).split())


def _numeric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): float(item) for key, item in value.items()}
