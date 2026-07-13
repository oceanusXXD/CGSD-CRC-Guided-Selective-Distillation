from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from mias_dcms.records import RunRecord
from mias_dcms.statistics import bootstrap_mean_ci


REQUIRED_COST_METRICS = (
    "seed_label_count",
    "active_label_count",
    "evaluation_label_count",
    "judge_calls",
    "train_tokens",
    "selector_compute_seconds",
)


def validate_run_record_for_paper_table(
    run: RunRecord,
    *,
    required_selection_metrics: Sequence[str],
    required_evaluation_metrics: Sequence[str],
    required_training_metrics: Sequence[str] = (),
    required_cost_metrics: Sequence[str] = REQUIRED_COST_METRICS,
) -> None:
    if not run.dataset:
        raise ValueError("run record must include dataset")
    if not run.model:
        raise ValueError("run record must include model")
    if not run.method:
        raise ValueError("run record must include method")
    if run.budget < 0:
        raise ValueError("run record budget must be non-negative")
    if run.selected_count < 0:
        raise ValueError("run record selected_count must be non-negative")
    if not run.config_hash:
        raise ValueError("run record must include config_hash")

    _require_metric_group(run.selection_metrics, required_selection_metrics, "selection_metrics")
    _require_metric_group(run.training_metrics, required_training_metrics, "training_metrics")
    _require_metric_group(run.evaluation_metrics, required_evaluation_metrics, "evaluation_metrics")
    _require_metric_group(run.cost_metrics, required_cost_metrics, "cost_metrics")
    for key in required_cost_metrics:
        value = float(run.cost_metrics[key])
        if value < 0.0:
            raise ValueError(f"cost metric {key!r} must be non-negative")


def aggregate_paper_metric_table(
    runs: Iterable[RunRecord],
    *,
    evaluation_metrics: Sequence[str],
    selection_metrics: Sequence[str] = (),
    training_metrics: Sequence[str] = (),
    cost_metrics: Sequence[str] = (),
    required_cost_metrics: Sequence[str] = REQUIRED_COST_METRICS,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    run_list = list(runs)
    if not run_list:
        raise ValueError("runs must not be empty")

    for run in run_list:
        validate_run_record_for_paper_table(
            run,
            required_selection_metrics=selection_metrics,
            required_training_metrics=training_metrics,
            required_evaluation_metrics=evaluation_metrics,
            required_cost_metrics=tuple(required_cost_metrics),
        )

    by_method: dict[str, list[RunRecord]] = defaultdict(list)
    for run in run_list:
        by_method[run.method].append(run)

    table: list[dict[str, Any]] = []
    for method_index, method in enumerate(sorted(by_method)):
        method_runs = by_method[method]
        table.append(
            {
                "method": method,
                "run_count": len(method_runs),
                "datasets": sorted({run.dataset for run in method_runs}),
                "models": sorted({run.model for run in method_runs}),
                "budgets": sorted({run.budget for run in method_runs}),
                "seeds": sorted({run.seed for run in method_runs}),
                "config_hashes": sorted({run.config_hash for run in method_runs}),
                "required_cost_metrics": list(required_cost_metrics),
                "selection_metrics": _aggregate_metric_group(
                    method_runs,
                    metric_names=selection_metrics,
                    source="selection_metrics",
                    confidence=confidence,
                    resamples=resamples,
                    seed=seed + method_index * 1000,
                ),
                "training_metrics": _aggregate_metric_group(
                    method_runs,
                    metric_names=training_metrics,
                    source="training_metrics",
                    confidence=confidence,
                    resamples=resamples,
                    seed=seed + method_index * 1000 + 100,
                ),
                "evaluation_metrics": _aggregate_metric_group(
                    method_runs,
                    metric_names=evaluation_metrics,
                    source="evaluation_metrics",
                    confidence=confidence,
                    resamples=resamples,
                    seed=seed + method_index * 1000 + 200,
                ),
                "cost_metrics": _aggregate_metric_group(
                    method_runs,
                    metric_names=cost_metrics,
                    source="cost_metrics",
                    confidence=confidence,
                    resamples=resamples,
                    seed=seed + method_index * 1000 + 300,
                ),
            }
        )
    return table


def _require_metric_group(
    metrics: Mapping[str, Any],
    required_names: Sequence[str],
    group_name: str,
) -> None:
    missing = [name for name in required_names if name not in metrics]
    if missing:
        raise ValueError(f"{group_name} missing required metrics: {missing}")
    for name in required_names:
        float(metrics[name])


def _aggregate_metric_group(
    runs: Sequence[RunRecord],
    *,
    metric_names: Sequence[str],
    source: str,
    confidence: float,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    return {
        metric_name: bootstrap_mean_ci(
            [_metric_value(run, source, metric_name) for run in runs],
            confidence=confidence,
            resamples=resamples,
            seed=seed + index,
        ).as_dict()
        for index, metric_name in enumerate(metric_names)
    }


def _metric_value(run: RunRecord, source: str, metric_name: str) -> float:
    metrics = getattr(run, source)
    return float(metrics[metric_name])
