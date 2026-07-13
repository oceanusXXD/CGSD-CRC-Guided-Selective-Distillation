from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from mias_dcms.records import RunRecord
from mias_dcms.statistics import bootstrap_mean_ci, paired_permutation_test


@dataclass(frozen=True)
class RunMetricComparisonReport:
    baseline_method: str
    treatment_methods: list[str]
    evaluation_metrics: list[str]
    selection_metrics: list[str]
    training_metrics: list[str]
    cost_metrics: list[str]
    comparisons: list[dict[str, Any]]
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "baseline_method": self.baseline_method,
            "treatment_methods": list(self.treatment_methods),
            "evaluation_metrics": list(self.evaluation_metrics),
            "selection_metrics": list(self.selection_metrics),
            "training_metrics": list(self.training_metrics),
            "cost_metrics": list(self.cost_metrics),
            "comparisons": [dict(row) for row in self.comparisons],
            "issues": [dict(issue) for issue in self.issues],
        }


def compare_run_metrics_to_baseline(
    runs: Iterable[RunRecord | Mapping[str, Any]],
    *,
    baseline_method: str,
    treatment_methods: Sequence[str],
    evaluation_metrics: Sequence[str],
    selection_metrics: Sequence[str] = (),
    training_metrics: Sequence[str] = (),
    cost_metrics: Sequence[str] = (),
    expected_seeds: Sequence[int] | None = None,
    minimum_paired_seeds: int = 1,
    confidence: float = 0.95,
    resamples: int = 1000,
    permutations: int = 1000,
    seed: int = 0,
) -> RunMetricComparisonReport:
    run_list = [_coerce_run(row) for row in runs]
    metric_specs = [
        *[("evaluation_metrics", metric) for metric in evaluation_metrics],
        *[("selection_metrics", metric) for metric in selection_metrics],
        *[("training_metrics", metric) for metric in training_metrics],
        *[("cost_metrics", metric) for metric in cost_metrics],
    ]
    issues: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    by_method_seed: dict[tuple[str, int], RunRecord] = {}

    for row_index, run in enumerate(run_list):
        key = (run.method, run.seed)
        if key in by_method_seed:
            issues.append(
                {
                    "code": "duplicate_method_seed",
                    "method": run.method,
                    "seed": run.seed,
                    "row_index": row_index,
                }
            )
        by_method_seed[key] = run

    expected_seed_list = (
        sorted({int(value) for value in expected_seeds})
        if expected_seeds is not None
        else sorted({run.seed for run in run_list})
    )

    for treatment_method in treatment_methods:
        for expected_seed in expected_seed_list:
            if (baseline_method, expected_seed) not in by_method_seed:
                issues.append(
                    {
                        "code": "missing_baseline_seed",
                        "baseline_method": baseline_method,
                        "treatment_method": treatment_method,
                        "seed": expected_seed,
                    }
                )
            if (str(treatment_method), expected_seed) not in by_method_seed:
                issues.append(
                    {
                        "code": "missing_treatment_seed",
                        "baseline_method": baseline_method,
                        "treatment_method": str(treatment_method),
                        "seed": expected_seed,
                    }
                )

        for metric_index, (metric_group, metric_name) in enumerate(metric_specs):
            paired = []
            paired_seeds = []
            for pair_seed in expected_seed_list:
                baseline_run = by_method_seed.get((baseline_method, pair_seed))
                treatment_run = by_method_seed.get((str(treatment_method), pair_seed))
                if baseline_run is None or treatment_run is None:
                    continue
                baseline_value = _metric_value_or_issue(
                    baseline_run,
                    group_name=metric_group,
                    metric_name=metric_name,
                    method=baseline_method,
                    seed=pair_seed,
                    issues=issues,
                )
                treatment_value = _metric_value_or_issue(
                    treatment_run,
                    group_name=metric_group,
                    metric_name=metric_name,
                    method=str(treatment_method),
                    seed=pair_seed,
                    issues=issues,
                )
                if baseline_value is None or treatment_value is None:
                    continue
                paired.append((baseline_value, treatment_value))
                paired_seeds.append(pair_seed)

            if len(paired) < int(minimum_paired_seeds):
                issues.append(
                    {
                        "code": "insufficient_paired_seeds",
                        "baseline_method": baseline_method,
                        "treatment_method": str(treatment_method),
                        "metric_group": metric_group,
                        "metric": metric_name,
                        "paired_count": len(paired),
                        "minimum_paired_seeds": int(minimum_paired_seeds),
                    }
                )
                continue

            baseline_values = [pair[0] for pair in paired]
            treatment_values = [pair[1] for pair in paired]
            deltas = [treatment - baseline for baseline, treatment in paired]
            ci = bootstrap_mean_ci(
                deltas,
                confidence=confidence,
                resamples=resamples,
                seed=seed + metric_index,
            )
            permutation = paired_permutation_test(
                baseline=baseline_values,
                treatment=treatment_values,
                permutations=permutations,
                seed=seed + metric_index + 1000,
            )
            comparisons.append(
                {
                    "baseline_method": baseline_method,
                    "treatment_method": str(treatment_method),
                    "metric_group": metric_group,
                    "metric": metric_name,
                    "baseline_mean": sum(baseline_values) / len(baseline_values),
                    "treatment_mean": sum(treatment_values) / len(treatment_values),
                    "delta_mean": ci.mean,
                    "delta_ci_low": ci.ci_low,
                    "delta_ci_high": ci.ci_high,
                    "confidence": ci.confidence,
                    "paired_count": ci.count,
                    "paired_seeds": list(paired_seeds),
                    "p_value": permutation.p_value,
                    "permutations": permutation.permutations,
                }
            )

    return RunMetricComparisonReport(
        baseline_method=str(baseline_method),
        treatment_methods=[str(method) for method in treatment_methods],
        evaluation_metrics=[str(metric) for metric in evaluation_metrics],
        selection_metrics=[str(metric) for metric in selection_metrics],
        training_metrics=[str(metric) for metric in training_metrics],
        cost_metrics=[str(metric) for metric in cost_metrics],
        comparisons=comparisons,
        issues=issues,
    )


def _coerce_run(row: RunRecord | Mapping[str, Any]) -> RunRecord:
    if isinstance(row, RunRecord):
        return row
    return RunRecord(**dict(row))


def _metric_value_or_issue(
    run: RunRecord,
    *,
    group_name: str,
    metric_name: str,
    method: str,
    seed: int,
    issues: list[dict[str, Any]],
) -> float | None:
    metrics = getattr(run, group_name)
    if metric_name not in metrics:
        issues.append(
            {
                "code": "missing_metric",
                "method": method,
                "seed": seed,
                "metric_group": group_name,
                "metric": f"{group_name}.{metric_name}",
            }
        )
        return None
    try:
        return float(metrics[metric_name])
    except (TypeError, ValueError):
        issues.append(
            {
                "code": "non_numeric_metric",
                "method": method,
                "seed": seed,
                "metric_group": group_name,
                "metric": f"{group_name}.{metric_name}",
            }
        )
        return None
