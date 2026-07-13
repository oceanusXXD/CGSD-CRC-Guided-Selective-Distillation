from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    ci_low: float
    ci_high: float
    count: int
    confidence: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean": self.mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "count": self.count,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class PermutationTestResult:
    observed_delta: float
    p_value: float
    paired_count: int
    permutations: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "observed_delta": self.observed_delta,
            "p_value": self.p_value,
            "paired_count": self.paired_count,
            "permutations": self.permutations,
        }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 0,
) -> MetricSummary:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("values must not be empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    observed_mean = _mean(numbers)
    rng = random.Random(seed)
    bootstrap_means = [
        _mean([numbers[rng.randrange(len(numbers))] for _ in numbers])
        for _ in range(int(resamples))
    ]
    alpha = 1.0 - confidence
    return MetricSummary(
        mean=observed_mean,
        ci_low=_quantile(bootstrap_means, alpha / 2.0),
        ci_high=_quantile(bootstrap_means, 1.0 - alpha / 2.0),
        count=len(numbers),
        confidence=float(confidence),
    )


def paired_mean_delta(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline_method: str,
    treatment_method: str,
    metric_field: str,
    method_field: str = "method",
    seed_field: str = "seed",
) -> float:
    paired = _paired_values(
        rows,
        baseline_method=baseline_method,
        treatment_method=treatment_method,
        metric_field=metric_field,
        method_field=method_field,
        seed_field=seed_field,
    )
    return _mean([treatment - baseline for baseline, treatment in paired])


def paired_permutation_test(
    *,
    baseline: Sequence[float],
    treatment: Sequence[float],
    permutations: int = 1000,
    seed: int = 0,
) -> PermutationTestResult:
    baseline_values = [float(value) for value in baseline]
    treatment_values = [float(value) for value in treatment]
    if len(baseline_values) != len(treatment_values):
        raise ValueError("baseline and treatment must have equal length")
    if not baseline_values:
        raise ValueError("baseline and treatment must not be empty")
    if permutations <= 0:
        raise ValueError("permutations must be positive")

    deltas = [treatment_item - baseline_item for baseline_item, treatment_item in zip(baseline_values, treatment_values)]
    observed = _mean(deltas)
    threshold = abs(observed)
    rng = random.Random(seed)
    extreme_count = 0
    for _ in range(int(permutations)):
        permuted = [delta if rng.random() < 0.5 else -delta for delta in deltas]
        if abs(_mean(permuted)) >= threshold - 1e-12:
            extreme_count += 1
    p_value = (extreme_count + 1) / (int(permutations) + 1)
    return PermutationTestResult(
        observed_delta=observed,
        p_value=p_value,
        paired_count=len(deltas),
        permutations=int(permutations),
    )


def summarize_metric_by_method(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric_field: str,
    method_field: str = "method",
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 0,
) -> dict[str, MetricSummary]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[method_field])].append(float(row[metric_field]))
    if not grouped:
        raise ValueError("rows must not be empty")
    return {
        method: bootstrap_mean_ci(
            values,
            confidence=confidence,
            resamples=resamples,
            seed=seed + index,
        )
        for index, (method, values) in enumerate(sorted(grouped.items()))
    }


def _paired_values(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline_method: str,
    treatment_method: str,
    metric_field: str,
    method_field: str,
    seed_field: str,
) -> list[tuple[float, float]]:
    by_seed: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_seed[str(row[seed_field])][str(row[method_field])] = float(row[metric_field])
    paired = []
    for seed in sorted(by_seed):
        methods = by_seed[seed]
        if baseline_method in methods and treatment_method in methods:
            paired.append((methods[baseline_method], methods[treatment_method]))
    if not paired:
        raise ValueError("no paired seeds found for the requested methods")
    return paired


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight
