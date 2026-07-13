from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class UtilityQuantileProfile:
    mean: float
    quantile_midpoints: list[float]
    bin_means: list[float]

    def as_dict(self) -> dict[str, float | list[float]]:
        return {
            "mean": self.mean,
            "quantile_midpoints": list(self.quantile_midpoints),
            "bin_means": list(self.bin_means),
        }


@dataclass(frozen=True)
class MatchedUtilityReport:
    baseline_count: int
    treatment_count: int
    baseline_utility: UtilityQuantileProfile
    treatment_utility: UtilityQuantileProfile
    mean_delta: float
    max_quantile_delta: float
    utility_matched: bool
    baseline_coverage_deviation: float
    treatment_coverage_deviation: float

    def as_dict(self) -> dict[str, object]:
        return {
            "baseline_count": self.baseline_count,
            "treatment_count": self.treatment_count,
            "baseline_utility": self.baseline_utility.as_dict(),
            "treatment_utility": self.treatment_utility.as_dict(),
            "mean_delta": self.mean_delta,
            "max_quantile_delta": self.max_quantile_delta,
            "utility_matched": self.utility_matched,
            "baseline_coverage_deviation": self.baseline_coverage_deviation,
            "treatment_coverage_deviation": self.treatment_coverage_deviation,
        }


def utility_quantile_profile(
    utilities: Sequence[float],
    *,
    bins: int = 4,
) -> UtilityQuantileProfile:
    values = sorted(float(value) for value in utilities)
    if not values:
        raise ValueError("utilities must not be empty")
    if bins <= 0:
        raise ValueError("bins must be positive")
    if bins > len(values):
        raise ValueError("bins must not exceed utility count")

    bin_means: list[float] = []
    midpoints: list[float] = []
    for index in range(bins):
        start = round(index * len(values) / bins)
        end = round((index + 1) * len(values) / bins)
        chunk = values[start:end]
        if not chunk:
            raise ValueError("empty quantile bin; reduce bin count")
        bin_means.append(_mean(chunk))
        midpoints.append((index + 0.5) / bins)
    return UtilityQuantileProfile(
        mean=_mean(values),
        quantile_midpoints=midpoints,
        bin_means=bin_means,
    )


def coverage_deviation(
    *,
    observed_moments: Mapping[str, float],
    target_moments: Mapping[str, float],
) -> float:
    groups = set(observed_moments) | set(target_moments)
    return 0.5 * sum(
        abs(float(observed_moments.get(group, 0.0)) - float(target_moments.get(group, 0.0)))
        for group in groups
    )


def matched_utility_report(
    *,
    baseline_utilities: Sequence[float],
    treatment_utilities: Sequence[float],
    baseline_moments: Mapping[str, float],
    treatment_moments: Mapping[str, float],
    target_moments: Mapping[str, float],
    mean_tolerance: float = 0.02,
    quantile_tolerance: float = 0.05,
    bins: int = 4,
) -> MatchedUtilityReport:
    if len(baseline_utilities) != len(treatment_utilities):
        raise ValueError("baseline and treatment batches must have the same size")
    if not baseline_utilities:
        raise ValueError("utility batches must not be empty")
    baseline_profile = utility_quantile_profile(baseline_utilities, bins=min(bins, len(baseline_utilities)))
    treatment_profile = utility_quantile_profile(treatment_utilities, bins=min(bins, len(treatment_utilities)))
    mean_delta = treatment_profile.mean - baseline_profile.mean
    max_quantile_delta = max(
        (
            abs(treatment_value - baseline_value)
            for baseline_value, treatment_value in zip(
                baseline_profile.bin_means,
                treatment_profile.bin_means,
            )
        ),
        default=0.0,
    )
    return MatchedUtilityReport(
        baseline_count=len(baseline_utilities),
        treatment_count=len(treatment_utilities),
        baseline_utility=baseline_profile,
        treatment_utility=treatment_profile,
        mean_delta=mean_delta,
        max_quantile_delta=max_quantile_delta,
        utility_matched=abs(mean_delta) <= mean_tolerance and max_quantile_delta <= quantile_tolerance,
        baseline_coverage_deviation=coverage_deviation(
            observed_moments=baseline_moments,
            target_moments=target_moments,
        ),
        treatment_coverage_deviation=coverage_deviation(
            observed_moments=treatment_moments,
            target_moments=target_moments,
        ),
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)
