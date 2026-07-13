from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mias_dcms.selection.dcms import DCMSResult


@dataclass(frozen=True)
class AcquisitionRecord:
    sample_id: str
    split: str
    round: int
    method: str
    model: str
    seed: int
    base_score: float
    normalized_score: float
    q_propensity: float
    selected: bool
    observable_groups: dict[str, Any] = field(default_factory=dict)
    oracle_label: Any | None = None
    train_tokens: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "round": self.round,
            "method": self.method,
            "model": self.model,
            "seed": self.seed,
            "base_score": self.base_score,
            "normalized_score": self.normalized_score,
            "q_propensity": self.q_propensity,
            "selected": self.selected,
            "observable_groups": dict(self.observable_groups),
            "oracle_label": self.oracle_label,
            "train_tokens": self.train_tokens,
        }


@dataclass(frozen=True)
class RunRecord:
    dataset: str
    model: str
    method: str
    budget: int
    seed: int
    selected_count: int
    config_hash: str
    selection_metrics: dict[str, Any] = field(default_factory=dict)
    training_metrics: dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: dict[str, Any] = field(default_factory=dict)
    cost_metrics: dict[str, Any] = field(default_factory=dict)
    continuous_moments: dict[str, float] = field(default_factory=dict)
    rounded_moments: dict[str, float] = field(default_factory=dict)
    robust_lower_moments: dict[str, float] = field(default_factory=dict)
    robust_upper_moments: dict[str, float] = field(default_factory=dict)
    utility_retained: float = 0.0
    max_constraint_violation: float = 0.0
    solver_status: str = ""
    selected_slack: float | None = None
    rounding_seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "method": self.method,
            "budget": self.budget,
            "seed": self.seed,
            "selected_count": self.selected_count,
            "config_hash": self.config_hash,
            "selection_metrics": dict(self.selection_metrics),
            "training_metrics": dict(self.training_metrics),
            "evaluation_metrics": dict(self.evaluation_metrics),
            "cost_metrics": dict(self.cost_metrics),
            "continuous_moments": dict(self.continuous_moments),
            "rounded_moments": dict(self.rounded_moments),
            "robust_lower_moments": dict(self.robust_lower_moments),
            "robust_upper_moments": dict(self.robust_upper_moments),
            "utility_retained": self.utility_retained,
            "max_constraint_violation": self.max_constraint_violation,
            "solver_status": self.solver_status,
            "selected_slack": self.selected_slack,
            "rounding_seed": self.rounding_seed,
        }


def build_acquisition_record(
    *,
    sample_id: str,
    split: str,
    round_index: int,
    method: str,
    model: str,
    seed: int,
    base_score: float,
    normalized_score: float,
    q_propensity: float,
    selected: bool,
    observable_groups: dict[str, Any] | None = None,
    oracle_label: Any | None = None,
    train_tokens: int = 0,
) -> AcquisitionRecord:
    if not sample_id:
        raise ValueError("sample_id must not be empty")
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    if not 0.0 <= q_propensity <= 1.0:
        raise ValueError("q_propensity must be between 0 and 1")
    if train_tokens < 0:
        raise ValueError("train_tokens must be non-negative")
    if split == "active_pool" and not selected and oracle_label is not None:
        raise ValueError("unselected active_pool records must not expose oracle_label")
    return AcquisitionRecord(
        sample_id=str(sample_id),
        split=str(split),
        round=int(round_index),
        method=str(method),
        model=str(model),
        seed=int(seed),
        base_score=float(base_score),
        normalized_score=float(normalized_score),
        q_propensity=float(q_propensity),
        selected=bool(selected),
        observable_groups=dict(observable_groups or {}),
        oracle_label=oracle_label,
        train_tokens=int(train_tokens),
    )


def build_records_from_dcms(
    *,
    sample_ids: Sequence[str],
    base_scores: Sequence[float],
    normalized_scores: Sequence[float],
    observable_groups: Sequence[Mapping[str, Any]],
    dcms_result: DCMSResult,
    split: str,
    round_index: int,
    method: str,
    model: str,
    seed: int,
    revealed_oracle_labels: Mapping[str, Any] | None = None,
    train_tokens: Mapping[str, int] | None = None,
) -> list[AcquisitionRecord]:
    if len(sample_ids) != len(base_scores) or len(sample_ids) != len(normalized_scores):
        raise ValueError("sample_ids, base_scores, and normalized_scores must have equal length")
    if len(sample_ids) != len(observable_groups):
        raise ValueError("observable_groups must match sample count")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")

    labels = dict(revealed_oracle_labels or {})
    token_counts = dict(train_tokens or {})
    records: list[AcquisitionRecord] = []
    for sample_id, base_score, normalized_score, groups in zip(
        sample_ids,
        base_scores,
        normalized_scores,
        observable_groups,
    ):
        sample_key = str(sample_id)
        selected = bool(dcms_result.selection_indicator.get(sample_key, 0))
        oracle_label = labels.get(sample_key) if selected else None
        records.append(
            build_acquisition_record(
                sample_id=sample_key,
                split=split,
                round_index=round_index,
                method=method,
                model=model,
                seed=seed,
                base_score=float(base_score),
                normalized_score=float(normalized_score),
                q_propensity=float(dcms_result.q_propensity.get(sample_key, 0.0)),
                selected=selected,
                observable_groups=dict(groups),
                oracle_label=oracle_label,
                train_tokens=int(token_counts.get(sample_key, 0)) if selected else 0,
            )
        )
    return records


def build_run_record(
    *,
    dataset: str,
    model: str,
    method: str,
    budget: int,
    seed: int,
    dcms_result: DCMSResult,
    config_hash: str,
    selection_metrics: Mapping[str, Any] | None = None,
    training_metrics: Mapping[str, Any] | None = None,
    evaluation_metrics: Mapping[str, Any] | None = None,
    cost_metrics: Mapping[str, Any] | None = None,
) -> RunRecord:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    selected_count = sum(int(value) for value in dcms_result.selection_indicator.values())
    return RunRecord(
        dataset=str(dataset),
        model=str(model),
        method=str(method),
        budget=int(budget),
        seed=int(seed),
        selected_count=selected_count,
        config_hash=str(config_hash),
        selection_metrics=dict(selection_metrics or {}),
        training_metrics=dict(training_metrics or {}),
        evaluation_metrics=dict(evaluation_metrics or {}),
        cost_metrics=dict(cost_metrics or {}),
        continuous_moments=dict(dcms_result.continuous_moments),
        rounded_moments=dict(dcms_result.rounded_moments),
        robust_lower_moments=dict(dcms_result.robust_lower_moments),
        robust_upper_moments=dict(dcms_result.robust_upper_moments),
        utility_retained=float(dcms_result.utility_retained),
        max_constraint_violation=float(dcms_result.max_constraint_violation),
        solver_status=str(dcms_result.solver_status),
        selected_slack=dcms_result.selected_slack,
        rounding_seed=dcms_result.rounding_seed,
    )
