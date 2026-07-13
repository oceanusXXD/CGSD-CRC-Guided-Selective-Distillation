from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class BudgetInputs:
    method: str
    seed_label_count: int = 0
    active_label_count: int = 0
    guide_label_count: int = 0
    calibration_label_count: int = 0
    group_estimator_label_count: int = 0
    evaluation_label_count: int = 0
    certification_label_count: int = 0
    judge_calls: int = 0
    train_tokens: int = 0
    selector_compute_seconds: float = 0.0


@dataclass(frozen=True)
class BudgetReport:
    method: str
    seed_label_count: int
    active_label_count: int
    guide_label_count: int
    calibration_label_count: int
    group_estimator_label_count: int
    supervision_budget_total: int
    evaluation_label_count: int
    certification_label_count: int
    evaluation_resource_total: int
    judge_calls: int
    train_tokens: int
    selector_compute_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "seed_label_count": self.seed_label_count,
            "active_label_count": self.active_label_count,
            "guide_label_count": self.guide_label_count,
            "calibration_label_count": self.calibration_label_count,
            "group_estimator_label_count": self.group_estimator_label_count,
            "supervision_budget_total": self.supervision_budget_total,
            "evaluation_label_count": self.evaluation_label_count,
            "certification_label_count": self.certification_label_count,
            "evaluation_resource_total": self.evaluation_resource_total,
            "judge_calls": self.judge_calls,
            "train_tokens": self.train_tokens,
            "selector_compute_seconds": self.selector_compute_seconds,
        }


def build_budget_report(inputs: BudgetInputs) -> BudgetReport:
    values = {
        "seed_label_count": inputs.seed_label_count,
        "active_label_count": inputs.active_label_count,
        "guide_label_count": inputs.guide_label_count,
        "calibration_label_count": inputs.calibration_label_count,
        "group_estimator_label_count": inputs.group_estimator_label_count,
        "evaluation_label_count": inputs.evaluation_label_count,
        "certification_label_count": inputs.certification_label_count,
        "judge_calls": inputs.judge_calls,
        "train_tokens": inputs.train_tokens,
        "selector_compute_seconds": inputs.selector_compute_seconds,
    }
    for name, value in values.items():
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative")

    supervision_total = (
        int(inputs.seed_label_count)
        + int(inputs.active_label_count)
        + int(inputs.guide_label_count)
        + int(inputs.calibration_label_count)
        + int(inputs.group_estimator_label_count)
    )
    evaluation_total = int(inputs.evaluation_label_count) + int(inputs.certification_label_count)
    return BudgetReport(
        method=str(inputs.method),
        seed_label_count=int(inputs.seed_label_count),
        active_label_count=int(inputs.active_label_count),
        guide_label_count=int(inputs.guide_label_count),
        calibration_label_count=int(inputs.calibration_label_count),
        group_estimator_label_count=int(inputs.group_estimator_label_count),
        supervision_budget_total=supervision_total,
        evaluation_label_count=int(inputs.evaluation_label_count),
        certification_label_count=int(inputs.certification_label_count),
        evaluation_resource_total=evaluation_total,
        judge_calls=int(inputs.judge_calls),
        train_tokens=int(inputs.train_tokens),
        selector_compute_seconds=float(inputs.selector_compute_seconds),
    )


def compare_budget_reports(
    reports: Iterable[BudgetReport],
    *,
    train_token_tolerance: int = 0,
) -> dict[str, Any]:
    report_list = list(reports)
    if not report_list:
        raise ValueError("reports must not be empty")
    if train_token_tolerance < 0:
        raise ValueError("train_token_tolerance must be non-negative")

    supervision_by_method = {
        report.method: report.supervision_budget_total
        for report in sorted(report_list, key=lambda item: item.method)
    }
    evaluation_by_method = {
        report.method: report.evaluation_resource_total
        for report in sorted(report_list, key=lambda item: item.method)
    }
    judge_calls_by_method = {
        report.method: report.judge_calls
        for report in sorted(report_list, key=lambda item: item.method)
    }
    train_tokens_by_method = {
        report.method: report.train_tokens
        for report in sorted(report_list, key=lambda item: item.method)
    }
    selector_compute_by_method = {
        report.method: report.selector_compute_seconds
        for report in sorted(report_list, key=lambda item: item.method)
    }
    train_tokens = list(train_tokens_by_method.values())
    return {
        "method_count": len(report_list),
        "supervision_budget_by_method": supervision_by_method,
        "evaluation_resource_by_method": evaluation_by_method,
        "judge_calls_by_method": judge_calls_by_method,
        "train_tokens_by_method": train_tokens_by_method,
        "selector_compute_seconds_by_method": selector_compute_by_method,
        "supervision_budget_equal": len(set(supervision_by_method.values())) == 1,
        "train_tokens_within_tolerance": max(train_tokens) - min(train_tokens) <= int(train_token_tolerance),
        "train_token_tolerance": int(train_token_tolerance),
    }
