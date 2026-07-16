from __future__ import annotations

from typing import Any


def recommend_algorithm_action(
    *,
    shift_analysis: dict[str, Any],
    classification_comparison: dict[str, Any] | None = None,
    preference_comparison: dict[str, Any] | None = None,
    min_performance_drop: float = 0.01,
    min_order_bias_improvement: float = 0.05,
) -> dict[str, Any]:
    classification = shift_analysis.get("classification", {})
    preference = shift_analysis.get("preference", {})
    classification_established = bool(classification.get("phenomenon_established"))
    preference_established = bool(preference.get("dpo_shift_established"))
    classification_method = classification.get("recommended_uncertainty_method")
    preference_method = preference.get("recommended_uncertainty_method")
    stable_classes = sorted(
        classification.get("methods", {})
        .get(classification_method, {})
        .get("stable_enriched_classes", []),
        key=_label_sort_key,
    )
    preference_domains = sorted(
        preference.get("dpo_methods", {})
        .get(preference_method, {})
        .get("material_domains", [])
    )
    classification_deltas = _comparison_deltas(
        classification_comparison,
        expected_task="classification",
        required_metrics=("accuracy", "macro_f1"),
    )
    preference_deltas = _comparison_deltas(
        preference_comparison,
        expected_task="preference",
        required_metrics=("accuracy_excluding_ties",),
    )
    classification_underperforms = _has_performance_drop(
        classification_deltas, ("accuracy", "macro_f1"), min_performance_drop
    )
    preference_underperforms = _has_performance_drop(
        preference_deltas, ("accuracy_excluding_ties",), min_performance_drop
    )
    order_delta = preference_deltas.get("mean_order_disagreement")

    evidence = {
        "classification_shift_established": classification_established,
        "classification_uncertainty_method": classification_method,
        "stable_enriched_classes": stable_classes,
        "classification_uncertainty_minus_random": classification_deltas,
        "preference_shift_established": preference_established,
        "preference_uncertainty_method": preference_method,
        "preference_material_domains": preference_domains,
        "preference_uncertainty_minus_random": preference_deltas,
        "position_bias_warning": bool(preference.get("position_bias_warning")),
        "position_bias_pool_mean": preference.get("position_bias_pool_mean"),
    }

    if bool(preference.get("position_bias_warning")) and (
        order_delta is None or order_delta > -min_order_bias_improvement
    ):
        return _decision(
            "repair_pairwise_scoring_first",
            [
                "Dual-order scoring has material position bias that is not demonstrably reduced by the uncertainty-trained model.",
                "Selection and DPO conclusions are not auditable until the pairwise scoring signal is order-robust.",
            ],
            evidence,
            ["repeat dual-order scoring after repairing or calibrating position bias"],
        )

    if not classification_established and not preference_established:
        return _decision(
            "keep_current_algorithm",
            ["Neither the multiclass nor preference shift gate is established."],
            evidence,
            ["collect full-budget diagnostics before refining DCMS constraints or adding WSR"],
        )

    missing_comparisons: list[str] = []
    if classification_established and classification_comparison is None:
        missing_comparisons.append("classification Random-vs-Uncertainty LoRA comparison")
    if preference_established and preference_comparison is None:
        missing_comparisons.append("preference Random-DPO-vs-Uncertainty-DPO comparison")
    if missing_comparisons:
        return _decision(
            "defer_algorithm_change_until_training",
            ["A selection shift is established, but downstream training evidence is incomplete."],
            evidence,
            missing_comparisons,
        )

    if (
        classification_established
        and preference_established
        and classification_underperforms
        and preference_underperforms
    ):
        return _decision(
            "restructure_algorithm",
            [
                "Uncertainty selection creates stable class-prior distortion and multi-domain preference distortion.",
                "The distortions also reduce downstream performance in both LoRA and DPO comparisons.",
            ],
            evidence,
            ["design a joint selection objective rather than extending a binary prior correction"],
        )

    if preference_established and preference_underperforms:
        return _decision(
            "introduce_wsr",
            [
                "Preference uncertainty selection shifts multiple response or prompt domains and underperforms Random-DPO.",
                "A weighted sampling or reweighting correction is better matched than class-only prior correction.",
            ],
            evidence,
            ["define WSR weights from the material preference domains and re-run the same DPO budget"],
        )

    if classification_established and classification_underperforms:
        return _decision(
            "refine_dcms_constraints",
            [
                "Stable class-prior distortion is concentrated in repeatedly enriched classes and hurts LoRA performance.",
                "Refine DCMS target moments and slack constraints before considering a full redesign.",
            ],
            evidence,
            ["run the same budget with audited multiclass DCMS target moments"],
        )

    return _decision(
        "keep_current_algorithm_with_monitoring",
        [
            "A measurable selection shift exists, but Uncertainty does not materially underperform Random after training.",
            "Changing the algorithm is not justified without a downstream cost signal.",
        ],
        evidence,
        ["monitor the same shift metrics on additional seeds or budgets"],
    )


def _comparison_deltas(
    comparison: dict[str, Any] | None,
    *,
    expected_task: str,
    required_metrics: tuple[str, ...],
) -> dict[str, float]:
    if comparison is None:
        return {}
    task = comparison.get("task")
    if task is not None and task != expected_task:
        raise ValueError(
            f"{expected_task} comparison task must be {expected_task!r}, got {task!r}"
        )
    raw_deltas = comparison.get("uncertainty_minus_random")
    if not isinstance(raw_deltas, dict):
        raise ValueError(f"{expected_task} comparison requires uncertainty_minus_random")
    deltas = {
        key: float(value)
        for key, value in raw_deltas.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not any(metric in deltas for metric in required_metrics):
        joined_metrics = ", ".join(required_metrics)
        raise ValueError(
            f"{expected_task} comparison requires at least one of: {joined_metrics}"
        )
    return deltas


def _has_performance_drop(
    deltas: dict[str, float], metrics: tuple[str, ...], threshold: float
) -> bool:
    if threshold < 0.0:
        raise ValueError("min_performance_drop must be non-negative")
    return any(deltas.get(metric, 0.0) <= -threshold for metric in metrics)


def _decision(
    recommendation: str,
    rationale: list[str],
    evidence: dict[str, Any],
    required_next_evidence: list[str],
) -> dict[str, Any]:
    return {
        "primary_recommendation": recommendation,
        "rationale": rationale,
        "evidence": evidence,
        "required_next_evidence": required_next_evidence,
    }


def _label_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10**9, value
