from __future__ import annotations

from statistics import mean
from typing import Any


def analyze_classification_shift(
    report: dict[str, Any],
    *,
    min_tv_delta: float = 0.02,
    min_enrichment_delta: float = 0.10,
    required_budgets: int = 2,
) -> dict[str, Any]:
    methods = report.get("methods", {})
    if "random" not in methods:
        raise ValueError("classification diagnostics must include random")
    if required_budgets <= 0:
        raise ValueError("required_budgets must be positive")
    random_reports = methods["random"]
    calibrated = "random_baseline" in report
    analyses: dict[str, Any] = {}
    for method, method_reports in methods.items():
        if method == "random":
            continue
        budgets = sorted(set(random_reports) & set(method_reports), key=int)
        if not budgets:
            raise ValueError(f"method {method!r} has no budgets shared with random")
        tv_delta_by_budget = {}
        budgets_meeting_tv = []
        for budget in budgets:
            budget_report = method_reports[budget]
            if calibrated and "excess_tv_vs_random_mean" in budget_report:
                tv_delta = float(budget_report["excess_tv_vs_random_mean"])
                meets_tv = bool(budget_report.get("global_envelope_exceeded", False))
            else:
                tv_delta = float(budget_report["category_tv"]) - float(
                    random_reports[budget]["category_tv"]
                )
                meets_tv = tv_delta >= min_tv_delta
            tv_delta_by_budget[budget] = tv_delta
            if meets_tv:
                budgets_meeting_tv.append(budget)
        classes = sorted(
            {
                label
                for budget in budgets
                for label in method_reports[budget].get("per_class", {})
            },
            key=_label_sort_key,
        )
        enrichment_delta_by_class: dict[str, dict[str, float]] = {}
        joint_budgets_by_class: dict[str, list[str]] = {}
        stable_enriched_classes: list[str] = []
        for label in classes:
            by_budget: dict[str, float] = {}
            enrichment_exceeds_random_envelope: dict[str, bool] = {}
            for budget in budgets:
                class_report = method_reports[budget].get("per_class", {}).get(label, {})
                method_value = class_report.get("enrichment")
                if calibrated and class_report.get("random_enrichment_mean") is not None:
                    random_value = class_report.get("random_enrichment_mean")
                    enrichment_exceeds_random_envelope[budget] = (
                        float(class_report.get("enrichment_excess_vs_random_q95", float("-inf")))
                        >= 0.0
                    )
                else:
                    random_value = (
                        random_reports[budget].get("per_class", {}).get(label, {}).get("enrichment")
                    )
                    enrichment_exceeds_random_envelope[budget] = True
                if method_value is None or random_value is None:
                    continue
                by_budget[budget] = float(method_value) - float(random_value)
            enrichment_delta_by_class[label] = by_budget
            joint_budgets = [
                budget
                for budget in budgets_meeting_tv
                if by_budget.get(budget, float("-inf")) >= min_enrichment_delta
                and enrichment_exceeds_random_envelope.get(budget, False)
            ]
            joint_budgets_by_class[label] = joint_budgets
            if len(joint_budgets) >= required_budgets:
                stable_enriched_classes.append(label)
        established = (
            len(budgets_meeting_tv) >= required_budgets and bool(stable_enriched_classes)
        )
        analyses[method] = {
            "budgets": budgets,
            "tv_delta_vs_random": tv_delta_by_budget,
            "mean_tv_delta_vs_random": mean(tv_delta_by_budget.values()),
            "budgets_meeting_tv": budgets_meeting_tv,
            "enrichment_delta_vs_random": enrichment_delta_by_class,
            "joint_budgets_by_class": joint_budgets_by_class,
            "stable_enriched_classes": stable_enriched_classes,
            "calibration": (
                "random_global_envelope" if calibrated else "single_random_trajectory"
            ),
            "aas_centered": report.get("method_summaries", {})
            .get(method, {})
            .get("aas_centered"),
            "established": established,
        }
    established_methods = [method for method, values in analyses.items() if values["established"]]
    recommended = _recommended_classification_method(
        analyses,
        established_methods,
        comparable_tv_delta=min_tv_delta,
    )
    leading = (
        max(analyses, key=lambda method: (analyses[method]["mean_tv_delta_vs_random"], method))
        if analyses
        else None
    )
    return {
        "thresholds": {
            "min_tv_delta": min_tv_delta,
            "min_enrichment_delta": min_enrichment_delta,
            "required_budgets": required_budgets,
        },
        "methods": analyses,
        "phenomenon_established": bool(established_methods),
        "recommended_uncertainty_method": recommended,
        "leading_method_without_gate": leading,
        "next_step": (
            "run_random_vs_uncertainty_lora" if recommended else "collect_more_classification_evidence"
        ),
    }


def _recommended_classification_method(
    analyses: dict[str, Any],
    methods: list[str],
    *,
    comparable_tv_delta: float,
) -> str | None:
    if not methods:
        return None
    leading = max(
        methods,
        key=lambda method: (
            analyses[method]["mean_tv_delta_vs_random"],
            len(analyses[method]["stable_enriched_classes"]),
            method,
        ),
    )
    if "entropy" not in methods:
        return leading
    leading_tv = float(analyses[leading]["mean_tv_delta_vs_random"])
    entropy_tv = float(analyses["entropy"]["mean_tv_delta_vs_random"])
    if leading_tv - entropy_tv <= comparable_tv_delta:
        return "entropy"
    return leading


def analyze_preference_shift(
    report: dict[str, Any],
    *,
    min_relative_length_shift: float = 0.10,
    min_attribute_shift: float = 0.25,
    min_direction_tv_delta: float = 0.05,
    min_prompt_js_delta: float = 0.01,
    min_position_bias: float = 0.20,
    required_domains: int = 2,
) -> dict[str, Any]:
    if required_domains <= 0:
        raise ValueError("required_domains must be positive")
    thresholds = {
        "min_relative_length_shift": min_relative_length_shift,
        "min_attribute_shift": min_attribute_shift,
        "min_direction_tv_delta": min_direction_tv_delta,
        "min_prompt_js_delta": min_prompt_js_delta,
        "min_position_bias": min_position_bias,
        "required_domains": required_domains,
    }
    all_pair_analysis = _analyze_preference_method_group(
        report.get("methods", {}), thresholds=thresholds
    )
    dpo_group = report.get("dpo_methods") or report.get("methods", {})
    dpo_analysis = _analyze_preference_method_group(dpo_group, thresholds=thresholds)
    random_report = dpo_group.get("random", {})
    order_pool_mean = (
        random_report.get("scoring", {})
        .get("order_disagreement", {})
        .get("pool_mean")
    )
    position_bias_warning = (
        order_pool_mean is not None and float(order_pool_mean) >= min_position_bias
    )
    scoring_reliable = not position_bias_warning
    established_methods = [
        method for method, values in dpo_analysis.items() if values["established"]
    ]
    recommended = (
        max(
            established_methods,
            key=lambda method: (
                dpo_analysis[method]["gate_score"],
                len(dpo_analysis[method]["material_domains"]),
                method,
            ),
        )
        if established_methods
        else None
    )
    dpo_training_ready = bool(established_methods) and scoring_reliable
    return {
        "thresholds": thresholds,
        "all_pair_methods": all_pair_analysis,
        "dpo_methods": dpo_analysis,
        "dpo_shift_established": bool(established_methods),
        "recommended_uncertainty_method": recommended,
        "position_bias_pool_mean": float(order_pool_mean) if order_pool_mean is not None else None,
        "position_bias_warning": position_bias_warning,
        "scoring_reliable": scoring_reliable,
        "dpo_training_ready": dpo_training_ready,
        "next_step": (
            "repair_pairwise_scoring_first"
            if position_bias_warning
            else "run_random_vs_uncertainty_dpo"
            if recommended
            else "collect_more_preference_evidence"
        ),
    }


def _analyze_preference_method_group(
    methods: dict[str, Any], *, thresholds: dict[str, float | int]
) -> dict[str, Any]:
    if not methods:
        return {}
    if "random" not in methods:
        raise ValueError("preference diagnostics must include random")
    random_report = methods["random"]
    analyses: dict[str, Any] = {}
    for method, method_report in methods.items():
        if method == "random":
            continue
        calibrated = (
            "domain_effects" in method_report
            and "domain_excess_vs_random_q95" in method_report
        )
        effects = (
            {domain: float(value) for domain, value in method_report["domain_effects"].items()}
            if calibrated
            else _preference_domain_effects(method_report, random_report)
        )
        significance = (
            {
                domain: float(value) > 0.0
                for domain, value in method_report["domain_excess_vs_random_q95"].items()
            }
            if calibrated
            else {domain: True for domain in effects}
        )
        material_domains = [
            domain
            for domain, value in effects.items()
            if domain in {"length", "attributes", "preference_direction", "prompt_distribution"}
            and value >= _preference_domain_threshold(domain, thresholds)
            and significance.get(domain, False)
        ]
        gate_score = sum(
            effects[domain] / _preference_domain_threshold(domain, thresholds)
            for domain in material_domains
        )
        analyses[method] = {
            "effects_vs_random": effects,
            "material_domains": material_domains,
            "gate_score": gate_score,
            "calibration": (
                "random_domain_envelope" if calibrated else "single_random_selection"
            ),
            "established": len(material_domains) >= int(thresholds["required_domains"]),
        }
    return analyses


def _preference_domain_effects(
    method_report: dict[str, Any], random_report: dict[str, Any]
) -> dict[str, float]:
    length_effect = 0.0
    for field, method_values in method_report.get("length", {}).items():
        random_values = random_report.get("length", {}).get(field, {})
        method_delta = _float_or_zero(method_values.get("delta"))
        random_delta = _float_or_zero(random_values.get("delta"))
        pool_mean = method_values.get("pool_mean", random_values.get("pool_mean"))
        scale = max(abs(_float_or_zero(pool_mean)), 1.0)
        length_effect = max(length_effect, abs(method_delta - random_delta) / scale)

    attribute_effect = 0.0
    for group, attributes in method_report.get("attributes", {}).items():
        random_attributes = random_report.get("attributes", {}).get(group, {})
        for attribute, method_values in attributes.items():
            random_values = random_attributes.get(attribute, {})
            attribute_effect = max(
                attribute_effect,
                abs(
                    _float_or_zero(method_values.get("delta"))
                    - _float_or_zero(random_values.get("delta"))
                ),
            )

    direction_effect = abs(
        _float_or_zero(method_report.get("preference_direction", {}).get("tv"))
        - _float_or_zero(random_report.get("preference_direction", {}).get("tv"))
    )
    prompt_effect = abs(
        _float_or_zero(
            method_report.get("prompt_distribution", {}).get("token_js_divergence")
        )
        - _float_or_zero(
            random_report.get("prompt_distribution", {}).get("token_js_divergence")
        )
    )
    order_effect = abs(
        _float_or_zero(
            method_report.get("scoring", {}).get("order_disagreement", {}).get("delta")
        )
        - _float_or_zero(
            random_report.get("scoring", {}).get("order_disagreement", {}).get("delta")
        )
    )
    return {
        "length": length_effect,
        "attributes": attribute_effect,
        "preference_direction": direction_effect,
        "prompt_distribution": prompt_effect,
        "order_disagreement": order_effect,
    }


def _preference_domain_threshold(
    domain: str, thresholds: dict[str, float | int]
) -> float:
    mapping = {
        "length": "min_relative_length_shift",
        "attributes": "min_attribute_shift",
        "preference_direction": "min_direction_tv_delta",
        "prompt_distribution": "min_prompt_js_delta",
    }
    return float(thresholds[mapping[domain]])


def _float_or_zero(value: Any) -> float:
    return float(value) if value is not None else 0.0


def _label_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10**9, value
