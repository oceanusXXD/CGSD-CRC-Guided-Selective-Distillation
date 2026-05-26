from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .binary_protocol import normalize_binary_label


DEFAULT_LAMBDA_GRID = [round(0.50 + index * 0.01, 2) for index in range(51)]


@dataclass(frozen=True)
class CRCResult:
    alpha: float
    temperature: float
    lambda_hat: float
    empirical_risk: float
    risk_bound: float
    guide_count: int
    guide_accept_count: int
    guide_defer_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "temperature": float(self.temperature),
            "lambda_hat": float(self.lambda_hat),
            "empirical_risk": float(self.empirical_risk),
            "risk_bound": float(self.risk_bound),
            "guide_count": int(self.guide_count),
            "guide_accept_count": int(self.guide_accept_count),
            "guide_defer_count": int(self.guide_defer_count),
        }


@dataclass(frozen=True)
class CRCErrorMassPlan:
    temperature: float
    alpha: float
    lambda_hat: float
    tau_crc: float
    budget: int
    r_U: float
    r_C: float
    e_all: float
    e_defer: float
    c_crc: float
    eta_crc: float
    s_accept: float
    s_defer: float
    B_accept: int
    B_defer: int
    pool_accept_count: int
    pool_defer_count: int
    guide_count: int
    guide_defer_count: int
    guide_error_count: int
    guide_defer_error_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": float(self.temperature),
            "alpha": float(self.alpha),
            "lambda_hat": float(self.lambda_hat),
            "tau_crc": float(self.tau_crc),
            "r_U": float(self.r_U),
            "r_C": float(self.r_C),
            "e_all": float(self.e_all),
            "e_defer": float(self.e_defer),
            "c_crc": float(self.c_crc),
            "eta_crc": float(self.eta_crc),
            "s_accept": float(self.s_accept),
            "s_defer": float(self.s_defer),
            "pool_accept_count": int(self.pool_accept_count),
            "pool_defer_count": int(self.pool_defer_count),
            "guide_count": int(self.guide_count),
            "guide_defer_count": int(self.guide_defer_count),
            "guide_error_count": int(self.guide_error_count),
            "guide_defer_error_count": int(self.guide_defer_error_count),
        }


@dataclass(frozen=True)
class PCSSPlan:
    temperature: float
    alpha: float
    lambda_hat: float
    tau_crc: float
    budget: int
    p_hat_1: float
    target_label0_budget: int
    target_label1_budget: int
    B_label0: int
    B_label1: int
    guide_count: int
    guide_label0_count: int
    guide_label1_count: int
    pool_label0_count: int
    pool_label1_count: int
    pool_accept_count: int
    pool_defer_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": float(self.temperature),
            "alpha": float(self.alpha),
            "lambda_hat": float(self.lambda_hat),
            "tau_crc": float(self.tau_crc),
            "budget": int(self.budget),
            "p_hat_1": float(self.p_hat_1),
            "target_label0_budget": int(self.target_label0_budget),
            "target_label1_budget": int(self.target_label1_budget),
            "B_label0": int(self.B_label0),
            "B_label1": int(self.B_label1),
            "guide_count": int(self.guide_count),
            "guide_label0_count": int(self.guide_label0_count),
            "guide_label1_count": int(self.guide_label1_count),
            "pool_label0_count": int(self.pool_label0_count),
            "pool_label1_count": int(self.pool_label1_count),
            "pool_accept_count": int(self.pool_accept_count),
            "pool_defer_count": int(self.pool_defer_count),
        }


@dataclass(frozen=True)
class SelectionResult:
    method: str
    selected_ids: list[str]
    accept_ids: list[str]
    defer_ids: list[str]
    label0_ids: list[str]
    label1_ids: list[str]
    requested_budget: int
    selected_budget: int
    requested_accept_budget: int
    requested_defer_budget: int
    requested_label0_budget: int
    requested_label1_budget: int
    accept_candidate_count: int
    defer_candidate_count: int
    label0_candidate_count: int
    label1_candidate_count: int
    shortfall: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "selected_ids": list(self.selected_ids),
            "accept_ids": list(self.accept_ids),
            "defer_ids": list(self.defer_ids),
            "label0_ids": list(self.label0_ids),
            "label1_ids": list(self.label1_ids),
            "requested_budget": int(self.requested_budget),
            "selected_budget": int(self.selected_budget),
            "requested_accept_budget": int(self.requested_accept_budget),
            "requested_defer_budget": int(self.requested_defer_budget),
            "requested_label0_budget": int(self.requested_label0_budget),
            "requested_label1_budget": int(self.requested_label1_budget),
            "accept_candidate_count": int(self.accept_candidate_count),
            "defer_candidate_count": int(self.defer_candidate_count),
            "label0_candidate_count": int(self.label0_candidate_count),
            "label1_candidate_count": int(self.label1_candidate_count),
            "shortfall": bool(self.shortfall),
        }


def _row_id(row: Mapping[str, Any]) -> str:
    sample_id = row.get("id", row.get("sample_id"))
    if sample_id is None or str(sample_id) == "":
        raise ValueError(f"row missing id/sample_id: {row!r}")
    return str(sample_id)


def _row_label(row: Mapping[str, Any]) -> int:
    return normalize_binary_label(row.get("label", row.get("groundtruth")), field_name="row label")


def _row_prediction(row: Mapping[str, Any]) -> int:
    if "prediction" in row:
        return normalize_binary_label(row["prediction"], field_name="row prediction")
    return 1 if float(row.get("score", 0.0) or 0.0) > 0.0 else 0


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def routing_score_from_margin(score: float, temperature: float) -> float:
    temp = float(temperature)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError("temperature must be a positive finite number")
    value = max(-60.0, min(60.0, abs(float(score)) / temp))
    return float(1.0 / (1.0 + math.exp(-value)))


def attach_routing_scores(rows: Sequence[Mapping[str, Any]], *, temperature: float) -> list[dict[str, Any]]:
    routed: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["routing_score"] = routing_score_from_margin(float(item["score"]), temperature)
        item["routing_temperature"] = float(temperature)
        routed.append(item)
    return routed


def crc_risk_bound(empirical_risk: float, n: int) -> float:
    if n <= 0:
        raise ValueError("CRC guide set cannot be empty")
    return float((n / (n + 1.0)) * float(empirical_risk) + (1.0 / (n + 1.0)))


def calibrate_crc(
    guide_predictions: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    temperature: float,
    lambda_grid: Iterable[float] = DEFAULT_LAMBDA_GRID,
) -> CRCResult:
    guide_rows = attach_routing_scores(guide_predictions, temperature=temperature)
    if not guide_rows:
        raise ValueError("guide_predictions cannot be empty")
    guide_count = len(guide_rows)
    last_result: CRCResult | None = None
    for lambda_value in lambda_grid:
        threshold = float(lambda_value)
        accept_rows = [row for row in guide_rows if float(row["routing_score"]) >= threshold]
        losses = sum(1 for row in accept_rows if _row_prediction(row) != _row_label(row))
        empirical_risk = float(losses / guide_count)
        bound = crc_risk_bound(empirical_risk, guide_count)
        result = CRCResult(
            alpha=float(alpha),
            temperature=float(temperature),
            lambda_hat=threshold,
            empirical_risk=empirical_risk,
            risk_bound=bound,
            guide_count=guide_count,
            guide_accept_count=len(accept_rows),
            guide_defer_count=guide_count - len(accept_rows),
        )
        last_result = result
        if bound <= float(alpha):
            return result
    assert last_result is not None
    return CRCResult(
        alpha=float(alpha),
        temperature=float(temperature),
        lambda_hat=1.01,
        empirical_risk=last_result.empirical_risk,
        risk_bound=last_result.risk_bound,
        guide_count=guide_count,
        guide_accept_count=0,
        guide_defer_count=guide_count,
    )


def crc_margin_cutoff(lambda_hat: float, temperature: float) -> float:
    threshold = float(lambda_hat)
    if threshold <= 0.0:
        return float("-inf")
    if threshold >= 1.0:
        return float("inf")
    return float(temperature) * math.log(threshold / (1.0 - threshold))


def apply_crc_defer_set(
    predictions: Sequence[Mapping[str, Any]],
    *,
    lambda_hat: float,
    temperature: float,
) -> list[dict[str, Any]]:
    threshold = float(lambda_hat)
    decisions: list[dict[str, Any]] = []
    for row in attach_routing_scores(predictions, temperature=temperature):
        item = dict(row)
        if threshold > 1.0:
            decision = "defer"
        else:
            decision = "accept" if float(item["routing_score"]) >= threshold else "defer"
        item["crc_decision"] = decision
        item["defer"] = decision == "defer"
        item["decision_threshold"] = threshold
        item["tau_crc"] = crc_margin_cutoff(threshold, temperature)
        decisions.append(item)
    return decisions


def summarize_crc_decisions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    defer_count = sum(1 for row in rows if bool(row.get("defer", False)))
    accept_rows = [row for row in rows if not bool(row.get("defer", False))]
    errors = sum(1 for row in rows if _row_prediction(row) != _row_label(row))
    accept_errors = sum(1 for row in accept_rows if _row_prediction(row) != _row_label(row))
    return {
        "total": int(total),
        "accept_count": int(total - defer_count),
        "defer_count": int(defer_count),
        "defer_rate": _rate(defer_count, total),
        "error_count": int(errors),
        "error_rate": _rate(errors, total),
        "accept_error_count": int(accept_errors),
        "accept_error_rate": _rate(accept_errors, len(accept_rows)),
    }


def compute_crc_error_mass_plan(
    guide_decisions: Sequence[Mapping[str, Any]],
    pool_decisions: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    temperature: float,
    lambda_hat: float,
    alpha: float,
) -> CRCErrorMassPlan:
    requested_budget = int(budget)
    guide_count = len(guide_decisions)
    if guide_count <= 0:
        raise ValueError("guide_decisions cannot be empty")
    pool_total = len(pool_decisions)
    pool_defer_count = sum(1 for row in pool_decisions if bool(row.get("defer", False)))
    pool_accept_count = pool_total - pool_defer_count
    guide_defer_rows = [row for row in guide_decisions if bool(row.get("defer", False))]
    guide_defer_count = len(guide_defer_rows)
    guide_error_count = sum(1 for row in guide_decisions if _row_prediction(row) != _row_label(row))
    guide_defer_error_count = sum(1 for row in guide_defer_rows if _row_prediction(row) != _row_label(row))
    r_U = _rate(pool_defer_count, pool_total)
    r_C = _rate(guide_defer_count, guide_count)
    e_all = _rate(guide_error_count, guide_count)
    e_defer = _rate(guide_defer_error_count, guide_defer_count)
    if guide_defer_count == 0 or guide_error_count == 0 or e_all <= 0.0:
        c_crc = 1.0
        eta_crc = 0.0
    else:
        c_crc = e_defer / e_all
        if c_crc <= 1.0 or r_C <= 0.0 or r_C >= 1.0:
            eta_crc = 0.0
        else:
            eta_crc = max(0.0, min(1.0, math.log(c_crc) / math.log(1.0 / r_C)))
    s_defer = max(0.0, min(1.0, r_U + eta_crc * ((1.0 - r_U) ** 2)))
    s_accept = 1.0 - s_defer
    B_defer = max(0, min(requested_budget, _round_half_up(requested_budget * s_defer)))
    B_accept = requested_budget - B_defer
    return CRCErrorMassPlan(
        temperature=float(temperature),
        alpha=float(alpha),
        lambda_hat=float(lambda_hat),
        tau_crc=crc_margin_cutoff(lambda_hat, temperature),
        budget=requested_budget,
        r_U=r_U,
        r_C=r_C,
        e_all=e_all,
        e_defer=e_defer,
        c_crc=c_crc,
        eta_crc=eta_crc,
        s_accept=s_accept,
        s_defer=s_defer,
        B_accept=B_accept,
        B_defer=B_defer,
        pool_accept_count=pool_accept_count,
        pool_defer_count=pool_defer_count,
        guide_count=guide_count,
        guide_defer_count=guide_defer_count,
        guide_error_count=guide_error_count,
        guide_defer_error_count=guide_defer_error_count,
    )


def _allocate_label_budgets(
    *,
    budget: int,
    p_hat_1: float,
    label0_count: int,
    label1_count: int,
) -> tuple[int, int, int, int]:
    requested_budget = max(0, int(budget))
    positive_rate = max(0.0, min(1.0, float(p_hat_1)))
    target_label1_budget = max(0, min(requested_budget, _round_half_up(requested_budget * positive_rate)))
    target_label0_budget = requested_budget - target_label1_budget

    label0_capacity = max(0, int(label0_count))
    label1_capacity = max(0, int(label1_count))
    label0_budget = min(target_label0_budget, label0_capacity)
    label1_budget = min(target_label1_budget, label1_capacity)

    feasible_budget = min(requested_budget, label0_capacity + label1_capacity)
    remaining = feasible_budget - label0_budget - label1_budget
    if remaining > 0:
        label0_extra = min(remaining, label0_capacity - label0_budget)
        label0_budget += label0_extra
        remaining -= label0_extra
    if remaining > 0:
        label1_extra = min(remaining, label1_capacity - label1_budget)
        label1_budget += label1_extra

    return target_label0_budget, target_label1_budget, label0_budget, label1_budget


def compute_pcss_plan(
    guide_decisions: Sequence[Mapping[str, Any]],
    pool_decisions: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    temperature: float,
    lambda_hat: float,
    alpha: float,
) -> PCSSPlan:
    requested_budget = int(budget)
    guide_count = len(guide_decisions)
    if guide_count <= 0:
        raise ValueError("guide_decisions cannot be empty")
    guide_label1_count = sum(1 for row in guide_decisions if _row_label(row) == 1)
    guide_label0_count = guide_count - guide_label1_count
    p_hat_1 = _rate(guide_label1_count, guide_count)

    pool_label1_count = sum(1 for row in pool_decisions if _row_label(row) == 1)
    pool_label0_count = len(pool_decisions) - pool_label1_count
    pool_defer_count = sum(1 for row in pool_decisions if bool(row.get("defer", False)))
    pool_accept_count = len(pool_decisions) - pool_defer_count
    target_label0_budget, target_label1_budget, label0_budget, label1_budget = _allocate_label_budgets(
        budget=requested_budget,
        p_hat_1=p_hat_1,
        label0_count=pool_label0_count,
        label1_count=pool_label1_count,
    )
    return PCSSPlan(
        temperature=float(temperature),
        alpha=float(alpha),
        lambda_hat=float(lambda_hat),
        tau_crc=crc_margin_cutoff(lambda_hat, temperature),
        budget=requested_budget,
        p_hat_1=p_hat_1,
        target_label0_budget=target_label0_budget,
        target_label1_budget=target_label1_budget,
        B_label0=label0_budget,
        B_label1=label1_budget,
        guide_count=guide_count,
        guide_label0_count=guide_label0_count,
        guide_label1_count=guide_label1_count,
        pool_label0_count=pool_label0_count,
        pool_label1_count=pool_label1_count,
        pool_accept_count=pool_accept_count,
        pool_defer_count=pool_defer_count,
    )


def _unique_ids(rows: Sequence[Mapping[str, Any]], *, blocked_ids: set[str]) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for row in rows:
        sample_id = _row_id(row)
        if sample_id in blocked_ids or sample_id in seen:
            continue
        seen.add(sample_id)
        ids.append(sample_id)
    return ids


def _random_ids(candidate_ids: Sequence[str], *, k: int, seed: int) -> list[str]:
    ids = list(candidate_ids)
    rng = random.Random(int(seed))
    rng.shuffle(ids)
    return ids[: max(0, min(int(k), len(ids)))]


def _high_confidence_ids(
    candidate_ids: Sequence[str],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    *,
    k: int,
) -> list[str]:
    return sorted(
        list(candidate_ids),
        key=lambda sample_id: (-float(rows_by_id[sample_id].get("routing_score", 0.0) or 0.0), sample_id),
    )[: max(0, int(k))]


def _row_routing_score(row: Mapping[str, Any], *, temperature: float) -> float:
    if "routing_score" in row and row.get("routing_score") is not None:
        routing_score = float(row["routing_score"])
    elif "score" in row:
        routing_score = routing_score_from_margin(float(row["score"]), temperature)
    else:
        raise ValueError(f"row missing routing_score/score for difficulty selection: {row!r}")
    if not math.isfinite(routing_score):
        raise ValueError(f"routing_score must be finite, got {routing_score!r}")
    return routing_score


def _uncertain_ids(
    candidate_ids: Sequence[str],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    *,
    k: int,
    temperature: float,
) -> list[str]:
    return sorted(
        list(candidate_ids),
        key=lambda sample_id: (_row_routing_score(rows_by_id[sample_id], temperature=temperature), sample_id),
    )[: max(0, int(k))]


def select_training_ids(
    pool_decisions: Sequence[Mapping[str, Any]],
    *,
    method: str,
    budget: int,
    seed: int,
    blocked_ids: set[str] | None = None,
    crc_error_mass_plan: CRCErrorMassPlan | None = None,
    pcss_plan: PCSSPlan | None = None,
    accept_strategy: str = "random",
    defer_strategy: str = "random",
) -> SelectionResult:
    method_name = str(method)
    if method_name not in {"random", "pcss", "crc-error-mass"}:
        raise ValueError("method must be one of {'random', 'pcss', 'crc-error-mass'}")
    if accept_strategy not in {"random", "high-confidence"}:
        raise ValueError("accept_strategy must be one of {'random', 'high-confidence'}")
    if defer_strategy not in {"random", "high-confidence"}:
        raise ValueError("defer_strategy must be one of {'random', 'high-confidence'}")
    blocked = set(blocked_ids or set())
    rows_by_id = {_row_id(row): row for row in pool_decisions}
    all_ids = _unique_ids(pool_decisions, blocked_ids=blocked)
    accept_ids = [sample_id for sample_id in all_ids if not bool(rows_by_id[sample_id].get("defer", False))]
    defer_ids = [sample_id for sample_id in all_ids if bool(rows_by_id[sample_id].get("defer", False))]
    label0_ids = [sample_id for sample_id in all_ids if _row_label(rows_by_id[sample_id]) == 0]
    label1_ids = [sample_id for sample_id in all_ids if _row_label(rows_by_id[sample_id]) == 1]
    requested_budget = int(budget)

    if method_name == "random":
        selected = _random_ids(all_ids, k=requested_budget, seed=seed)
        accept_id_set = set(accept_ids)
        defer_id_set = set(defer_ids)
        label0_id_set = set(label0_ids)
        label1_id_set = set(label1_ids)
        selected_accept = [sample_id for sample_id in selected if sample_id in accept_id_set]
        selected_defer = [sample_id for sample_id in selected if sample_id in defer_id_set]
        selected_label0 = [sample_id for sample_id in selected if sample_id in label0_id_set]
        selected_label1 = [sample_id for sample_id in selected if sample_id in label1_id_set]
        return SelectionResult(
            method=method_name,
            selected_ids=selected,
            accept_ids=selected_accept,
            defer_ids=selected_defer,
            label0_ids=selected_label0,
            label1_ids=selected_label1,
            requested_budget=requested_budget,
            selected_budget=len(selected),
            requested_accept_budget=0,
            requested_defer_budget=0,
            requested_label0_budget=0,
            requested_label1_budget=0,
            accept_candidate_count=len(accept_ids),
            defer_candidate_count=len(defer_ids),
            label0_candidate_count=len(label0_ids),
            label1_candidate_count=len(label1_ids),
            shortfall=len(selected) < requested_budget,
        )

    if method_name == "pcss":
        if pcss_plan is None:
            raise ValueError("pcss_plan is required for method='pcss'")
        _, _, label0_budget, label1_budget = _allocate_label_budgets(
            budget=requested_budget,
            p_hat_1=pcss_plan.p_hat_1,
            label0_count=len(label0_ids),
            label1_count=len(label1_ids),
        )
        selected_label0 = _uncertain_ids(
            label0_ids,
            rows_by_id,
            k=label0_budget,
            temperature=pcss_plan.temperature,
        )
        selected_label1 = _uncertain_ids(
            label1_ids,
            rows_by_id,
            k=label1_budget,
            temperature=pcss_plan.temperature,
        )
        selected = [*selected_label1, *selected_label0]
        accept_id_set = set(accept_ids)
        defer_id_set = set(defer_ids)
        selected_accept = [sample_id for sample_id in selected if sample_id in accept_id_set]
        selected_defer = [sample_id for sample_id in selected if sample_id in defer_id_set]
        return SelectionResult(
            method=method_name,
            selected_ids=selected,
            accept_ids=selected_accept,
            defer_ids=selected_defer,
            label0_ids=selected_label0,
            label1_ids=selected_label1,
            requested_budget=requested_budget,
            selected_budget=len(selected),
            requested_accept_budget=0,
            requested_defer_budget=0,
            requested_label0_budget=label0_budget,
            requested_label1_budget=label1_budget,
            accept_candidate_count=len(accept_ids),
            defer_candidate_count=len(defer_ids),
            label0_candidate_count=len(label0_ids),
            label1_candidate_count=len(label1_ids),
            shortfall=len(selected) < requested_budget,
        )

    if crc_error_mass_plan is None:
        raise ValueError("crc_error_mass_plan is required for method='crc-error-mass'")
    accept_budget = min(int(crc_error_mass_plan.B_accept), len(accept_ids))
    defer_budget = min(int(crc_error_mass_plan.B_defer), len(defer_ids))
    if accept_strategy == "high-confidence":
        selected_accept = _high_confidence_ids(accept_ids, rows_by_id, k=accept_budget)
    else:
        selected_accept = _random_ids(accept_ids, k=accept_budget, seed=seed)
    if defer_strategy == "high-confidence":
        selected_defer = _high_confidence_ids(defer_ids, rows_by_id, k=defer_budget)
    else:
        selected_defer = _random_ids(defer_ids, k=defer_budget, seed=seed + 1)
    selected = [*selected_accept, *selected_defer]
    label0_id_set = set(label0_ids)
    label1_id_set = set(label1_ids)
    selected_label0 = [sample_id for sample_id in selected if sample_id in label0_id_set]
    selected_label1 = [sample_id for sample_id in selected if sample_id in label1_id_set]
    return SelectionResult(
        method=method_name,
        selected_ids=selected,
        accept_ids=selected_accept,
        defer_ids=selected_defer,
        label0_ids=selected_label0,
        label1_ids=selected_label1,
        requested_budget=requested_budget,
        selected_budget=len(selected),
        requested_accept_budget=int(crc_error_mass_plan.B_accept),
        requested_defer_budget=int(crc_error_mass_plan.B_defer),
        requested_label0_budget=0,
        requested_label1_budget=0,
        accept_candidate_count=len(accept_ids),
        defer_candidate_count=len(defer_ids),
        label0_candidate_count=len(label0_ids),
        label1_candidate_count=len(label1_ids),
        shortfall=len(selected) < requested_budget,
    )
