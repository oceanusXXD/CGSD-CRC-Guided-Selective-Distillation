from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
from .binary_protocol import normalize_binary_label
DEFAULT_LAMBDA_GRID = [round(0.5 + index * 0.01, 2) for index in range(51)]

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
        return {'alpha': float(self.alpha), 'temperature': float(self.temperature), 'lambda_hat': float(self.lambda_hat), 'empirical_risk': float(self.empirical_risk), 'risk_bound': float(self.risk_bound), 'guide_count': int(self.guide_count), 'guide_accept_count': int(self.guide_accept_count), 'guide_defer_count': int(self.guide_defer_count)}

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
        return {'temperature': float(self.temperature), 'alpha': float(self.alpha), 'lambda_hat': float(self.lambda_hat), 'tau_crc': float(self.tau_crc), 'r_U': float(self.r_U), 'r_C': float(self.r_C), 'e_all': float(self.e_all), 'e_defer': float(self.e_defer), 'c_crc': float(self.c_crc), 'eta_crc': float(self.eta_crc), 's_accept': float(self.s_accept), 's_defer': float(self.s_defer), 'pool_accept_count': int(self.pool_accept_count), 'pool_defer_count': int(self.pool_defer_count), 'guide_count': int(self.guide_count), 'guide_defer_count': int(self.guide_defer_count), 'guide_error_count': int(self.guide_error_count), 'guide_defer_error_count': int(self.guide_defer_error_count)}

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
        return {'temperature': float(self.temperature), 'alpha': float(self.alpha), 'lambda_hat': float(self.lambda_hat), 'tau_crc': float(self.tau_crc), 'budget': int(self.budget), 'p_hat_1': float(self.p_hat_1), 'target_label0_budget': int(self.target_label0_budget), 'target_label1_budget': int(self.target_label1_budget), 'B_label0': int(self.B_label0), 'B_label1': int(self.B_label1), 'guide_count': int(self.guide_count), 'guide_label0_count': int(self.guide_label0_count), 'guide_label1_count': int(self.guide_label1_count), 'pool_label0_count': int(self.pool_label0_count), 'pool_label1_count': int(self.pool_label1_count), 'pool_accept_count': int(self.pool_accept_count), 'pool_defer_count': int(self.pool_defer_count)}

@dataclass(frozen=True)
class AdaptiveSelectionPlan:
    temperature: float
    alpha: float
    lambda_hat: float
    tau_crc: float
    budget: int
    risk_strength: float
    risk_level: str
    guide_true_yes_rate: float
    guide_pred_yes_rate: float
    guide_soft_yes_rate: float
    guide_calibrated_yes_rate: float
    guide_proxy_balanced_accuracy: float
    guide_proxy_reliability: float
    guide_score_brier: float
    guide_base_brier: float
    guide_score_reliability: float
    label_probability_reliability: float
    score_calibration_slope: float
    score_calibration_intercept: float
    pool_pred_yes_rate: float
    pool_soft_yes_rate: float
    pool_calibrated_yes_rate: float
    target_yes_rate: float
    target_label0_mass: float
    target_label1_mass: float
    target_label0_budget: int
    target_label1_budget: int
    w_uncertainty: float
    w_alignment: float
    w_diversity: float
    bucket_count: int
    guide_count: int
    pool_count: int
    guide_accept_count: int
    pool_accept_count: int
    pool_defer_count: int

    def to_dict(self) -> dict[str, Any]:
        return {'temperature': float(self.temperature), 'alpha': float(self.alpha), 'lambda_hat': float(self.lambda_hat), 'tau_crc': float(self.tau_crc), 'budget': int(self.budget), 'risk_strength': float(self.risk_strength), 'risk_level': self.risk_level, 'guide_true_yes_rate': float(self.guide_true_yes_rate), 'guide_pred_yes_rate': float(self.guide_pred_yes_rate), 'guide_soft_yes_rate': float(self.guide_soft_yes_rate), 'guide_calibrated_yes_rate': float(self.guide_calibrated_yes_rate), 'guide_proxy_balanced_accuracy': float(self.guide_proxy_balanced_accuracy), 'guide_proxy_reliability': float(self.guide_proxy_reliability), 'guide_score_brier': float(self.guide_score_brier), 'guide_base_brier': float(self.guide_base_brier), 'guide_score_reliability': float(self.guide_score_reliability), 'label_probability_reliability': float(self.label_probability_reliability), 'score_calibration_slope': float(self.score_calibration_slope), 'score_calibration_intercept': float(self.score_calibration_intercept), 'pool_pred_yes_rate': float(self.pool_pred_yes_rate), 'pool_soft_yes_rate': float(self.pool_soft_yes_rate), 'pool_calibrated_yes_rate': float(self.pool_calibrated_yes_rate), 'target_yes_rate': float(self.target_yes_rate), 'target_label0_mass': float(self.target_label0_mass), 'target_label1_mass': float(self.target_label1_mass), 'target_label0_budget': int(self.target_label0_budget), 'target_label1_budget': int(self.target_label1_budget), 'w_uncertainty': float(self.w_uncertainty), 'w_alignment': float(self.w_alignment), 'w_diversity': float(self.w_diversity), 'bucket_count': int(self.bucket_count), 'guide_count': int(self.guide_count), 'pool_count': int(self.pool_count), 'guide_accept_count': int(self.guide_accept_count), 'pool_accept_count': int(self.pool_accept_count), 'pool_defer_count': int(self.pool_defer_count)}

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
        return {'method': self.method, 'selected_ids': list(self.selected_ids), 'accept_ids': list(self.accept_ids), 'defer_ids': list(self.defer_ids), 'label0_ids': list(self.label0_ids), 'label1_ids': list(self.label1_ids), 'requested_budget': int(self.requested_budget), 'selected_budget': int(self.selected_budget), 'requested_accept_budget': int(self.requested_accept_budget), 'requested_defer_budget': int(self.requested_defer_budget), 'requested_label0_budget': int(self.requested_label0_budget), 'requested_label1_budget': int(self.requested_label1_budget), 'accept_candidate_count': int(self.accept_candidate_count), 'defer_candidate_count': int(self.defer_candidate_count), 'label0_candidate_count': int(self.label0_candidate_count), 'label1_candidate_count': int(self.label1_candidate_count), 'shortfall': bool(self.shortfall)}

def _row_id(row: Mapping[str, Any]) -> str:
    sample_id = row.get('id', row.get('sample_id'))
    if sample_id is None or str(sample_id) == '':
        raise ValueError(f'row missing id/sample_id: {row!r}')
    return str(sample_id)

def _row_label(row: Mapping[str, Any]) -> int:
    return normalize_binary_label(row.get('label', row.get('groundtruth')), field_name='row label')

def _row_prediction(row: Mapping[str, Any]) -> int:
    if 'prediction' in row:
        return normalize_binary_label(row['prediction'], field_name='row prediction')
    return 1 if float(row.get('score', 0.0) or 0.0) > 0.0 else 0

def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0

def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))

def routing_score_from_margin(score: float, temperature: float) -> float:
    temp = float(temperature)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError('temperature must be a positive finite number')
    value = max(-60.0, min(60.0, abs(float(score)) / temp))
    return float(1.0 / (1.0 + math.exp(-value)))

def attach_routing_scores(rows: Sequence[Mapping[str, Any]], *, temperature: float) -> list[dict[str, Any]]:
    routed: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item['routing_score'] = routing_score_from_margin(float(item['score']), temperature)
        item['routing_temperature'] = float(temperature)
        routed.append(item)
    return routed

def crc_risk_bound(empirical_risk: float, n: int) -> float:
    if n <= 0:
        raise ValueError('CRC guide set cannot be empty')
    return float(n / (n + 1.0) * float(empirical_risk) + 1.0 / (n + 1.0))

def calibrate_crc(guide_predictions: Sequence[Mapping[str, Any]], *, alpha: float, temperature: float, lambda_grid: Iterable[float]=DEFAULT_LAMBDA_GRID) -> CRCResult:
    guide_rows = attach_routing_scores(guide_predictions, temperature=temperature)
    if not guide_rows:
        raise ValueError('guide_predictions cannot be empty')
    guide_count = len(guide_rows)
    last_result: CRCResult | None = None
    for lambda_value in lambda_grid:
        threshold = float(lambda_value)
        accept_rows = [row for row in guide_rows if float(row['routing_score']) >= threshold]
        losses = sum((1 for row in accept_rows if _row_prediction(row) != _row_label(row)))
        empirical_risk = float(losses / guide_count)
        bound = crc_risk_bound(empirical_risk, guide_count)
        result = CRCResult(alpha=float(alpha), temperature=float(temperature), lambda_hat=threshold, empirical_risk=empirical_risk, risk_bound=bound, guide_count=guide_count, guide_accept_count=len(accept_rows), guide_defer_count=guide_count - len(accept_rows))
        last_result = result
        if bound <= float(alpha):
            return result
    assert last_result is not None
    return CRCResult(alpha=float(alpha), temperature=float(temperature), lambda_hat=1.01, empirical_risk=last_result.empirical_risk, risk_bound=last_result.risk_bound, guide_count=guide_count, guide_accept_count=0, guide_defer_count=guide_count)

def crc_margin_cutoff(lambda_hat: float, temperature: float) -> float:
    threshold = float(lambda_hat)
    if threshold <= 0.0:
        return float('-inf')
    if threshold >= 1.0:
        return float('inf')
    return float(temperature) * math.log(threshold / (1.0 - threshold))

def apply_crc_defer_set(predictions: Sequence[Mapping[str, Any]], *, lambda_hat: float, temperature: float) -> list[dict[str, Any]]:
    threshold = float(lambda_hat)
    decisions: list[dict[str, Any]] = []
    for row in attach_routing_scores(predictions, temperature=temperature):
        item = dict(row)
        if threshold > 1.0:
            decision = 'defer'
        else:
            decision = 'accept' if float(item['routing_score']) >= threshold else 'defer'
        item['crc_decision'] = decision
        item['defer'] = decision == 'defer'
        item['decision_threshold'] = threshold
        item['tau_crc'] = crc_margin_cutoff(threshold, temperature)
        decisions.append(item)
    return decisions

def summarize_crc_decisions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    defer_count = sum((1 for row in rows if bool(row.get('defer', False))))
    accept_rows = [row for row in rows if not bool(row.get('defer', False))]
    errors = sum((1 for row in rows if _row_prediction(row) != _row_label(row)))
    accept_errors = sum((1 for row in accept_rows if _row_prediction(row) != _row_label(row)))
    return {'total': int(total), 'accept_count': int(total - defer_count), 'defer_count': int(defer_count), 'defer_rate': _rate(defer_count, total), 'error_count': int(errors), 'error_rate': _rate(errors, total), 'accept_error_count': int(accept_errors), 'accept_error_rate': _rate(accept_errors, len(accept_rows))}

def compute_crc_error_mass_plan(guide_decisions: Sequence[Mapping[str, Any]], pool_decisions: Sequence[Mapping[str, Any]], *, budget: int, temperature: float, lambda_hat: float, alpha: float) -> CRCErrorMassPlan:
    requested_budget = int(budget)
    guide_count = len(guide_decisions)
    if guide_count <= 0:
        raise ValueError('guide_decisions cannot be empty')
    pool_total = len(pool_decisions)
    pool_defer_count = sum((1 for row in pool_decisions if bool(row.get('defer', False))))
    pool_accept_count = pool_total - pool_defer_count
    guide_defer_rows = [row for row in guide_decisions if bool(row.get('defer', False))]
    guide_defer_count = len(guide_defer_rows)
    guide_error_count = sum((1 for row in guide_decisions if _row_prediction(row) != _row_label(row)))
    guide_defer_error_count = sum((1 for row in guide_defer_rows if _row_prediction(row) != _row_label(row)))
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
    s_defer = max(0.0, min(1.0, r_U + eta_crc * (1.0 - r_U) ** 2))
    s_accept = 1.0 - s_defer
    B_defer = max(0, min(requested_budget, _round_half_up(requested_budget * s_defer)))
    B_accept = requested_budget - B_defer
    return CRCErrorMassPlan(temperature=float(temperature), alpha=float(alpha), lambda_hat=float(lambda_hat), tau_crc=crc_margin_cutoff(lambda_hat, temperature), budget=requested_budget, r_U=r_U, r_C=r_C, e_all=e_all, e_defer=e_defer, c_crc=c_crc, eta_crc=eta_crc, s_accept=s_accept, s_defer=s_defer, B_accept=B_accept, B_defer=B_defer, pool_accept_count=pool_accept_count, pool_defer_count=pool_defer_count, guide_count=guide_count, guide_defer_count=guide_defer_count, guide_error_count=guide_error_count, guide_defer_error_count=guide_defer_error_count)

def _allocate_label_budgets(*, budget: int, p_hat_1: float, label0_count: int, label1_count: int) -> tuple[int, int, int, int]:
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
    return (target_label0_budget, target_label1_budget, label0_budget, label1_budget)

def compute_pcss_plan(guide_decisions: Sequence[Mapping[str, Any]], pool_decisions: Sequence[Mapping[str, Any]], *, budget: int, temperature: float, lambda_hat: float, alpha: float) -> PCSSPlan:
    requested_budget = int(budget)
    guide_count = len(guide_decisions)
    if guide_count <= 0:
        raise ValueError('guide_decisions cannot be empty')
    guide_label1_count = sum((1 for row in guide_decisions if _row_label(row) == 1))
    guide_label0_count = guide_count - guide_label1_count
    p_hat_1 = _rate(guide_label1_count, guide_count)
    pool_label1_count = sum((1 for row in pool_decisions if _row_prediction(row) == 1))
    pool_label0_count = len(pool_decisions) - pool_label1_count
    pool_defer_count = sum((1 for row in pool_decisions if bool(row.get('defer', False))))
    pool_accept_count = len(pool_decisions) - pool_defer_count
    target_label0_budget, target_label1_budget, label0_budget, label1_budget = _allocate_label_budgets(budget=requested_budget, p_hat_1=p_hat_1, label0_count=pool_label0_count, label1_count=pool_label1_count)
    return PCSSPlan(temperature=float(temperature), alpha=float(alpha), lambda_hat=float(lambda_hat), tau_crc=crc_margin_cutoff(lambda_hat, temperature), budget=requested_budget, p_hat_1=p_hat_1, target_label0_budget=target_label0_budget, target_label1_budget=target_label1_budget, B_label0=label0_budget, B_label1=label1_budget, guide_count=guide_count, guide_label0_count=guide_label0_count, guide_label1_count=guide_label1_count, pool_label0_count=pool_label0_count, pool_label1_count=pool_label1_count, pool_accept_count=pool_accept_count, pool_defer_count=pool_defer_count)

def _prior_risk_strength(*, abs_label_gap_pp: float, pred_skew_pp: float) -> float:
    gap_term = min(1.0, max(0.0, float(abs_label_gap_pp) / 30.0))
    skew_term = min(1.0, max(0.0, float(pred_skew_pp) / 40.0))
    return float(max(gap_term, skew_term))

def _risk_level_from_strength(value: float) -> str:
    strength = float(value)
    if strength >= 0.75:
        return 'severe'
    if strength >= 0.35:
        return 'moderate'
    return 'low'

def _soft_label1_probability(row: Mapping[str, Any], *, temperature: float) -> float:
    if row.get('score') is None:
        return float(_row_prediction(row))
    temp = float(temperature)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError('temperature must be a positive finite number')
    value = max(-60.0, min(60.0, float(row['score']) / temp))
    return float(1.0 / (1.0 + math.exp(-value)))

def _score_feature(row: Mapping[str, Any], *, temperature: float) -> float:
    if row.get('score') is None:
        return 1.0 if _row_prediction(row) == 1 else -1.0
    temp = float(temperature)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError('temperature must be a positive finite number')
    return max(-20.0, min(20.0, float(row['score']) / temp))

def _logit_probability(value: float) -> float:
    clipped = min(1.0 - 1e-06, max(1e-06, float(value)))
    return math.log(clipped / (1.0 - clipped))

def _sigmoid(value: float) -> float:
    clipped = max(-60.0, min(60.0, float(value)))
    return float(1.0 / (1.0 + math.exp(-clipped)))

def _fit_score_calibration(guide_decisions: Sequence[Mapping[str, Any]], *, temperature: float) -> tuple[float, float]:
    if not guide_decisions:
        raise ValueError('guide_decisions cannot be empty')
    features = [_score_feature(row, temperature=temperature) for row in guide_decisions]
    labels = [float(_row_label(row)) for row in guide_decisions]
    smoothed_yes_rate = (sum(labels) + 0.5) / (len(labels) + 1.0)
    slope = 1.0
    intercept = _logit_probability(smoothed_yes_rate)
    l2 = 0.01
    for _ in range(50):
        grad_slope = 0.0
        grad_intercept = 0.0
        h_ss = l2
        h_si = 0.0
        h_ii = 0.0
        for feature, label in zip(features, labels):
            probability = _sigmoid(slope * feature + intercept)
            residual = probability - label
            weight = probability * (1.0 - probability)
            grad_slope += residual * feature
            grad_intercept += residual
            h_ss += weight * feature * feature
            h_si += weight * feature
            h_ii += weight
        n = float(len(labels))
        grad_slope = grad_slope / n + l2 * slope
        grad_intercept /= n
        h_ss = h_ss / n + l2
        h_si /= n
        h_ii /= n
        determinant = h_ss * h_ii - h_si * h_si
        if abs(determinant) < 1e-12:
            break
        step_slope = (grad_slope * h_ii - grad_intercept * h_si) / determinant
        step_intercept = (h_ss * grad_intercept - h_si * grad_slope) / determinant
        step_slope = max(-5.0, min(5.0, step_slope))
        step_intercept = max(-5.0, min(5.0, step_intercept))
        slope -= step_slope
        intercept -= step_intercept
        slope = max(-20.0, min(20.0, slope))
        intercept = max(-20.0, min(20.0, intercept))
        if abs(step_slope) + abs(step_intercept) < 1e-07:
            break
    return (float(slope), float(intercept))

def _calibrated_label1_probability(row: Mapping[str, Any], *, temperature: float, slope: float, intercept: float) -> float:
    return _sigmoid(float(slope) * _score_feature(row, temperature=temperature) + float(intercept))

def adaptive_label1_probability(row: Mapping[str, Any], plan: AdaptiveSelectionPlan) -> float:
    calibrated_probability = _calibrated_label1_probability(row, temperature=plan.temperature, slope=plan.score_calibration_slope, intercept=plan.score_calibration_intercept)
    reliability = min(1.0, max(0.0, float(plan.label_probability_reliability)))
    return float(reliability * calibrated_probability + (1.0 - reliability) * float(plan.target_yes_rate))

def _guide_proxy_balanced_accuracy(guide_decisions: Sequence[Mapping[str, Any]]) -> float:
    recalls: list[float] = []
    for label in (0, 1):
        label_rows = [row for row in guide_decisions if _row_label(row) == label]
        if not label_rows:
            recalls.append(0.0)
            continue
        correct = sum((1 for row in label_rows if _row_prediction(row) == label))
        recalls.append(_rate(correct, len(label_rows)))
    return float(sum(recalls) / 2.0)

def _proxy_reliability_from_balanced_accuracy(value: float) -> float:
    return min(1.0, max(0.0, 2.0 * (float(value) - 0.5)))

def _brier_score(rows: Sequence[Mapping[str, Any]], *, probability: float | None=None, temperature: float | None=None, slope: float | None=None, intercept: float | None=None) -> float:
    if not rows:
        return 0.0
    losses: list[float] = []
    for row in rows:
        if probability is None:
            assert temperature is not None
            assert slope is not None
            assert intercept is not None
            predicted = _calibrated_label1_probability(row, temperature=temperature, slope=slope, intercept=intercept)
        else:
            predicted = float(probability)
        error = predicted - float(_row_label(row))
        losses.append(error * error)
    return float(sum(losses) / len(losses))

def _score_reliability_from_brier(*, base_brier: float, score_brier: float) -> float:
    if base_brier <= 1e-12:
        return 0.0
    return min(1.0, max(0.0, (float(base_brier) - float(score_brier)) / float(base_brier)))

def compute_adaptive_selection_plan(guide_decisions: Sequence[Mapping[str, Any]], pool_decisions: Sequence[Mapping[str, Any]], *, budget: int, temperature: float, lambda_hat: float, alpha: float, bucket_count: int=5) -> AdaptiveSelectionPlan:
    requested_budget = max(0, int(budget))
    guide_count = len(guide_decisions)
    pool_count = len(pool_decisions)
    if guide_count <= 0:
        raise ValueError('guide_decisions cannot be empty')
    if pool_count <= 0:
        raise ValueError('pool_decisions cannot be empty')
    guide_true_yes_rate = _rate(sum((1 for row in guide_decisions if _row_label(row) == 1)), guide_count)
    guide_pred_yes_rate = _rate(sum((1 for row in guide_decisions if _row_prediction(row) == 1)), guide_count)
    guide_soft_yes_rate = _rate(sum((_soft_label1_probability(row, temperature=temperature) for row in guide_decisions)), guide_count)
    calibration_slope, calibration_intercept = _fit_score_calibration(guide_decisions, temperature=temperature)
    guide_calibrated_yes_rate = _rate(sum((_calibrated_label1_probability(row, temperature=temperature, slope=calibration_slope, intercept=calibration_intercept) for row in guide_decisions)), guide_count)
    pool_pred_yes_rate = _rate(sum((1 for row in pool_decisions if _row_prediction(row) == 1)), pool_count)
    pool_soft_yes_rate = _rate(sum((_soft_label1_probability(row, temperature=temperature) for row in pool_decisions)), pool_count)
    pool_calibrated_yes_rate = _rate(sum((_calibrated_label1_probability(row, temperature=temperature, slope=calibration_slope, intercept=calibration_intercept) for row in pool_decisions)), pool_count)
    abs_gap_pp = 100.0 * abs(guide_pred_yes_rate - guide_true_yes_rate)
    pred_skew_pp = abs(100.0 * guide_pred_yes_rate - 50.0)
    risk_strength = _prior_risk_strength(abs_label_gap_pp=abs_gap_pp, pred_skew_pp=pred_skew_pp)
    proxy_balanced_accuracy = _guide_proxy_balanced_accuracy(guide_decisions)
    proxy_reliability = _proxy_reliability_from_balanced_accuracy(proxy_balanced_accuracy)
    base_brier = _brier_score(guide_decisions, probability=guide_true_yes_rate)
    score_brier = _brier_score(guide_decisions, temperature=temperature, slope=calibration_slope, intercept=calibration_intercept)
    score_reliability = _score_reliability_from_brier(base_brier=base_brier, score_brier=score_brier)
    label_probability_reliability = max(proxy_reliability, score_reliability)
    target_yes_rate = risk_strength * guide_true_yes_rate + (1.0 - risk_strength) * pool_calibrated_yes_rate
    target_yes_rate = min(1.0, max(0.0, target_yes_rate))
    target_label1_mass = float(requested_budget) * target_yes_rate
    target_label0_mass = float(requested_budget) - target_label1_mass
    target_label1_budget = max(0, min(requested_budget, _round_half_up(requested_budget * target_yes_rate)))
    target_label0_budget = requested_budget - target_label1_budget
    weight_alignment = 0.2 + 0.35 * risk_strength
    weight_uncertainty = 0.65 - 0.35 * risk_strength
    weight_diversity = 0.15
    pool_defer_count = sum((1 for row in pool_decisions if bool(row.get('defer', False))))
    return AdaptiveSelectionPlan(temperature=float(temperature), alpha=float(alpha), lambda_hat=float(lambda_hat), tau_crc=crc_margin_cutoff(lambda_hat, temperature), budget=requested_budget, risk_strength=risk_strength, risk_level=_risk_level_from_strength(risk_strength), guide_true_yes_rate=guide_true_yes_rate, guide_pred_yes_rate=guide_pred_yes_rate, guide_soft_yes_rate=guide_soft_yes_rate, guide_calibrated_yes_rate=guide_calibrated_yes_rate, guide_proxy_balanced_accuracy=proxy_balanced_accuracy, guide_proxy_reliability=proxy_reliability, guide_score_brier=score_brier, guide_base_brier=base_brier, guide_score_reliability=score_reliability, label_probability_reliability=label_probability_reliability, score_calibration_slope=calibration_slope, score_calibration_intercept=calibration_intercept, pool_pred_yes_rate=pool_pred_yes_rate, pool_soft_yes_rate=pool_soft_yes_rate, pool_calibrated_yes_rate=pool_calibrated_yes_rate, target_yes_rate=target_yes_rate, target_label0_mass=target_label0_mass, target_label1_mass=target_label1_mass, target_label0_budget=target_label0_budget, target_label1_budget=target_label1_budget, w_uncertainty=weight_uncertainty, w_alignment=weight_alignment, w_diversity=weight_diversity, bucket_count=max(1, int(bucket_count)), guide_count=guide_count, pool_count=pool_count, guide_accept_count=sum((1 for row in guide_decisions if not bool(row.get('defer', False)))), pool_accept_count=pool_count - pool_defer_count, pool_defer_count=pool_defer_count)

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
    return ids[:max(0, min(int(k), len(ids)))]

def _high_confidence_ids(candidate_ids: Sequence[str], rows_by_id: Mapping[str, Mapping[str, Any]], *, k: int) -> list[str]:
    return sorted(list(candidate_ids), key=lambda sample_id: (-float(rows_by_id[sample_id].get('routing_score', 0.0) or 0.0), sample_id))[:max(0, int(k))]

def _row_routing_score(row: Mapping[str, Any], *, temperature: float) -> float:
    if 'routing_score' in row and row.get('routing_score') is not None:
        routing_score = float(row['routing_score'])
    elif 'score' in row:
        routing_score = routing_score_from_margin(float(row['score']), temperature)
    else:
        raise ValueError(f'row missing routing_score/score for difficulty selection: {row!r}')
    if not math.isfinite(routing_score):
        raise ValueError(f'routing_score must be finite, got {routing_score!r}')
    return routing_score

def _uncertain_ids(candidate_ids: Sequence[str], rows_by_id: Mapping[str, Mapping[str, Any]], *, k: int, temperature: float) -> list[str]:
    return sorted(list(candidate_ids), key=lambda sample_id: (_row_routing_score(rows_by_id[sample_id], temperature=temperature), sample_id))[:max(0, int(k))]

def _confidence_bucket_index(routing_score: float, *, bucket_count: int) -> int:
    if bucket_count <= 1:
        return 0
    clipped = min(0.999999, max(0.5, float(routing_score)))
    fraction = (clipped - 0.5) / 0.5
    bucket = int(fraction * bucket_count)
    return min(bucket_count - 1, max(0, bucket))

def _soft_label_budget_distance(*, selected_count: int, label1_mass: float, target_label0_mass: float, target_label1_mass: float) -> float:
    label0_mass = float(selected_count) - float(label1_mass)
    return abs(label0_mass - float(target_label0_mass)) + abs(float(label1_mass) - float(target_label1_mass))

def _adaptive_marginal_gain(*, sample_id: str, rows_by_id: Mapping[str, Mapping[str, Any]], routing_scores: Mapping[str, float], selected_bucket_counts: Mapping[int, int], selected_count: int, selected_label1_mass: float, target_label0_mass: float, target_label1_mass: float, adaptive_plan: AdaptiveSelectionPlan) -> float:
    label1_probability = adaptive_label1_probability(rows_by_id[sample_id], adaptive_plan)
    routing_score = float(routing_scores[sample_id])
    uncertainty_gain = 1.0 - routing_score
    bucket = _confidence_bucket_index(routing_score, bucket_count=adaptive_plan.bucket_count)
    diversity_gain = 1.0 / float(1 + int(selected_bucket_counts.get(bucket, 0)))
    distance_before = _soft_label_budget_distance(selected_count=selected_count, label1_mass=selected_label1_mass, target_label0_mass=target_label0_mass, target_label1_mass=target_label1_mass)
    distance_after = _soft_label_budget_distance(selected_count=selected_count + 1, label1_mass=selected_label1_mass + label1_probability, target_label0_mass=target_label0_mass, target_label1_mass=target_label1_mass)
    balance_gain = float(distance_before - distance_after)
    return adaptive_plan.w_uncertainty * uncertainty_gain + adaptive_plan.w_alignment * balance_gain + adaptive_plan.w_diversity * diversity_gain

def _adaptive_greedy_ids(*, candidate_ids: Sequence[str], rows_by_id: Mapping[str, Mapping[str, Any]], routing_scores: Mapping[str, float], budget: int, target_label0_mass: float, target_label1_mass: float, adaptive_plan: AdaptiveSelectionPlan) -> list[str]:
    selected: list[str] = []
    remaining = set(candidate_ids)
    selected_bucket_counts: dict[int, int] = {}
    selected_label1_mass = 0.0
    requested_budget = max(0, min(int(budget), len(remaining)))
    while len(selected) < requested_budget and remaining:
        sample_id = min(remaining, key=lambda sample_id: (-_adaptive_marginal_gain(sample_id=sample_id, rows_by_id=rows_by_id, routing_scores=routing_scores, selected_bucket_counts=selected_bucket_counts, selected_count=len(selected), selected_label1_mass=selected_label1_mass, target_label0_mass=target_label0_mass, target_label1_mass=target_label1_mass, adaptive_plan=adaptive_plan), routing_scores[sample_id], sample_id))
        selected.append(sample_id)
        remaining.remove(sample_id)
        selected_label1_mass += adaptive_label1_probability(rows_by_id[sample_id], adaptive_plan)
        bucket = _confidence_bucket_index(routing_scores[sample_id], bucket_count=adaptive_plan.bucket_count)
        selected_bucket_counts[bucket] = selected_bucket_counts.get(bucket, 0) + 1
    return selected

def _harmonic_number(count: int) -> float:
    return float(sum((1.0 / index for index in range(1, max(0, int(count)) + 1))))

def _adaptive_swap_refine_ids(*, selected_ids: Sequence[str], candidate_ids: Sequence[str], rows_by_id: Mapping[str, Mapping[str, Any]], routing_scores: Mapping[str, float], target_label0_mass: float, target_label1_mass: float, adaptive_plan: AdaptiveSelectionPlan, target_proxy_label1_count: int | None=None, max_swaps: int=5, min_gain: float=1e-12) -> list[str]:
    selected = list(selected_ids)
    selected_set = set(selected)
    if not selected or len(selected_set) != len(selected):
        return selected
    label1_probabilities = {sample_id: adaptive_label1_probability(rows_by_id[sample_id], adaptive_plan) for sample_id in candidate_ids}
    uncertainty_values = {sample_id: 1.0 - float(routing_scores[sample_id]) for sample_id in candidate_ids}
    bucket_by_id = {sample_id: _confidence_bucket_index(routing_scores[sample_id], bucket_count=adaptive_plan.bucket_count) for sample_id in candidate_ids}
    proxy_label1_by_id = {sample_id: 1 if _row_prediction(rows_by_id[sample_id]) == 1 else 0 for sample_id in candidate_ids}

    def _state(ids: Sequence[str]) -> tuple[float, float, dict[int, int], float]:
        label1_mass = sum((label1_probabilities[sample_id] for sample_id in ids))
        uncertainty = sum((uncertainty_values[sample_id] for sample_id in ids))
        bucket_counts: dict[int, int] = {}
        for sample_id in ids:
            bucket = bucket_by_id[sample_id]
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        distance = _soft_label_budget_distance(selected_count=len(ids), label1_mass=label1_mass, target_label0_mass=target_label0_mass, target_label1_mass=target_label1_mass)
        diversity = sum((_harmonic_number(count) for count in bucket_counts.values()))
        objective = adaptive_plan.w_uncertainty * uncertainty - adaptive_plan.w_alignment * distance + adaptive_plan.w_diversity * diversity
        return (label1_mass, uncertainty, bucket_counts, objective)
    candidate_id_set = set(candidate_ids)
    max_accepted_swaps = max(0, int(max_swaps))
    accepted_swaps = 0
    while accepted_swaps < max_accepted_swaps:
        label1_mass, uncertainty, bucket_counts, objective = _state(selected)
        current_proxy_label1_count = sum((proxy_label1_by_id[sample_id] for sample_id in selected))
        current_diversity = sum((_harmonic_number(count) for count in bucket_counts.values()))
        unselected = [sample_id for sample_id in candidate_id_set if sample_id not in selected_set]
        ranked_unselected = sorted(unselected, key=lambda sample_id: (float(routing_scores[sample_id]), sample_id))
        best_gain = float(min_gain)
        best_swap: tuple[int, str] | None = None
        for out_index, out_id in enumerate(selected):
            out_bucket = bucket_by_id[out_id]
            out_bucket_count = bucket_counts[out_bucket]
            for in_id in ranked_unselected:
                if target_proxy_label1_count is not None:
                    new_proxy_label1_count = current_proxy_label1_count - proxy_label1_by_id[out_id] + proxy_label1_by_id[in_id]
                    if new_proxy_label1_count != int(target_proxy_label1_count):
                        continue
                in_bucket = bucket_by_id[in_id]
                new_label1_mass = label1_mass - label1_probabilities[out_id] + label1_probabilities[in_id]
                new_uncertainty = uncertainty - uncertainty_values[out_id] + uncertainty_values[in_id]
                new_distance = _soft_label_budget_distance(selected_count=len(selected), label1_mass=new_label1_mass, target_label0_mass=target_label0_mass, target_label1_mass=target_label1_mass)
                if in_bucket == out_bucket:
                    new_diversity = current_diversity
                else:
                    in_bucket_count = bucket_counts.get(in_bucket, 0)
                    new_diversity = current_diversity
                    new_diversity -= _harmonic_number(out_bucket_count)
                    new_diversity += _harmonic_number(out_bucket_count - 1)
                    new_diversity -= _harmonic_number(in_bucket_count)
                    new_diversity += _harmonic_number(in_bucket_count + 1)
                new_objective = adaptive_plan.w_uncertainty * new_uncertainty - adaptive_plan.w_alignment * new_distance + adaptive_plan.w_diversity * new_diversity
                gain = new_objective - objective
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (out_index, in_id)
        if best_swap is None:
            break
        out_index, in_id = best_swap
        out_id = selected[out_index]
        selected_set.remove(out_id)
        selected[out_index] = in_id
        selected_set.add(in_id)
        accepted_swaps += 1
    return selected

def _repair_label_balance(*, selected_ids: list[str], pool_ids: set[str], rows_by_id: Mapping[str, Mapping[str, Any]], utilities: Mapping[str, float], target_label1_budget: int, temperature: float) -> list[str]:
    selected = list(selected_ids)
    selected_set = set(selected)
    selected_label1 = [sample_id for sample_id in selected if _row_prediction(rows_by_id[sample_id]) == 1]
    selected_label0 = [sample_id for sample_id in selected if _row_prediction(rows_by_id[sample_id]) == 0]
    unselected_label1 = [sample_id for sample_id in pool_ids if sample_id not in selected_set and _row_prediction(rows_by_id[sample_id]) == 1]
    unselected_label0 = [sample_id for sample_id in pool_ids if sample_id not in selected_set and _row_prediction(rows_by_id[sample_id]) == 0]

    def _drop_id(ids: list[str]) -> str | None:
        if not ids:
            return None
        return min(ids, key=lambda sample_id: (utilities[sample_id], -_row_routing_score(rows_by_id[sample_id], temperature=temperature), sample_id))

    def _add_id(ids: list[str]) -> str | None:
        if not ids:
            return None
        return min(ids, key=lambda sample_id: (-utilities[sample_id], _row_routing_score(rows_by_id[sample_id], temperature=temperature), sample_id))
    current_label1 = len(selected_label1)
    target_label1_budget = max(0, int(target_label1_budget))
    while current_label1 < target_label1_budget and selected_label0 and unselected_label1:
        drop_id = _drop_id(selected_label0)
        add_id = _add_id(unselected_label1)
        if drop_id is None or add_id is None:
            break
        selected.remove(drop_id)
        selected.append(add_id)
        selected_set.remove(drop_id)
        selected_set.add(add_id)
        selected_label0.remove(drop_id)
        selected_label1.append(add_id)
        unselected_label0.append(drop_id)
        unselected_label1.remove(add_id)
        current_label1 += 1
    while current_label1 > target_label1_budget and selected_label1 and unselected_label0:
        drop_id = _drop_id(selected_label1)
        add_id = _add_id(unselected_label0)
        if drop_id is None or add_id is None:
            break
        selected.remove(drop_id)
        selected.append(add_id)
        selected_set.remove(drop_id)
        selected_set.add(add_id)
        selected_label1.remove(drop_id)
        selected_label0.append(add_id)
        unselected_label1.append(drop_id)
        unselected_label0.remove(add_id)
        current_label1 -= 1
    return selected

def _build_selection_result(*, method: str, selected_ids: Sequence[str], accept_ids: Sequence[str], defer_ids: Sequence[str], label0_ids: Sequence[str], label1_ids: Sequence[str], requested_budget: int, requested_accept_budget: int=0, requested_defer_budget: int=0, requested_label0_budget: int=0, requested_label1_budget: int=0) -> SelectionResult:
    selected = list(selected_ids)
    accept_id_set = set(accept_ids)
    defer_id_set = set(defer_ids)
    label0_id_set = set(label0_ids)
    label1_id_set = set(label1_ids)
    return SelectionResult(
        method=method,
        selected_ids=selected,
        accept_ids=[sample_id for sample_id in selected if sample_id in accept_id_set],
        defer_ids=[sample_id for sample_id in selected if sample_id in defer_id_set],
        label0_ids=[sample_id for sample_id in selected if sample_id in label0_id_set],
        label1_ids=[sample_id for sample_id in selected if sample_id in label1_id_set],
        requested_budget=int(requested_budget),
        selected_budget=len(selected),
        requested_accept_budget=int(requested_accept_budget),
        requested_defer_budget=int(requested_defer_budget),
        requested_label0_budget=int(requested_label0_budget),
        requested_label1_budget=int(requested_label1_budget),
        accept_candidate_count=len(accept_ids),
        defer_candidate_count=len(defer_ids),
        label0_candidate_count=len(label0_ids),
        label1_candidate_count=len(label1_ids),
        shortfall=len(selected) < int(requested_budget),
    )

def select_training_ids(pool_decisions: Sequence[Mapping[str, Any]], *, method: str, budget: int, seed: int, blocked_ids: set[str] | None=None, crc_error_mass_plan: CRCErrorMassPlan | None=None, pcss_plan: PCSSPlan | None=None, adaptive_plan: AdaptiveSelectionPlan | None=None, accept_strategy: str='random', defer_strategy: str='random') -> SelectionResult:
    method_name = str(method)
    if method_name not in {'random', 'pcss', 'crc-error-mass', 'adaptive'}:
        raise ValueError("method must be one of {'random', 'pcss', 'crc-error-mass', 'adaptive'}")
    if accept_strategy not in {'random', 'high-confidence'}:
        raise ValueError("accept_strategy must be one of {'random', 'high-confidence'}")
    if defer_strategy not in {'random', 'high-confidence'}:
        raise ValueError("defer_strategy must be one of {'random', 'high-confidence'}")
    blocked = set(blocked_ids or set())
    rows_by_id = {_row_id(row): row for row in pool_decisions}
    all_ids = _unique_ids(pool_decisions, blocked_ids=blocked)
    accept_ids = [sample_id for sample_id in all_ids if not bool(rows_by_id[sample_id].get('defer', False))]
    defer_ids = [sample_id for sample_id in all_ids if bool(rows_by_id[sample_id].get('defer', False))]
    label0_ids = [sample_id for sample_id in all_ids if _row_prediction(rows_by_id[sample_id]) == 0]
    label1_ids = [sample_id for sample_id in all_ids if _row_prediction(rows_by_id[sample_id]) == 1]
    requested_budget = int(budget)
    if method_name == 'random':
        selected = _random_ids(all_ids, k=requested_budget, seed=seed)
        return _build_selection_result(method=method_name, selected_ids=selected, accept_ids=accept_ids, defer_ids=defer_ids, label0_ids=label0_ids, label1_ids=label1_ids, requested_budget=requested_budget)
    if method_name == 'pcss':
        if pcss_plan is None:
            raise ValueError("pcss_plan is required for method='pcss'")
        _, _, label0_budget, label1_budget = _allocate_label_budgets(budget=requested_budget, p_hat_1=pcss_plan.p_hat_1, label0_count=len(label0_ids), label1_count=len(label1_ids))
        selected_label0 = _uncertain_ids(label0_ids, rows_by_id, k=label0_budget, temperature=pcss_plan.temperature)
        selected_label1 = _uncertain_ids(label1_ids, rows_by_id, k=label1_budget, temperature=pcss_plan.temperature)
        selected = [*selected_label1, *selected_label0]
        return _build_selection_result(method=method_name, selected_ids=selected, accept_ids=accept_ids, defer_ids=defer_ids, label0_ids=label0_ids, label1_ids=label1_ids, requested_budget=requested_budget, requested_label0_budget=label0_budget, requested_label1_budget=label1_budget)
    if method_name == 'adaptive':
        if adaptive_plan is None:
            raise ValueError("adaptive_plan is required for method='adaptive'")
        if not all_ids:
            return _build_selection_result(method=method_name, selected_ids=[], accept_ids=[], defer_ids=[], label0_ids=[], label1_ids=[], requested_budget=requested_budget)
        feasible_budget = min(max(0, requested_budget), len(all_ids))
        if adaptive_plan.guide_proxy_reliability >= 0.5:
            _, _, label0_budget, label1_budget = _allocate_label_budgets(budget=requested_budget, p_hat_1=adaptive_plan.target_yes_rate, label0_count=len(label0_ids), label1_count=len(label1_ids))
        else:
            label1_budget = max(0, min(feasible_budget, _round_half_up(feasible_budget * adaptive_plan.target_yes_rate)))
            label0_budget = feasible_budget - label1_budget
        target_label1_mass = float(feasible_budget) * float(adaptive_plan.target_yes_rate)
        target_label0_mass = float(feasible_budget) - target_label1_mass
        routing_scores = {sample_id: _row_routing_score(rows_by_id[sample_id], temperature=adaptive_plan.temperature) for sample_id in all_ids}
        bucket_counts: dict[int, int] = {}
        for sample_id in all_ids:
            bucket = _confidence_bucket_index(routing_scores[sample_id], bucket_count=adaptive_plan.bucket_count)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        utilities: dict[str, float] = {}
        for sample_id in all_ids:
            utilities[sample_id] = _adaptive_marginal_gain(sample_id=sample_id, rows_by_id=rows_by_id, routing_scores=routing_scores, selected_bucket_counts=bucket_counts, selected_count=0, selected_label1_mass=0.0, target_label0_mass=target_label0_mass, target_label1_mass=target_label1_mass, adaptive_plan=adaptive_plan)
        selected = _adaptive_greedy_ids(candidate_ids=all_ids, rows_by_id=rows_by_id, routing_scores=routing_scores, budget=requested_budget, target_label0_mass=target_label0_mass, target_label1_mass=target_label1_mass, adaptive_plan=adaptive_plan)
        if adaptive_plan.guide_proxy_reliability >= 0.5:
            selected = _repair_label_balance(selected_ids=selected, pool_ids=set(all_ids), rows_by_id=rows_by_id, utilities=utilities, target_label1_budget=label1_budget, temperature=adaptive_plan.temperature)
        selected = _adaptive_swap_refine_ids(selected_ids=selected, candidate_ids=all_ids, rows_by_id=rows_by_id, routing_scores=routing_scores, target_label0_mass=target_label0_mass, target_label1_mass=target_label1_mass, adaptive_plan=adaptive_plan, target_proxy_label1_count=label1_budget if adaptive_plan.guide_proxy_reliability >= 0.5 else None)
        return _build_selection_result(method=method_name, selected_ids=selected, accept_ids=accept_ids, defer_ids=defer_ids, label0_ids=label0_ids, label1_ids=label1_ids, requested_budget=requested_budget, requested_label0_budget=label0_budget, requested_label1_budget=label1_budget)
    if crc_error_mass_plan is None:
        raise ValueError("crc_error_mass_plan is required for method='crc-error-mass'")
    accept_budget = min(int(crc_error_mass_plan.B_accept), len(accept_ids))
    defer_budget = min(int(crc_error_mass_plan.B_defer), len(defer_ids))
    if accept_strategy == 'high-confidence':
        selected_accept = _high_confidence_ids(accept_ids, rows_by_id, k=accept_budget)
    else:
        selected_accept = _random_ids(accept_ids, k=accept_budget, seed=seed)
    if defer_strategy == 'high-confidence':
        selected_defer = _high_confidence_ids(defer_ids, rows_by_id, k=defer_budget)
    else:
        selected_defer = _random_ids(defer_ids, k=defer_budget, seed=seed + 1)
    selected = [*selected_accept, *selected_defer]
    return _build_selection_result(method=method_name, selected_ids=selected, accept_ids=accept_ids, defer_ids=defer_ids, label0_ids=label0_ids, label1_ids=label1_ids, requested_budget=requested_budget, requested_accept_budget=crc_error_mass_plan.B_accept, requested_defer_budget=crc_error_mass_plan.B_defer)
