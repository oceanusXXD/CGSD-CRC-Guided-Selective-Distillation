"""CGSD 的自包含核心算法。

该模块只放纯算法逻辑，不加载模型、不调用外部 API，也不依赖项目外的
cascade 代码。脚本层负责把模型预测、teacher 标签和 embedding 转成这里的
输入行，算法层只负责 CRC 校准、defer 集识别、k-Center 选样和部署决策。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_LAMBDA_GRID = tuple(round(index / 100.0, 2) for index in range(50, 101))
DEFAULT_TEMPERATURE_GRID = (5.0, 10.0, 15.0, 20.0)
NEIGHBOR_SUPPORT_EPS = 1e-6
QUERY_REFERENCE_SHRINKAGE_K = 12.0
THRESHOLD_MULTIPLIER_GAMMA = 0.5


class CGSDEmbeddingError(RuntimeError):
    """k-Center 选择缺少真实 pair embedding 时抛出的错误。"""


@dataclass(frozen=True)
class CRCResult:
    """一次 CRC 校准后的阈值和风险摘要。"""

    alpha: float
    temperature: float
    lambda_hat: float
    n_calibration: int
    accepted_count: int
    wrong_accept_count: int
    empirical_risk: float
    risk_bound: float
    grid_feasible: bool
    neighbor_support_enabled: bool = False
    neighbor_support_reference: float | None = None
    query_reference_supports: dict[str, float] | None = None
    threshold_function: str = "global_threshold"

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "temperature": float(self.temperature),
            "lambda_hat": float(self.lambda_hat),
            "n_calibration": int(self.n_calibration),
            "accepted_count": int(self.accepted_count),
            "wrong_accept_count": int(self.wrong_accept_count),
            "empirical_risk": float(self.empirical_risk),
            "risk_bound": float(self.risk_bound),
            "grid_feasible": bool(self.grid_feasible),
            "method": "crc_wrong_accept_risk_v1",
            "loss": "1{routing_score>=lambda}*1{prediction!=teacher_label}",
            "neighbor_support_enabled": bool(self.neighbor_support_enabled),
            "neighbor_support_reference": (
                None
                if self.neighbor_support_reference is None
                else float(self.neighbor_support_reference)
            ),
            "query_reference_supports": dict(self.query_reference_supports or {}),
            "threshold_function": str(self.threshold_function),
        }


@dataclass(frozen=True)
class TemperatureChoice:
    """温度扫描的结果。"""

    temperature: float
    crc: CRCResult
    pool_defer_rate: float
    pool_defer_count: int
    candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": float(self.temperature),
            "crc": self.crc.to_dict(),
            "pool_defer_rate": float(self.pool_defer_rate),
            "pool_defer_count": int(self.pool_defer_count),
            "candidates": list(self.candidates),
        }


@dataclass(frozen=True)
class DeferSelection:
    """从当前 defer 集选出的蒸馏样本。"""

    distillation_ids: list[str]
    requested_budget: int
    selected_budget: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "distillation_ids": list(self.distillation_ids),
            "requested_budget": int(self.requested_budget),
            "selected_budget": int(self.selected_budget),
            "selection_method": "defer_k_center",
        }


@dataclass(frozen=True)
class AdaptiveSamplingPlan:
    """按 CRC defer rate 和错误浓缩度得到的本轮选样预算。"""

    temperature: float
    alpha: float | None
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
    calibration_count: int
    calibration_defer_count: int
    calibration_error_count: int
    calibration_defer_error_count: int

    def to_dict(self) -> dict[str, Any]:
        """导出 CRC Error-Mass 诊断量。

        `budget/B_accept/B_defer` 是 selection stage 的临时执行量，不写入
        公式诊断记录，避免和 CRC 校准字段混在一起。
        """
        return {
            "T": float(self.temperature),
            "alpha": None if self.alpha is None else float(self.alpha),
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
            "calibration_count": int(self.calibration_count),
            "calibration_defer_count": int(self.calibration_defer_count),
            "calibration_error_count": int(self.calibration_error_count),
            "calibration_defer_error_count": int(self.calibration_defer_error_count),
        }


@dataclass(frozen=True)
class AdaptiveSelection:
    """按自适应 accept/defer 预算选出的蒸馏样本。"""

    distillation_ids: list[str]
    accept_ids: list[str]
    defer_ids: list[str]
    requested_budget: int
    selected_budget: int
    requested_accept_budget: int
    requested_defer_budget: int
    selected_accept_budget: int
    selected_defer_budget: int
    accept_strategy: str
    defer_strategy: str
    selection_method: str = "adaptive_accept_defer"
    pool_candidate_count: int = 0
    accept_candidate_count: int = 0
    defer_candidate_count: int = 0
    shortfall: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "distillation_ids": list(self.distillation_ids),
            "accept_ids": list(self.accept_ids),
            "defer_ids": list(self.defer_ids),
            "requested_budget": int(self.requested_budget),
            "selected_budget": int(self.selected_budget),
            "requested_accept_budget": int(self.requested_accept_budget),
            "requested_defer_budget": int(self.requested_defer_budget),
            "selected_accept_budget": int(self.selected_accept_budget),
            "selected_defer_budget": int(self.selected_defer_budget),
            "accept_strategy": str(self.accept_strategy),
            "defer_strategy": str(self.defer_strategy),
            "selection_method": str(self.selection_method),
            "pool_candidate_count": int(self.pool_candidate_count),
            "accept_candidate_count": int(self.accept_candidate_count),
            "defer_candidate_count": int(self.defer_candidate_count),
            "shortfall": bool(self.shortfall),
        }


def _row_id(row: Mapping[str, Any]) -> str:
    value = row.get("id", row.get("sample_id", ""))
    return str(value)


def _binary_to_int(value: Any, *, field_name: str) -> int:
    """把外部输入里的 yes/no/true/false/1/0 统一收敛成 1/0。

    CGSD 算法层不直接消费字符串标签。模型协议里可以用 yes/no token，
    但进入 CRC、选样、迭代停止和部署决策前，标签和预测必须都是整数，
    这样不同阶段写出的 JSONL 不会出现混合口径。
    """
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "y", "true", "1"}:
            return 1
        if normalized in {"no", "n", "false", "0"}:
            return 0
    try:
        label = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be binary 0/1, got {value!r}") from exc
    if label not in {0, 1}:
        raise ValueError(f"{field_name} must be binary 0/1, got {value!r}")
    return label


def _row_label(row: Mapping[str, Any]) -> int:
    value = row.get("label", row.get("groundtruth"))
    if value is None:
        raise ValueError(f"row is missing label/groundtruth: {_row_id(row)!r}")
    return _binary_to_int(value, field_name="binary label")


def _row_prediction(row: Mapping[str, Any]) -> int:
    value = row.get("prediction")
    if value is None:
        score = float(row.get("score", 0.0) or 0.0)
        return 1 if score > 0.0 else 0
    return _binary_to_int(value, field_name="binary prediction")


def _row_query_id(row: Mapping[str, Any]) -> str:
    value = row.get("query_id")
    if value is not None and str(value).strip():
        return str(value).strip()
    sample_id = _row_id(row)
    if ":" in sample_id:
        return sample_id.rsplit(":", 1)[-1].strip()
    return ""


def _clip_unit(value: Any) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number) or number <= 0.0:
        return 0.0
    if number >= 1.0:
        return 1.0
    return float(number)


def _threshold_multiplier(neighbor_support: float, reference_support: float) -> float:
    """把局部 neighbor support 转成阈值缩放因子。

    support 高于参考值时，阈值会降低一些，允许更多局部一致的样本 accept；
    support 低于参考值时，阈值会升高，要求模型 margin 更强才 accept。
    """
    support = _clip_unit(neighbor_support)
    base = max(NEIGHBOR_SUPPORT_EPS, _clip_unit(reference_support))
    ratio = (base + NEIGHBOR_SUPPORT_EPS) / (support + NEIGHBOR_SUPPORT_EPS)
    return float(max(NEIGHBOR_SUPPORT_EPS, ratio) ** THRESHOLD_MULTIPLIER_GAMMA)


def _adaptive_threshold(lambda_hat: float, neighbor_support: float, reference_support: float) -> float:
    return _clip_unit(float(lambda_hat) * _threshold_multiplier(neighbor_support, reference_support))


def _lambda_transition(routing_score: float, neighbor_support: float, reference_support: float) -> float:
    """NS 自适应阈值下的候选 lambda 转折点。

    启用 neighbor support 时，每条样本的决策阈值不是全局 lambda，
    而是 lambda 乘上局部缩放因子。扫描这些转折点即可覆盖所有 accept
    集变化位置，比固定 0.00-1.00 网格更精确。
    """
    score = _clip_unit(routing_score)
    if score >= 1.0 - 1e-12:
        return 1.0
    multiplier = _threshold_multiplier(neighbor_support, reference_support)
    if multiplier <= 1e-12:
        return 1.0
    return _clip_unit(score / multiplier)


def _embedding_for_id(embeddings_by_id: Mapping[str, np.ndarray], sample_id: str) -> np.ndarray:
    if sample_id not in embeddings_by_id:
        raise CGSDEmbeddingError(
            "CRC neighbor support requires real precomputed pair embeddings; "
            f"missing id: {sample_id!r}"
        )
    vector = np.asarray(embeddings_by_id[sample_id], dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise CGSDEmbeddingError(f"invalid embedding vector for id: {sample_id!r}")
    return vector


def _support_bank_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    bank: list[dict[str, Any]] = []
    for row in rows:
        if row.get("label", row.get("groundtruth")) is None:
            continue
        item = dict(row)
        item["id"] = _row_id(row)
        item["prediction"] = _row_prediction(row)
        item["label"] = _row_label(row)
        bank.append(item)
    return bank


def _attach_neighbor_support_scores(
    target_rows: Sequence[Mapping[str, Any]],
    *,
    embeddings_by_id: Mapping[str, np.ndarray],
    support_rows: Sequence[Mapping[str, Any]],
    exclude_self: bool,
) -> list[dict[str, Any]]:
    """给样本补充局部邻域可靠度 N_i。

    N_i 的含义是：在 embedding 空间里，和当前样本预测方向相同的邻居中，
    有多少相似度权重来自“预测也确实正确”的样本。它只作为阈值自适应的
    局部证据，不改变原始 student score。
    """
    output = [dict(row) for row in target_rows]
    bank_rows = _support_bank_rows(support_rows)
    if not output or not bank_rows:
        for row in output:
            row["neighbor_support"] = 0.0
        return output

    bank_ids = [_row_id(row) for row in bank_rows]
    target_ids = [_row_id(row) for row in output]
    bank_matrix = _normalize_embedding_matrix(
        np.vstack([_embedding_for_id(embeddings_by_id, sample_id) for sample_id in bank_ids])
    )
    target_matrix = _normalize_embedding_matrix(
        np.vstack([_embedding_for_id(embeddings_by_id, sample_id) for sample_id in target_ids])
    )
    bank_preds = np.asarray([_row_prediction(row) for row in bank_rows], dtype=np.int64)
    bank_labels = np.asarray([_row_label(row) for row in bank_rows], dtype=np.int64)
    bank_index_by_id = {sample_id: index for index, sample_id in enumerate(bank_ids)}
    sim_matrix = np.matmul(target_matrix, bank_matrix.T)
    if exclude_self:
        for row_index, sample_id in enumerate(target_ids):
            bank_index = bank_index_by_id.get(sample_id)
            if bank_index is not None:
                sim_matrix[row_index, bank_index] = -np.inf

    for row_index, row in enumerate(output):
        pred = _row_prediction(row)
        similarities = sim_matrix[row_index]
        finite_indices = np.flatnonzero(np.isfinite(similarities))
        same_pred_indices = finite_indices[bank_preds[finite_indices] == pred]
        if same_pred_indices.size == 0:
            row["neighbor_support"] = 0.0
            continue
        weights = np.clip(similarities[same_pred_indices], a_min=0.0, a_max=None).astype(np.float64)
        total_weight = float(np.sum(weights))
        if total_weight <= 1e-12:
            row["neighbor_support"] = 0.0
            continue
        support_weight = float(np.sum(weights[bank_labels[same_pred_indices] == pred]))
        row["neighbor_support"] = float(max(0.0, min(1.0, support_weight / total_weight)))
    return output


def _neighbor_support_reference(rows: Sequence[Mapping[str, Any]]) -> float:
    values = [_clip_unit(row.get("neighbor_support", 0.0)) for row in rows]
    if not values:
        return NEIGHBOR_SUPPORT_EPS
    return float(max(NEIGHBOR_SUPPORT_EPS, min(1.0, float(np.median(np.asarray(values, dtype=np.float64))))))


def _query_reference_supports(
    rows: Sequence[Mapping[str, Any]],
    *,
    global_reference_support: float,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    global_ref = max(NEIGHBOR_SUPPORT_EPS, _clip_unit(global_reference_support))
    for row in rows:
        query_id = _row_query_id(row)
        if not query_id:
            continue
        grouped.setdefault(query_id, []).append(_clip_unit(row.get("neighbor_support", 0.0)))

    references: dict[str, float] = {}
    for query_id, values in grouped.items():
        local_ref = max(
            NEIGHBOR_SUPPORT_EPS,
            min(1.0, float(np.median(np.asarray(values, dtype=np.float64)))),
        )
        weight = float(len(values) / (len(values) + QUERY_REFERENCE_SHRINKAGE_K))
        references[query_id] = float(
            max(NEIGHBOR_SUPPORT_EPS, min(1.0, (weight * local_ref) + ((1.0 - weight) * global_ref)))
        )
    return references


def _row_reference_support(
    row: Mapping[str, Any],
    *,
    global_reference_support: float,
    query_reference_supports: Mapping[str, float] | None,
) -> float:
    query_id = _row_query_id(row)
    if query_reference_supports and query_id in query_reference_supports:
        return max(NEIGHBOR_SUPPORT_EPS, _clip_unit(query_reference_supports[query_id]))
    return max(NEIGHBOR_SUPPORT_EPS, _clip_unit(global_reference_support))


def sigmoid_abs_margin(score: float, temperature: float) -> float:
    """把 logit margin 映射到 CRC 路由分数 R_i。

    R_i 只表示 student 对自己预测方向的确信度，所以使用 abs(score)；
    temperature 越大，分数越接近 0.5，CRC 阈值会更保守。
    """
    temp = float(temperature)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError("temperature must be a positive finite number")
    value = abs(float(score)) / temp
    if value >= 0.0:
        exp_term = math.exp(-value)
        return float(1.0 / (1.0 + exp_term))
    exp_term = math.exp(value)
    return float(exp_term / (1.0 + exp_term))


def attach_routing_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    temperature: float,
) -> list[dict[str, Any]]:
    """返回带 routing_score 的预测行副本。

    输入行必须已经包含 student 的二分类 logit margin `score`。
    这里统一把 score 转成文档中的 R_i=sigma(|ell_i|/T)，后续 CRC、
    选样和部署判断都只消费这个同一口径的 `routing_score`。
    """
    routed: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["id"] = _row_id(row)
        item["prediction"] = _row_prediction(row)
        item["score"] = float(row.get("score", 0.0) or 0.0)
        item["routing_score"] = sigmoid_abs_margin(float(item["score"]), temperature)
        routed.append(item)
    return routed


def crc_risk_bound(empirical_risk: float, n: int) -> float:
    """CRC 有限样本修正：n/(n+1)*risk + 1/(n+1)。"""
    if int(n) <= 0:
        raise ValueError("CRC calibration needs at least one calibration row")
    return float((n / (n + 1.0)) * float(empirical_risk) + (1.0 / (n + 1.0)))


def calibrate_crc(
    calibration_rows: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    temperature: float,
    lambda_grid: Iterable[float] = DEFAULT_LAMBDA_GRID,
    embeddings_by_id: Mapping[str, np.ndarray] | None = None,
) -> CRCResult:
    """在校准集上搜索最宽松且满足 wrong-accept 风险的 CRC 阈值。

    核心损失为 L_i(lambda)=1{R_i>=lambda}*1{prediction_i!=label_i}。
    lambda 越小，accept 越多；因此按升序扫描网格，第一个可行 lambda
    就是 defer 最少的合法阈值。
    """
    rows = attach_routing_scores(calibration_rows, temperature=temperature)
    n = len(rows)
    if n == 0:
        raise ValueError("CRC calibration rows cannot be empty")
    risk_target = float(alpha)
    if not (0.0 <= risk_target <= 1.0):
        raise ValueError("alpha must be within [0, 1]")

    if embeddings_by_id is not None:
        # 中间轮里 calibration_rows 就是 D_guide。这里只用 D_guide 估计
        # 自适应 CRC 的 neighbor support，并排除样本自身 embedding。
        # D_cert 不能进入此路径；它只能用于最终 CRC/评估层。
        rows = _attach_neighbor_support_scores(
            rows,
            embeddings_by_id=embeddings_by_id,
            support_rows=rows,
            exclude_self=True,
        )
        reference_support = _neighbor_support_reference(rows)
        query_refs = _query_reference_supports(
            rows,
            global_reference_support=reference_support,
        )
        for row in rows:
            row["query_reference_support"] = _row_reference_support(
                row,
                global_reference_support=reference_support,
                query_reference_supports=query_refs,
            )
            row["lambda_transition"] = _lambda_transition(
                float(row.get("routing_score", 0.0) or 0.0),
                float(row.get("neighbor_support", 0.0) or 0.0),
                float(row["query_reference_support"]),
            )

        lambda_values = sorted({0.0, 1.0, *[float(row["lambda_transition"]) for row in rows]})
        threshold_function = "clip(lambda*(((b_q+eps)/(neighbor_support+eps))**gamma),0,1)"
    else:
        reference_support = None
        query_refs = {}
        lambda_values = sorted(float(value) for value in lambda_grid)
        threshold_function = "global_threshold"

    best: CRCResult | None = None
    for lambda_value in lambda_values:
        # 对每个 lambda 显式构造 accept 集，再计算
        # mean(1{accept and wrong})。分母固定为 n_calibration，
        # 这正是文档公式里的 (1/n) sum_j L_j(lambda)，不是
        # accept 条件下的错误率。
        if embeddings_by_id is not None:
            accepted = [
                row
                for row in rows
                if float(row["routing_score"])
                >= _adaptive_threshold(
                    float(lambda_value),
                    float(row.get("neighbor_support", 0.0) or 0.0),
                    float(row.get("query_reference_support", reference_support or NEIGHBOR_SUPPORT_EPS) or NEIGHBOR_SUPPORT_EPS),
                )
                - 1e-12
            ]
        else:
            accepted = [row for row in rows if float(row["routing_score"]) >= lambda_value]
        wrong_accept_count = sum(1 for row in accepted if _row_prediction(row) != _row_label(row))
        empirical_risk = wrong_accept_count / float(n)
        bound = crc_risk_bound(empirical_risk, n)
        if accepted and bound <= risk_target + 1e-12:
            best = CRCResult(
                alpha=risk_target,
                temperature=float(temperature),
                lambda_hat=float(lambda_value),
                n_calibration=n,
                accepted_count=len(accepted),
                wrong_accept_count=int(wrong_accept_count),
                empirical_risk=float(empirical_risk),
                risk_bound=float(bound),
                grid_feasible=True,
                neighbor_support_enabled=embeddings_by_id is not None,
                neighbor_support_reference=reference_support,
                query_reference_supports=query_refs,
                threshold_function=threshold_function,
            )
            break

    if best is not None:
        return best

    # 网格上不可行时，按方案约定设置 1.01，让所有样本 defer。
    # 此时 empirical risk 为 0，有限样本上界为 1/(n+1)。
    return CRCResult(
        alpha=risk_target,
        temperature=float(temperature),
        lambda_hat=1.01,
        n_calibration=n,
        accepted_count=0,
        wrong_accept_count=0,
        empirical_risk=0.0,
        risk_bound=crc_risk_bound(0.0, n),
        grid_feasible=False,
        neighbor_support_enabled=embeddings_by_id is not None,
        neighbor_support_reference=reference_support,
        query_reference_supports=query_refs,
        threshold_function=threshold_function,
    )


def apply_crc_decisions(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    lambda_hat: float,
    temperature: float,
    embeddings_by_id: Mapping[str, np.ndarray] | None = None,
    support_rows: Sequence[Mapping[str, Any]] | None = None,
    crc_result: CRCResult | Mapping[str, Any] | None = None,
    neighbor_exclude_self: bool = False,
) -> list[dict[str, Any]]:
    """用固定 lambda 和 temperature 给样本打 accept/defer 决策。

    这是部署阶段同样使用的唯一判定规则：
    R_i >= lambda_hat 时 accept，否则 defer。
    """
    routed = attach_routing_scores(prediction_rows, temperature=temperature)
    threshold = float(lambda_hat)
    force_all_defer = threshold > 1.0
    neighbor_enabled = bool(
        embeddings_by_id is not None
        and support_rows is not None
        and crc_result is not None
        and (
            bool(getattr(crc_result, "neighbor_support_enabled", False))
            if isinstance(crc_result, CRCResult)
            else bool(crc_result.get("neighbor_support_enabled", False))
        )
    )
    if neighbor_enabled:
        support_routed = attach_routing_scores(support_rows or [], temperature=temperature)
        # 中间轮决策时，调用方传入的 support_rows 是 D_guide。
        # D_cert 仍然隔离给最终 CRC/评估使用，不能作为这里的邻居库。
        routed = _attach_neighbor_support_scores(
            routed,
            embeddings_by_id=embeddings_by_id or {},
            support_rows=support_routed,
            exclude_self=bool(neighbor_exclude_self),
        )
        if isinstance(crc_result, CRCResult):
            reference_support = float(crc_result.neighbor_support_reference or NEIGHBOR_SUPPORT_EPS)
            query_refs = dict(crc_result.query_reference_supports or {})
        else:
            reference_support = float(crc_result.get("neighbor_support_reference", NEIGHBOR_SUPPORT_EPS) or NEIGHBOR_SUPPORT_EPS)
            query_refs = dict(crc_result.get("query_reference_supports", {}) or {})
    decided: list[dict[str, Any]] = []
    for row in routed:
        item = dict(row)
        if force_all_defer:
            decision_threshold = threshold
            decision = "defer"
        elif neighbor_enabled:
            reference = _row_reference_support(
                item,
                global_reference_support=reference_support,
                query_reference_supports=query_refs,
            )
            decision_threshold = _adaptive_threshold(
                threshold,
                float(item.get("neighbor_support", 0.0) or 0.0),
                reference,
            )
            decision = "accept" if float(item["routing_score"]) >= float(decision_threshold) - 1e-12 else "defer"
        else:
            decision_threshold = threshold
            decision = "accept" if float(item["routing_score"]) >= float(decision_threshold) - 1e-12 else "defer"
        item["crc_decision"] = decision
        item["defer"] = decision == "defer"
        item["crc_lambda"] = threshold
        item["crc_temperature"] = float(temperature)
        item["decision_threshold"] = float(decision_threshold)
        decided.append(item)
    return decided


def summarize_crc_decisions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """汇总 CRC 决策结果，用于记录每轮 defer 率和诊断错误率。"""
    total = len(rows)
    defer_count = sum(1 for row in rows if bool(row.get("defer", False)))
    accept_rows = [row for row in rows if not bool(row.get("defer", False))]
    wrong_accept_count = 0
    labeled_accept_count = 0
    for row in accept_rows:
        if row.get("label", row.get("groundtruth")) is None:
            continue
        labeled_accept_count += 1
        wrong_accept_count += int(_row_prediction(row) != _row_label(row))
    return {
        "total": int(total),
        "accept_count": int(total - defer_count),
        "defer_count": int(defer_count),
        "defer_rate": float(defer_count / total) if total else 0.0,
        "labeled_accept_count": int(labeled_accept_count),
        "wrong_accept_count": int(wrong_accept_count),
        "accept_error_rate": (
            float(wrong_accept_count / labeled_accept_count)
            if labeled_accept_count
            else 0.0
        ),
    }


def crc_margin_cutoff(lambda_hat: float, temperature: float) -> float:
    """把 routing-score 阈值换算成原始 score 空间的 margin cutoff。

    文档里的公式是 tau_crc = T * logit(lambda)。当 lambda 超出
    [0, 1] 边界时，用无穷值表达全 accept / 全 defer 的极限情况。
    """
    temp = float(temperature)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError("temperature must be a positive finite number")
    threshold = float(lambda_hat)
    if not math.isfinite(threshold):
        raise ValueError("lambda_hat must be finite")
    if threshold <= 0.0:
        return float("-inf")
    if threshold >= 1.0:
        return float("inf")
    return float(temp * math.log(threshold / (1.0 - threshold)))


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _rate(count: int, total: int) -> float:
    return float(count / total) if int(total) > 0 else 0.0


def compute_adaptive_sampling_plan(
    calibration_decisions: Sequence[Mapping[str, Any]],
    pool_decisions: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    temperature: float,
    lambda_hat: float,
    alpha: float | None = None,
) -> AdaptiveSamplingPlan:
    """根据更新文档计算 accept/defer 自适应选样比例和预算。

    输入必须已经是 CRC 判定后的行，至少包含 `defer`，并且校准行需要
    带有 label/groundtruth 与 prediction/score。这里严格按文档公式记录
    r_U、r_C、e_all、e_defer、c_crc、eta_crc、s_accept/s_defer 和预算。
    """
    requested_budget = int(max(0, budget))
    calibration_rows = [dict(row) for row in calibration_decisions]
    pool_rows = [dict(row) for row in pool_decisions]
    n_calibration = len(calibration_rows)
    if n_calibration == 0:
        raise ValueError("adaptive sampling needs non-empty calibration decisions")

    pool_total = len(pool_rows)
    pool_defer_count = sum(1 for row in pool_rows if bool(row.get("defer", False)))
    pool_accept_count = int(pool_total - pool_defer_count)
    r_U = _rate(pool_defer_count, pool_total)

    calibration_defer_rows = [row for row in calibration_rows if bool(row.get("defer", False))]
    calibration_defer_count = len(calibration_defer_rows)
    calibration_error_count = sum(1 for row in calibration_rows if _row_prediction(row) != _row_label(row))
    calibration_defer_error_count = sum(
        1 for row in calibration_defer_rows if _row_prediction(row) != _row_label(row)
    )
    r_C = _rate(calibration_defer_count, n_calibration)
    e_all = _rate(calibration_error_count, n_calibration)
    e_defer = _rate(calibration_defer_error_count, calibration_defer_count)

    if calibration_defer_count == 0 or calibration_error_count == 0 or e_all <= 0.0:
        c_crc = 1.0
        eta_crc = 0.0
    else:
        c_crc = float(e_defer / e_all) if e_all > 0.0 else 1.0
        if c_crc <= 1.0 or r_C <= 0.0 or r_C >= 1.0:
            eta_crc = 0.0
        else:
            denominator = math.log(1.0 / r_C)
            eta_crc = 0.0 if denominator <= 0.0 else math.log(c_crc) / denominator
            eta_crc = float(max(0.0, min(1.0, eta_crc)))

    s_defer = float(r_U + eta_crc * ((1.0 - r_U) ** 2))
    s_defer = float(max(0.0, min(1.0, s_defer)))
    s_accept = float(1.0 - s_defer)
    B_defer = max(0, min(requested_budget, _round_half_up(requested_budget * s_defer)))
    B_accept = int(requested_budget - B_defer)

    return AdaptiveSamplingPlan(
        temperature=float(temperature),
        alpha=None if alpha is None else float(alpha),
        lambda_hat=float(lambda_hat),
        tau_crc=crc_margin_cutoff(lambda_hat, temperature),
        budget=requested_budget,
        r_U=r_U,
        r_C=r_C,
        e_all=e_all,
        e_defer=e_defer,
        c_crc=float(c_crc),
        eta_crc=float(eta_crc),
        s_accept=s_accept,
        s_defer=s_defer,
        B_accept=B_accept,
        B_defer=B_defer,
        pool_accept_count=pool_accept_count,
        pool_defer_count=pool_defer_count,
        calibration_count=n_calibration,
        calibration_defer_count=calibration_defer_count,
        calibration_error_count=calibration_error_count,
        calibration_defer_error_count=calibration_defer_error_count,
    )


def choose_temperature(
    calibration_rows: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    temperatures: Iterable[float] = DEFAULT_TEMPERATURE_GRID,
    lambda_grid: Iterable[float] = DEFAULT_LAMBDA_GRID,
    embeddings_by_id: Mapping[str, np.ndarray] | None = None,
) -> TemperatureChoice:
    """扫描温度，选择 CRC 校准集 accept 最多的温度。

    严格实验默认不应使用这个路径，而应在 CLI 显式传固定温度。
    这里保留给诊断实验：不读取 pool 的 defer 结果参与排序，只把
    pool defer 率作为诊断字段输出。
    排序键依次为：
    1. D_cal 上 CRC 可接受样本数最多；
    2. CRC 风险上界更低；
    3. 温度更小，保证完全确定性。
    """
    candidates: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_choice: TemperatureChoice | None = None
    for temperature in [float(value) for value in temperatures]:
        crc = calibrate_crc(
            calibration_rows,
            alpha=alpha,
            temperature=temperature,
            lambda_grid=lambda_grid,
            embeddings_by_id=embeddings_by_id,
        )
        pool_decisions = apply_crc_decisions(
            pool_rows,
            lambda_hat=crc.lambda_hat,
            temperature=temperature,
            embeddings_by_id=embeddings_by_id,
            support_rows=calibration_rows,
            crc_result=crc,
        )
        summary = summarize_crc_decisions(pool_decisions)
        candidate = {
            "temperature": float(temperature),
            "lambda_hat": float(crc.lambda_hat),
            "pool_defer_rate": float(summary["defer_rate"]),
            "pool_defer_count": int(summary["defer_count"]),
            "risk_bound": float(crc.risk_bound),
            "grid_feasible": bool(crc.grid_feasible),
        }
        candidates.append(candidate)
        key = (-float(crc.accepted_count), float(crc.risk_bound), float(temperature))
        if best_key is None or key < best_key:
            best_key = key
            best_choice = TemperatureChoice(
                temperature=temperature,
                crc=crc,
                pool_defer_rate=float(summary["defer_rate"]),
                pool_defer_count=int(summary["defer_count"]),
                candidates=[],
            )
    if best_choice is None:
        raise ValueError("temperature grid cannot be empty")
    return TemperatureChoice(
        temperature=best_choice.temperature,
        crc=best_choice.crc,
        pool_defer_rate=best_choice.pool_defer_rate,
        pool_defer_count=best_choice.pool_defer_count,
        candidates=candidates,
    )


def split_calibration_pool_ids(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_calibration: int,
    seed: int = 42,
    stratified: bool = False,
) -> tuple[list[str], list[str]]:
    """固定随机划分 D_cal 和 U_pool。

    文档第 7 节要求算法启动前做一次性随机划分，并强制
    S_train 与 D_cal 不相交。这里默认不按标签分层；`stratified`
    仅保留为显式实验扩展，CGSD 主入口会固定传 False。
    """
    if n_calibration <= 0:
        raise ValueError("n_calibration must be positive")
    if n_calibration >= len(rows):
        raise ValueError("n_calibration must be smaller than the dataset size")
    rng = random.Random(seed)
    if not stratified:
        ids = [_row_id(row) for row in rows]
        rng.shuffle(ids)
        return ids[:n_calibration], ids[n_calibration:]

    by_label: dict[int, list[str]] = {0: [], 1: []}
    for row in rows:
        by_label[_row_label(row)].append(_row_id(row))
    for ids in by_label.values():
        rng.shuffle(ids)

    total = sum(len(ids) for ids in by_label.values())
    desired = {
        label: int(round(n_calibration * len(ids) / float(total)))
        for label, ids in by_label.items()
    }
    for label, ids in by_label.items():
        if ids and desired[label] == 0:
            desired[label] = 1
        desired[label] = min(desired[label], len(ids))

    while sum(desired.values()) > n_calibration:
        label = max(desired, key=lambda key: desired[key])
        if desired[label] > 0:
            desired[label] -= 1
    while sum(desired.values()) < n_calibration:
        candidates = [label for label, ids in by_label.items() if desired[label] < len(ids)]
        if not candidates:
            break
        label = max(candidates, key=lambda key: len(by_label[key]) - desired[key])
        desired[label] += 1

    calibration_ids: list[str] = []
    for label in sorted(by_label):
        calibration_ids.extend(by_label[label][: desired[label]])
    calibration_set = set(calibration_ids)
    pool_ids = [_row_id(row) for row in rows if _row_id(row) not in calibration_set]
    rng.shuffle(calibration_ids)
    return calibration_ids, pool_ids


def _normalize_embedding_matrix(matrix: np.ndarray) -> np.ndarray:
    """把 pair embedding 单位化，让欧氏距离和余弦相似度口径稳定。"""
    if matrix.size == 0:
        return matrix.astype(np.float32)
    arr = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms <= 1e-12] = 1.0
    return (arr / norms).astype(np.float32)


def k_center_greedy(
    candidate_ids: Sequence[str],
    embeddings_by_id: Mapping[str, np.ndarray],
    *,
    k: int,
    seed: int = 42,
) -> list[str]:
    """在 embedding 空间内做 k-Center Greedy 多样性选择。

    初始化选择离全体中心最远的点，之后每次加入“到已选集合最近距离最大”的点。
    这样近似最大化被选训练子集对候选 band 的覆盖半径。
    """
    requested = int(max(0, k))
    if requested == 0 or not candidate_ids:
        return []

    ids_with_embeddings = [str(sample_id) for sample_id in candidate_ids if str(sample_id) in embeddings_by_id]
    missing_ids = [str(sample_id) for sample_id in candidate_ids if str(sample_id) not in embeddings_by_id]
    if missing_ids:
        raise CGSDEmbeddingError(
            "k-Center selection requires real precomputed pair embeddings for every candidate; "
            f"missing ids: {missing_ids[:5]}"
        )

    matrix = _normalize_embedding_matrix(
        np.vstack([np.asarray(embeddings_by_id[sample_id], dtype=np.float32) for sample_id in ids_with_embeddings])
    )
    if len(ids_with_embeddings) <= requested:
        return ids_with_embeddings[:requested]

    rng = random.Random(seed)
    # 第一个中心选“离候选整体质心最远”的点，避免随机起点锁在局部密集区。
    center = np.mean(matrix, axis=0, keepdims=True)
    first_distances = np.sum((matrix - center) ** 2, axis=1)
    max_distance = float(np.max(first_distances))
    first_candidates = np.flatnonzero(np.isclose(first_distances, max_distance))
    first_index = int(rng.choice(first_candidates.tolist()))

    selected_indices = [first_index]
    min_distances = np.sum((matrix - matrix[first_index]) ** 2, axis=1)
    min_distances[first_index] = -1.0

    while len(selected_indices) < requested and len(selected_indices) < len(ids_with_embeddings):
        # 维护每个候选点到已选集合的最近距离；每次选最近距离最大的点，
        # 等价于贪心缩小当前最坏覆盖半径。
        next_index = int(np.argmax(min_distances))
        selected_indices.append(next_index)
        candidate_distances = np.sum((matrix - matrix[next_index]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, candidate_distances)
        min_distances[selected_indices] = -1.0

    selected_ids = [ids_with_embeddings[index] for index in selected_indices]
    return selected_ids[:requested]


def _unique_ids_in_order(rows: Sequence[Mapping[str, Any]], *, defer: bool, blocked: set[str]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = _row_id(row)
        if sample_id in seen or sample_id in blocked:
            continue
        if bool(row.get("defer", False)) != bool(defer):
            continue
        seen.add(sample_id)
        ids.append(sample_id)
    return ids


def _select_random_ids(candidate_ids: Sequence[str], *, k: int, seed: int) -> list[str]:
    ids = sorted(str(sample_id) for sample_id in candidate_ids)
    random.Random(seed).shuffle(ids)
    return ids[: int(max(0, k))]


def _select_high_confidence_accept_ids(
    candidate_ids: Sequence[str],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    *,
    k: int,
) -> list[str]:
    return sorted(
        (str(sample_id) for sample_id in candidate_ids),
        key=lambda sample_id: (
            -float(rows_by_id[sample_id].get("routing_score", 0.0) or 0.0),
            sample_id,
        ),
    )[: int(max(0, k))]


def _select_accept_ids(
    candidate_ids: Sequence[str],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    *,
    k: int,
    seed: int,
    strategy: str,
) -> list[str]:
    if int(k) <= 0:
        return []
    if strategy == "random":
        return _select_random_ids(candidate_ids, k=k, seed=seed)
    if strategy == "high-confidence":
        return _select_high_confidence_accept_ids(candidate_ids, rows_by_id, k=k)
    raise ValueError("accept_strategy must be one of {'random', 'high-confidence'}")


def _select_defer_ids(
    candidate_ids: Sequence[str],
    embeddings_by_id: Mapping[str, np.ndarray] | None,
    *,
    k: int,
    seed: int,
    strategy: str,
) -> list[str]:
    if int(k) <= 0:
        return []
    if strategy == "random":
        return _select_random_ids(candidate_ids, k=k, seed=seed)
    if strategy == "k-center":
        if embeddings_by_id is None:
            raise ValueError("embeddings_by_id is required when defer_strategy='k-center'")
        return k_center_greedy(candidate_ids, embeddings_by_id, k=int(k), seed=int(seed))
    raise ValueError("defer_strategy must be one of {'random', 'k-center'}")


def _selection_from_ids(
    *,
    selected_ids: list[str],
    accept_ids: list[str],
    defer_ids: list[str],
    requested_budget: int,
    requested_accept_budget: int,
    requested_defer_budget: int,
    accept_strategy: str,
    defer_strategy: str,
    selection_method: str,
    pool_candidate_count: int,
    accept_candidate_count: int,
    defer_candidate_count: int,
) -> AdaptiveSelection:
    shortfall = len(selected_ids) < int(requested_budget)
    return AdaptiveSelection(
        distillation_ids=list(selected_ids),
        accept_ids=list(accept_ids),
        defer_ids=list(defer_ids),
        requested_budget=int(requested_budget),
        selected_budget=len(selected_ids),
        requested_accept_budget=int(requested_accept_budget),
        requested_defer_budget=int(requested_defer_budget),
        selected_accept_budget=len(accept_ids),
        selected_defer_budget=len(defer_ids),
        accept_strategy=str(accept_strategy),
        defer_strategy=str(defer_strategy),
        selection_method=str(selection_method),
        pool_candidate_count=int(pool_candidate_count),
        accept_candidate_count=int(accept_candidate_count),
        defer_candidate_count=int(defer_candidate_count),
        shortfall=bool(shortfall),
    )


def _allocated_side_counts(
    *,
    requested_accept: int,
    requested_defer: int,
    available_accept: int,
    available_defer: int,
) -> tuple[int, int]:
    accept_count = min(int(requested_accept), int(available_accept))
    defer_count = min(int(requested_defer), int(available_defer))
    return accept_count, defer_count


def select_adaptive_distillation_samples(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    sampling_plan: AdaptiveSamplingPlan,
    already_selected_ids: Iterable[str],
    embeddings_by_id: Mapping[str, np.ndarray] | None = None,
    seed: int = 42,
    accept_strategy: str = "random",
    defer_strategy: str = "random",
) -> AdaptiveSelection:
    """按自适应预算从 accept/defer 两侧选择本轮蒸馏样本。

    accept 侧作为 easy anchor，defer 侧作为 hard / informative samples。
    若某一侧候选不足，只采样该侧全部候选并记录 shortfall，不跨侧补样。
    """
    rows = [dict(row) for row in prediction_rows]
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        rows_by_id.setdefault(_row_id(row), row)
    blocked = {str(sample_id) for sample_id in already_selected_ids}
    accept_candidates = _unique_ids_in_order(rows, defer=False, blocked=blocked)
    defer_candidates = _unique_ids_in_order(rows, defer=True, blocked=blocked)
    accept_count, defer_count = _allocated_side_counts(
        requested_accept=int(sampling_plan.B_accept),
        requested_defer=int(sampling_plan.B_defer),
        available_accept=len(accept_candidates),
        available_defer=len(defer_candidates),
    )
    accept_ids = _select_accept_ids(
        accept_candidates,
        rows_by_id,
        k=accept_count,
        seed=int(seed),
        strategy=str(accept_strategy),
    )
    defer_ids = _select_defer_ids(
        defer_candidates,
        embeddings_by_id,
        k=defer_count,
        seed=int(seed) + 1,
        strategy=str(defer_strategy),
    )
    selected_ids = [*accept_ids, *defer_ids]
    return AdaptiveSelection(
        distillation_ids=selected_ids,
        accept_ids=accept_ids,
        defer_ids=defer_ids,
        requested_budget=int(sampling_plan.budget),
        selected_budget=len(selected_ids),
        requested_accept_budget=int(sampling_plan.B_accept),
        requested_defer_budget=int(sampling_plan.B_defer),
        selected_accept_budget=len(accept_ids),
        selected_defer_budget=len(defer_ids),
        accept_strategy=str(accept_strategy),
        defer_strategy=str(defer_strategy),
        selection_method="adaptive_accept_defer",
        pool_candidate_count=len(rows),
        accept_candidate_count=len(accept_candidates),
        defer_candidate_count=len(defer_candidates),
        shortfall=len(selected_ids) < int(sampling_plan.budget),
    )


def _within_pool(rows: Sequence[Mapping[str, Any]], *, blocked_ids: set[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        sample_id = _row_id(row)
        if sample_id in blocked_ids:
            continue
        output.append(dict(row))
    return output


def select_documented_training_samples(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    budget: int,
    seed: int,
    blocked_ids: Iterable[str] = (),
    sampling_plan: AdaptiveSamplingPlan | None = None,
    accept_strategy: str = "random",
    defer_strategy: str = "random",
    embeddings_by_id: Mapping[str, np.ndarray] | None = None,
) -> AdaptiveSelection:
    """按文档中的五种采样方法从同一个 pool 里取样本。

    方法定义：
    - pool-random: 从整个 pool 随机取样
    - pure-accept: 只从 accept 集随机取样
    - pure-defer: 只从 defer 集随机取样
    - fixed-15-85: 固定 15% / 85% 混合
    - crc-error-mass: 按 CRC Error-Mass 自适应预算取样
    """
    method_name = str(method)
    rows = [dict(row) for row in prediction_rows]
    blocked = {str(sample_id) for sample_id in blocked_ids}
    pool_rows = _within_pool(rows, blocked_ids=blocked)
    rows_by_id = {_row_id(row): row for row in pool_rows}
    pool_ids = [_row_id(row) for row in pool_rows]
    accept_ids = _unique_ids_in_order(pool_rows, defer=False, blocked=set())
    defer_ids = _unique_ids_in_order(pool_rows, defer=True, blocked=set())

    if method_name == "pool-random":
        selected_ids = _select_random_ids(pool_ids, k=int(budget), seed=int(seed))
        selected_accept_ids = [sample_id for sample_id in selected_ids if not bool(rows_by_id[sample_id].get("defer", False))]
        selected_defer_ids = [sample_id for sample_id in selected_ids if bool(rows_by_id[sample_id].get("defer", False))]
        return _selection_from_ids(
            selected_ids=selected_ids,
            accept_ids=selected_accept_ids,
            defer_ids=selected_defer_ids,
            requested_budget=int(budget),
            requested_accept_budget=int(budget),
            requested_defer_budget=0,
            accept_strategy="random",
            defer_strategy="random",
            selection_method=method_name,
            pool_candidate_count=len(pool_ids),
            accept_candidate_count=len(accept_ids),
            defer_candidate_count=len(defer_ids),
        )

    if method_name == "pure-accept":
        selected_accept_ids = _select_random_ids(accept_ids, k=int(budget), seed=int(seed))
        return _selection_from_ids(
            selected_ids=selected_accept_ids,
            accept_ids=selected_accept_ids,
            defer_ids=[],
            requested_budget=int(budget),
            requested_accept_budget=int(budget),
            requested_defer_budget=0,
            accept_strategy="random",
            defer_strategy="random",
            selection_method=method_name,
            pool_candidate_count=len(pool_ids),
            accept_candidate_count=len(accept_ids),
            defer_candidate_count=len(defer_ids),
        )

    if method_name == "pure-defer":
        selected_defer_ids = _select_random_ids(defer_ids, k=int(budget), seed=int(seed))
        return _selection_from_ids(
            selected_ids=selected_defer_ids,
            accept_ids=[],
            defer_ids=selected_defer_ids,
            requested_budget=int(budget),
            requested_accept_budget=0,
            requested_defer_budget=int(budget),
            accept_strategy="random",
            defer_strategy="random",
            selection_method=method_name,
            pool_candidate_count=len(pool_ids),
            accept_candidate_count=len(accept_ids),
            defer_candidate_count=len(defer_ids),
        )

    if method_name == "fixed-15-85":
        accept_budget = int(round(int(budget) * 0.15))
        defer_budget = int(budget) - accept_budget
        selected_accept_ids = _select_random_ids(accept_ids, k=accept_budget, seed=int(seed))
        selected_defer_ids = _select_random_ids(defer_ids, k=defer_budget, seed=int(seed) + 1)
        selected_ids = [*selected_accept_ids, *selected_defer_ids]
        return _selection_from_ids(
            selected_ids=selected_ids,
            accept_ids=selected_accept_ids,
            defer_ids=selected_defer_ids,
            requested_budget=int(budget),
            requested_accept_budget=accept_budget,
            requested_defer_budget=defer_budget,
            accept_strategy="random",
            defer_strategy="random",
            selection_method=method_name,
            pool_candidate_count=len(pool_ids),
            accept_candidate_count=len(accept_ids),
            defer_candidate_count=len(defer_ids),
        )

    if method_name == "crc-error-mass":
        if sampling_plan is None:
            raise ValueError("crc-error-mass requires sampling_plan")
        if int(sampling_plan.budget) != int(budget):
            raise ValueError("sampling_plan budget must match the requested budget")
        accept_count = min(int(sampling_plan.B_accept), len(accept_ids))
        defer_count = min(int(sampling_plan.B_defer), len(defer_ids))
        selected_accept_ids = _select_accept_ids(
            accept_ids,
            rows_by_id,
            k=accept_count,
            seed=int(seed),
            strategy=str(accept_strategy),
        )
        selected_defer_ids = _select_defer_ids(
            defer_ids,
            embeddings_by_id,
            k=defer_count,
            seed=int(seed) + 1,
            strategy=str(defer_strategy),
        )
        selected_ids = [*selected_accept_ids, *selected_defer_ids]
        return _selection_from_ids(
            selected_ids=selected_ids,
            accept_ids=selected_accept_ids,
            defer_ids=selected_defer_ids,
            requested_budget=int(budget),
            requested_accept_budget=int(sampling_plan.B_accept),
            requested_defer_budget=int(sampling_plan.B_defer),
            accept_strategy=str(accept_strategy),
            defer_strategy=str(defer_strategy),
            selection_method=method_name,
            pool_candidate_count=len(pool_ids),
            accept_candidate_count=len(accept_ids),
            defer_candidate_count=len(defer_ids),
        )

    raise ValueError("method must be one of {'pool-random', 'pure-accept', 'pure-defer', 'fixed-15-85', 'crc-error-mass'}")


def select_defer_k_center_samples(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    defer_ids: Iterable[str],
    already_selected_ids: Iterable[str],
    budget: int,
    embeddings_by_id: Mapping[str, np.ndarray],
    seed: int = 42,
) -> DeferSelection:
    """从当前 defer 集中用 k-Center Greedy 选择蒸馏样本。

    这保留 CKD 的核心选择逻辑：CRC 先定义 student 不可靠的 defer 集，
    然后 k-Center 在固定 embedding 空间里覆盖 defer 集的不同区域。
    """
    blocked = {str(sample_id) for sample_id in already_selected_ids}
    defer_set = {str(sample_id) for sample_id in defer_ids} - blocked
    rows_by_id = {_row_id(row): dict(row) for row in prediction_rows}
    candidate_ids = [sample_id for sample_id in defer_set if sample_id in rows_by_id]
    candidate_ids.sort(key=lambda sample_id: (-float(rows_by_id[sample_id].get("routing_score", 0.0) or 0.0), sample_id))
    selected = k_center_greedy(
        candidate_ids,
        embeddings_by_id,
        k=int(budget),
        seed=seed,
    )

    return DeferSelection(
        distillation_ids=selected,
        requested_budget=int(budget),
        selected_budget=len(selected),
    )


def teacher_weight(confidence: float, beta: float) -> float:
    """把 teacher 置信度转成训练样本权重。

    文档中的权重是 (c_T)^beta。数据文件若有 parsed_confidence 或
    teacher_confidence，脚本层会传入这里；groundtruth置信度时按 1.0 处理，
    等价于 teacher 对离线标签完全确信。
    """
    value = max(0.0, min(1.0, float(confidence)))
    return float(value**float(beta)) if float(beta) != 0.0 else 1.0


def build_deployment_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    train_label_by_id: Mapping[str, int],
    lambda_hat: float,
    temperature: float,
    embeddings_by_id: Mapping[str, np.ndarray] | None = None,
    support_rows: Sequence[Mapping[str, Any]] | None = None,
    crc_result: CRCResult | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """构造最终部署决策表。

    已经 teacher 标注并进入训练的样本直接复用 teacher 标签；其余样本按 CRC
    阈值 accept 或 defer。离线数据有 groundtruth 时，defer 行会把它记录为
    teacher_label，方便复现实验成本与精度。
    """
    decided = apply_crc_decisions(
        prediction_rows,
        lambda_hat=lambda_hat,
        temperature=temperature,
        embeddings_by_id=embeddings_by_id,
        support_rows=support_rows,
        crc_result=crc_result,
    )
    output: list[dict[str, Any]] = []
    for row in decided:
        sample_id = _row_id(row)
        item = dict(row)
        # 文档 Phase 3 第一条：训练集中已有 teacher 标签的样本
        # 直接复用 teacher 输出，不再用 student 或 CRC 决策覆盖。
        if sample_id in train_label_by_id:
            item["deployment_source"] = "teacher_train_label"
            item["teacher_required"] = False
            item["output_label"] = _binary_to_int(
                train_label_by_id[sample_id],
                field_name="teacher train label",
            )
        elif bool(row.get("defer", False)):
            # defer 行在真实部署中会调用 teacher；离线实验数据已有标签时，
            # 这里把 label 记录为 teacher_label 以便计算成本和最终输出。
            item["deployment_source"] = "teacher_defer"
            item["teacher_required"] = True
            item["teacher_label"] = _row_label(row) if row.get("label", row.get("groundtruth")) is not None else None
            item["output_label"] = item["teacher_label"]
        else:
            item["deployment_source"] = "student_accept"
            item["teacher_required"] = False
            item["output_label"] = _row_prediction(row)
        output.append(item)
    return output


__all__ = [
    "AdaptiveSamplingPlan",
    "AdaptiveSelection",
    "CRCResult",
    "CGSDEmbeddingError",
    "DeferSelection",
    "DEFAULT_LAMBDA_GRID",
    "DEFAULT_TEMPERATURE_GRID",
    "TemperatureChoice",
    "apply_crc_decisions",
    "attach_routing_scores",
    "build_deployment_rows",
    "calibrate_crc",
    "choose_temperature",
    "compute_adaptive_sampling_plan",
    "crc_margin_cutoff",
    "crc_risk_bound",
    "k_center_greedy",
    "select_adaptive_distillation_samples",
    "select_defer_k_center_samples",
    "select_documented_training_samples",
    "sigmoid_abs_margin",
    "split_calibration_pool_ids",
    "summarize_crc_decisions",
    "teacher_weight",
]
