"""Step2 的 CRC 判定与邻域支持计算。

这一层执行 CRC 接收/延迟策略。
它只使用 Step1 的信号，再结合邻域支持度，决定样本是：

- 直接接受
- 继续 defer 到 Step3
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from .cascade_local import (
    STEP1_ROUTABLE_SIGNAL_QUALITIES,
    Sample,
    normalize_binary_decision_int,
    normalize_binary_decision_text,
    split_sample_id,
    neighbor_support_summary_from_samples,
    numeric_summary,
    signal_quality_distribution_from_samples,
)


def stable_key(value: Any) -> str:
    """把可能来自 numpy / pandas 的 key 归一成稳定字符串。"""
    item = value
    if hasattr(item, "item"):
        try:
            item = item.item()
        except Exception:
            pass
    return str(item or "").strip()


def _binary_text(value: Any) -> str:
    """归一化内部比较用的二元文本标签。"""
    return normalize_binary_decision_text(value)


def _binary_int(value: Any) -> Optional[int]:
    """归一化外部工件用的二元整数标签。"""
    return normalize_binary_decision_int(value)


class NeighborSupportBank:
    """用于估计 Step2 邻域支持度的向量检索库。"""

    def __init__(self, *, embedding_model: str = "", enable_neighbor_support: bool = True):
        """初始化邻域检索库配置，不在构造阶段加载向量。"""
        self.enabled = bool(enable_neighbor_support)
        self.embedding_model = str(embedding_model or "").strip()

        self._embeddings = np.zeros((0, 0), dtype=np.float32)
        self._preds = np.asarray([], dtype=object)
        self._labels = np.asarray([], dtype=object)
        self._sample_index_by_id: Dict[str, int] = {}
        self._last_pair_cache_read: Dict[str, Any] = {}
        self._fitted = False

    def reset(self) -> None:
        """清空已拟合的向量、标签和索引缓存。"""
        # 每次重新拟合前都先清空已有检索库，避免上一轮校准样本泄漏到当前结果。
        self._embeddings = np.zeros((0, 0), dtype=np.float32)
        self._preds = np.asarray([], dtype=object)
        self._labels = np.asarray([], dtype=object)
        self._sample_index_by_id = {}
        self._last_pair_cache_read = {}
        self._fitted = False

    @staticmethod
    def _normalize_embeddings(arr: np.ndarray) -> np.ndarray:
        """把二维向量矩阵按行归一化。"""
        # 保持向量单位化，后续点积才能稳定等价为余弦相似度。
        if arr.size == 0:
            return arr.astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms <= 1e-12] = 1.0
        return (arr / norms).astype(np.float32)

    @staticmethod
    def _sample_cached_pair_vector(sample: Sample) -> Optional[np.ndarray]:
        """读取 prepare 阶段挂到 Sample 上的 pair embedding。"""
        value = getattr(sample, "_pair_embedding", None)
        if not isinstance(value, np.ndarray):
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return None
        return arr

    def _embeddings_from_prepared_samples(self, samples: List[Sample]) -> np.ndarray:
        """从 Sample 列表抽取并归一化已准备好的 pair embedding。"""
        if not samples:
            return np.zeros((0, 0), dtype=np.float32)
        if not self.enabled:
            return np.zeros((0, 0), dtype=np.float32)

        cached_pair_vectors = [
            self._sample_cached_pair_vector(sample)
            for sample in samples
        ]
        if cached_pair_vectors and all(vector is not None for vector in cached_pair_vectors):
            self._last_pair_cache_read = {
                "source": "prepared_artifact_pair_embeddings",
                "sample_count": int(len(samples)),
            }
            return self._normalize_embeddings(
                np.vstack(
                    [
                        np.asarray(vector, dtype=np.float32)
                        for vector in cached_pair_vectors
                        if vector is not None
                    ]
                )
            )

        raise RuntimeError(
            "Step2 neighbor support requires prepared pair embeddings; "
            "missing or invalid prepared embedding rows were found"
        )

    def fit(self, samples: List[Sample]) -> None:
        """用已标注且 Step1 有二元预测的样本构建邻域支持库。"""
        self.reset()
        if not self.enabled:
            return
        # 只有同时具备 yes/no 真值且 Step1 也产出 yes/no 预测的样本，才允许写入检索库。
        bank_samples = [
            sample
            for sample in samples
            if _binary_text(sample.label) in {"yes", "no"}
            and _binary_text(sample.answer_4b) in {"yes", "no"}
        ]
        if not bank_samples:
            return

        embeddings = self._embeddings_from_prepared_samples(bank_samples)
        if embeddings.size == 0:
            return

        self._embeddings = embeddings
        self._preds = np.asarray(
            [_binary_text(sample.answer_4b) for sample in bank_samples],
            dtype=object,
        )
        self._labels = np.asarray(
            [_binary_text(sample.label) for sample in bank_samples],
            dtype=object,
        )
        self._sample_index_by_id = {
            str(sample.sample_id): idx
            for idx, sample in enumerate(bank_samples)
            if str(sample.sample_id or "").strip()
        }
        self._fitted = bool(self._embeddings.shape[0] > 0)

    def _support_from_similarities(self, similarities: np.ndarray, pred: str) -> float:
        """把相似度向量压缩成当前预测标签的局部支持度。"""
        # 这里把相似样本检索结果压缩成邻域支持度 N_i。
        # 具体做法是：对学习库中所有可用样本计算相似度，再保留预测标签与当前样本一致的样本，
        # 最后统计这些样本中“真实标签也等于 pred”的非负相似度权重占比。
        # 因此 N_i 表示的是局部经验可靠性，而不是样本密度、文本长度或全局类别频率。
        if pred not in {"yes", "no"} or similarities.size == 0:
            return 0.0
        finite_mask = np.isfinite(similarities)
        if not np.any(finite_mask):
            return 0.0

        finite_indices = np.flatnonzero(finite_mask)
        same_pred_mask = self._preds[finite_indices] == pred
        if not np.any(same_pred_mask):
            return 0.0
        same_pred_indices = finite_indices[same_pred_mask]
        weights = np.clip(similarities[same_pred_indices], a_min=0.0, a_max=None).astype(np.float64)
        total_weight = float(np.sum(weights))
        if total_weight <= 1e-12:
            return 0.0
        # 只统计真实标签也与当前预测一致的权重，得到当前预测在局部邻域中的支持比例。
        support_weight = float(np.sum(weights[self._labels[same_pred_indices] == pred]))
        return float(max(0.0, min(1.0, support_weight / total_weight)))

    def attach(self, samples: List[Sample], *, exclude_self: bool = False) -> None:
        """为样本写入 `step1_neighbor_support`。"""
        # 先把已有 support 清零，避免复用 Sample 对象时残留过期特征。
        for sample in samples:
            sample.step1_neighbor_support = 0.0

        if not self.enabled or not samples or not self._fitted:
            return

        target_samples = [
            sample
            for sample in samples
            if _binary_text(sample.answer_4b) in {"yes", "no"}
        ]
        if not target_samples:
            return

        query_embeddings = self._embeddings_from_prepared_samples(target_samples)
        if query_embeddings.size == 0:
            return

        # 向量已经单位化，因此这里的点积可以直接视为余弦相似度矩阵。
        sim_matrix = np.matmul(query_embeddings, self._embeddings.T)
        if exclude_self:
            # 校准阶段必须排除自身邻居，否则样本会把自己当成最强支持证据。
            for row_idx, sample in enumerate(target_samples):
                sample_id = str(sample.sample_id or "").strip()
                bank_idx = self._sample_index_by_id.get(sample_id)
                if bank_idx is not None and bank_idx < sim_matrix.shape[1]:
                    sim_matrix[row_idx, bank_idx] = -np.inf

        for row_idx, sample in enumerate(target_samples):
            pred = _binary_text(sample.answer_4b)
            sample.step1_neighbor_support = self._support_from_similarities(sim_matrix[row_idx], pred)

    def snapshot(self) -> Dict[str, Any]:
        """返回邻域检索库的轻量诊断快照。"""
        # 只暴露检索库的摘要状态，不返回原始向量，避免快照过大且泄漏内部实现。
        return {
            "enabled": bool(self.enabled),
            "is_fitted": bool(self._fitted),
            "embedding_model": str(self.embedding_model),
            "embedding_source": str(
                self._last_pair_cache_read.get("source", "prepared_artifact_pair_embeddings")
                or "prepared_artifact_pair_embeddings"
            ),
            "last_pair_cache_read": dict(self._last_pair_cache_read),
            "bank_size": int(self._embeddings.shape[0] if self._embeddings.ndim == 2 else 0),
        }


class Step2CRC:
    """学习并执行 Step2 的 CRC 接收/延迟规则。"""

    _WRONG_ACCEPT_LOSS_BOUND_B = 1.0
    _DECISION_TOL = 1e-12
    _NEIGHBOR_SUPPORT_EPS = 1e-6
    _QUERY_REFERENCE_SHRINKAGE_K = 12.0
    _THRESHOLD_MULTIPLIER_GAMMA = 0.5
    # 论文 / 规格基线：
    #   S_i(lambda) = clip(lambda * b / (N_i + eps), 0, 1)
    # 其中 b 是单个全局参考支持度。
    #
    # 当前默认实现保留同样的 "lambda * f(N_i)" 结构，
    # 但把纯全局倒数替换成按查询自适应的温和乘子：
    #   f(N_i) = ((b_q + eps) / (N_i + eps)) ** gamma
    # 其中 b_q 是向全局参考点收缩后的查询级参考支持度；
    # gamma 用来压低倒数项的敏感度，减少阈值抖动。
    #
    def __init__(
        self,
        *,
        risk_target: float,
        embedding_model: str = "",
        enable_neighbor_support: bool = True,
    ):
        """初始化 CRC 风险目标和邻域支持组件。"""
        self.mode = "global_crc_step2_with_neighbor_support_adaptive_threshold"
        self.lambda_search_method = "direct_threshold_comparison_over_lambda_candidates"
        self.threshold_function = "clip(lambda*(((b_q+eps)/(neighbor_support+eps))**gamma),0,1)"
        self.calibration_method = "conformal_risk_control_wrong_accept_risk"
        self.risk_target = float(risk_target)
        if not (0.0 <= self.risk_target <= 1.0):
            raise RuntimeError("risk_target must be within [0, 1]")

        self._calibrated = False
        self._threshold_state: Dict[str, Any] | None = None
        self._support_bank = NeighborSupportBank(
            embedding_model=str(embedding_model or ""),
            enable_neighbor_support=bool(enable_neighbor_support),
        )
        self._signal_quality_distribution: Dict[str, int] = {}
        self._neighbor_support_summary: Dict[str, float] = numeric_summary([])
        self._lambda_transition_summary: Dict[str, float] = numeric_summary([])
        self._neighbor_support_reference: Optional[float] = None

    @staticmethod
    def _clip_unit(value: Any) -> float:
        """把数值裁剪到 [0, 1]，非法值按 0 处理。"""
        value = float(value or 0.0)
        if np.isnan(value) or value <= 0.0:
            return 0.0
        if value >= 1.0:
            return 1.0
        return value

    @classmethod
    def _neighbor_support_reference_from_samples(cls, samples: List[Sample]) -> float:
        """从校准样本估计全局邻域支持参考点。"""
        # 这里从校准样本的 N_i 分布中估计参考点 b。
        # 当前实现取中位数，是为了减弱极端 support 对阈值映射的影响。
        values = [
            cls._clip_unit(sample.step1_neighbor_support or 0.0)
            for sample in samples
            if _binary_text(sample.answer_4b) in {"yes", "no"}
        ]
        if not values:
            return float(cls._NEIGHBOR_SUPPORT_EPS)
        return float(
            max(
                cls._NEIGHBOR_SUPPORT_EPS,
                min(1.0, float(np.median(np.asarray(values, dtype=np.float64)))),
            )
        )

    @staticmethod
    def _sample_query_id(sample: Sample) -> str:
        """从 sample_id 中提取 query_id。"""
        try:
            _, query_id = split_sample_id(sample.sample_id)
        except Exception:
            return ""
        return stable_key(query_id)

    @classmethod
    def _query_reference_supports_from_rows(
        cls,
        rows: List[Dict[str, Any]],
        *,
        global_reference_support: float,
    ) -> Dict[str, float]:
        """按 query 估计收缩后的邻域支持参考点。"""
        global_ref = max(cls._NEIGHBOR_SUPPORT_EPS, cls._clip_unit(global_reference_support))
        grouped: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            query_id = stable_key(row.get("query_id"))
            if not query_id:
                continue
            grouped[query_id].append(cls._clip_unit(float(row.get("neighbor_support", 0.0) or 0.0)))

        shrinkage_k = max(0.0, float(cls._QUERY_REFERENCE_SHRINKAGE_K))
        references: Dict[str, float] = {}
        for query_id, values in grouped.items():
            if not values:
                continue
            local_ref = max(
                cls._NEIGHBOR_SUPPORT_EPS,
                min(1.0, float(np.median(np.asarray(values, dtype=np.float64)))),
            )
            weight = 1.0 if shrinkage_k <= 0.0 else float(len(values) / (len(values) + shrinkage_k))
            references[query_id] = float(
                max(
                    cls._NEIGHBOR_SUPPORT_EPS,
                    min(1.0, (weight * local_ref) + ((1.0 - weight) * global_ref)),
                )
            )
        return references

    @classmethod
    def _attach_query_reference_supports(
        cls,
        rows: List[Dict[str, Any]],
        *,
        global_reference_support: float,
        query_reference_supports: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """把 query 级参考支持度和 lambda 转折点写回校准行。"""
        references = (
            dict(query_reference_supports)
            if query_reference_supports is not None
            else cls._query_reference_supports_from_rows(
                rows,
                global_reference_support=global_reference_support,
            )
        )
        fallback = max(cls._NEIGHBOR_SUPPORT_EPS, cls._clip_unit(global_reference_support))
        for row in rows:
            query_id = stable_key(row.get("query_id"))
            reference_support = float(references.get(query_id, fallback))
            row["query_reference_support"] = reference_support
            # 诊断字段也必须使用和校准/线上判定相同的 b_q；
            # 否则转折摘要描述的是另一套阈值族。
            row["lambda_transition"] = cls._lambda_transition_from_values(
                float(row.get("routing_score", 0.0) or 0.0),
                float(row.get("neighbor_support", 0.0) or 0.0),
                reference_support=reference_support,
            )
        return references

    @classmethod
    def _threshold_multiplier_from_values(
        cls,
        neighbor_support: float,
        *,
        reference_support: float,
    ) -> float:
        """计算邻域支持对样本级阈值的乘子。"""
        # 保持 CRC 的 `lambda * f(N_i)` 结构不变，只把全局纯倒数替换成
        # 基于查询级收缩参考点的温和倒数：
        #   f(N_i) = ((b_q + eps) / (N_i + eps)) ** gamma
        # 其中 b_q 是查询级支持度中位数向全局参考点收缩后的结果。（未优化的主要地方）
        support = cls._clip_unit(neighbor_support)
        base = max(cls._NEIGHBOR_SUPPORT_EPS, cls._clip_unit(reference_support))
        gamma = max(cls._NEIGHBOR_SUPPORT_EPS, float(cls._THRESHOLD_MULTIPLIER_GAMMA))
        ratio = (base + cls._NEIGHBOR_SUPPORT_EPS) / (support + cls._NEIGHBOR_SUPPORT_EPS)
        return float(max(cls._NEIGHBOR_SUPPORT_EPS, ratio) ** gamma)

    @classmethod
    def _adaptive_threshold_from_values(
        cls,
        lambda_hat: float,
        neighbor_support: float,
        *,
        reference_support: float,
    ) -> float:
        """把全局 lambda 映射成单样本 Step2 接收阈值。"""
        # 这里把全局基准阈值 lambda_hat 映射成样本级门槛 S_i。
        # 因为 S_i 直接由 lambda_hat 和 f(N_i) 共同决定，
        # 所以上面 f(N_i) 的任何调整都会同步影响：
        # 1. 校准阶段学出来的 lambda_hat；
        # 2. 线上每个样本的最终 accept/defer 比较门槛。
        # 最终结果会 clip 到 [0, 1]，保证门槛始终是合法概率阈值。
        base_lambda = cls._clip_unit(lambda_hat)
        multiplier = cls._threshold_multiplier_from_values(
            neighbor_support,
            reference_support=reference_support,
        )
        return cls._clip_unit(base_lambda * multiplier)

    @classmethod
    def _state_reference_support(
        cls,
        threshold_state: Dict[str, Any],
        *,
        query_id: str = "",
    ) -> float:
        """从阈值状态中读取当前 query 的参考支持度。"""
        query_refs = threshold_state.get("query_reference_supports", {})
        if isinstance(query_refs, dict):
            query_ref = query_refs.get(stable_key(query_id))
            if query_ref is not None:
                return max(cls._NEIGHBOR_SUPPORT_EPS, cls._clip_unit(query_ref))
        return max(
            cls._NEIGHBOR_SUPPORT_EPS,
            cls._clip_unit(threshold_state.get("neighbor_support_reference", cls._NEIGHBOR_SUPPORT_EPS)),
        )

    @classmethod
    def _lambda_transition_from_values(
        cls,
        routing_score: float,
        neighbor_support: float,
        *,
        reference_support: float,
    ) -> float:
        """计算样本在 lambda 搜索空间中的 accept/defer 转折点。"""
        # 这里把方程 R_i = T_lambda(N_i) 反解成候选 lambda 转折点。
        # 这些转折点只用于构造有限候选集；真正搜索时仍逐个 lambda 显式计算
        # `R_i >= S_i`，而不是先把 `R_i` 和 `N_i` 合成为单一接受分。
        # 对固定样本而言，这个转折点就是它从 accept 切换到 defer 的临界位置。
        score = cls._clip_unit(routing_score)
        if score >= 1.0 - cls._DECISION_TOL:
            return 1.0
        multiplier = cls._threshold_multiplier_from_values(
            neighbor_support,
            reference_support=reference_support,
        )
        if multiplier <= cls._DECISION_TOL:
            return 1.0
        return cls._clip_unit(score / multiplier)

    @classmethod
    def _lambda_candidates_from_rows(
        cls,
        rows: List[Dict[str, Any]],
        *,
        reference_support: float,
    ) -> np.ndarray:
        """从校准行构造有限 lambda 候选集合。"""
        candidate_values = [0.0, 1.0]
        for row in rows:
            candidate_values.append(
                cls._lambda_transition_from_values(
                    float(row.get("routing_score", 0.0) or 0.0),
                    float(row.get("neighbor_support", 0.0) or 0.0),
                    reference_support=float(row.get("query_reference_support", reference_support) or reference_support),
                )
            )
        return np.unique(np.asarray(candidate_values, dtype=np.float64))

    def _decision_threshold_for_state(
        self,
        neighbor_support: float,
        threshold_state: Dict[str, Any],
        *,
        query_id: str = "",
    ) -> Optional[float]:
        """基于已学习状态计算当前样本的 Step2 阈值。"""
        lambda_hat = threshold_state.get("lambda_hat")
        if lambda_hat is None:
            return None
        # 这里显式构造样本级门槛 S_i = T_{lambda_hat}(N_i)。
        # S_i 只依赖全局 lambda_hat、当前样本的 N_i 和查询级参考点 b_q，
        # 不直接读取 R_i，这样“阈值映射”和“最终比较”两步保持分离。
        return self._adaptive_threshold_from_values(
            float(lambda_hat),
            float(neighbor_support or 0.0),
            reference_support=self._state_reference_support(threshold_state, query_id=query_id),
        )

    def _accepts_with_state(
        self,
        routing_score: float,
        neighbor_support: float,
        threshold_state: Dict[str, Any],
        *,
        query_id: str = "",
    ) -> tuple[bool, Optional[float]]:
        """执行 `routing_score >= adaptive_threshold` 的最终比较。"""
        # 这里执行最终的 accept/defer 判断：accept 当且仅当 R_i >= S_i。
        # _DECISION_TOL 用于吸收浮点临界误差，保持算法本身的方向。
        decision_threshold = self._decision_threshold_for_state(
            neighbor_support,
            threshold_state,
            query_id=query_id,
        )
        if decision_threshold is None:
            return False, None
        accepted = self._clip_unit(routing_score) >= float(decision_threshold) - self._DECISION_TOL
        return bool(accepted), float(decision_threshold)

    @classmethod
    def _wrong_accept_bound(cls, empirical_risk: Any, n: int) -> np.ndarray:
        """计算 CRC 有限样本 wrong-accept 风险上界。"""
        # 对取值范围在 [0, B] 的损失应用 CRC 的有限样本风险修正。
        # 当前损失定义为 1{accept and wrong}，因此这里的 B 固定为 1。
        risk_arr = np.asarray(empirical_risk, dtype=np.float64)
        if int(n) <= 0:
            return np.full_like(risk_arr, np.inf, dtype=np.float64)
        bound_B = float(cls._WRONG_ACCEPT_LOSS_BOUND_B)
        n_float = float(n)
        return (n_float / (n_float + 1.0)) * risk_arr + (bound_B / (n_float + 1.0))

    def annotate_neighbor_support(self, samples: List[Sample], *, exclude_self: bool = False) -> None:
        """为样本批量标注邻域支持度。"""
        # 对外只暴露 Step2 接口，不让调用方直接操作内部检索库。
        self._support_bank.attach(samples, exclude_self=exclude_self)

    @staticmethod
    def _is_step2_eligible_sample(sample: Sample) -> bool:
        """判断样本是否可进入 Step2 校准集合。"""
        label = _binary_text(sample.label)
        pred = _binary_text(sample.answer_4b)
        signal_quality = str(sample.step1_signal_quality or "").strip()
        return (
            label in {"yes", "no"}
            and pred in {"yes", "no"}
            and signal_quality in STEP1_ROUTABLE_SIGNAL_QUALITIES
        )

    def _collect_step2_eligible_samples(self, samples: List[Sample]) -> List[Sample]:
        """筛选可用于 Step2 CRC 校准的样本。"""
        # Step2 只使用线上真正可能进入当前阶段判定的已标注样本来做校准。
        return [sample for sample in samples if self._is_step2_eligible_sample(sample)]

    def _build_step2_rows(
        self,
        samples: List[Sample],
        *,
        reference_support: float,
    ) -> List[Dict[str, Any]]:
        """把校准样本转成 CRC 学习需要的紧凑行。"""
        rows: List[Dict[str, Any]] = []
        for sample in samples:
            pred = _binary_text(sample.answer_4b)
            pred_label = _binary_int(pred)
            signal_quality = str(sample.step1_signal_quality or "").strip()
            query_id = self._sample_query_id(sample)
            rows.append(
                {
                    "query_id": query_id,
                    "pred": pred_label,
                    "signal_quality": signal_quality,
                    "routing_score": self._clip_unit(float(sample.step1_routing_score or 0.0)),
                    "neighbor_support": self._clip_unit(float(sample.step1_neighbor_support or 0.0)),
                    "lambda_transition": self._lambda_transition_from_values(
                        float(sample.step1_routing_score or 0.0),
                        float(sample.step1_neighbor_support or 0.0),
                        reference_support=reference_support,
                    ),
                    "wrong": bool(pred_label != _binary_int(sample.label)),
                }
            )
        return rows

    def _disabled_threshold_state(
        self,
        *,
        reason: str,
        reference_support: float,
        fit_support: int = 0,
    ) -> Dict[str, Any]:
        """构造不可用阈值状态，并保留禁用原因。"""
        return {
            "fit_support": int(fit_support),
            "accepted_count": 0,
            "accepted_rate": 0.0,
            "wrong_accept_count": 0,
            "empirical_risk_at_lambda": None,
            "wrong_accept_bound_at_lambda": None,
            "lambda_hat": None,
            "risk_target": float(self.risk_target),
            "enabled": False,
            "disable_reason": str(reason),
            "method": str(self.calibration_method),
            "lambda_search_method": str(self.lambda_search_method),
            "threshold_function": str(self.threshold_function),
            "neighbor_support_reference": float(reference_support),
            "neighbor_support_eps": float(self._NEIGHBOR_SUPPORT_EPS),
            "query_reference_supports": {},
            "query_reference_shrinkage_k": float(self._QUERY_REFERENCE_SHRINKAGE_K),
            "threshold_multiplier_gamma": float(self._THRESHOLD_MULTIPLIER_GAMMA),
        }

    def _learn_threshold_state(
        self,
        *,
        rows: List[Dict[str, Any]],
        reference_support: float,
    ) -> Dict[str, Any]:
        """枚举 lambda 候选并学习满足风险约束的阈值状态。"""
        # 这里是 CRC 校准的核心：直接枚举候选 lambda，并对每个 lambda
        # 显式构造全部样本的样本级阈值 S_i，再按 `R_i >= S_i` 计算接受集合。
        # 随后在该接受集合上统计 wrong_accept_count，得到经验风险与 CRC 风险上界。
        # 当前实现保留论文里的“阈值映射 + 最终比较”两步结构，
        # 不把 `R_i` 与 `N_i` 预先折叠成单一接受分。
        fit_support = int(len(rows))
        if not rows:
            return self._disabled_threshold_state(
                reason="crc_no_labeled_rows",
                reference_support=reference_support,
                fit_support=0,
            )

        routing_scores = np.asarray(
            [float(row.get("routing_score", 0.0) or 0.0) for row in rows],
            dtype=np.float64,
        )
        # chunk 模式下，R_i 已经在 Step1 由 --step1-chunk-score-temperature
        # 将 paired Y/N logprob gap 映射成 routing_score；Step2 只消费落盘后的 R_i。
        neighbor_supports = np.asarray(
            [float(row.get("neighbor_support", 0.0) or 0.0) for row in rows],
            dtype=np.float64,
        )
        wrong_indicators = np.asarray(
            [1 if bool(row.get("wrong", False)) else 0 for row in rows],
            dtype=np.int64,
        )

        multipliers = np.asarray(
            [
                self._threshold_multiplier_from_values(
                    float(neighbor_support),
                    reference_support=float(row.get("query_reference_support", reference_support) or reference_support),
                )
                for neighbor_support, row in zip(neighbor_supports, rows, strict=True)
            ],
            dtype=np.float64,
        )
        # 这里显式把 f(N_i) 作用到 CRC 学习过程里。
        # f(N_i) 直接参与 accept 集合的形成，因此会影响最终学出的 lambda_hat。
        lambda_candidates = self._lambda_candidates_from_rows(
            rows,
            reference_support=reference_support,
        )
        accepted_counts = np.zeros(lambda_candidates.shape[0], dtype=np.int64)
        wrong_accept_counts = np.zeros(lambda_candidates.shape[0], dtype=np.int64)
        for lambda_idx, lambda_candidate in enumerate(lambda_candidates):
            decision_thresholds = np.clip(float(lambda_candidate) * multipliers, 0.0, 1.0)
            accept_mask = routing_scores >= (decision_thresholds - self._DECISION_TOL)
            accepted_counts[lambda_idx] = int(np.sum(accept_mask, dtype=np.int64))
            wrong_accept_counts[lambda_idx] = int(np.sum(wrong_indicators[accept_mask], dtype=np.int64))
        # 当前风险口径固定为 mean(1{accept and wrong})，因此分母始终是整个 fit_support。
        empirical_risks = wrong_accept_counts.astype(np.float64) / float(fit_support)
        risk_bounds = self._wrong_accept_bound(empirical_risks, fit_support)
        feasible_mask = (
            (accepted_counts > 0)
            & np.isfinite(risk_bounds)
            & (risk_bounds <= float(self.risk_target) + self._DECISION_TOL)
        )

        if not np.any(feasible_mask):
            return self._disabled_threshold_state(
                reason="crc_risk_target_not_met",
                reference_support=reference_support,
                fit_support=fit_support,
            )

        # lambda 候选按升序排列。
        # 在当前阈值函数下，lambda 越小，整体门槛越低，accept 集越大。
        # 因此第一个满足风险约束的 lambda，就是“最宽松且仍合法”的全局阈值。
        selected_idx = int(np.flatnonzero(feasible_mask)[0])
        accepted_count = int(accepted_counts[selected_idx])
        wrong_accept_count = int(wrong_accept_counts[selected_idx])
        empirical_risk = float(empirical_risks[selected_idx])
        risk_bound = float(risk_bounds[selected_idx])
        lambda_hat = float(lambda_candidates[selected_idx])

        return {
            "fit_support": int(fit_support),
            "accepted_count": int(accepted_count),
            "accepted_rate": float(accepted_count / fit_support) if fit_support else 0.0,
            "wrong_accept_count": int(wrong_accept_count),
            "empirical_risk_at_lambda": float(empirical_risk),
            "wrong_accept_bound_at_lambda": float(risk_bound),
            "lambda_hat": float(lambda_hat),
            "risk_target": float(self.risk_target),
            "enabled": True,
            "disable_reason": "",
            "method": str(self.calibration_method),
            "lambda_search_method": str(self.lambda_search_method),
            "threshold_function": str(self.threshold_function),
            "neighbor_support_reference": float(reference_support),
            "neighbor_support_eps": float(self._NEIGHBOR_SUPPORT_EPS),
            "query_reference_supports": {},
            "query_reference_shrinkage_k": float(self._QUERY_REFERENCE_SHRINKAGE_K),
            "threshold_multiplier_gamma": float(self._THRESHOLD_MULTIPLIER_GAMMA),
        }

    def calibrate(self, samples: List[Sample]) -> None:
        """用已标注样本完成 Step2 CRC 校准。"""
        # 每次重新校准都从干净状态开始，避免上一轮检索库和阈值状态残留。
        self._threshold_state = None
        self._calibrated = False
        self._signal_quality_distribution = {}
        self._neighbor_support_summary = numeric_summary([])
        self._lambda_transition_summary = numeric_summary([])
        self._neighbor_support_reference = None

        eligible_samples = self._collect_step2_eligible_samples(samples)
        if not eligible_samples:
            self._support_bank.reset()
            self._threshold_state = None
            return

        # 这里开始构建最终上线使用的校准结果，而不是继续沿用留出集临时产物。
        self._support_bank.fit(eligible_samples)
        self.annotate_neighbor_support(eligible_samples, exclude_self=True)
        self._neighbor_support_reference = self._neighbor_support_reference_from_samples(eligible_samples)
        step2_rows = self._build_step2_rows(
            eligible_samples,
            reference_support=float(self._neighbor_support_reference),
        )
        query_reference_supports = self._attach_query_reference_supports(
            step2_rows,
            global_reference_support=float(self._neighbor_support_reference),
        )
        self._signal_quality_distribution = signal_quality_distribution_from_samples(
            eligible_samples,
            labeled_only=True,
        )
        self._neighbor_support_summary = neighbor_support_summary_from_samples(
            eligible_samples,
            labeled_only=True,
        )
        self._lambda_transition_summary = numeric_summary(
            [float(row.get("lambda_transition", 0.0) or 0.0) for row in step2_rows]
        )
        self._threshold_state = self._learn_threshold_state(
            rows=step2_rows,
            reference_support=float(self._neighbor_support_reference),
        )
        self._threshold_state["query_reference_supports"] = dict(query_reference_supports)
        self._calibrated = bool(self._threshold_state.get("enabled", False))

    def _normalized_threshold_state(self) -> Dict[str, Any] | None:
        """返回 JSON 友好的阈值状态副本。"""
        state = self._threshold_state
        if not isinstance(state, dict):
            return None
        return {
            "fit_support": int(state.get("fit_support", 0) or 0),
            "accepted_count": int(state.get("accepted_count", 0) or 0),
            "accepted_rate": float(state.get("accepted_rate", 0.0) or 0.0),
            "wrong_accept_count": int(state.get("wrong_accept_count", 0) or 0),
            "empirical_risk_at_lambda": state.get("empirical_risk_at_lambda"),
            "wrong_accept_bound_at_lambda": state.get("wrong_accept_bound_at_lambda"),
            "lambda_hat": (
                None
                if state.get("lambda_hat") is None
                else float(state.get("lambda_hat", 0.0) or 0.0)
            ),
            "neighbor_support_reference": float(
                state.get(
                    "neighbor_support_reference",
                    self._neighbor_support_reference or self._NEIGHBOR_SUPPORT_EPS,
                )
                or self._NEIGHBOR_SUPPORT_EPS
            ),
            "neighbor_support_eps": float(
                state.get("neighbor_support_eps", self._NEIGHBOR_SUPPORT_EPS)
                or self._NEIGHBOR_SUPPORT_EPS
            ),
            "risk_target": float(state.get("risk_target", self.risk_target) or self.risk_target),
            "enabled": bool(state.get("enabled", False)),
            "disable_reason": str(state.get("disable_reason", "")),
            "method": str(state.get("method", self.calibration_method)),
            "lambda_search_method": str(
                state.get("lambda_search_method", self.lambda_search_method)
            ),
            "threshold_function": str(state.get("threshold_function", self.threshold_function)),
            "query_reference_supports": dict(state.get("query_reference_supports", {}) or {}),
            "query_reference_shrinkage_k": float(
                state.get("query_reference_shrinkage_k", self._QUERY_REFERENCE_SHRINKAGE_K)
                or self._QUERY_REFERENCE_SHRINKAGE_K
            ),
            "threshold_multiplier_gamma": float(
                state.get("threshold_multiplier_gamma", self._THRESHOLD_MULTIPLIER_GAMMA)
                or self._THRESHOLD_MULTIPLIER_GAMMA
            ),
        }

    def snapshot(self) -> Dict[str, Any]:
        """返回 Step2 CRC 的完整诊断快照。"""
        threshold_state = self._normalized_threshold_state()
        summary = {
            "enabled": bool((threshold_state or {}).get("enabled", False)),
            "neighbor_support_reference": self._neighbor_support_reference,
            "signal_quality_distribution": dict(self._signal_quality_distribution),
            "neighbor_support_summary": dict(self._neighbor_support_summary),
            "lambda_transition_summary": dict(self._lambda_transition_summary),
        }
        return {
            "is_calibrated": bool(self._calibrated),
            "mode": str(self.mode),
            "lambda_search_method": str(self.lambda_search_method),
            "threshold_function": str(self.threshold_function),
            "calibration_method": str(self.calibration_method),
            "risk_target": float(self.risk_target),
            "neighbor_support_reference": self._neighbor_support_reference,
            "neighbor_support_eps": float(self._NEIGHBOR_SUPPORT_EPS),
            "query_reference_shrinkage_k": float(self._QUERY_REFERENCE_SHRINKAGE_K),
            "threshold_multiplier_gamma": float(self._THRESHOLD_MULTIPLIER_GAMMA),
            "neighbor_support_bank": self._support_bank.snapshot(),
            "signal_quality_distribution": dict(self._signal_quality_distribution),
            "neighbor_support_summary": dict(self._neighbor_support_summary),
            "lambda_transition_summary": dict(self._lambda_transition_summary),
            "threshold_state": threshold_state,
            "summary": summary,
        }

    def _eligibility_status(self, sample: Sample) -> Dict[str, Any]:
        """检查样本是否能进入 Step2 算法决策。"""
        raw_pred = sample.prediction
        if raw_pred is None or str(raw_pred).strip() == "":
            raw_pred = sample.answer_4b
        pred = _binary_text(raw_pred)
        pred_label = _binary_int(pred)
        if pred not in {"yes", "no"}:
            return {
                "eligible": False,
                "reason": "invalid_step1_prediction",
                "pred": None,
                "signal_quality": str(sample.step1_signal_quality or "").strip(),
            }

        signal_quality = str(sample.step1_signal_quality or "").strip()
        if signal_quality not in STEP1_ROUTABLE_SIGNAL_QUALITIES:
            return {
                "eligible": False,
                "reason": "unsupported_signal_quality",
                "pred": pred_label,
                "signal_quality": signal_quality,
            }
        # 这里做的是 Step2 的入口资格检查；核心比较仍是 `R_i >= S_i`。
        return {
            "eligible": True,
            "reason": "eligible",
            "pred": pred_label,
            "pred_text": pred,
            "signal_quality": signal_quality,
        }

    def _algorithmic_decision(
        self,
        sample: Sample,
        *,
        pred: str,
        signal_quality: str,
        threshold_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """基于已校准状态生成 accept/defer 决策明细。"""
        query_id = self._sample_query_id(sample)
        lambda_hat = threshold_state.get("lambda_hat")
        pred_label = _binary_int(pred)
        if lambda_hat is None:
            return {
                "decision": "defer",
                "reason": "lambda_unavailable",
                "decision_source": "step2_state",
                "step2_eligible": True,
                "eligibility_reason": "eligible",
                "pred": pred_label,
                "signal_quality": signal_quality,
                "routing_score": float(self._clip_unit(sample.step1_routing_score or 0.0)),
                "neighbor_support": float(self._clip_unit(sample.step1_neighbor_support or 0.0)),
                "decision_threshold": None,
                "lambda_hat": None,
                "threshold_state": threshold_state,
            }

        accepted, decision_threshold = self._accepts_with_state(
            float(sample.step1_routing_score or 0.0),
            float(sample.step1_neighbor_support or 0.0),
            threshold_state,
            query_id=query_id,
        )
        decision = "accept" if accepted else "defer"
        # 从这里开始才进入论文中的核心规则，最终主体比较就是 `R_i >= S_i`。
        return {
            "decision": decision,
            "reason": "accepted" if decision == "accept" else "routing_score_below_adaptive_threshold",
            "decision_source": "algorithmic_rule",
            "step2_eligible": True,
            "eligibility_reason": "eligible",
            "pred": pred_label,
            "signal_quality": signal_quality,
            "routing_score": float(self._clip_unit(sample.step1_routing_score or 0.0)),
            "neighbor_support": float(self._clip_unit(sample.step1_neighbor_support or 0.0)),
            "decision_threshold": decision_threshold,
            "lambda_hat": float(lambda_hat),
            "threshold_state": threshold_state,
        }

    def decide(self, sample: Sample) -> Dict[str, Any]:
        """对单个样本执行 Step2 判定，返回可写入工件的路由结果。"""
        eligibility = self._eligibility_status(sample)
        routing_score = float(self._clip_unit(sample.step1_routing_score or 0.0))
        neighbor_support = float(self._clip_unit(sample.step1_neighbor_support or 0.0))
        if not bool(eligibility.get("eligible", False)):
            return {
                "decision": "defer",
                "reason": str(eligibility.get("reason", "step2_ineligible")),
                "decision_source": "eligibility_gate",
                "step2_eligible": False,
                "eligibility_reason": str(eligibility.get("reason", "step2_ineligible")),
                "pred": eligibility.get("pred"),
                "signal_quality": str(eligibility.get("signal_quality", "") or ""),
                "routing_score": routing_score,
                "neighbor_support": neighbor_support,
                "decision_threshold": None,
                "lambda_hat": None,
                "threshold_state": None,
            }

        if not self._calibrated:
            return {
                "decision": "defer",
                "reason": "calibration_required",
                "decision_source": "step2_state",
                "step2_eligible": True,
                "eligibility_reason": "eligible",
                "pred": eligibility.get("pred"),
                "signal_quality": str(eligibility.get("signal_quality", "") or ""),
                "routing_score": routing_score,
                "neighbor_support": neighbor_support,
                "decision_threshold": None,
                "lambda_hat": None,
                "threshold_state": None,
            }

        threshold_state = self._threshold_state
        if threshold_state is None:
            return {
                "decision": "defer",
                "reason": "threshold_state_missing",
                "decision_source": "step2_state",
                "step2_eligible": True,
                "eligibility_reason": "eligible",
                "pred": eligibility.get("pred"),
                "signal_quality": str(eligibility.get("signal_quality", "") or ""),
                "routing_score": routing_score,
                "neighbor_support": neighbor_support,
                "decision_threshold": None,
                "lambda_hat": None,
                "threshold_state": None,
            }

        if not bool(threshold_state.get("enabled", False)):
            return {
                "decision": "defer",
                "reason": "threshold_state_disabled",
                "decision_source": "step2_state",
                "step2_eligible": True,
                "eligibility_reason": "eligible",
                "pred": eligibility.get("pred"),
                "signal_quality": str(eligibility.get("signal_quality", "") or ""),
                "routing_score": routing_score,
                "neighbor_support": neighbor_support,
                "disable_reason": str(threshold_state.get("disable_reason", "")),
                "decision_threshold": None,
                "lambda_hat": None,
                "threshold_state": threshold_state,
            }

        return self._algorithmic_decision(
            sample,
            pred=str(eligibility.get("pred_text", "") or ""),
            signal_quality=str(eligibility.get("signal_quality", "") or ""),
            threshold_state=threshold_state,
        )


__all__ = [
    "NeighborSupportBank",
    "Step2CRC",
]
