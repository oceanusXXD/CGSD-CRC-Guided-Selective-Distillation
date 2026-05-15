"""本地化的 cascade 算法共享类型和工具。

这些定义来自当前项目需要的最小算法接口，用来让 `algorithms/step*.py`
保持在 LLM_layer_test 内部自包含，避免从仓库外的级联实现或脚本包
交叉导入。这里不包含模型调用或 IO，只提供纯数据结构、标签归一化和摘要统计。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional

import numpy as np


DEFAULT_STEP1_CHUNK_SCORE_TEMPERATURE = 0.0
SAMPLE_ID_SEPARATOR = ":"
STEP1_ROUTABLE_SIGNAL_QUALITIES = {
    "paired_exact",
    "single_sided_lb",
    "batchprocess_chunk",
}

_BINARY_TEXT_TRUE_VALUES = {"y", "yes", "true", "1"}
_BINARY_TEXT_FALSE_VALUES = {
    "n",
    "no",
    "false",
    "0",
    "refutes",
    "not enough info",
    "not enough information",
}


class Action(str, Enum):
    """样本在运行期的统一动作标记。"""

    ACCEPT = "accept"
    CONTINUE = "continue"
    DEFER = "defer"


class Source(str, Enum):
    """最终判定来源，用于报表和审计。"""

    BLACKLIST_YES = "blacklist_yes"
    BLACKLIST_NO = "blacklist_no"
    STEP1_LOCAL = "step1_local"
    STEP3_LARGE_MODEL = "step3_large_model"
    STEP3_GROUND_TRUTH = "step3_ground_truth"


@dataclass
class Sample:
    """贯穿 cascade 算法链路的本地样本对象。"""

    sample_id: str
    text: str
    query: str
    label: Optional[str] = None
    raw_query_text: str = ""
    prompt_style: str = "query_document"
    prompt_task_label: str = "Query"
    prompt_evidence_label: str = "Document"
    claim_text: str = ""
    prompt_task_text: str = ""
    prompt_instruction_text: str = ""
    prompt_evidence_text: str = ""
    embedding_task_text: str = ""
    embedding_instruction_text: str = ""
    embedding_evidence_text: str = ""

    prediction: Optional[str] = None
    action: Action = Action.CONTINUE
    source: Optional[Source] = None

    answer_4b: Optional[str] = None
    step1_raw_label: str = ""
    step1_parse_failed: bool = False
    step1_parse_failure_reason: str = ""
    step1_yes_logprob: float = 0.0
    step1_no_logprob: float = 0.0
    step1_paired_exact_prob: float = 0.0
    step1_signal_quality: str = ""
    step1_signal_source: str = ""
    step1_neighbor_support: float = 0.0
    step1_routing_score: float = 0.0
    step1_binary_entropy: float = 0.0
    defer_reason: str = ""
    step2_decision_threshold_used: Optional[float] = None
    step2_lambda_hat_used: Optional[float] = None

    path: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def document(self) -> str:
        """`text` 的语义别名。"""
        return str(self.text or "")

    @document.setter
    def document(self, value: Any) -> None:
        self.text = str(value or "")

    @property
    def document_text(self) -> str:
        """文档文本别名。"""
        return str(self.text or "")

    @document_text.setter
    def document_text(self, value: Any) -> None:
        self.text = str(value or "")

    @property
    def query_text(self) -> str:
        """查询文本别名。"""
        return str(self.query or "")

    @query_text.setter
    def query_text(self, value: Any) -> None:
        self.query = str(value or "")


def normalize_id(value: Any) -> str:
    """把任意输入标准化成去空白字符串 ID。"""
    return str(value or "").strip()


def split_sample_id(sample_id: Any) -> tuple[str, str]:
    """把 sample_id 拆成 `(document_id, query_id)`。"""
    text = normalize_id(sample_id)
    if not text:
        return "", ""
    parts = text.split(SAMPLE_ID_SEPARATOR)
    if len(parts) == 1:
        return parts[0], ""
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"invalid sample_id: {sample_id!r}")
    return parts[0], parts[1]


def normalize_binary_decision_text(value: Any) -> str:
    """把常见二元标签和模型输出归一成 `yes` / `no`。"""
    raw = str("" if value is None else value).strip()
    if not raw:
        return ""

    candidates = [raw]
    compact = re.sub(r"\s+", "", raw)
    if compact and compact != raw:
        candidates.append(compact)

    for candidate in candidates:
        cleaned = str(candidate).strip()
        previous = None
        while cleaned and cleaned != previous:
            previous = cleaned
            if len(cleaned) >= 2 and cleaned[0] in {'"', "'", "`"} and cleaned[-1] == cleaned[0]:
                cleaned = cleaned[1:-1].strip()
                continue
            break

        for normalized_candidate in (
            cleaned.lower(),
            re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", cleaned).strip().lower(),
        ):
            if normalized_candidate in _BINARY_TEXT_TRUE_VALUES:
                return "yes"
            if normalized_candidate in _BINARY_TEXT_FALSE_VALUES:
                return "no"
    return ""


def normalize_binary_decision_int(value: Any) -> Optional[int]:
    """把常见二元标签和模型输出归一成整数 `1` / `0`。

    级联算法内部仍有一些地方需要 yes/no 文本做 token 或旧接口兼容；
    但只要字段会写入外部工件，就优先使用这个整数归一化结果。
    """
    label = normalize_binary_decision_text(value)
    if label == "yes":
        return 1
    if label == "no":
        return 0
    return None


def normalize_binary_label(value: Any) -> str:
    """把支持的二元标签写法归一成 `yes` / `no`。"""
    return normalize_binary_decision_text(value)


def is_binary_label(value: Any) -> bool:
    """判断输入是否能归一成 `yes` 或 `no`。"""
    return bool(normalize_binary_decision_text(value))


def value_distribution(values: Iterable[str]) -> dict[str, int]:
    """把字符串序列汇总成稳定排序的频次表。"""
    dist: dict[str, int] = {}
    for value in values:
        key = str(value or "").strip() or "unknown"
        dist[key] = int(dist.get(key, 0)) + 1
    return dict(sorted(dist.items(), key=lambda item: item[0]))


def numeric_summary(values: Iterable[float]) -> dict[str, float]:
    """返回校准和阶段摘要使用的紧凑数值统计。"""
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def _sample_signal_quality_key(sample: Any) -> str:
    if bool(getattr(sample, "step1_parse_failed", False)):
        return "parse_failed"
    return str(getattr(sample, "step1_signal_quality", "") or "").strip() or "unknown"


def signal_quality_distribution_from_samples(
    samples: Iterable[Any],
    *,
    labeled_only: bool = False,
) -> dict[str, int]:
    """汇总样本里的 Step1 signal quality。"""
    values: list[str] = []
    for sample in samples:
        if labeled_only and not is_binary_label(getattr(sample, "label", "")):
            continue
        values.append(_sample_signal_quality_key(sample))
    return value_distribution(values)


def neighbor_support_summary_from_samples(
    samples: Iterable[Any],
    *,
    labeled_only: bool = False,
) -> dict[str, float]:
    """汇总二元 Step1 输出样本的邻域支持度。"""
    values: list[float] = []
    for sample in samples:
        if labeled_only and not is_binary_label(getattr(sample, "label", "")):
            continue
        if is_binary_label(getattr(sample, "answer_4b", "")):
            values.append(float(getattr(sample, "step1_neighbor_support", 0.0) or 0.0))
    return numeric_summary(values)


__all__ = [
    "Action",
    "DEFAULT_STEP1_CHUNK_SCORE_TEMPERATURE",
    "STEP1_ROUTABLE_SIGNAL_QUALITIES",
    "Sample",
    "Source",
    "is_binary_label",
    "neighbor_support_summary_from_samples",
    "normalize_binary_decision_int",
    "normalize_binary_decision_text",
    "normalize_binary_label",
    "numeric_summary",
    "split_sample_id",
    "value_distribution",
    "signal_quality_distribution_from_samples",
]
