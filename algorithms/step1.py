"""Chunk Step1 的 prompt 协议、输出解析和 routing score 归一化。"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from .cascade_local import DEFAULT_STEP1_CHUNK_SCORE_TEMPERATURE


CHUNK_STEP1_SYSTEM_PROMPT = (
    "You are a strict chunk-level binary classifier. "
    "For each indexed chunk, decide whether that chunk alone contains direct evidence that the answer to the Query is YES. "
    "Return exactly one line per index in strict ascending order using only the format '<index>: Y' or '<index>: N'. "
    "Do not return JSON, explanations, or extra text."
)

# Step1 chunk 协议限制成逐行 `index: Y/N`：
# 这样可以把模型调用留在 commands 层，把“批内第几个样本对应哪个标签”的校验留在算法层。
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_CRC_CHUNK_LABEL_LINE_RE = re.compile(
    r"^\s*(?:\[(?P<bracket_idx>\d+)\]|(?P<plain_idx>\d+))\s*[:.-]?\s*(?P<label>[A-Za-z0-9]+)\s*$",
    flags=re.IGNORECASE,
)


def _strip_markdown_code_fence(text: str) -> str:
    """移除模型输出外层的 markdown 代码块和 think 片段。"""
    clean = (text or "").strip()
    clean = _THINK_BLOCK_RE.sub("", clean).strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
    return clean.strip()


def _normalize_binary_batch_label(value: Any) -> str:
    """把批量协议中的 Y/N/yes/no 变体归一成 yes/no。"""
    raw = str(value or "").strip().lower()
    if raw in {"y", "yes", "1", "true"}:
        return "yes"
    if raw in {"n", "no", "0", "false"}:
        return "no"
    return ""


def _render_chunk_evidence_text(item: dict[str, Any]) -> str:
    """从 chunk item 中取出用于 prompt 的证据文本。"""
    return str(item.get("evidence_text", "") or item.get("top_chunk", "") or "").strip()


def create_chunk_step1_prompt(
    query: str,
    batch: list[dict[str, Any]],
    *,
    task_label: str = "Query",
    evidence_label: str = "Document",
    instruction_text: str = "",
) -> str:
    """构造 chunk 批判定 prompt。

    batch 中每个 item 独立判定，输出仍必须按 1..N 顺序返回；后续解析会依赖这个顺序
    将标签和 logprob 对齐到原始 sample。
    """
    lines = [
        f"Task: For each indexed item, decide whether the {evidence_label} supports that item's {task_label}.",
    ]
    if str(instruction_text or "").strip():
        lines.append(f"Instruction: {str(instruction_text or '').strip()}")
    elif str(query or "").strip():
        lines.append(f"Query: {query}")
    lines.extend(
        [
            "Rules:",
            "- Evaluate each indexed item independently.",
            "- Output exactly one label for every index from 1 to N in strict ascending order.",
            "- Use Y when the Evidence supports the Claim.",
            "- Use N when the Evidence contradicts the Claim or does not provide enough evidence to support it.",
            "- Treat concise, compressed, or paraphrased evidence as sufficient when it clearly supports the Claim.",
            "- Do not require the Evidence to repeat every word from the Claim when entity and relation match.",
            "- Do not guess from outside knowledge; use only the Claim and Evidence in that item.",
            "- Return plain text only. Do not return JSON, explanations, or any extra text.",
            "- Return exactly one line per index.",
            "- Use ASCII punctuation only.",
            "- Put exactly one space after each colon.",
            "Output format:",
            "1: N",
            "2: Y",
            "3: Y",
        ]
    )
    for display_idx, item in enumerate(batch, start=1):
        evidence_text = str(item.get("evidence_text", "") or _render_chunk_evidence_text(item)).strip()
        task_text = str(item.get("task_text", "") or item.get("claim", "") or "").strip()
        if task_text:
            item_task_label = task_label
            item_evidence_label = evidence_label
            if task_label == "Query" and evidence_label == "Document":
                item_task_label = "Claim"
                item_evidence_label = "Evidence"
            lines.append(f"[{display_idx}]")
            lines.append(f"{item_task_label}: {task_text}")
            lines.append(f"{item_evidence_label}:")
            lines.append(evidence_text)
        elif "\n" in evidence_text:
            lines.append(f"[{display_idx}]")
            lines.append(evidence_text)
        else:
            lines.append(f"[{display_idx}] {evidence_text}")
    return "\n".join(lines)


def parse_chunk_step1_output(response: str, item_count: int) -> dict[int, str]:
    """解析并严格校验 Step1 chunk 批输出。

    这里不接受乱序、漏项或重复下标，因为 Step1 的 routing score 会继续进入 Step2 CRC；
    下标错位比解析失败更危险，所以直接抛错让运行态中止。
    """
    clean = _strip_markdown_code_fence(response or "")
    rows = [row.strip() for row in clean.replace(";", "\n").replace(",", "\n").splitlines() if row.strip()]
    label_by_index: dict[int, str] = {}
    for row in rows:
        match = _CRC_CHUNK_LABEL_LINE_RE.fullmatch(row)
        if match is None:
            raise ValueError(f"unparseable chunk step1 label row: {row!r}")
        raw_idx = match.group("bracket_idx") or match.group("plain_idx") or ""
        parsed_idx = int(raw_idx)
        label = _normalize_binary_batch_label(match.group("label") or "")
        if label not in {"yes", "no"}:
            raise ValueError(f"invalid chunk step1 label row: {row!r}")
        if parsed_idx in label_by_index:
            raise ValueError(f"duplicate chunk step1 index: {parsed_idx}")
        expected_next = len(label_by_index) + 1
        if parsed_idx != expected_next:
            raise ValueError(f"out_of_order chunk step1 index: {parsed_idx}!={expected_next}")
        label_by_index[parsed_idx] = label
    expected_indices = list(range(1, int(item_count) + 1))
    if sorted(label_by_index) != expected_indices:
        raise ValueError(f"unexpected chunk step1 indices: {sorted(label_by_index)!r}")
    return {index: label_by_index[index] for index in expected_indices}


def _normalize_chunk_logprob_token(token: Any) -> str:
    """把 vLLM 返回的候选 token 归一成 Y/N。"""
    # vLLM logprobs 返回的 token 可能带空格、换行或标点，这里只把可明确归一为 Y/N 的 token 纳入打分。
    raw = str(token or "")
    if not raw:
        return ""
    stripped = raw.strip().upper()
    if stripped in {"YES", "TRUE"}:
        return "Y"
    if stripped in {"NO", "FALSE"}:
        return "N"
    if stripped in {"Y", "N"}:
        return stripped
    edge_cleaned = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", stripped).upper()
    if edge_cleaned in {"YES", "TRUE"}:
        return "Y"
    if edge_cleaned in {"NO", "FALSE"}:
        return "N"
    if edge_cleaned in {"Y", "N"}:
        return edge_cleaned
    alpha_only = re.sub(r"[^A-Za-z]", "", stripped).upper()
    if alpha_only in {"YES", "TRUE"}:
        return "Y"
    if alpha_only in {"NO", "FALSE"}:
        return "N"
    if alpha_only in {"Y", "N"}:
        return alpha_only
    return ""


def _collect_chunk_binary_logprobs(top_logprobs: Any) -> dict[str, float]:
    """从一个输出位置的 top_logprobs 中收集 Y/N 的最高 logprob。"""
    # 同一输出位置可能在 top_logprobs 中出现多个可归一到 Y/N 的 token；
    # 取 logprob 更高的一个，保留模型在该位置对二元标签的最强显式信号。
    observed: dict[str, float] = {}
    if not top_logprobs:
        return observed
    for item in top_logprobs:
        value = getattr(item, "logprob", None)
        if value is None:
            continue
        token_norm = _normalize_chunk_logprob_token(getattr(item, "token", ""))
        if token_norm not in {"Y", "N"}:
            continue
        lp = float(value)
        prev = observed.get(token_norm)
        if prev is None or lp > prev:
            observed[token_norm] = lp
    return observed


def _content_item_text(content_item: Any) -> str:
    """读取 vLLM logprob content item 的 token 文本。"""
    return str(getattr(content_item, "token", "") or "")


def _find_chunk_label_token_positions(content_items: list[Any]) -> list[int]:
    """定位批输出中真正承载 Y/N 标签的 token 位置。"""
    # 只把最终答案里的 Y/N token 当作标签位置，避免把 prompt echo 或标点纳入分数对齐。
    positions: list[int] = []
    for index, content_item in enumerate(content_items):
        token_norm = _normalize_chunk_logprob_token(_content_item_text(content_item))
        if token_norm in {"Y", "N"}:
            positions.append(index)
    return positions


def extract_chunk_step1_label_scores(raw_response: Any, expected_labels: list[str]) -> dict[str, Any]:
    """从 vLLM 原始响应里抽取每个批内标签的 Y/N logprob。

    `expected_labels` 来自文本协议解析结果。先解析文本，再按标签 token 位置抽分数，
    可以确保“模型输出的标签”和“用于 Step2 的分数”来自同一批输出。
    """
    try:
        choice = list(getattr(raw_response, "choices", []) or [])[0]
    except Exception:
        return {"score_ok": False, "items": [], "error": "missing_choice"}
    logprobs_obj = getattr(choice, "logprobs", None)
    content_items = list(getattr(logprobs_obj, "content", []) or [])
    if not content_items:
        return {"score_ok": False, "items": [], "error": "missing_logprobs_content"}

    expected = [str(label or "").strip().lower() for label in list(expected_labels or [])]
    items: list[dict[str, Any]] = []
    label_positions = _find_chunk_label_token_positions(content_items)
    for matched_idx, token_position in enumerate(label_positions):
        if matched_idx >= len(expected):
            break
        content_item = content_items[token_position]
        token_text = str(getattr(content_item, "token", "") or "")
        observed = _collect_chunk_binary_logprobs(getattr(content_item, "top_logprobs", None))
        token_logprob = getattr(content_item, "logprob", None)
        token_norm = _normalize_chunk_logprob_token(token_text)
        if token_norm in {"Y", "N"} and token_logprob is not None and token_norm not in observed:
            observed[token_norm] = float(token_logprob)
        if token_norm not in {"Y", "N"}:
            continue
        y_lp = observed.get("Y")
        n_lp = observed.get("N")
        pred = "yes" if token_norm == "Y" else "no"
        pred_lp = y_lp if pred == "yes" else n_lp
        # 有些服务只返回生成 token 本身的 logprob，没有同时给另一个标签的 top_logprob；
        # 这种行只能作为单侧下界信号，后续不能计算 paired exact probability。
        if pred_lp is None and token_logprob is not None:
            pred_lp = float(token_logprob)
            if pred == "yes" and y_lp is None:
                y_lp = float(token_logprob)
            if pred == "no" and n_lp is None:
                n_lp = float(token_logprob)
        if pred_lp is None:
            continue
        if y_lp is not None and n_lp is not None:
            items.append({"pred": pred, "yes_logprob": float(y_lp), "no_logprob": float(n_lp)})
        else:
            items.append(
                {
                    "pred": pred,
                    "yes_logprob": (None if y_lp is None else float(y_lp)),
                    "no_logprob": (None if n_lp is None else float(n_lp)),
                    "score_signal_quality": "single_sided_lb",
                }
            )
    if len(items) != len(expected):
        return {
            "score_ok": False,
            "items": items,
            "error": f"score_count_mismatch:{len(items)}!={len(expected)}",
            "content_item_count": int(len(content_items)),
            "label_token_positions": list(label_positions),
        }
    predicted_labels = [str(item.get("pred", "") or "") for item in items]
    if predicted_labels != expected:
        return {
            "score_ok": False,
            "items": items,
            "error": f"score_label_mismatch:{predicted_labels!r}!={expected!r}",
        }
    return {"score_ok": True, "items": items, "error": None}


def _resolve_chunk_score_temperature(
    score_temperature: float,
    *,
    batch_item_count: int | None = None,
) -> float:
    """解析 chunk routing score 的温度参数。"""
    # 温度影响 paired Y/N gap 到 routing score 的压缩强度，并保持模型原始标签。
    temperature = float(score_temperature)
    if not math.isfinite(temperature):
        raise ValueError("chunk score temperature must be finite")
    if temperature > 0.0:
        return temperature
    if batch_item_count is not None and int(batch_item_count) > 0:
        return float(max(1, int(batch_item_count)))
    return 1.0


def _sigmoid_from_gap(gap: float) -> float:
    """把 logprob gap 转成稳定的 sigmoid 分数。"""
    # 使用数值稳定写法，避免大 gap 下 exp 溢出。
    value = float(gap)
    if value >= 0.0:
        exp_term = math.exp(-value)
        return float(1.0 / (1.0 + exp_term))
    exp_term = math.exp(value)
    return float(exp_term / (1.0 + exp_term))


def normalize_chunk_step1_score_row(
    row: dict[str, Any],
    prediction: str,
    *,
    score_temperature: float = DEFAULT_STEP1_CHUNK_SCORE_TEMPERATURE,
    batch_item_count: int | None = None,
) -> dict[str, Any]:
    """把单个 chunk 标签的 logprob 信号转成 Step2 可消费的稳定特征。"""
    pred = _normalize_binary_batch_label(prediction)
    if pred not in {"yes", "no"}:
        raise ValueError(f"invalid chunk step1 prediction: {prediction!r}")

    yes_logprob_raw = row.get("yes_logprob")
    no_logprob_raw = row.get("no_logprob")
    yes_logprob = None if yes_logprob_raw is None else float(yes_logprob_raw)
    no_logprob = None if no_logprob_raw is None else float(no_logprob_raw)
    signal_quality = str(row.get("score_signal_quality", "") or "")
    prob_yes = 0.0
    prob_no = 0.0
    paired_exact_prob = 0.0
    routing_score = 0.0
    binary_entropy = 0.0
    if signal_quality != "single_sided_lb" and yes_logprob is not None and no_logprob is not None:
        # paired Y/N 同时存在时，先在二元标签空间内重新归一化，再用预测方向的 gap 计算 routing score。
        # 这让 Step2 消费的是“模型在 yes/no 之间的相对确信度”，而不是受全词表概率稀释的绝对概率。
        normalizer = np.logaddexp(float(yes_logprob), float(no_logprob))
        prob_yes = float(np.exp(float(yes_logprob) - normalizer))
        prob_no = float(np.exp(float(no_logprob) - normalizer))
        paired_exact_prob = float(prob_yes if pred == "yes" else prob_no)
        gap = float((yes_logprob - no_logprob) if pred == "yes" else (no_logprob - yes_logprob))
        temperature = _resolve_chunk_score_temperature(
            float(score_temperature),
            batch_item_count=batch_item_count,
        )
        routing_score = _sigmoid_from_gap(gap / temperature)
        if prob_yes > 0.0:
            binary_entropy -= float(prob_yes) * float(math.log(max(prob_yes, 1e-12)))
        if prob_no > 0.0:
            binary_entropy -= float(prob_no) * float(math.log(max(prob_no, 1e-12)))
        signal_quality = "batchprocess_chunk"
    else:
        # 单侧信号只说明模型生成了当前标签，不能证明另一个标签的相对概率；
        # 因此只给一个保守的下界分数，并把 signal_quality 显式标出来。
        pred_logprob = yes_logprob if pred == "yes" else no_logprob
        if pred_logprob is not None:
            routing_score = float(max(0.0, min(1.0, math.exp(float(pred_logprob)))))
        signal_quality = "single_sided_lb"
    return {
        "prediction": 1 if pred == "yes" else 0,
        "signal_quality": signal_quality,
        "yes_logprob": 0.0 if yes_logprob is None else float(yes_logprob),
        "no_logprob": 0.0 if no_logprob is None else float(no_logprob),
        "prob_yes": float(prob_yes),
        "prob_no": float(prob_no),
        "paired_exact_prob": float(paired_exact_prob),
        "routing_score": float(routing_score),
        "binary_entropy": float(binary_entropy),
    }


__all__ = [
    "CHUNK_STEP1_SYSTEM_PROMPT",
    "create_chunk_step1_prompt",
    "extract_chunk_step1_label_scores",
    "normalize_chunk_step1_score_row",
    "parse_chunk_step1_output",
]
