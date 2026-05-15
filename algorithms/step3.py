"""Step3 批量输出协议解析。"""

from __future__ import annotations

import json
import re

from .cascade_local import normalize_binary_decision_int, normalize_binary_decision_text

_STEP3_BATCH_LABEL_RE = re.compile(
    r"^\s*(?:\[(?P<bracket_idx>\d+)\]|(?P<plain_idx>\d+))\s*[:.-]?\s*(?P<label>[A-Za-z0-9]+)\s*$",
    flags=re.IGNORECASE,
)


def extract_json_candidate(text: str) -> str:
    """从 Step3 输出里截取 JSON 主体。

    这里允许模型在外层包代码块，但只接受最外层 object 作为有效协议载体。
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    if raw.startswith("```"):
        lines = [line for line in raw.splitlines() if not line.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return ""
    return raw[start : end + 1]


def parse_step3_batch_output(text: str, document_count: int, *, uses_chunk_json_protocol: bool) -> list[int]:
    """解析 Step3 批输出。

    `chunk` 模式使用 JSON 协议返回 `all_no` / `has_yes`，`document` 模式继续使用
    逐行 `index:Y/N`。返回值统一是 `1/0`，避免后续工件混入 yes/no 字符串。
    """
    if uses_chunk_json_protocol:
        # JSON 协议只表达“全无命中”或“哪些下标命中 yes”，因此这里必须先验证结构，再还原成逐条标签。
        candidate = extract_json_candidate(text)
        if not candidate:
            raise ValueError("missing_batch_json")
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise ValueError("batch_json_not_object")
        if payload.get("result") == "all_no":
            return [0 for _ in range(int(document_count))]
        if payload.get("result") != "has_yes":
            raise ValueError("unexpected_batch_json_result")
        yes_indices_raw = payload.get("yes_indices", [])
        if not isinstance(yes_indices_raw, list):
            raise ValueError("yes_indices_not_list")
        yes_indices = {int(idx) for idx in yes_indices_raw if isinstance(idx, int)}
        return [1 if idx in yes_indices else 0 for idx in range(int(document_count))]

    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty_batch_response")
    if raw.startswith("```"):
        # 允许代码块包裹，但最终仍然按纯文本协议逐行校验。
        rows = [line for line in raw.splitlines() if not line.strip().startswith("```")]
        raw = "\n".join(rows).strip()
    observed_indices: list[int] = []
    labels: list[int] = []
    for row in raw.replace(";", "\n").splitlines():
        item = str(row or "").strip()
        if not item:
            continue
        match = _STEP3_BATCH_LABEL_RE.fullmatch(item)
        if match is None:
            continue
        raw_idx = match.group("bracket_idx") or match.group("plain_idx") or ""
        parsed_idx = int(raw_idx)
        label = normalize_binary_decision_text(match.group("label") or "")
        if label not in {"yes", "no"}:
            raise ValueError(f"invalid_batch_label:{item!r}")
        label_int = normalize_binary_decision_int(label)
        if label_int is None:
            raise ValueError(f"invalid_batch_label:{item!r}")
        observed_indices.append(parsed_idx)
        labels.append(label_int)
    expected = list(range(1, int(document_count) + 1))
    # 逐行协议必须严格覆盖 1..N，任何缺项、乱序或多余项都要直接失败。
    if observed_indices != expected:
        raise ValueError(f"unexpected_batch_indices:{observed_indices!r}")
    return labels


__all__ = [
    "extract_json_candidate",
    "parse_step3_batch_output",
]
