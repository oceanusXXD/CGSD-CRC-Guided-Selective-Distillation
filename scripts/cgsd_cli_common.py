"""CGSD 独立 stage CLI 的共享工具。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import PairExample, filter_examples_by_ids, load_examples  # noqa: E402
from src.utils import read_json, resolve_input_path, resolve_output_path, write_json  # noqa: E402


CACHE_POLICIES = ("reuse", "overwrite", "fail")


@dataclass(frozen=True)
class StageCacheDecision:
    """独立 stage 写文件前的缓存判定结果。"""

    stage_name: str
    cache_policy: str
    cache_hit: bool
    action: str
    existing_outputs: list[str]
    missing_outputs: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "cache_policy": self.cache_policy,
            "cache_hit": bool(self.cache_hit),
            "action": self.action,
            "existing_outputs": list(self.existing_outputs),
            "missing_outputs": list(self.missing_outputs),
        }


def binary_to_int(value: Any, *, field_name: str) -> int:
    """把 yes/no/true/false/1/0 统一成整数 1/0。"""
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


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取一个 stage 边界上的 JSONL 文件。"""
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def output_dir_from_arg(value: str | Path) -> Path:
    return resolve_output_path(value, PROJECT_ROOT)


def input_artifact_path(value: str | Path | None, default: Path) -> Path:
    """解析独立 stage 的输入 artifact 路径。"""
    return resolve_input_path(value, PROJECT_ROOT) if value is not None else default


def output_artifact_path(value: str | Path | None, default: Path) -> Path:
    """解析独立 stage 的输出 artifact 路径。"""
    return resolve_output_path(value, PROJECT_ROOT) if value is not None else default


def stage_cache_decision(
    *,
    stage_name: str,
    required_outputs: Iterable[str | Path],
    cache_policy: str,
) -> StageCacheDecision:
    """判断独立 stage 应该运行还是复用已有完整输出。

    `reuse` 必须严格：输出全在才复用，输出全无才运行，只有一部分输出
    存在就报错。这样被中断的 stage 不会把旧产物和新产物静默混在一起。
    """
    policy = str(cache_policy)
    if policy not in CACHE_POLICIES:
        raise ValueError(f"cache_policy must be one of {CACHE_POLICIES}, got {cache_policy!r}")
    output_paths = [Path(path) for path in required_outputs]
    existing = [str(path) for path in output_paths if path.exists()]
    missing = [str(path) for path in output_paths if not path.exists()]
    if policy == "overwrite":
        return StageCacheDecision(stage_name, policy, False, "run", existing, missing)
    if existing and missing:
        raise RuntimeError(
            f"{stage_name} partial cache: existing outputs={existing}; missing outputs={missing}. "
            "Use --cache_policy overwrite to regenerate this stage, or remove the partial outputs."
        )
    if existing and not missing:
        if policy == "fail":
            raise FileExistsError(f"{stage_name} outputs already exist: {existing}")
        return StageCacheDecision(stage_name, policy, True, "reuse", existing, missing)
    return StageCacheDecision(stage_name, policy, False, "run", existing, missing)


def add_stage_cache_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache_policy", choices=CACHE_POLICIES, default="reuse")
    parser.add_argument("--show_result", action="store_true", default=False)


def print_existing_stage_result(*, stage_name: str, summary_path: str | Path | None) -> None:
    if summary_path is None:
        print(json.dumps({"stage_name": stage_name, "cache_hit": True}, ensure_ascii=False, sort_keys=True))
        return
    source = Path(summary_path)
    if not source.exists():
        raise FileNotFoundError(f"{stage_name} result does not exist: {source}")
    if source.suffix == ".jsonl":
        rows = read_jsonl(source)
        payload: dict[str, Any] = {"stage_name": stage_name, "rows": len(rows), "result_path": str(source)}
    else:
        payload = read_json(source)
        payload.setdefault("stage_name", stage_name)
        payload.setdefault("result_path", str(source))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def estimate_text_tokens(text: str) -> int:
    """为 usage 账本提供确定性的低成本 token 估算。"""
    stripped = str(text or "").strip()
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / 4.0))


def estimate_query_document_prompt_tokens(rows: Iterable[dict[str, Any]]) -> int:
    total = 0
    for row in rows:
        total += estimate_text_tokens(str(row.get("query", "")))
        total += estimate_text_tokens(str(row.get("document", "")))
        total += 16
    return int(total)


def summarize_teacher_label_usage(rows: Iterable[dict[str, Any]], *, purpose: str) -> dict[str, Any]:
    materialized = list(rows)
    api_calls = sum(1 for row in materialized if row.get("teacher_source") == "teacher_api_file")
    groundtruth_calls = sum(
        1
        for row in materialized
        if row.get("teacher_source") == "groundtruth_substitute_for_real_teacher_api"
    )
    unknown_calls = sum(
        1
        for row in materialized
        if row.get("teacher_source") not in {"teacher_api_file", "groundtruth_substitute_for_real_teacher_api"}
    )
    return {
        "purpose": purpose,
        "teacher_calls": int(api_calls + groundtruth_calls + unknown_calls),
        "teacher_api_file_calls": int(api_calls),
        "groundtruth_substitute_calls": int(groundtruth_calls),
        "unknown_teacher_source_calls": int(unknown_calls),
        "estimated_prompt_tokens": estimate_query_document_prompt_tokens(materialized),
        "estimated_completion_tokens": int(len(materialized)),
    }


def embedding_usage_payload(
    *,
    embedding_source: str | Path,
    row_count: int,
    embedding_dim: int,
    purpose: str,
) -> dict[str, Any]:
    return {
        "purpose": purpose,
        "embedding_source": str(embedding_source),
        "embedding_rows": int(row_count),
        "embedding_dim": int(embedding_dim),
        "embedding_value_count": int(row_count) * int(embedding_dim),
    }


def write_stage_usage(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(payload, path)


def load_split_ids(output_dir: str | Path) -> dict[str, Any]:
    return read_json(output_dir_from_arg(output_dir) / "cgsd_split_ids.json")


def load_stage_examples(
    *,
    data_path: str | Path,
    query_field: str,
    document_field: str,
    label_field: str,
) -> list[PairExample]:
    """按 CLI 参数加载当前 stage 需要的样本。"""
    return load_examples(
        resolve_input_path(data_path, PROJECT_ROOT),
        query_field=str(query_field),
        document_field=str(document_field),
        label_field=str(label_field),
    )


def split_examples(
    examples: list[PairExample],
    split_payload: dict[str, Any],
) -> tuple[list[PairExample], list[PairExample]]:
    calibration_ids = {str(sample_id) for sample_id in split_payload["calibration_ids"]}
    pool_ids = {str(sample_id) for sample_id in split_payload["pool_ids"]}
    return (
        filter_examples_by_ids(examples, calibration_ids),
        filter_examples_by_ids(examples, pool_ids),
    )


def selected_train_rows_path(output_dir: str | Path) -> Path:
    return output_dir_from_arg(output_dir) / "cgsd_train_rows.jsonl"


def load_selected_train_rows(output_dir: str | Path) -> list[dict[str, Any]]:
    path = selected_train_rows_path(output_dir)
    if not path.exists():
        return []
    return read_jsonl(path)


def train_label_snapshot(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return {
        str(row["id"]): binary_to_int(row.get("label", row.get("groundtruth")), field_name="train label snapshot")
        for row in rows
    }


def _first_present(row: dict[str, Any], field_names: Iterable[Any]) -> Any:
    for field_name in field_names:
        if field_name in row and row[field_name] is not None:
            return row[field_name]
    return None


def _teacher_logit_margin(row: dict[str, Any]) -> float | None:
    """从常见 API 输出结构里提取 yes-vs-no teacher logit margin。"""
    value = _first_present(
        row,
        (
            "teacher_logit_margin",
            "teacher_logit",
            "logit_margin",
            "answer_logit_margin",
        ),
    )
    if value is not None:
        return float(value)

    logits = _first_present(row, ("teacher_logits", "logits"))
    if isinstance(logits, dict):
        yes_value = _first_present(logits, ("yes", "Yes", "1", 1))
        no_value = _first_present(logits, ("no", "No", "0", 0))
        if yes_value is not None and no_value is not None:
            return float(yes_value) - float(no_value)
    return None


def _confidence_from_logit_margin(margin: float, temperature: float) -> float:
    """按 sigmoid(abs(margin)/T) 把 teacher logit margin 转成置信度。"""
    temp = float(temperature)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError("teacher_temperature must be a positive finite number")
    value = abs(float(margin)) / temp
    return float(1.0 / (1.0 + math.exp(-value)))


def _clamped_confidence(value: Any) -> float:
    confidence = float(value)
    if not math.isfinite(confidence):
        raise ValueError(f"teacher confidence must be finite, got {value!r}")
    return float(max(0.0, min(1.0, confidence)))


def _teacher_row_id(row: dict[str, Any], *, source: str) -> str:
    sample_id = row.get("id", row.get("sample_id"))
    if sample_id is None or str(sample_id) == "":
        raise ValueError(f"{source} teacher row is missing id/sample_id")
    return str(sample_id)


def _normalize_teacher_row(row: dict[str, Any], *, source: str, teacher_temperature: float) -> dict[str, Any]:
    sample_id = _teacher_row_id(row, source=source)
    margin = _teacher_logit_margin(row)
    label_value = _first_present(
        row,
        (
            "teacher_label",
            "teacher_answer",
            "parsed_answer",
            "answer",
            "label",
            "groundtruth",
        ),
    )
    if label_value is None:
        if margin is None:
            raise ValueError(f"{source} teacher row {sample_id!r} needs teacher_label or teacher_logit_margin")
        label = 1 if margin > 0.0 else 0
        label_source = "teacher_logit_margin_sign"
    else:
        label = binary_to_int(label_value, field_name=f"{source} teacher label")
        label_source = "teacher_label"

    confidence_value = _first_present(
        row,
        (
            "teacher_confidence",
            "confidence",
            "parsed_confidence",
            "answer_confidence",
        ),
    )
    if confidence_value is None:
        confidence = _confidence_from_logit_margin(margin, teacher_temperature) if margin is not None else 1.0
        confidence_source = "teacher_logit_margin" if margin is not None else "default_1.0"
    else:
        confidence = _clamped_confidence(confidence_value)
        confidence_source = "teacher_confidence"

    payload: dict[str, Any] = {
        "id": sample_id,
        "teacher_label": int(label),
        "teacher_confidence": float(confidence),
        "teacher_source": "teacher_api_file",
        "teacher_label_source": label_source,
        "teacher_confidence_source": confidence_source,
    }
    if margin is not None:
        payload["teacher_logit_margin"] = float(margin)
    return payload


def _iter_teacher_json_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        yield from read_jsonl(path)
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError(f"{path} teacher JSON list entries must be objects")
            yield row
        return

    if not isinstance(payload, dict):
        raise ValueError(f"{path} teacher JSON must be a list, dict, or JSONL")
    for key in ("rows", "data", "items", "predictions", "teacher_labels"):
        value = payload.get(key)
        if isinstance(value, list):
            for row in value:
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{key} entries must be objects")
                yield row
            return
    for sample_id, value in payload.items():
        if isinstance(value, dict):
            row = dict(value)
            row.setdefault("id", sample_id)
            yield row
        else:
            yield {"id": sample_id, "teacher_label": value}


def load_teacher_labels(path: str | Path, *, teacher_temperature: float = 1.0) -> dict[str, dict[str, Any]]:
    """从 JSON/JSONL 读取真实 teacher API 输出。

    真实 API 可以输出 `teacher_label`，也可以只输出 yes/no logit margin；
    后者会用 margin 的符号得到标签，并用 sigmoid(abs(margin)/T) 得到
    teacher confidence。没有传这个文件时，预测阶段会用 groundtruth 代替
    真实 teacher API，并把 teacher confidence 固定为 1.0。
    """
    source = resolve_input_path(path, PROJECT_ROOT)
    if not source.exists():
        raise FileNotFoundError(f"teacher_labels_path does not exist: {source}")
    labels: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(_iter_teacher_json_rows(source), start=1):
        normalized = _normalize_teacher_row(
            dict(row),
            source=f"{source}:{index}",
            teacher_temperature=teacher_temperature,
        )
        sample_id = str(normalized["id"])
        if sample_id in labels:
            raise ValueError(f"{source}:{index} duplicate teacher id: {sample_id!r}")
        labels[sample_id] = normalized
    return labels


def apply_teacher_label(row: dict[str, Any], teacher_labels_by_id: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    """给单条预测结果附加 CRC、DBDS 和训练使用的 teacher 标签。"""
    sample_id = str(row.get("id", row.get("sample_id", "")))
    teacher_payload = (teacher_labels_by_id or {}).get(sample_id)
    if teacher_payload is not None:
        label = binary_to_int(teacher_payload["teacher_label"], field_name="teacher API label")
        row["label"] = label
        row["groundtruth"] = label
        row["teacher_label"] = label
        row["teacher_confidence"] = float(teacher_payload.get("teacher_confidence", 1.0))
        row["teacher_source"] = str(teacher_payload.get("teacher_source", "teacher_api_file"))
        row["teacher_label_source"] = str(teacher_payload.get("teacher_label_source", "teacher_label"))
        row["teacher_confidence_source"] = str(teacher_payload.get("teacher_confidence_source", "teacher_confidence"))
        if "teacher_logit_margin" in teacher_payload:
            row["teacher_logit_margin"] = float(teacher_payload["teacher_logit_margin"])
        return row

    # 离线实验没有真实 teacher API 输出时，用数据集 groundtruth 代替真实 API。
    # 这是实验替代，不是部署逻辑；因此 teacher_confidence 明确记为 1.0。
    label = binary_to_int(row.get("groundtruth", row.get("label")), field_name="groundtruth teacher substitute")
    row["label"] = label
    row["groundtruth"] = label
    row["teacher_label"] = label
    row["teacher_confidence"] = 1.0
    row["teacher_source"] = "groundtruth_substitute_for_real_teacher_api"
    row["teacher_label_source"] = "groundtruth"
    row["teacher_confidence_source"] = "fixed_1.0_groundtruth_substitute"
    return row


def _ids_from_json_payload(payload: Any) -> list[str]:
    if isinstance(payload, list):
        ids: list[str] = []
        for item in payload:
            if isinstance(item, dict):
                sample_id = item.get("id", item.get("sample_id"))
            else:
                sample_id = item
            if sample_id is not None and str(sample_id) != "":
                ids.append(str(sample_id))
        return ids
    if isinstance(payload, dict):
        for key in ("anchor_ids", "ids", "candidate_ids", "anchor_candidate_ids"):
            value = payload.get(key)
            if isinstance(value, list):
                return _ids_from_json_payload(value)
        return [str(key) for key in payload.keys()]
    raise ValueError("anchor id JSON must be a list or object")


def load_anchor_ids(path: str | Path) -> list[str]:
    """从 JSON、JSONL 或一行一个 ID 的文本读取可复用 anchor 候选集。"""
    source = resolve_input_path(path, PROJECT_ROOT)
    if not source.exists():
        raise FileNotFoundError(f"anchor_ids_path does not exist: {source}")
    if source.suffix == ".jsonl":
        raw_ids = [str(row.get("id", row.get("sample_id", ""))) for row in read_jsonl(source)]
    elif source.suffix == ".json":
        raw_ids = _ids_from_json_payload(json.loads(source.read_text(encoding="utf-8")))
    else:
        raw_ids = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()]

    seen: set[str] = set()
    ids: list[str] = []
    for sample_id in raw_ids:
        if not sample_id or sample_id in seen:
            continue
        seen.add(sample_id)
        ids.append(sample_id)
    return ids


def runtime_args_from_cli(cli_args: argparse.Namespace) -> argparse.Namespace:
    """从 CLI 参数构造模型推理和训练需要的运行参数。"""
    def value(name: str, default: Any = None) -> Any:
        override = getattr(cli_args, name, None)
        return default if override is None else override

    return argparse.Namespace(
        max_length=int(value("max_length", 512)),
        batch_size=int(value("batch_size", 16)),
        eval_batch_size=int(value("eval_batch_size", 32)),
        epochs=int(value("epochs", 3)),
        lr=float(value("lr", 2e-4)),
        weight_decay=float(value("weight_decay", 0.01)),
        gradient_accumulation_steps=int(value("gradient_accumulation_steps", 1)),
        max_grad_norm=float(value("max_grad_norm", 1.0)),
        warmup_ratio=float(value("warmup_ratio", 0.1)),
        early_stopping_patience=value("early_stopping_patience", None),
        early_stopping_min_delta=float(value("early_stopping_min_delta", 0.0)),
        threshold=float(value("threshold", 0.0)),
        num_workers=int(value("num_workers", 2)),
        prefetch_factor=int(value("prefetch_factor", 2)),
        pad_to_multiple_of=int(value("pad_to_multiple_of", 8)),
        cache_tokenization=bool(value("cache_tokenization", True)),
        pin_memory=bool(value("pin_memory", True)),
        tf32=bool(value("tf32", True)),
        torch_dtype=str(value("torch_dtype", "auto")),
        trust_remote_code=bool(value("trust_remote_code", True)),
        lora_r=int(value("lora_r", 1)),
        lora_alpha=int(value("lora_alpha", 16)),
        lora_dropout=float(value("lora_dropout", 0.05)),
        lora_target_modules=str(value("lora_target_modules", "qv")),
        lora_layer_scope=str(value("lora_layer_scope", "all")),
        balance_train_classes=bool(value("balance_train_classes", False)),
        seed=int(value("seed", 42)),
    )


def add_runtime_overrides(parser: argparse.ArgumentParser) -> None:
    """给需要模型运行的 stage 添加通用可选覆盖参数。"""
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--pad_to_multiple_of", type=int, default=None)
    parser.add_argument("--torch_dtype", choices=["auto", "none", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--trust_remote_code", action="store_true", default=None)
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_target_modules", default=None)
    parser.add_argument("--lora_layer_scope", default=None)
    parser.add_argument("--balance_train_classes", action="store_true", default=None)
    parser.add_argument("--no_balance_train_classes", dest="balance_train_classes", action="store_false")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_cache_tokenization", dest="cache_tokenization", action="store_false", default=None)
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false", default=None)
    parser.add_argument("--no_tf32", dest="tf32", action="store_false", default=None)
