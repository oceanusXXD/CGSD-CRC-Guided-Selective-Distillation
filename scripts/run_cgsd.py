#!/usr/bin/env python
"""CGSD 单步 CLI 共享函数。

本文件不再提供一体化流水线入口。各阶段必须通过
`cgsd_prepare.py`、`cgsd_predict.py`、`cgsd_calibrate.py`、
`cgsd_select.py`、`cgsd_train_round.py` 和 `cgsd_finalize.py`
单独启动、单独停止、单独产出结果。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cgsd_cli_common import apply_teacher_label, binary_to_int, read_jsonl
from src.data import (
    GenerationPairCollator,
    GenerationQueryDocumentDataset,
    PairExample,
)
from src.model import QwenGenerativeModel
from src.trainer import fit, predict_model
from src.utils import (
    parse_torch_dtype,
    read_json,
    write_jsonl,
)


def _embedding_ids_from_sidecar(path: Path, row_count: int) -> list[str]:
    """从本地 CGSD sidecar 文件恢复 .npy embedding 的样本顺序。"""
    candidates: list[Path] = []
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        meta = read_json(meta_path)
        source_file = meta.get("source_selected_chunks_file")
        if source_file:
            source_path = Path(str(source_file))
            if not source_path.is_absolute():
                candidates.append(path.parent / source_path.name)
                candidates.append(PROJECT_ROOT.parent / source_path)
            else:
                candidates.append(source_path)
    candidates.extend(
        [
            path.with_suffix(".ids.jsonl"),
            path.with_suffix(".ids.json"),
            path.parent / "evidence_rows.jsonl",
            path.parent / "selected_chunks.jsonl",
            path.parent / "embedding_rows.jsonl",
        ]
    )

    seen_candidates: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.exists() else candidate
        if candidate in seen_candidates or not candidate.exists():
            continue
        seen_candidates.add(candidate)
        if candidate.suffix == ".jsonl":
            rows = read_jsonl(candidate)
            ids = [
                str(row.get("id", row.get("sample_id", row.get("review_id", row.get("document_id", "")))))
                for row in rows
            ]
        else:
            payload = read_json(candidate)
            raw_ids = payload.get("ids", payload.get("embedding_ids", payload)) if isinstance(payload, dict) else payload
            ids = [str(item.get("id", item.get("sample_id", "")) if isinstance(item, dict) else item) for item in raw_ids]
        ids = [sample_id for sample_id in ids if sample_id]
        if len(ids) == int(row_count):
            if len(set(ids)) != len(ids):
                raise ValueError(f"{candidate} contains duplicate embedding ids")
            return ids
    raise ValueError(
        f"{path} is a .npy embedding matrix with {row_count} rows, but no matching id sidecar was found. "
        "Expected evidence_rows.jsonl, selected_chunks.jsonl, *.ids.jsonl, or meta source_selected_chunks_file."
    )


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    """读取真实预计算 pair embedding。

    文件必须覆盖所有样本 ID；缺失会在主流程中直接报错。这样 DBDS 的
    k-Center 始终运行在文档要求的固定 query-aware embedding 空间，
    不会退化成哈希、随机或其他启发式占位向量。
    """
    if path.suffix == ".npy":
        matrix = np.load(path, allow_pickle=False)
        if matrix.ndim != 2:
            raise ValueError(f"{path} must contain a 2D embedding matrix, got shape {matrix.shape}")
        ids = _embedding_ids_from_sidecar(path, int(matrix.shape[0]))
        return {
            sample_id: np.asarray(matrix[index], dtype=np.float32)
            for index, sample_id in enumerate(ids)
        }
    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        return {str(key): np.asarray(payload[key], dtype=np.float32) for key in payload.files}
    if path.suffix == ".json":
        data = read_json(path)
        return {str(key): np.asarray(value, dtype=np.float32) for key, value in data.items()}

    embeddings: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            sample_id = str(row.get("id", row.get("sample_id", "")))
            vector = row.get("embedding")
            if not sample_id or vector is None:
                raise ValueError(f"{path}:{line_number} must contain id/sample_id and embedding")
            if sample_id in embeddings:
                raise ValueError(f"{path}:{line_number} duplicate embedding id: {sample_id!r}")
            embeddings[sample_id] = np.asarray(vector, dtype=np.float32)
    return embeddings


def assert_embedding_coverage(
    embeddings_by_id: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    *,
    expected_dim: int,
) -> None:
    """校验 embedding 是否覆盖全量 U。

    文档 Phase 0 要求对所有文档一次性计算 pair embedding；这里把这个
    要求做成硬约束，避免迭代中某个 band 因缺向量而静默跳过样本。
    """
    row_ids = [str(row["id"]) for row in rows]
    missing = [sample_id for sample_id in row_ids if sample_id not in embeddings_by_id]
    if missing:
        raise ValueError(f"embeddings_path is missing {len(missing)} sample ids, examples: {missing[:5]}")
    bad_shape: list[tuple[str, tuple[int, ...]]] = []
    for sample_id in row_ids:
        vector = np.asarray(embeddings_by_id[sample_id], dtype=np.float32)
        if vector.ndim != 1 or (expected_dim > 0 and vector.shape[0] != int(expected_dim)):
            bad_shape.append((sample_id, tuple(vector.shape)))
    if bad_shape:
        raise ValueError(
            f"embeddings_path has vectors with invalid shape; expected ({expected_dim},), "
            f"examples: {bad_shape[:5]}"
        )


def examples_to_rows(examples: Iterable[PairExample]) -> list[dict[str, Any]]:
    """把数据加载器的 PairExample 转成算法层使用的普通行。"""
    rows: list[dict[str, Any]] = []
    for example in examples:
        row = dict(example.metadata)
        row.update(
            {
                "id": example.sample_id,
                "query": example.query,
                "document": example.document,
                "label": int(example.label),
                "groundtruth": int(example.label),
                "sample_weight": float(example.sample_weight),
            }
        )
        rows.append(row)
    return rows


def examples_from_rows(rows: Iterable[dict[str, Any]]) -> list[PairExample]:
    """把累计 teacher 标注行还原成 LoRA 训练数据。"""
    examples: list[PairExample] = []
    for row in rows:
        label = binary_to_int(row.get("label", row.get("groundtruth")), field_name="training row label")
        examples.append(
            PairExample(
                sample_id=str(row["id"]),
                query=str(row.get("query", "")),
                document=str(row.get("document", "")),
                label=label,
                sample_weight=float(row.get("sample_weight", 1.0) or 1.0),
                metadata=dict(row),
            )
        )
    return examples


def count_labels(examples: list[PairExample]) -> dict[int, int]:
    return dict(sorted(Counter(example.label for example in examples).items()))


def compute_balanced_class_weights(examples: list[PairExample]) -> dict[int, float]:
    """计算类别权重；默认不启用，仅保留为显式消融实验选项。"""
    label_counts = Counter(example.label for example in examples)
    total = sum(label_counts.values())
    if total == 0:
        return {}
    return {label: total / max(2.0 * label_counts.get(label, 0), 1.0) for label in (0, 1)}


def get_single_token_id(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(f"Expected {text!r} to encode to one token, got {token_ids}")
    return int(token_ids[0])


def build_class_token_weights(tokenizer: Any, class_weights: dict[int, float]) -> dict[int, float]:
    # CGSD 对外标签统一是 1/0，但 Qwen chat 蒸馏协议的 assistant
    # 回复仍是 yes/no token；类别权重必须绑定到实际参与 loss 的首个 token。
    label_tokens = {0: "no", 1: "yes"}
    return {get_single_token_id(tokenizer, label_tokens[int(label)]): weight for label, weight in class_weights.items()}


def build_dataloader(
    examples: list[PairExample],
    tokenizer: Any,
    *,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    pad_to_multiple_of: int,
    cache_tokenization: bool,
) -> DataLoader:
    dataset = GenerationQueryDocumentDataset(
        examples,
        tokenizer=tokenizer,
        max_length=max_length,
        cache_tokenization=cache_tokenization,
        input_format="cgsd_chat_yes_no_v1",
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": GenerationPairCollator(
            tokenizer,
            pad_to_multiple_of=pad_to_multiple_of if pad_to_multiple_of > 0 else None,
        ),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def predict_examples(
    *,
    model: torch.nn.Module,
    examples: list[PairExample],
    tokenizer: Any,
    device: Any,
    args: Any,
    predictions_path: Path | None = None,
    round_index: int | None = None,
    teacher_labels_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """对一批样本执行 student 推理并合并原始元数据。

    输出里的 `score` 是 log p(yes)-log p(no)，算法层会进一步用固定温度
    转成 CRC 需要的 routing score；写入工件的标签和预测统一是 1/0。
    如果传入真实 teacher API/logit 文件，则用 teacher 输出作为监督标签；
    否则离线实验用 groundtruth 代替真实 API，置信度固定记为 1.0。
    """
    dataloader = build_dataloader(
        examples,
        tokenizer,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory and device.type == "cuda",
        pad_to_multiple_of=args.pad_to_multiple_of,
        cache_tokenization=args.cache_tokenization,
    )
    prediction_rows = predict_model(
        model=model,
        dataloader=dataloader,
        device=device,
        tokenizer=tokenizer,
        threshold=args.threshold,
        negative_token_text="no",
        positive_token_text="yes",
        predictions_path=None,
    )
    metadata_by_id = {example.sample_id: example for example in examples}
    merged: list[dict[str, Any]] = []
    for row in prediction_rows:
        example = metadata_by_id[str(row["id"])]
        item = dict(example.metadata)
        item.update(row)
        item["query"] = example.query
        item["document"] = example.document
        item["label"] = int(example.label)
        item["groundtruth"] = int(example.label)
        item["prediction"] = binary_to_int(row["prediction"], field_name="prediction row prediction")
        item["generated_text"] = str(item["prediction"])
        if round_index is not None:
            item["round_index"] = int(round_index)
        apply_teacher_label(item, teacher_labels_by_id)
        merged.append(item)
    if predictions_path is not None:
        write_jsonl(merged, predictions_path)
    return merged


def train_round_model(
    *,
    train_examples: list[PairExample],
    eval_examples: list[PairExample],
    tokenizer: Any,
    model_path: Path,
    init_adapter_path: Path | None = None,
    output_dir: Path,
    device: Any,
    args: Any,
    round_index: int,
    run_config: dict[str, Any],
) -> QwenGenerativeModel:
    """从基座重新训练一轮 LoRA，并返回固定后的 round 模型。

    注意：这里显式传 `eval_loader=None`，不使用 D_cal 做 early stopping
    或最佳 epoch 选择。D_cal 只在模型固定后用于 CRC 校准，以满足文档中
    “训练集与校准集独立”的不变式。
    """
    train_loader = build_dataloader(
        train_examples,
        tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory and device.type == "cuda",
        pad_to_multiple_of=args.pad_to_multiple_of,
        cache_tokenization=args.cache_tokenization,
    )
    model = QwenGenerativeModel(
        model_path=str(model_path),
        mode="lora_attention_mlp",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_layer_scope=args.lora_layer_scope,
        adapter_path=init_adapter_path,
        adapters_trainable=True,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
    )
    class_weights = compute_balanced_class_weights(train_examples) if args.balance_train_classes else {}
    class_token_weights = build_class_token_weights(tokenizer, class_weights) if class_weights else {}
    round_config = dict(run_config)
    round_config.update(
        {
            "round_index": int(round_index),
            "train_size": len(train_examples),
            "eval_size": 0,
            "calibration_size_held_out": len(eval_examples),
            "calibration_used_for_training_or_model_selection": False,
            "train_label_counts": count_labels(train_examples),
            "class_weights": class_weights,
            "class_token_weights": class_token_weights,
        }
    )
    fit(
        model=model,
        train_loader=train_loader,
        eval_loader=None,
        tokenizer=tokenizer,
        output_dir=output_dir,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        threshold=args.threshold,
        run_config=round_config,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        class_token_weights=class_token_weights if args.balance_train_classes else None,
        scheduler_type="cosine",
    )
    # 训练对象保存完成后先移回 CPU，再重载落盘 adapter 作为下一轮固定模型。
    # 这样 GPU 上不会同时保留训练模型和重载后的推理模型。
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # 保存后重新加载 adapter，确保下一轮推理使用的是落盘的固定模型，
    # 而不是仍处于训练对象状态中的内存模型。
    trained_model = QwenGenerativeModel.load_from_checkpoint(
        output_dir,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        model_path=model_path,
    )
    trained_model.to(device)
    return trained_model


if __name__ == "__main__":
    raise SystemExit(
        "scripts/run_cgsd.py 只提供 CGSD 单步 CLI 共享函数。"
        "请分别运行 cgsd_prepare.py、cgsd_predict.py、cgsd_calibrate.py、"
        "cgsd_select.py、cgsd_train_round.py、cgsd_finalize.py。"
    )
