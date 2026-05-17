#!/usr/bin/env python
"""只对上一轮 defer 集执行本轮预测和 CRC 校准。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.cgsd import apply_crc_decisions, calibrate_crc, summarize_crc_decisions
from scripts.cgsd_cli_common import (
    add_runtime_overrides,
    binary_to_int,
    estimate_query_document_prompt_tokens,
    load_stage_examples,
    read_jsonl,
    runtime_args_from_cli,
    summarize_teacher_label_usage,
    write_stage_usage,
)
from scripts.run_cgsd import predict_examples
from src.metrics import compute_binary_metrics
from src.model import QwenGenerativeModel
from src.utils import (
    configure_torch_performance,
    disable_tokenizer_thinking,
    ensure_tokenizer_padding,
    get_device,
    parse_torch_dtype,
    read_json,
    write_json,
    write_jsonl,
)
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_index", type=int, required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--split_ids_path", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=15.0)
    parser.add_argument("--query_field", default="query")
    parser.add_argument("--document_field", default="document")
    parser.add_argument("--label_field", default="groundtruth")
    add_runtime_overrides(parser)
    return parser.parse_args()


def row_id(row: dict[str, Any]) -> str:
    return str(row["id"])


def load_model_and_tokenizer(
    *,
    checkpoint_dir: Path,
    model_path: Path,
    runtime_args: Any,
    device: Any,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir if checkpoint_dir.exists() else model_path,
        trust_remote_code=runtime_args.trust_remote_code,
        local_files_only=True,
    )
    ensure_tokenizer_padding(tokenizer)
    disable_tokenizer_thinking(tokenizer)
    model = QwenGenerativeModel.load_from_checkpoint(
        checkpoint_dir,
        torch_dtype=parse_torch_dtype(runtime_args.torch_dtype),
        model_path=model_path,
    )
    if hasattr(model.backbone, "merge_and_unload"):
        model.backbone = model.backbone.merge_and_unload()
    model.to(device)
    return model, tokenizer


def labels_and_scores(rows: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    return (
        [binary_to_int(row["label"], field_name="label") for row in rows],
        [float(row["score"]) for row in rows],
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    round_index = int(args.round_index)
    if round_index <= 0:
        raise ValueError("--round_index must be > 0 for defer-only prediction")
    previous_round = round_index - 1
    round_dir = output_dir / f"round_{round_index}"
    previous_dir = output_dir / f"round_{previous_round}"
    round_dir.mkdir(parents=True, exist_ok=True)

    runtime_args = runtime_args_from_cli(args)
    configure_torch_performance(enable_tf32=runtime_args.tf32)
    device = get_device(args.device)
    model_path = Path(args.model_path)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else round_dir / "model"
    split_payload = read_json(Path(args.split_ids_path) if args.split_ids_path else output_dir / "cgsd_split_ids.json")

    examples = load_stage_examples(
        data_path=args.data_path,
        query_field=args.query_field,
        document_field=args.document_field,
        label_field=args.label_field,
    )
    examples_by_id = {str(example.sample_id): example for example in examples}
    calibration_ids = [str(sample_id) for sample_id in split_payload["calibration_ids"]]
    calibration_examples = [examples_by_id[sample_id] for sample_id in calibration_ids]

    previous_pool_rows = read_jsonl(previous_dir / "pool_crc_predictions.jsonl")
    previous_by_id = {row_id(row): row for row in previous_pool_rows}
    previous_defer_ids = [sample_id for sample_id, row in previous_by_id.items() if bool(row.get("defer", False))]
    prediction_examples = calibration_examples + [examples_by_id[sample_id] for sample_id in previous_defer_ids]

    model, tokenizer = load_model_and_tokenizer(
        checkpoint_dir=checkpoint_dir,
        model_path=model_path,
        runtime_args=runtime_args,
        device=device,
    )
    predictions = predict_examples(
        model=model,
        examples=prediction_examples,
        tokenizer=tokenizer,
        device=device,
        args=runtime_args,
        predictions_path=round_dir / "defer_only_subset_predictions.jsonl",
        round_index=round_index,
        teacher_labels_by_id={},
    )
    by_id = {row_id(row): row for row in predictions}
    calibration_predictions = [by_id[sample_id] for sample_id in calibration_ids]
    rerouted_rows = [by_id[sample_id] for sample_id in previous_defer_ids]
    crc = calibrate_crc(calibration_predictions, alpha=float(args.alpha), temperature=float(args.temperature))
    rerouted_decisions = apply_crc_decisions(
        rerouted_rows,
        lambda_hat=float(crc.lambda_hat),
        temperature=float(args.temperature),
    )
    rerouted_by_id = {row_id(row): row for row in rerouted_decisions}

    merged_pool_rows: list[dict[str, Any]] = []
    newly_accepted = 0
    for sample_id, previous_row in previous_by_id.items():
        if sample_id in rerouted_by_id:
            updated = dict(rerouted_by_id[sample_id])
            updated["defer_only_source"] = f"round_{round_index}_rerouted_from_previous_defer"
            if not bool(updated.get("defer", False)):
                newly_accepted += 1
            merged_pool_rows.append(updated)
        else:
            frozen = dict(previous_row)
            frozen["defer_only_source"] = "frozen_previous_accept"
            merged_pool_rows.append(frozen)

    labels, scores = labels_and_scores(merged_pool_rows)
    pool_summary = summarize_crc_decisions(merged_pool_rows)
    record = {
        "round_index": round_index,
        "mode": "defer_only_prediction",
        "previous_round_index": previous_round,
        "temperature": float(args.temperature),
        "lambda_hat": float(crc.lambda_hat),
        "crc": crc.to_dict(),
        "pool_summary": pool_summary,
        "pool_metrics": compute_binary_metrics(labels, scores, threshold=0.0),
        "defer_only": {
            "previous_defer_count": len(previous_defer_ids),
            "student_model_calls": len(predictions),
            "calibration_model_calls": len(calibration_predictions),
            "rerouted_pool_model_calls": len(rerouted_rows),
            "frozen_previous_accept_count": len(previous_by_id) - len(previous_defer_ids),
            "newly_accepted_from_previous_defer": newly_accepted,
        },
    }

    write_jsonl(predictions, round_dir / "all_student_predictions.jsonl")
    write_jsonl(calibration_predictions, round_dir / "calibration_student_predictions.jsonl")
    write_jsonl([], round_dir / "final_calibration_student_predictions.jsonl")
    write_jsonl(merged_pool_rows, round_dir / "pool_student_predictions.jsonl")
    write_jsonl(merged_pool_rows, round_dir / "pool_crc_predictions.jsonl")
    write_json(record, round_dir / "round_summary.json")
    teacher_usage = summarize_teacher_label_usage(predictions, purpose="defer_only_prediction_teacher_labels")
    write_stage_usage(
        round_dir / "predict_usage.json",
        {
            "stage_name": "cgsd_predict_defer_only_round",
            "round_index": round_index,
            "student_model_calls": len(predictions),
            "student_model_role": "round_lora_adapter_defer_only",
            "previous_defer_count": len(previous_defer_ids),
            "rerouted_pool_model_calls": len(rerouted_rows),
            "frozen_previous_accept_count": len(previous_by_id) - len(previous_defer_ids),
            "estimated_student_prompt_tokens": estimate_query_document_prompt_tokens(predictions),
            "teacher_label_usage": teacher_usage,
            "groundtruth_substitute_calls": teacher_usage["groundtruth_substitute_calls"],
            "teacher_api_file_calls": teacher_usage["teacher_api_file_calls"],
        },
    )
    write_stage_usage(
        round_dir / "calibrate_usage.json",
        {
            "stage_name": "cgsd_calibrate_defer_only_round",
            "round_index": round_index,
            "student_model_calls": 0,
            "crc_calibration_rows": len(calibration_predictions),
            "pool_decision_rows": len(merged_pool_rows),
            "pool_crc_predictions_path": str(round_dir / "pool_crc_predictions.jsonl"),
            "round_summary_path": str(round_dir / "round_summary.json"),
        },
    )
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
