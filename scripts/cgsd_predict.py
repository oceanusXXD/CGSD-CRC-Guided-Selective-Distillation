#!/usr/bin/env python
"""对单个 CGSD round 执行非 vLLM student 预测。

该脚本直接在当前 Python 进程中加载 Hugging Face 基座模型或 LoRA
checkpoint，输出格式与 vLLM 预测脚本一致。它适合调试和较小规模评估；
全量 pool 高吞吐推理优先使用 `cgsd_predict_vllm_openai.py`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cgsd_cli_common import (
    add_runtime_overrides,
    add_stage_cache_args,
    estimate_query_document_prompt_tokens,
    input_artifact_path,
    load_selected_train_rows,
    load_split_ids,
    load_stage_examples,
    load_teacher_labels,
    output_dir_from_arg,
    output_artifact_path,
    print_existing_stage_result,
    read_jsonl,
    runtime_args_from_cli,
    stage_cache_decision,
    split_examples,
    summarize_teacher_label_usage,
    train_label_snapshot,
    write_stage_usage,
)
from scripts.run_cgsd import predict_examples
from src.model import QwenGenerativeModel, set_use_cache_false
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
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_index", type=int, required=True)
    parser.add_argument("--model_path", default="model/qwen3-0.6b")
    parser.add_argument("--data_path", default="datasets")
    parser.add_argument("--query_field", default="query")
    parser.add_argument("--document_field", default="document")
    parser.add_argument("--label_field", default="groundtruth")
    parser.add_argument("--split_ids_path", default=None)
    parser.add_argument("--selected_train_rows_path", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--teacher_labels_path", default=None)
    parser.add_argument("--teacher_temperature", type=float, default=1.0)
    parser.add_argument("--all_predictions_path", default=None)
    parser.add_argument("--partial_predictions_path", default=None)
    parser.add_argument("--calibration_predictions_path", default=None)
    parser.add_argument("--final_calibration_predictions_path", default=None)
    parser.add_argument("--pool_predictions_path", default=None)
    parser.add_argument("--train_label_snapshot_path", default=None)
    parser.add_argument("--usage_path", default=None)
    add_runtime_overrides(parser)
    add_stage_cache_args(parser)
    return parser.parse_args()


def checkpoint_for_round(output_dir: Path, round_index: int, explicit: str | None) -> Path | None:
    if explicit:
        return input_artifact_path(explicit, output_dir / f"round_{round_index}" / "model")
    if round_index <= 0:
        return None
    return output_dir / f"round_{round_index}" / "model"


def main() -> None:
    args = parse_args()
    output_dir = output_dir_from_arg(args.output_dir)
    round_dir = output_dir / f"round_{args.round_index}"
    all_predictions_path = output_artifact_path(
        args.all_predictions_path,
        round_dir / "all_student_predictions.jsonl",
    )
    calibration_predictions_path = output_artifact_path(
        args.calibration_predictions_path,
        round_dir / "calibration_student_predictions.jsonl",
    )
    pool_predictions_path = output_artifact_path(
        args.pool_predictions_path,
        round_dir / "pool_student_predictions.jsonl",
    )
    final_calibration_predictions_path = output_artifact_path(
        args.final_calibration_predictions_path,
        round_dir / "final_calibration_student_predictions.jsonl",
    )
    train_label_snapshot_path = output_artifact_path(
        args.train_label_snapshot_path,
        round_dir / "predict_train_label_snapshot.json",
    )
    usage_path = output_artifact_path(args.usage_path, round_dir / "predict_usage.json")
    partial_predictions_path = output_artifact_path(
        args.partial_predictions_path,
        round_dir / "all_student_predictions.partial.jsonl",
    )
    if args.show_result:
        print_existing_stage_result(stage_name="cgsd_predict", summary_path=usage_path)
        return
    cache_decision = stage_cache_decision(
        stage_name="cgsd_predict",
        required_outputs=[
            all_predictions_path,
            calibration_predictions_path,
            final_calibration_predictions_path,
            pool_predictions_path,
            train_label_snapshot_path,
            usage_path,
        ],
        cache_policy=args.cache_policy,
    )
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name="cgsd_predict", summary_path=usage_path)
        return
    if args.cache_policy == "overwrite" and partial_predictions_path.exists():
        partial_predictions_path.unlink()

    round_dir.mkdir(parents=True, exist_ok=True)
    runtime_args = runtime_args_from_cli(args)
    split_payload = (
        read_json(input_artifact_path(args.split_ids_path, output_dir / "cgsd_split_ids.json"))
        if args.split_ids_path
        else load_split_ids(output_dir)
    )
    configure_torch_performance(enable_tf32=runtime_args.tf32)

    device = get_device(args.device)
    model_path = input_artifact_path(args.model_path, PROJECT_ROOT / "model" / "qwen3-0.6b")
    checkpoint_dir = checkpoint_for_round(output_dir, args.round_index, args.checkpoint_dir)
    tokenizer_source = checkpoint_dir if checkpoint_dir is not None and checkpoint_dir.exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=runtime_args.trust_remote_code,
        local_files_only=True,
    )
    ensure_tokenizer_padding(tokenizer)
    disable_tokenizer_thinking(tokenizer)

    if checkpoint_dir is None:
        # 第 0 轮是未蒸馏的基座模型评估，不加载 LoRA adapter。
        # 后续轮次才加载上一条训练 CLI 保存的 LoRA checkpoint。
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=parse_torch_dtype(runtime_args.torch_dtype),
            trust_remote_code=runtime_args.trust_remote_code,
            local_files_only=True,
        )
        set_use_cache_false(model)
    else:
        model = QwenGenerativeModel.load_from_checkpoint(
            checkpoint_dir,
            torch_dtype=parse_torch_dtype(runtime_args.torch_dtype),
            model_path=model_path,
        )
        if hasattr(model.backbone, "merge_and_unload"):
            model.backbone = model.backbone.merge_and_unload()
    model.to(device)

    examples = load_stage_examples(
        data_path=args.data_path,
        query_field=args.query_field,
        document_field=args.document_field,
        label_field=args.label_field,
    )
    # 每轮都写出三份固定切分：D_guide 用于中间 CRC，D_cert 只保留给
    # 最终认证，pool 用于部署候选和 active selection。
    calibration_examples, pool_examples = split_examples(examples, split_payload)
    examples_by_id = {str(example.sample_id): example for example in examples}
    final_calibration_ids = [str(sample_id) for sample_id in split_payload.get("final_calibration_ids", [])]
    final_calibration_examples = [examples_by_id[sample_id] for sample_id in final_calibration_ids]
    prediction_examples = calibration_examples + final_calibration_examples + pool_examples
    teacher_labels_by_id = (
        load_teacher_labels(
            args.teacher_labels_path,
            teacher_temperature=float(args.teacher_temperature),
        )
        if args.teacher_labels_path
        else {}
    )

    selected_train_rows = (
        read_jsonl(input_artifact_path(args.selected_train_rows_path, output_dir / "cgsd_train_rows.jsonl"))
        if args.selected_train_rows_path
        else load_selected_train_rows(output_dir)
    )
    write_json(train_label_snapshot(selected_train_rows), train_label_snapshot_path)
    predictions = predict_examples(
        model=model,
        examples=prediction_examples,
        tokenizer=tokenizer,
        device=device,
        args=runtime_args,
        predictions_path=all_predictions_path,
        partial_predictions_path=partial_predictions_path,
        round_index=args.round_index,
        teacher_labels_by_id=teacher_labels_by_id,
    )
    by_id = {str(row["id"]): row for row in predictions}
    calibration_predictions = [by_id[str(sample_id)] for sample_id in split_payload["calibration_ids"]]
    final_calibration_predictions = [by_id[str(sample_id)] for sample_id in final_calibration_ids]
    pool_predictions = [by_id[str(sample_id)] for sample_id in split_payload["pool_ids"]]
    write_jsonl(calibration_predictions, calibration_predictions_path)
    write_jsonl(final_calibration_predictions, final_calibration_predictions_path)
    write_jsonl(pool_predictions, pool_predictions_path)
    teacher_usage = summarize_teacher_label_usage(predictions, purpose="predict_teacher_label_attachment")
    write_stage_usage(
        usage_path,
        {
            "stage_name": "cgsd_predict",
            "round_index": int(args.round_index),
            "cache": cache_decision.to_dict(),
            "student_model_calls": len(predictions),
            "student_model_role": "base_model" if checkpoint_dir is None else "round_lora_adapter",
            "estimated_student_prompt_tokens": estimate_query_document_prompt_tokens(predictions),
            "estimated_student_completion_tokens": len(predictions),
            "teacher_label_usage": teacher_usage,
            "groundtruth_substitute_calls": teacher_usage["groundtruth_substitute_calls"],
            "teacher_api_file_calls": teacher_usage["teacher_api_file_calls"],
            "all_predictions_path": str(all_predictions_path),
            "partial_predictions_path": str(partial_predictions_path),
            "calibration_predictions_path": str(calibration_predictions_path),
            "final_calibration_predictions_path": str(final_calibration_predictions_path),
            "pool_predictions_path": str(pool_predictions_path),
        },
    )
    print(json.dumps({"round_index": args.round_index, "predicted": len(predictions)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
