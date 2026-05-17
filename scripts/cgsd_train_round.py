#!/usr/bin/env python
"""用累计选中样本训练单个 CGSD LoRA round。

训练输入是累计的 `cgsd_train_rows.jsonl`，而不是仅本轮新增样本。
round 0 表示未训练基座模型，因此本脚本只允许训练 round >= 1。
"""

from __future__ import annotations

import argparse
import json
import math
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
    load_stage_examples,
    output_dir_from_arg,
    output_artifact_path,
    print_existing_stage_result,
    read_jsonl,
    runtime_args_from_cli,
    stage_cache_decision,
    summarize_teacher_label_usage,
    train_label_snapshot,
    write_stage_usage,
)
from scripts.run_cgsd import examples_from_rows, train_round_model
from src.data import filter_examples_by_ids
from src.utils import (
    configure_torch_performance,
    disable_tokenizer_thinking,
    ensure_tokenizer_padding,
    get_device,
    read_json,
    write_json,
    write_jsonl,
)
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_index", type=int, required=True, help="要产出的 LoRA round，例如 select round0 后训练 round1")
    parser.add_argument("--model_path", default="model/qwen3-0.6b")
    parser.add_argument("--data_path", default="datasets")
    parser.add_argument("--query_field", default="query")
    parser.add_argument("--document_field", default="document")
    parser.add_argument("--label_field", default="groundtruth")
    parser.add_argument("--split_ids_path", default=None)
    parser.add_argument("--train_rows_path", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--init_adapter_path", default=None)
    parser.add_argument("--training_rows_used_path", default=None)
    parser.add_argument("--train_label_snapshot_path", default=None)
    parser.add_argument("--training_summary_path", default=None)
    parser.add_argument("--usage_path", default=None)
    add_runtime_overrides(parser)
    add_stage_cache_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.round_index) <= 0:
        raise ValueError("round 0 is the base model and has no LoRA checkpoint; train round_index must be >= 1")
    output_dir = output_dir_from_arg(args.output_dir)
    round_dir = output_dir / f"round_{args.round_index}"
    train_rows_path = input_artifact_path(args.train_rows_path, output_dir / "cgsd_train_rows.jsonl")
    checkpoint_dir = output_artifact_path(args.checkpoint_dir, round_dir / "model")
    training_rows_used_path = output_artifact_path(
        args.training_rows_used_path,
        round_dir / "training_rows_used.jsonl",
    )
    train_label_snapshot_path = output_artifact_path(
        args.train_label_snapshot_path,
        round_dir / "train_label_snapshot.json",
    )
    training_summary_path = output_artifact_path(
        args.training_summary_path,
        round_dir / "training_round_summary.json",
    )
    usage_path = output_artifact_path(args.usage_path, round_dir / "train_usage.json")
    if args.show_result:
        print_existing_stage_result(stage_name="cgsd_train_round", summary_path=training_summary_path)
        return
    cache_decision = stage_cache_decision(
        stage_name="cgsd_train_round",
        required_outputs=[
            checkpoint_dir / "model_config.json",
            training_rows_used_path,
            train_label_snapshot_path,
            training_summary_path,
            usage_path,
        ],
        cache_policy=args.cache_policy,
    )
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name="cgsd_train_round", summary_path=training_summary_path)
        return

    round_dir.mkdir(parents=True, exist_ok=True)
    runtime_args = runtime_args_from_cli(args)
    configure_torch_performance(enable_tf32=runtime_args.tf32)
    # 这里读取累计训练集：第一轮 250，后续轮会包含之前轮次已选样本。
    selected_rows = read_jsonl(train_rows_path) if args.train_rows_path else load_selected_train_rows(output_dir)
    if not selected_rows:
        raise RuntimeError(f"{train_rows_path} is empty; run cgsd_select.py first or pass --train_rows_path")
    training_rows_snapshot = [
        {
            **dict(row),
            "trained_for_round": int(args.round_index),
            "training_artifact_role": "cumulative_lora_train_row",
        }
        for row in selected_rows
    ]
    write_jsonl(training_rows_snapshot, training_rows_used_path)

    model_path = input_artifact_path(args.model_path, PROJECT_ROOT / "model" / "qwen3-0.6b")
    init_adapter_path = input_artifact_path(args.init_adapter_path, output_dir) if args.init_adapter_path else None
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=runtime_args.trust_remote_code,
        local_files_only=True,
    )
    ensure_tokenizer_padding(tokenizer)
    disable_tokenizer_thinking(tokenizer)

    train_examples = examples_from_rows(selected_rows)
    all_examples = load_stage_examples(
        data_path=args.data_path,
        query_field=args.query_field,
        document_field=args.document_field,
        label_field=args.label_field,
    )
    split_payload = read_json(input_artifact_path(args.split_ids_path, output_dir / "cgsd_split_ids.json"))
    calibration_ids = set(split_payload["calibration_ids"])
    calibration_examples = filter_examples_by_ids(all_examples, calibration_ids)
    device = get_device(args.device)
    model = train_round_model(
        train_examples=train_examples,
        eval_examples=calibration_examples,
        tokenizer=tokenizer,
        model_path=model_path,
        init_adapter_path=init_adapter_path,
        output_dir=checkpoint_dir,
        device=device,
        args=runtime_args,
        round_index=args.round_index,
        run_config=vars(args),
    )
    model.to("cpu")
    label_snapshot = train_label_snapshot(selected_rows)
    selection_round_counts: dict[str, int] = {}
    for row in selected_rows:
        key = str(row.get("selection_round", "unknown"))
        selection_round_counts[key] = selection_round_counts.get(key, 0) + 1
    write_json(label_snapshot, train_label_snapshot_path)
    write_json(
        {
            "round_index": int(args.round_index),
            "model_round_index": int(args.round_index),
            "checkpoint_dir": str(checkpoint_dir),
            "training_rows_path": str(training_rows_used_path),
            "train_size": len(selected_rows),
            "train_label_snapshot_size": len(label_snapshot),
            "selection_round_counts": selection_round_counts,
            "source_selection_rounds": sorted(selection_round_counts),
            "training_mode": "lora_sft_from_base_model",
            "init_adapter_path": str(init_adapter_path) if init_adapter_path is not None else None,
        },
        training_summary_path,
    )
    teacher_usage = summarize_teacher_label_usage(selected_rows, purpose="training_label_reuse")
    train_steps = int(math.ceil(len(selected_rows) / max(int(runtime_args.batch_size), 1)) * int(runtime_args.epochs))
    write_stage_usage(
        usage_path,
        {
            "stage_name": "cgsd_train_round",
            "round_index": int(args.round_index),
            "cache": cache_decision.to_dict(),
            "student_model_calls": 0,
            "student_model_train_steps_estimated": train_steps,
            "student_model_train_examples": int(len(selected_rows) * int(runtime_args.epochs)),
            "estimated_student_train_tokens": int(
                estimate_query_document_prompt_tokens(selected_rows) * int(runtime_args.epochs)
            ),
            "teacher_label_usage": teacher_usage,
            "groundtruth_substitute_calls": teacher_usage["groundtruth_substitute_calls"],
            "teacher_api_file_calls": teacher_usage["teacher_api_file_calls"],
            "checkpoint_dir": str(checkpoint_dir),
        },
    )
    print(
        json.dumps(
            {
                "round_index": args.round_index,
                "checkpoint_dir": str(checkpoint_dir),
                "train_size": len(selected_rows),
                "selection_round_counts": selection_round_counts,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
