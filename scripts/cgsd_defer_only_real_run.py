#!/usr/bin/env python
"""Run a real defer-only evaluation using existing CGSD checkpoints."""

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
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_path", default="experiments/inputs/lrobench/data.jsonl")
    parser.add_argument("--model_path", default="../model/qwen3-0.6b")
    parser.add_argument("--split_ids_path", default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.07)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--query_field", default="query")
    parser.add_argument("--document_field", default="document")
    parser.add_argument("--label_field", default="groundtruth")
    add_runtime_overrides(parser)
    return parser.parse_args()


def row_id(row: dict[str, Any]) -> str:
    return str(row["id"])


def labels_and_scores(rows: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    return (
        [binary_to_int(row["label"], field_name="label") for row in rows],
        [float(row["score"]) for row in rows],
    )


def summarize(rows: list[dict[str, Any]], *, round_index: int, crc: Any, temperature: float) -> dict[str, Any]:
    labels, scores = labels_and_scores(rows)
    return {
        "round_index": int(round_index),
        "lambda_hat": float(crc.lambda_hat),
        "temperature": float(temperature),
        "crc": crc.to_dict(),
        "pool_summary": summarize_crc_decisions(rows),
        "pool_metrics": compute_binary_metrics(labels, scores, threshold=0.0),
    }


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
    model.to(device)
    return model, tokenizer


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_args = runtime_args_from_cli(args)
    configure_torch_performance(enable_tf32=runtime_args.tf32)
    device = get_device(args.device)
    model_path = (PROJECT_ROOT / args.model_path).resolve() if not Path(args.model_path).is_absolute() else Path(args.model_path)
    split_payload = read_json(Path(args.split_ids_path) if args.split_ids_path else run_dir / "cgsd_split_ids.json")

    examples = load_stage_examples(
        data_path=args.data_path,
        query_field=args.query_field,
        document_field=args.document_field,
        label_field=args.label_field,
    )
    examples_by_id = {str(example.sample_id): example for example in examples}
    calibration_examples = [examples_by_id[str(sample_id)] for sample_id in split_payload["calibration_ids"]]

    current_rows = read_jsonl(run_dir / "round_0" / "pool_crc_predictions.jsonl")
    current_by_id = {row_id(row): row for row in current_rows}
    write_jsonl(current_rows, output_dir / "round_0_pool_crc_predictions.jsonl")

    records: list[dict[str, Any]] = []
    round0_summary = read_json(run_dir / "round_0" / "round_summary.json")
    records.append(
        {
            "round": 0,
            "mode": "zero_shot_full_pool_existing_run",
            "student_model_calls": int(read_json(run_dir / "round_0" / "predict_usage.json")["student_model_calls"]),
            "student_pool_calls": len(current_rows),
            "lambda_hat": float(round0_summary["lambda_hat"]),
            **round0_summary["pool_summary"],
            "raw_accuracy": float(round0_summary["pool_metrics"]["accuracy"]),
            "macro_F1": float(round0_summary["pool_metrics"]["macro_F1"]),
            "wrong_accept_risk": float(round0_summary["pool_summary"]["wrong_accept_count"])
            / float(round0_summary["pool_summary"]["total"]),
        }
    )

    defer_only_pool_calls_after_round0 = 0
    defer_only_model_calls_after_round0 = 0

    for round_index in range(1, int(args.rounds) + 1):
        previous_defer_ids = [sample_id for sample_id, row in current_by_id.items() if bool(row.get("defer", False))]
        defer_examples = [examples_by_id[sample_id] for sample_id in previous_defer_ids]
        prediction_examples = calibration_examples + defer_examples

        model, tokenizer = load_model_and_tokenizer(
            checkpoint_dir=run_dir / f"round_{round_index}" / "model",
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
            predictions_path=output_dir / f"round_{round_index}_real_subset_predictions.jsonl",
            round_index=round_index,
            teacher_labels_by_id={},
        )
        model.to("cpu")

        by_id = {row_id(row): row for row in predictions}
        calibration_predictions = [by_id[str(sample_id)] for sample_id in split_payload["calibration_ids"]]
        rerouted_rows = [by_id[sample_id] for sample_id in previous_defer_ids]
        crc = calibrate_crc(calibration_predictions, alpha=float(args.alpha), temperature=float(args.temperature))
        rerouted_decisions = apply_crc_decisions(
            rerouted_rows,
            lambda_hat=float(crc.lambda_hat),
            temperature=float(args.temperature),
        )
        rerouted_by_id = {row_id(row): row for row in rerouted_decisions}

        merged_rows: list[dict[str, Any]] = []
        newly_accepted = 0
        for sample_id, previous_row in current_by_id.items():
            if sample_id in rerouted_by_id:
                updated = dict(rerouted_by_id[sample_id])
                updated["defer_only_source"] = f"round_{round_index}_real_rerouted_from_previous_defer"
                if not bool(updated.get("defer", False)):
                    newly_accepted += 1
                merged_rows.append(updated)
            else:
                frozen = dict(previous_row)
                frozen["defer_only_source"] = "frozen_previous_accept"
                merged_rows.append(frozen)

        summary = summarize(merged_rows, round_index=round_index, crc=crc, temperature=float(args.temperature))
        write_json(summary, output_dir / f"round_{round_index}_summary.json")
        write_jsonl(merged_rows, output_dir / f"round_{round_index}_pool_crc_predictions.jsonl")

        student_model_calls = len(predictions)
        defer_only_model_calls_after_round0 += student_model_calls
        defer_only_pool_calls_after_round0 += len(previous_defer_ids)
        pool_summary = summary["pool_summary"]
        teacher_usage = summarize_teacher_label_usage(predictions, purpose="defer_only_real_prediction_label_attachment")
        records.append(
            {
                "round": round_index,
                "mode": "defer_only_real_run",
                "lambda_hat": float(summary["lambda_hat"]),
                "student_model_calls": int(student_model_calls),
                "student_calibration_calls": len(calibration_examples),
                "student_pool_calls": len(previous_defer_ids),
                "estimated_student_prompt_tokens": estimate_query_document_prompt_tokens(predictions),
                "teacher_label_usage": teacher_usage,
                "frozen_previous_accept_count": len(current_by_id) - len(previous_defer_ids),
                "newly_accepted_from_previous_defer": newly_accepted,
                **pool_summary,
                "raw_accuracy": float(summary["pool_metrics"]["accuracy"]),
                "macro_F1": float(summary["pool_metrics"]["macro_F1"]),
                "wrong_accept_risk": float(pool_summary["wrong_accept_count"]) / float(pool_summary["total"]),
            }
        )
        current_by_id = {row_id(row): row for row in merged_rows}

    full_run_calls = [
        int(read_json(run_dir / f"round_{round_index}" / "predict_usage.json")["student_model_calls"])
        for round_index in range(0, int(args.rounds) + 1)
    ]
    pool_size = len(current_rows)
    full_pool_calls_after_round0 = pool_size * int(args.rounds)
    summary_payload = {
        "run_dir": str(run_dir),
        "mode": "defer_only_real_run",
        "rounds": int(args.rounds),
        "alpha": float(args.alpha),
        "temperature": float(args.temperature),
        "pool_size": pool_size,
        "full_run_student_model_calls_total": sum(full_run_calls),
        "defer_only_student_model_calls_total_including_existing_round0": int(
            full_run_calls[0] + defer_only_model_calls_after_round0
        ),
        "model_call_savings_total": int(sum(full_run_calls) - (full_run_calls[0] + defer_only_model_calls_after_round0)),
        "full_pool_calls_after_round0": int(full_pool_calls_after_round0),
        "defer_only_pool_calls_after_round0": int(defer_only_pool_calls_after_round0),
        "pool_call_savings_after_round0": int(full_pool_calls_after_round0 - defer_only_pool_calls_after_round0),
        "records": records,
    }
    write_json(summary_payload, output_dir / "defer_only_real_summary.json")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
