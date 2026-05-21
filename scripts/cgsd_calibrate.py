#!/usr/bin/env python
"""校准单个 CGSD round 的 CRC 阈值。

输入是预测脚本写出的 `calibration_student_predictions.jsonl` 和
`pool_student_predictions.jsonl`。本脚本先在 D_guide 上扫描阈值
`lambda`，选择满足有限样本修正风险 `<= alpha` 的最小可行阈值，
再把同一阈值应用到 pool，写出 accept/defer 决策。
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
    add_stage_cache_args,
    binary_to_int,
    estimate_query_document_prompt_tokens,
    input_artifact_path,
    output_dir_from_arg,
    output_artifact_path,
    print_existing_stage_result,
    read_jsonl,
    stage_cache_decision,
    summarize_teacher_label_usage,
    write_stage_usage,
)
from algorithms.cgsd import (
    DEFAULT_TEMPERATURE_GRID,
    apply_crc_decisions,
    calibrate_crc,
    choose_temperature,
    crc_margin_cutoff,
    summarize_crc_decisions,
)
from scripts.run_cgsd import assert_embedding_coverage, load_embeddings
from src.metrics import compute_binary_metrics
from src.utils import read_json, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_index", type=int, required=True)
    parser.add_argument("--calibration_predictions_path", default=None)
    parser.add_argument("--pool_predictions_path", default=None)
    parser.add_argument("--pool_crc_predictions_path", default=None)
    parser.add_argument("--round_summary_path", default=None)
    parser.add_argument("--previous_round_summary_path", default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--temperatures", default=",".join(str(value) for value in DEFAULT_TEMPERATURE_GRID))
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=15.0)
    parser.add_argument("--embeddings_path", default=None)
    parser.add_argument("--embedding_dim", type=int, default=0)
    parser.add_argument("--usage_path", default=None)
    add_stage_cache_args(parser)
    return parser.parse_args()


def parse_float_csv(text: str) -> list[float]:
    """解析 CLI 传入的温度网格。"""
    values = [float(part.strip()) for part in str(text or "").split(",") if part.strip()]
    if not values:
        raise ValueError("--temperatures cannot be empty")
    return values


def _decision_error_count(rows: list[dict[str, object]]) -> int:
    count = 0
    for row in rows:
        prediction = binary_to_int(row.get("prediction", int(float(row.get("score", 0.0) or 0.0) > 0.0)), field_name="calibration prediction")
        label = binary_to_int(row.get("label", row.get("groundtruth")), field_name="calibration label")
        count += int(prediction != label)
    return count


def compute_crc_sampling_statistics(
    calibration_decisions: list[dict[str, object]],
    pool_decisions: list[dict[str, object]],
    *,
    temperature: float,
    lambda_hat: float,
) -> dict[str, float]:
    """计算和预算无关的 CRC Error-Mass 诊断量。

    这些字段描述当前 CRC 划分和 guide 错误浓缩度；真正的样本数量只在
    selection stage 根据显式 `--budget` 计算。
    """
    n_calibration = len(calibration_decisions)
    if n_calibration == 0:
        raise ValueError("calibration decisions cannot be empty")
    pool_total = len(pool_decisions)
    pool_defer_count = sum(1 for row in pool_decisions if bool(row.get("defer", False)))
    calibration_defer_rows = [row for row in calibration_decisions if bool(row.get("defer", False))]
    calibration_error_count = _decision_error_count(calibration_decisions)
    calibration_defer_error_count = _decision_error_count(calibration_defer_rows)
    r_u = float(pool_defer_count / pool_total) if pool_total else 0.0
    r_c = float(len(calibration_defer_rows) / n_calibration)
    e_all = float(calibration_error_count / n_calibration)
    e_defer = (
        float(calibration_defer_error_count / len(calibration_defer_rows))
        if calibration_defer_rows
        else 0.0
    )
    if not calibration_defer_rows or calibration_error_count == 0 or e_all <= 0.0:
        c_crc = 1.0
        eta_crc = 0.0
    else:
        c_crc = float(e_defer / e_all)
        if c_crc <= 1.0 or r_c <= 0.0 or r_c >= 1.0:
            eta_crc = 0.0
        else:
            eta_crc = max(0.0, min(1.0, math.log(c_crc) / math.log(1.0 / r_c)))
    s_defer = max(0.0, min(1.0, float(r_u + eta_crc * ((1.0 - r_u) ** 2))))
    return {
        "tau_crc": float(crc_margin_cutoff(lambda_hat, temperature)),
        "r_U": r_u,
        "r_C": r_c,
        "e_all": e_all,
        "e_defer": e_defer,
        "c_crc": float(c_crc),
        "eta_crc": float(eta_crc),
        "s_accept": float(1.0 - s_defer),
        "s_defer": s_defer,
    }


def fixed_temperature_for_round(
    *,
    round_index: int,
    explicit_temperature: float | None,
    previous_round_summary_path: Path | None,
) -> tuple[float | None, str]:
    """解析后续 round 必须复用的固定温度。

    严格实验应显式传 `--temperature 15`。round0 保留温度网格搜索只为
    兼容诊断实验；round1 及以后只能复用已固定温度，避免后续轮次
    悄悄重新扫描温度并改变 routing score 口径。
    """
    if explicit_temperature is not None:
        return float(explicit_temperature), "cli_arg"
    if int(round_index) == 0:
        return None, "round0_temperature_search"
    if previous_round_summary_path is not None and previous_round_summary_path.exists():
        previous_summary = read_json(previous_round_summary_path)
        if "temperature" in previous_summary:
            return float(previous_summary["temperature"]), "previous_round_summary"
    raise RuntimeError(
        "round_index > 0 requires a fixed temperature from --temperature or the previous round_summary.json"
    )


def main() -> None:
    args = parse_args()
    output_dir = output_dir_from_arg(args.output_dir)
    round_dir = output_dir / f"round_{args.round_index}"
    calibration_predictions_path = input_artifact_path(
        args.calibration_predictions_path,
        round_dir / "calibration_student_predictions.jsonl",
    )
    pool_predictions_path = input_artifact_path(args.pool_predictions_path, round_dir / "pool_student_predictions.jsonl")
    pool_crc_predictions_path = output_artifact_path(
        args.pool_crc_predictions_path,
        round_dir / "pool_crc_predictions.jsonl",
    )
    round_summary_path = output_artifact_path(args.round_summary_path, round_dir / "round_summary.json")
    usage_path = output_artifact_path(args.usage_path, round_dir / "calibrate_usage.json")
    if args.show_result:
        print_existing_stage_result(stage_name="cgsd_calibrate", summary_path=round_summary_path)
        return
    cache_decision = stage_cache_decision(
        stage_name="cgsd_calibrate",
        required_outputs=[pool_crc_predictions_path, round_summary_path, usage_path],
        cache_policy=args.cache_policy,
    )
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name="cgsd_calibrate", summary_path=round_summary_path)
        return

    calibration_predictions = read_jsonl(calibration_predictions_path)
    pool_predictions = read_jsonl(pool_predictions_path)
    # CRC 一律启用 neighbor-support：D_guide 既用于中间阈值校准，也作为
    # pool 决策时的局部支持参考库。没有 embedding 直接失败，避免旧的
    # global-threshold CRC 混入实验。D_cert 保存在
    # final_calibration_student_predictions.jsonl，只能用于最终认证阶段。
    if not args.embeddings_path:
        raise ValueError("--embeddings_path is required because all CRC uses neighbor-support")
    embeddings_path = input_artifact_path(args.embeddings_path, PROJECT_ROOT / str(args.embeddings_path))
    embeddings_by_id = load_embeddings(embeddings_path)
    assert_embedding_coverage(
        embeddings_by_id,
        [*calibration_predictions, *pool_predictions],
        expected_dim=int(args.embedding_dim),
    )

    previous_summary_path = (
        input_artifact_path(
            args.previous_round_summary_path,
            output_dir / f"round_{args.round_index - 1}" / "round_summary.json",
        )
        if args.previous_round_summary_path
        else output_dir / f"round_{args.round_index - 1}" / "round_summary.json"
    )
    fixed_temperature, temperature_source = fixed_temperature_for_round(
        round_index=int(args.round_index),
        explicit_temperature=args.temperature,
        previous_round_summary_path=previous_summary_path,
    )
    if fixed_temperature is not None:
        temperature = float(fixed_temperature)
        crc_result = calibrate_crc(
            calibration_predictions,
            alpha=float(args.alpha),
            temperature=temperature,
            embeddings_by_id=embeddings_by_id,
        )
        temperature_payload = {
            "temperature": temperature,
            "crc": crc_result.to_dict(),
            "source": temperature_source,
        }
    else:
        temperature_choice = choose_temperature(
            calibration_predictions,
            pool_predictions,
            alpha=float(args.alpha),
            temperatures=parse_float_csv(args.temperatures),
            embeddings_by_id=embeddings_by_id,
        )
        temperature = float(temperature_choice.temperature)
        crc_result = temperature_choice.crc
        temperature_payload = temperature_choice.to_dict()
        temperature_payload["source"] = temperature_source

    # apply_crc_decisions 会重新计算 routing_score，并在 NS 开启时为每条
    # pool 样本写出自适应 decision_threshold。
    pool_decisions = apply_crc_decisions(
        pool_predictions,
        lambda_hat=crc_result.lambda_hat,
        temperature=temperature,
        embeddings_by_id=embeddings_by_id,
        # 中间轮的 neighbor support 以 D_guide 为参考库。
        # 不要替换成 D_cert；D_cert 必须保留到最终认证阶段以保持隔离。
        support_rows=calibration_predictions,
        crc_result=crc_result,
    )
    calibration_decisions = apply_crc_decisions(
        calibration_predictions,
        lambda_hat=crc_result.lambda_hat,
        temperature=temperature,
        embeddings_by_id=embeddings_by_id,
        support_rows=calibration_predictions,
        crc_result=crc_result,
        neighbor_exclude_self=True,
    )
    summary = summarize_crc_decisions(pool_decisions)
    guide_summary = summarize_crc_decisions(calibration_decisions)
    sampling_stats = compute_crc_sampling_statistics(
        calibration_decisions,
        pool_decisions,
        temperature=temperature,
        lambda_hat=crc_result.lambda_hat,
    )
    metrics = compute_binary_metrics(
        [binary_to_int(row["label"], field_name="pool prediction label") for row in pool_predictions],
        [float(row["score"]) for row in pool_predictions],
        threshold=float(args.threshold),
    )
    record = {
        "round_index": int(args.round_index),
        "crc": crc_result.to_dict(),
        "temperature_choice": temperature_payload,
        "temperature": float(temperature),
        "T": float(temperature),
        "alpha": float(args.alpha),
        "lambda_hat": float(crc_result.lambda_hat),
        "pool_summary": summary,
        "guide_summary": guide_summary,
        "sampling_statistics": sampling_stats,
        "tau_crc": float(sampling_stats["tau_crc"]),
        "r_U": float(sampling_stats["r_U"]),
        "r_C": float(sampling_stats["r_C"]),
        "e_all": float(sampling_stats["e_all"]),
        "e_defer": float(sampling_stats["e_defer"]),
        "c_crc": float(sampling_stats["c_crc"]),
        "eta_crc": float(sampling_stats["eta_crc"]),
        "s_accept": float(sampling_stats["s_accept"]),
        "s_defer": float(sampling_stats["s_defer"]),
        "pool_metrics": metrics,
        "embeddings_path": str(embeddings_path),
    }
    write_jsonl(pool_decisions, pool_crc_predictions_path)
    write_json(record, round_summary_path)
    calibration_teacher_usage = summarize_teacher_label_usage(
        calibration_predictions,
        purpose="crc_calibration_teacher_labels",
    )
    write_stage_usage(
        usage_path,
        {
            "stage_name": "cgsd_calibrate",
            "round_index": int(args.round_index),
            "cache": cache_decision.to_dict(),
            "student_model_calls": 0,
            "crc_calibration_rows": len(calibration_predictions),
            "pool_decision_rows": len(pool_predictions),
            "estimated_crc_input_tokens": estimate_query_document_prompt_tokens(calibration_predictions),
            "teacher_label_usage": calibration_teacher_usage,
            "groundtruth_substitute_calls": calibration_teacher_usage["groundtruth_substitute_calls"],
            "teacher_api_file_calls": calibration_teacher_usage["teacher_api_file_calls"],
            "pool_crc_predictions_path": str(pool_crc_predictions_path),
            "round_summary_path": str(round_summary_path),
            "embeddings_path": str(embeddings_path),
        },
    )
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
