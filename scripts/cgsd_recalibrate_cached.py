"""基于已缓存预测文件重新计算 CGSD CRC 摘要。

该工具不会调用模型，只重新读取 calibration/pool 预测 JSONL，用固定温度
重算 CRC，并写出适合报告汇总的紧凑结果。
"""

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
from scripts.cgsd_cli_common import binary_to_int, read_jsonl
from src.metrics import compute_binary_metrics


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _binary_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = [
        binary_to_int(row.get("label", row.get("groundtruth")), field_name="label")
        for row in rows
    ]
    scores = [float(row.get("score", 0.0) or 0.0) for row in rows]
    return compute_binary_metrics(labels, scores)


def _wrong_accept_risk(decided_rows: list[dict[str, Any]]) -> float:
    if not decided_rows:
        return 0.0
    wrong_accept_count = 0
    for row in decided_rows:
        if bool(row.get("defer", False)):
            continue
        pred = binary_to_int(row.get("prediction"), field_name="prediction")
        label = binary_to_int(row.get("label", row.get("groundtruth")), field_name="label")
        wrong_accept_count += int(pred != label)
    return wrong_accept_count / float(len(decided_rows))


def recompute_round(
    *,
    run_dir: Path,
    output_dir: Path,
    round_index: int,
    alpha: float,
    temperature: float,
    eval_set: str,
    final_calibration: bool = False,
    same_final_calibration: bool = False,
) -> dict[str, Any]:
    round_dir = run_dir / f"round_{round_index}"
    cal_name = (
        "final_calibration_student_predictions.jsonl"
        if final_calibration and not same_final_calibration
        else "calibration_student_predictions.jsonl"
    )
    calibration_rows = read_jsonl(round_dir / cal_name)
    eval_name = {
        "pool": "pool_student_predictions.jsonl",
        "calibration": "calibration_student_predictions.jsonl",
        "final_calibration": "final_calibration_student_predictions.jsonl",
    }[eval_set]
    eval_rows = read_jsonl(round_dir / eval_name)
    crc = calibrate_crc(calibration_rows, alpha=alpha, temperature=temperature)
    decided_rows = apply_crc_decisions(
        eval_rows,
        lambda_hat=crc.lambda_hat,
        temperature=temperature,
    )
    decision_summary = summarize_crc_decisions(decided_rows)
    metrics = _binary_metrics(eval_rows)
    summary = {
        "source_run_dir": str(run_dir),
        "round": int(round_index),
        "alpha": float(alpha),
        "temperature": float(temperature),
        "calibration_source": (
            "calibration_as_final"
            if final_calibration and same_final_calibration
            else "final_calibration"
            if final_calibration
            else "calibration"
        ),
        "eval_set": str(eval_set),
        "calibration": crc.to_dict(),
        "eval_metrics": metrics,
        "eval_decision_summary": decision_summary,
        "eval_wrong_accept_risk": _wrong_accept_risk(decided_rows),
        "eval_prediction_rows": len(eval_rows),
        "calibration_prediction_rows": len(calibration_rows),
    }
    # 兼容旧报告片段中的字段名。
    summary["pool_metrics"] = metrics
    summary["pool_decision_summary"] = decision_summary
    summary["pool_wrong_accept_risk"] = summary["eval_wrong_accept_risk"]
    summary["pool_prediction_rows"] = summary["eval_prediction_rows"]
    stem = f"round_{round_index}_final" if final_calibration else f"round_{round_index}"
    _write_json(output_dir / f"{stem}_summary.json", summary)
    _write_jsonl(output_dir / f"{stem}_{eval_set}_crc_predictions.jsonl", decided_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.07)
    parser.add_argument("--temperature", type=float, default=15.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--eval_set", choices=("pool", "calibration", "final_calibration"), default="pool")
    parser.add_argument("--same_final_calibration", action="store_true", default=False)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    summaries = [
        recompute_round(
            run_dir=run_dir,
            output_dir=output_dir,
            round_index=round_index,
            alpha=args.alpha,
            temperature=args.temperature,
            eval_set=args.eval_set,
        )
        for round_index in range(args.rounds + 1)
    ]
    final_summary = recompute_round(
        run_dir=run_dir,
        output_dir=output_dir,
        round_index=args.rounds,
        alpha=args.alpha,
        temperature=args.temperature,
        eval_set=args.eval_set,
        final_calibration=True,
        same_final_calibration=args.same_final_calibration,
    )
    compact = {
        "source_run_dir": str(run_dir),
        "alpha": float(args.alpha),
        "temperature": float(args.temperature),
        "eval_set": str(args.eval_set),
        "same_final_calibration": bool(args.same_final_calibration),
        "rounds": summaries,
        "final": final_summary,
    }
    _write_json(output_dir / "t15_recalibration_summary.json", compact)
    print(json.dumps(compact, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
