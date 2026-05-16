#!/usr/bin/env python
"""Collect CGSD experiment run summaries into CSV/JSONL tables."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_optional(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def infer_dataset(run_dir: Path) -> str:
    parts = run_dir.resolve().parts
    if "runs" in parts:
        index = parts.index("runs")
        if index + 1 < len(parts):
            return parts[index + 1]
    return run_dir.parent.name


def choose_round(run_dir: Path, round_index: int | None = None) -> int:
    if round_index is not None:
        return int(round_index)
    summary = read_json_optional(run_dir / "cgsd_summary.json")
    if "best_round_index" in summary:
        return int(summary["best_round_index"])
    candidates: list[int] = []
    for path in run_dir.glob("round_*/round_summary.json"):
        try:
            candidates.append(int(path.parent.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if not candidates:
        raise FileNotFoundError(f"no round_*/round_summary.json under {run_dir}")
    return max(candidates)


def usage_totals(run_dir: Path) -> dict[str, int]:
    totals: defaultdict[str, int] = defaultdict(int)
    usage_paths = list(run_dir.glob("round_*/*_usage.json")) + list(run_dir.glob("*_usage.json"))
    for path in usage_paths:
        payload = read_json_optional(path)
        totals["student_model_calls_total"] += _as_int(payload.get("student_model_calls"))
        totals["estimated_student_prompt_tokens_total"] += _as_int(payload.get("estimated_student_prompt_tokens"))
        totals["estimated_student_completion_tokens_total"] += _as_int(payload.get("estimated_student_completion_tokens"))
        totals["estimated_student_train_tokens_total"] += _as_int(payload.get("estimated_student_train_tokens"))
        totals["estimated_defer_prompt_tokens_total"] += _as_int(payload.get("estimated_defer_prompt_tokens"))
        for key in ("teacher_label_usage", "teacher_defer_usage"):
            usage = payload.get(key)
            if isinstance(usage, dict):
                totals["teacher_calls_total"] += _as_int(usage.get("teacher_calls"))
                totals["teacher_api_file_calls_total"] += _as_int(usage.get("teacher_api_file_calls"))
                totals["groundtruth_substitute_calls_total"] += _as_int(usage.get("groundtruth_substitute_calls"))
                totals["estimated_teacher_prompt_tokens_total"] += _as_int(usage.get("estimated_prompt_tokens"))
                totals["estimated_teacher_completion_tokens_total"] += _as_int(usage.get("estimated_completion_tokens"))
    return dict(totals)


def collect_run(run_dir: str | Path, *, round_index: int | None = None, method: str | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    chosen_round = choose_round(run_path, round_index)
    round_dir = run_path / f"round_{chosen_round}"
    final_summary_path = round_dir / "final_round_summary.json"
    round_summary_path = final_summary_path if final_summary_path.exists() else round_dir / "round_summary.json"
    round_summary = read_json(round_summary_path)
    cgsd_summary = read_json_optional(run_path / "cgsd_summary.json")
    crc = round_summary.get("crc", {})
    pool_summary = round_summary.get("pool_summary", {})
    pool_metrics = round_summary.get("pool_metrics", {})
    total = _as_int(pool_summary.get("total"))
    wrong_accept_count = _as_int(pool_summary.get("wrong_accept_count"))
    record: dict[str, Any] = {
        "dataset": infer_dataset(run_path),
        "run_name": run_path.name,
        "run_dir": str(run_path),
        "method": method or run_path.name,
        "round_index": chosen_round,
        "summary_file": round_summary_path.name,
        "alpha": _as_float(crc.get("alpha")),
        "temperature": _as_float(round_summary.get("temperature")),
        "lambda_hat": _as_float(round_summary.get("lambda_hat")),
        "raw_accuracy": _as_float(pool_metrics.get("accuracy")),
        "defer_rate": _as_float(pool_summary.get("defer_rate")),
        "defer_count": _as_int(pool_summary.get("defer_count")),
        "pool_total": total,
        "wrong_accept_count": wrong_accept_count,
        "wrong_accept_rate": float(wrong_accept_count / total) if total else 0.0,
        "accept_error_rate": _as_float(pool_summary.get("accept_error_rate")),
        "crc_empirical_risk": _as_float(crc.get("empirical_risk")),
        "crc_risk_bound": _as_float(crc.get("risk_bound")),
        "crc_grid_feasible": bool(crc.get("grid_feasible", False)),
        "teacher_train_calls": _as_int(cgsd_summary.get("teacher_train_calls")),
        "teacher_defer_calls": _as_int(cgsd_summary.get("teacher_defer_calls")),
    }
    record.update(usage_totals(run_path))
    return record


def expand_run_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        paths.extend(matches or [Path(pattern)])
    return sorted({path for path in paths if path.exists()})


def summarize_crc_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record.get("dataset", "")), str(record.get("method", "")), _as_float(record.get("alpha")))].append(record)
    summaries: list[dict[str, Any]] = []
    for (dataset, method, alpha), rows in sorted(groups.items()):
        count = len(rows)
        violations = sum(1 for row in rows if _as_float(row.get("wrong_accept_rate")) > alpha)
        summaries.append(
            {
                "dataset": dataset,
                "method": method,
                "alpha": alpha,
                "runs": count,
                "mean_wrong_accept_rate": sum(_as_float(row.get("wrong_accept_rate")) for row in rows) / count,
                "mean_crc_risk_bound": sum(_as_float(row.get("crc_risk_bound")) for row in rows) / count,
                "mean_defer_rate": sum(_as_float(row.get("defer_rate")) for row in rows) / count,
                "violation_count": violations,
                "violation_rate": violations / count,
            }
        )
    return summaries


def apply_cost_estimate(
    record: dict[str, Any],
    *,
    teacher_call_cost: float,
    student_call_cost: float,
    teacher_prompt_cost_per_million: float,
    teacher_completion_cost_per_million: float,
    student_prompt_cost_per_million: float,
    student_completion_cost_per_million: float,
) -> dict[str, Any]:
    teacher_prompt_tokens = _as_int(record.get("estimated_teacher_prompt_tokens_total"))
    teacher_completion_tokens = _as_int(record.get("estimated_teacher_completion_tokens_total"))
    student_prompt_tokens = _as_int(record.get("estimated_student_prompt_tokens_total")) + _as_int(
        record.get("estimated_student_train_tokens_total")
    )
    student_completion_tokens = _as_int(record.get("estimated_student_completion_tokens_total"))
    estimated_cost = (
        _as_int(record.get("teacher_calls_total")) * float(teacher_call_cost)
        + _as_int(record.get("student_model_calls_total")) * float(student_call_cost)
        + teacher_prompt_tokens * float(teacher_prompt_cost_per_million) / 1_000_000.0
        + teacher_completion_tokens * float(teacher_completion_cost_per_million) / 1_000_000.0
        + student_prompt_tokens * float(student_prompt_cost_per_million) / 1_000_000.0
        + student_completion_tokens * float(student_completion_cost_per_million) / 1_000_000.0
    )
    item = dict(record)
    item["estimated_cost"] = float(estimated_cost)
    return item


def write_csv(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    handle = path.open("w", encoding="utf-8", newline="") if path else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if path:
            handle.close()


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Run dirs or glob patterns under experiments/runs")
    parser.add_argument("--round_index", type=int, default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--crc_summary_csv", default=None)
    parser.add_argument("--teacher_call_cost", type=float, default=0.0)
    parser.add_argument("--student_call_cost", type=float, default=0.0)
    parser.add_argument("--teacher_prompt_cost_per_million", type=float, default=0.0)
    parser.add_argument("--teacher_completion_cost_per_million", type=float, default=0.0)
    parser.add_argument("--student_prompt_cost_per_million", type=float, default=0.0)
    parser.add_argument("--student_completion_cost_per_million", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [collect_run(path, round_index=args.round_index, method=args.method) for path in expand_run_paths(args.runs)]
    records = [
        apply_cost_estimate(
            record,
            teacher_call_cost=args.teacher_call_cost,
            student_call_cost=args.student_call_cost,
            teacher_prompt_cost_per_million=args.teacher_prompt_cost_per_million,
            teacher_completion_cost_per_million=args.teacher_completion_cost_per_million,
            student_prompt_cost_per_million=args.student_prompt_cost_per_million,
            student_completion_cost_per_million=args.student_completion_cost_per_million,
        )
        for record in records
    ]
    if args.output_jsonl:
        write_jsonl(records, Path(args.output_jsonl))
    if args.crc_summary_csv:
        write_csv(summarize_crc_records(records), Path(args.crc_summary_csv))
    write_csv(records, Path(args.output_csv) if args.output_csv else None)


if __name__ == "__main__":
    main()
