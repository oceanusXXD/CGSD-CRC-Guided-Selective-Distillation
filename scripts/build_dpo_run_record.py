from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_run_summary import build_preference_run_record
from mias_dcms.utils import read_json, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one DPO run record from completed selection, reveal, training, evaluation, and cost artifacts."
    )
    parser.add_argument("--selection_summary_path", type=Path, required=True)
    parser.add_argument("--reveal_summary_path", type=Path, required=True)
    parser.add_argument("--training_rows_path", type=Path, required=True)
    parser.add_argument("--training_summary_path", type=Path, required=True)
    parser.add_argument("--evaluation_metrics_path", type=Path, required=True)
    parser.add_argument("--cost_report_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_hash", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_summary = read_json(args.training_summary_path)
    evaluation_payload = read_json(args.evaluation_metrics_path)
    cost_report = read_json(args.cost_report_path)
    record = build_preference_run_record(
        dataset=str(args.dataset),
        model=str(args.model),
        method=str(args.method),
        budget=int(args.budget),
        seed=int(args.seed),
        config_hash=str(args.config_hash),
        selection_summary=read_json(args.selection_summary_path),
        reveal_summary=read_json(args.reveal_summary_path),
        training_rows=read_jsonl(args.training_rows_path),
        training_metrics=_nested_mapping(training_summary, "training_metrics"),
        evaluation_metrics=_evaluation_metrics(evaluation_payload),
        seed_label_count=int(cost_report.get("seed_label_count", 0)),
        evaluation_label_count=int(cost_report.get("evaluation_label_count", 0)),
        judge_calls=int(cost_report.get("judge_calls", 0)),
        selector_compute_seconds=float(cost_report.get("selector_compute_seconds", 0.0)),
        train_tokens=(
            int(cost_report["train_tokens"])
            if cost_report.get("train_tokens") is not None
            else None
        ),
        oracle_label_calls=(
            int(cost_report["oracle_label_calls"])
            if cost_report.get("oracle_label_calls") is not None
            else None
        ),
    )
    payload = {**record.as_dict(), "run_status": "completed"}
    write_json(payload, args.output_path)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _nested_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return dict(value) if isinstance(value, Mapping) else dict(payload)


def _evaluation_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("evaluation_metrics")
    return dict(value) if isinstance(value, Mapping) else dict(payload)


if __name__ == "__main__":
    main()
