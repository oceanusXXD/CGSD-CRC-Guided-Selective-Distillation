from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_run_summary import build_preference_run_record
from mias_dcms.utils import read_json, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a run-level preference acquisition record from selection and reveal artifacts."
    )
    parser.add_argument("--selection_summary_path", type=Path, required=True)
    parser.add_argument("--reveal_summary_path", type=Path, required=True)
    parser.add_argument("--training_rows_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_hash", required=True)
    parser.add_argument("--training_metrics", default="{}")
    parser.add_argument("--evaluation_metrics", default="{}")
    parser.add_argument("--seed_label_count", type=int, default=0)
    parser.add_argument("--evaluation_label_count", type=int, default=0)
    parser.add_argument("--judge_calls", type=int, default=0)
    parser.add_argument("--selector_compute_seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = build_preference_run_record(
        dataset=str(args.dataset),
        model=str(args.model),
        method=str(args.method),
        budget=int(args.budget),
        seed=int(args.seed),
        config_hash=str(args.config_hash),
        selection_summary=read_json(args.selection_summary_path),
        reveal_summary=read_json(args.reveal_summary_path),
        training_rows=read_jsonl(args.training_rows_path),
        training_metrics=_parse_json_object(str(args.training_metrics), "training_metrics"),
        evaluation_metrics=_parse_json_object(str(args.evaluation_metrics), "evaluation_metrics"),
        seed_label_count=int(args.seed_label_count),
        evaluation_label_count=int(args.evaluation_label_count),
        judge_calls=int(args.judge_calls),
        selector_compute_seconds=float(args.selector_compute_seconds),
    )
    write_json(run.as_dict(), args.output_path)


def _parse_json_object(value: str, name: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


if __name__ == "__main__":
    main()
