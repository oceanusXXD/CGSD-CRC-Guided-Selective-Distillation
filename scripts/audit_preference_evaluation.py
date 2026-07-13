from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_evaluation import build_preference_evaluation_metrics
from mias_dcms.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DPO preference evaluation metrics for run-level aggregation."
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--preference_predictions_path", type=Path)
    parser.add_argument("--judge_rows_path", type=Path)
    parser.add_argument("--capability_rows_path", type=Path)
    parser.add_argument("--aulc_rows_path", type=Path)
    parser.add_argument("--group_field", default="observable_group")
    parser.add_argument("--length_bin_field", default="length_gap_bin")
    parser.add_argument("--label_field", default="oracle_preference")
    parser.add_argument("--prediction_field", default="predicted_preference")
    parser.add_argument("--win_field", default="judge_win")
    parser.add_argument("--baseline_field", default="baseline_score")
    parser.add_argument("--policy_field", default="policy_score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preference_rows = (
        read_jsonl(args.preference_predictions_path)
        if args.preference_predictions_path is not None
        else None
    )
    judge_rows = read_jsonl(args.judge_rows_path) if args.judge_rows_path is not None else None
    capability_rows = (
        read_jsonl(args.capability_rows_path)
        if args.capability_rows_path is not None
        else None
    )
    aulc_rows = read_jsonl(args.aulc_rows_path) if args.aulc_rows_path is not None else None
    metrics = build_preference_evaluation_metrics(
        preference_rows=preference_rows,
        judge_rows=judge_rows,
        capability_rows=capability_rows,
        aulc_rows=aulc_rows,
        group_field=str(args.group_field),
        length_bin_field=str(args.length_bin_field),
        label_field=str(args.label_field),
        prediction_field=str(args.prediction_field),
        win_field=str(args.win_field),
        baseline_field=str(args.baseline_field),
        policy_field=str(args.policy_field),
    )
    write_json(
        {
            "input_paths": {
                "preference_predictions_path": str(args.preference_predictions_path)
                if args.preference_predictions_path is not None
                else None,
                "judge_rows_path": str(args.judge_rows_path)
                if args.judge_rows_path is not None
                else None,
                "capability_rows_path": str(args.capability_rows_path)
                if args.capability_rows_path is not None
                else None,
                "aulc_rows_path": str(args.aulc_rows_path)
                if args.aulc_rows_path is not None
                else None,
            },
            "fields": {
                "group_field": str(args.group_field),
                "length_bin_field": str(args.length_bin_field),
                "label_field": str(args.label_field),
                "prediction_field": str(args.prediction_field),
                "win_field": str(args.win_field),
                "baseline_field": str(args.baseline_field),
                "policy_field": str(args.policy_field),
            },
            "evaluation_metrics": metrics,
        },
        args.output_path,
    )


if __name__ == "__main__":
    main()
