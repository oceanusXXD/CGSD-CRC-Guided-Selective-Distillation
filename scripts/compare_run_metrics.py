from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.run_metric_comparison import compare_run_metrics_to_baseline
from mias_dcms.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare run-level metrics against a baseline with paired seed tests."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--baseline_method", required=True)
    parser.add_argument("--treatment_methods", required=True)
    parser.add_argument("--evaluation_metrics", required=True)
    parser.add_argument("--selection_metrics", default="")
    parser.add_argument("--training_metrics", default="")
    parser.add_argument("--cost_metrics", default="")
    parser.add_argument("--expected_seeds", default="")
    parser.add_argument("--minimum_paired_seeds", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input_path))
    expected_seeds = [int(value) for value in _parse_csv(args.expected_seeds)] or None
    report = compare_run_metrics_to_baseline(
        rows,
        baseline_method=str(args.baseline_method),
        treatment_methods=_parse_csv(args.treatment_methods),
        evaluation_metrics=_parse_csv(args.evaluation_metrics),
        selection_metrics=_parse_csv(args.selection_metrics),
        training_metrics=_parse_csv(args.training_metrics),
        cost_metrics=_parse_csv(args.cost_metrics),
        expected_seeds=expected_seeds,
        minimum_paired_seeds=int(args.minimum_paired_seeds),
        confidence=float(args.confidence),
        resamples=int(args.resamples),
        permutations=int(args.permutations),
        seed=int(args.seed),
    )
    payload = report.as_dict()
    payload["input_path"] = str(args.input_path)
    write_json(payload, Path(args.output_path))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if report.is_ready else 1)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    main()
