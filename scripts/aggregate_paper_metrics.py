from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.records import RunRecord
from mias_dcms.result_aggregation import aggregate_paper_metric_table
from mias_dcms.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate run-level MIAS/DCMS JSONL records into paper-level metric tables."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--evaluation_metrics", required=True)
    parser.add_argument("--selection_metrics", default="")
    parser.add_argument("--training_metrics", default="")
    parser.add_argument("--cost_metrics", default="")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input_path))
    runs = [RunRecord(**row) for row in rows]
    evaluation_metrics = _parse_csv(str(args.evaluation_metrics))
    selection_metrics = _parse_csv(str(args.selection_metrics))
    training_metrics = _parse_csv(str(args.training_metrics))
    cost_metrics = _parse_csv(str(args.cost_metrics))
    table = aggregate_paper_metric_table(
        runs,
        evaluation_metrics=evaluation_metrics,
        selection_metrics=selection_metrics,
        training_metrics=training_metrics,
        cost_metrics=cost_metrics,
        confidence=float(args.confidence),
        resamples=int(args.resamples),
        seed=int(args.seed),
    )
    payload = {
        "input_path": str(args.input_path),
        "run_count": len(runs),
        "evaluation_metrics": evaluation_metrics,
        "selection_metrics": selection_metrics,
        "training_metrics": training_metrics,
        "cost_metrics": cost_metrics,
        "confidence": float(args.confidence),
        "resamples": int(args.resamples),
        "seed": int(args.seed),
        "paper_metric_table": table,
    }
    write_json(payload, Path(args.output_path))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
