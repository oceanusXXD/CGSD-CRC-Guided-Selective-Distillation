from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_run_evaluation import build_preference_run_evaluation_artifacts
from mias_dcms.utils import read_json, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize held-out DPO pairwise evaluation artifacts from policy/reference log-probs."
    )
    parser.add_argument("--heldout_pool_path", type=Path, required=True)
    parser.add_argument("--heldout_oracle_store_path", type=Path, required=True)
    parser.add_argument("--heldout_logprobs_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed_budget", type=int, required=True)
    parser.add_argument("--active_budget", type=int, required=True)
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--group_field", default="length_gap_bin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_preference_run_evaluation_artifacts(
        read_jsonl(args.heldout_pool_path),
        oracle_store=read_json(args.heldout_oracle_store_path),
        logprob_rows=read_jsonl(args.heldout_logprobs_path),
        seed_budget=int(args.seed_budget),
        active_budget=int(args.active_budget),
        id_field=str(args.id_field),
        group_field=str(args.group_field),
    )
    output_dir = args.output_dir
    write_jsonl(artifacts.preference_rows, output_dir / "heldout_preference_predictions.jsonl")
    write_jsonl(artifacts.judge_rows, output_dir / "judge_rows.jsonl")
    write_jsonl(artifacts.capability_rows, output_dir / "capability_rows.jsonl")
    write_jsonl(artifacts.aulc_rows, output_dir / "aulc_rows.jsonl")
    payload = {
        "input_paths": {
            "heldout_pool_path": str(args.heldout_pool_path),
            "heldout_oracle_store_path": str(args.heldout_oracle_store_path),
            "heldout_logprobs_path": str(args.heldout_logprobs_path),
        },
        "output_paths": {
            "preference_predictions_path": str(output_dir / "heldout_preference_predictions.jsonl"),
            "judge_rows_path": str(output_dir / "judge_rows.jsonl"),
            "capability_rows_path": str(output_dir / "capability_rows.jsonl"),
            "aulc_rows_path": str(output_dir / "aulc_rows.jsonl"),
        },
        "evaluation_metrics": artifacts.metrics,
        "initial_evaluation_metrics": artifacts.initial_metrics,
        "metadata": artifacts.metadata,
    }
    write_json(payload, output_dir / "evaluation_materialization.json")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
