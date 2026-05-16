#!/usr/bin/env python
"""准备固定的 CGSD 数据划分。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cgsd_cli_common import (
    add_stage_cache_args,
    embedding_usage_payload,
    output_dir_from_arg,
    output_artifact_path,
    print_existing_stage_result,
    stage_cache_decision,
    write_stage_usage,
)
from scripts.run_cgsd import (
    assert_embedding_coverage,
    examples_to_rows,
    load_embeddings,
)
from src.data import load_examples
from src.utils import resolve_input_path, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="datasets")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_ids_path", default=None)
    parser.add_argument("--usage_path", default=None)
    parser.add_argument("--query_field", default="query")
    parser.add_argument("--document_field", default="document")
    parser.add_argument("--label_field", default="groundtruth")
    parser.add_argument("--embeddings_path", required=True)
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--n_calibration", type=int, default=200)
    parser.add_argument("--n_final_calibration", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    add_stage_cache_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = output_dir_from_arg(args.output_dir)
    split_ids_path = output_artifact_path(args.split_ids_path, output_dir / "cgsd_split_ids.json")
    usage_path = output_artifact_path(args.usage_path, output_dir / "prepare_usage.json")
    if args.show_result:
        print_existing_stage_result(stage_name="cgsd_prepare", summary_path=split_ids_path)
        return
    cache_decision = stage_cache_decision(
        stage_name="cgsd_prepare",
        required_outputs=[split_ids_path, usage_path],
        cache_policy=args.cache_policy,
    )
    if cache_decision.cache_hit:
        print_existing_stage_result(stage_name="cgsd_prepare", summary_path=split_ids_path)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = resolve_input_path(args.data_path, PROJECT_ROOT)
    embeddings_path = resolve_input_path(args.embeddings_path, PROJECT_ROOT)

    examples = load_examples(
        data_path,
        query_field=args.query_field,
        document_field=args.document_field,
        label_field=args.label_field,
    )
    rows = examples_to_rows(examples)
    if int(args.n_calibration) <= 0:
        raise ValueError("--n_calibration must be positive")
    if int(args.n_final_calibration) < 0:
        raise ValueError("--n_final_calibration must be non-negative")
    if int(args.n_calibration) + int(args.n_final_calibration) >= len(rows):
        raise ValueError("--n_calibration + --n_final_calibration must be smaller than the dataset size")
    all_ids = [str(row["id"]) for row in rows]
    rng = random.Random(int(args.seed))
    rng.shuffle(all_ids)
    calibration_ids = all_ids[: int(args.n_calibration)]
    final_calibration_ids = all_ids[
        int(args.n_calibration) : int(args.n_calibration) + int(args.n_final_calibration)
    ]
    pool_ids = all_ids[int(args.n_calibration) + int(args.n_final_calibration) :]
    embeddings_by_id = load_embeddings(embeddings_path)
    assert_embedding_coverage(embeddings_by_id, rows, expected_dim=args.embedding_dim)

    split_payload = {
        "calibration_ids": calibration_ids,
        "pool_ids": pool_ids,
        "final_calibration_ids": final_calibration_ids,
        "n_calibration": args.n_calibration,
        "n_final_calibration": args.n_final_calibration,
        "seed": args.seed,
        "split_algorithm": "fixed_random_calibration_final_calibration_pool_v1",
    }
    write_json(split_payload, split_ids_path)
    write_stage_usage(
        usage_path,
        {
            "stage_name": "cgsd_prepare",
            "cache": cache_decision.to_dict(),
            "data_path": str(data_path),
            "split_ids_path": str(split_ids_path),
            "data_rows": len(rows),
            "calibration_size": len(calibration_ids),
            "final_calibration_size": len(final_calibration_ids),
            "pool_size": len(pool_ids),
            "embedding": embedding_usage_payload(
                embedding_source=embeddings_path,
                row_count=len(rows),
                embedding_dim=int(args.embedding_dim),
                purpose="prepare_validate_full_embedding_coverage",
            ),
            "student_model_calls": 0,
            "teacher_calls": 0,
            "groundtruth_substitute_calls": 0,
        },
    )
    print(json.dumps({"prepared": True, "output_dir": str(output_dir)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
