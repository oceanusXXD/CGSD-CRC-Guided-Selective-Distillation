from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.embeddings import load_embeddings
from mias_dcms.prompt_clusters import build_prompt_cluster_assignments
from mias_dcms.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign frozen prompt embedding clusters for preference/DCMS observable groups."
    )
    parser.add_argument("--active_pool_path", type=Path, required=True)
    parser.add_argument("--embeddings_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--summary_path", type=Path)
    parser.add_argument("--cluster_count", type=int, required=True)
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--max_iterations", type=int, default=50)
    parser.add_argument("--softmax_temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_prompt_cluster_assignments(
        rows=read_jsonl(args.active_pool_path),
        embeddings_by_id=load_embeddings(args.embeddings_path),
        cluster_count=int(args.cluster_count),
        id_field=str(args.id_field),
        max_iterations=int(args.max_iterations),
        softmax_temperature=float(args.softmax_temperature),
    )
    write_jsonl(result.rows, args.output_path)
    summary_path = args.summary_path or args.output_path.with_suffix(".summary.json")
    summary = {
        **result.summary,
        "active_pool_path": str(args.active_pool_path),
        "embeddings_path": str(args.embeddings_path),
        "output_path": str(args.output_path),
    }
    write_json(summary, summary_path)
    print(json.dumps(_compact_stdout_payload(summary), ensure_ascii=False, sort_keys=True))


def _compact_stdout_payload(summary: dict[str, object]) -> dict[str, object]:
    return {
        "row_count": summary.get("row_count"),
        "cluster_count": summary.get("cluster_count"),
        "cluster_counts": summary.get("cluster_counts"),
        "converged": summary.get("converged"),
        "iterations": summary.get("iterations"),
        "output_path": summary.get("output_path"),
    }


if __name__ == "__main__":
    main()
