from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_scoring import (
    SUPPORTED_PREFERENCE_SCORE_METHODS,
    build_preference_baseline_score_rows,
)
from mias_dcms.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build selector-safe Reward Margin, APL, and ActiveDPO preference baseline scores."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--summary_path", type=Path)
    parser.add_argument(
        "--metadata_path",
        action="append",
        default=[],
        help="Optional selector-safe JSONL metadata to merge by sample_id before scoring.",
    )
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument(
        "--methods",
        default=",".join(SUPPORTED_PREFERENCE_SCORE_METHODS),
        help="Comma-separated methods: reward_margin, apl, active_dpo.",
    )
    parser.add_argument("--prompt_entropy_weight", type=float, default=1.0)
    parser.add_argument(
        "--active_dpo_length_normalize",
        action="store_true",
        help="Use pair token count to length-normalize the fixed-pool ActiveDPO gradient proxy.",
    )
    parser.add_argument(
        "--active_dpo_novelty_weight",
        type=float,
        default=0.0,
        help="Weight for selector-safe prompt novelty in the fixed-pool ActiveDPO adaptation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = tuple(method.strip() for method in str(args.methods).split(",") if method.strip())
    rows = _merge_metadata(
        read_jsonl(args.input_path),
        metadata_paths=[Path(path) for path in args.metadata_path],
        id_field=str(args.id_field),
    )
    scored_rows = build_preference_baseline_score_rows(
        rows,
        methods=methods,
        prompt_entropy_weight=float(args.prompt_entropy_weight),
        active_dpo_length_normalize=bool(args.active_dpo_length_normalize),
        active_dpo_novelty_weight=float(args.active_dpo_novelty_weight),
        id_field=str(args.id_field),
    )
    write_jsonl(scored_rows, args.output_path)

    summary_path = args.summary_path or args.output_path.with_suffix(".summary.json")
    score_fields = [f"{method}_score" for method in methods]
    write_json(
        {
            "input_path": str(args.input_path),
            "metadata_paths": [str(path) for path in args.metadata_path],
            "output_path": str(args.output_path),
            "row_count": len(scored_rows),
            "methods": list(methods),
            "score_fields": score_fields,
            "prompt_entropy_weight": float(args.prompt_entropy_weight),
            "active_dpo_length_normalize": bool(args.active_dpo_length_normalize),
            "active_dpo_novelty_weight": float(args.active_dpo_novelty_weight),
        },
        summary_path,
    )


def _merge_metadata(
    rows: list[dict[str, object]],
    *,
    metadata_paths: list[Path],
    id_field: str,
) -> list[dict[str, object]]:
    merged = [dict(row) for row in rows]
    for path in metadata_paths:
        metadata_by_id: dict[str, dict[str, object]] = {}
        for row in read_jsonl(path):
            sample_id = str(row.get(id_field, row.get("id")))
            if not sample_id or sample_id == "None":
                raise ValueError(f"metadata row in {path} is missing id field {id_field!r}")
            if sample_id in metadata_by_id:
                raise ValueError(f"duplicate metadata row for sample id {sample_id!r} in {path}")
            metadata_by_id[sample_id] = dict(row)
        for row in merged:
            sample_id = str(row.get(id_field, row.get("id")))
            if sample_id in metadata_by_id:
                row.update(metadata_by_id[sample_id])
    return merged


if __name__ == "__main__":
    main()
