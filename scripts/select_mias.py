from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.selection.mias import (
    DEFAULT_KAPPA,
    DEFAULT_SLACK_GRID,
    MIASSelectionResult,
    select_mias_classification,
    select_mias_preference,
)
from mias_dcms.selection.features import merge_feature_rows
from mias_dcms.preference_selection_metrics import build_preference_selection_metrics
from mias_dcms.selectors import assert_selector_rows_are_label_safe
from mias_dcms.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a pre-training batch with seed-only expected validation influence."
    )
    parser.add_argument("--task", choices=("classification", "preference"), required=True)
    parser.add_argument("--seed_rows_path", type=Path, required=True)
    parser.add_argument("--candidate_rows_path", type=Path, required=True)
    parser.add_argument("--seed_feature_path", type=Path, required=True)
    parser.add_argument("--candidate_feature_path", type=Path, required=True)
    parser.add_argument(
        "--metadata_path",
        type=Path,
        action="append",
        default=[],
        help="Selector-safe candidate metadata to join by sample_id; may be repeated.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dcms", action="store_true")
    parser.add_argument("--label_field", default="label")
    parser.add_argument("--bootstrap_heads", type=int, default=20)
    parser.add_argument("--semantic_cluster_count", type=int)
    parser.add_argument(
        "--slack_grid",
        default=",".join(str(value) for value in DEFAULT_SLACK_GRID),
    )
    parser.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    return parser.parse_args()


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    seed_rows = merge_feature_rows(
        read_jsonl(args.seed_rows_path),
        read_jsonl(args.seed_feature_path),
        source_name="seed_feature_path",
    )
    candidate_rows = merge_feature_rows(
        read_jsonl(args.candidate_rows_path),
        read_jsonl(args.candidate_feature_path),
        source_name="candidate_feature_path",
    )
    for path in args.metadata_path:
        candidate_rows = _merge_optional_metadata(candidate_rows, read_jsonl(path), source_name=str(path))
    assert_selector_rows_are_label_safe(candidate_rows)

    common = {
        "seed_rows": seed_rows,
        "candidate_rows": candidate_rows,
        "budget": int(args.budget),
        "seed": int(args.seed),
        "use_dcms": bool(args.dcms),
        "bootstrap_heads": int(args.bootstrap_heads),
        "slack_grid": _parse_slack_grid(str(args.slack_grid)),
        "kappa": float(args.kappa),
    }
    if args.task == "classification":
        result = select_mias_classification(
            **common,
            label_field=str(args.label_field),
            semantic_cluster_count=args.semantic_cluster_count,
        )
    else:
        result = select_mias_preference(**common)
    _write_outputs(
        result,
        args=args,
        candidate_rows=candidate_rows,
        selector_compute_seconds=time.perf_counter() - started_at,
    )


def _write_outputs(
    result: MIASSelectionResult,
    *,
    args: argparse.Namespace,
    candidate_rows: list[dict[str, Any]],
    selector_compute_seconds: float,
) -> None:
    method = "mias_dcms" if args.dcms else "mias"
    selected_payload = {
        "selected_ids": list(result.selected_ids),
        "selected_count": len(result.selected_ids),
        "budget": int(args.budget),
        "method": method,
        "seed": int(args.seed),
    }
    score_rows = []
    for row in result.scoring.score_rows(result.selected_ids):
        sample_id = str(row["sample_id"])
        score_rows.append(
            {
                **row,
                "group_membership": dict(result.group_membership.get(sample_id, {})),
                "membership_lower": dict(result.membership_lower.get(sample_id, {})),
                "membership_upper": dict(result.membership_upper.get(sample_id, {})),
                "q_propensity": (
                    float(result.dcms.q_propensity.get(sample_id, 0.0))
                    if result.dcms is not None
                    else float(row["selected"])
                ),
            }
        )
    summary = {
        **result.summary_dict(method=method, budget=int(args.budget)),
        "task": str(args.task),
        "seed": int(args.seed),
        "seed_rows_path": str(args.seed_rows_path),
        "candidate_rows_path": str(args.candidate_rows_path),
        "seed_feature_path": str(args.seed_feature_path),
        "candidate_feature_path": str(args.candidate_feature_path),
        "metadata_paths": [str(path) for path in args.metadata_path],
        "selector_compute_seconds": float(selector_compute_seconds),
    }
    if args.task == "preference":
        summary["selection_metrics"] = build_preference_selection_metrics(
            candidate_rows,
            selected_ids=result.selected_ids,
            method=method,
            constraint_violation=float(summary["max_constraint_violation"]),
            utility_retained=float(summary["utility_retained"]),
        )
    write_json(selected_payload, args.output_dir / "selected_ids.json")
    write_jsonl(score_rows, args.output_dir / "mias_scores.jsonl")
    write_json(result.scoring.model_dict(), args.output_dir / "mias_selector_model.json")
    write_json(summary, args.output_dir / "selection_summary.json")
    print(
        json.dumps(
            {
                "algorithm": summary["algorithm"],
                "task": summary["task"],
                "pool_size": summary["pool_size"],
                "selected_count": summary["selected_count"],
                "temperature_status": result.scoring.temperature_status,
                "bootstrap_status": result.scoring.bootstrap_status,
                "selected_slack": result.dcms.selected_slack if result.dcms is not None else None,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _merge_optional_metadata(
    rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    *,
    source_name: str,
) -> list[dict[str, Any]]:
    metadata_by_id = _unique_rows_by_id(metadata_rows, source_name=source_name)
    missing = sorted({_row_id(row) for row in rows} - set(metadata_by_id))
    if missing:
        raise ValueError(f"metadata {source_name} is missing {len(missing)} candidate ids")
    return [{**row, **_without_id_fields(metadata_by_id[_row_id(row)])} for row in rows]


def _unique_rows_by_id(rows: list[dict[str, Any]], *, source_name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = _row_id(row)
        if sample_id in output:
            raise ValueError(f"{source_name} contains duplicate id {sample_id!r}")
        output[sample_id] = dict(row)
    return output


def _without_id_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"id", "sample_id"}}


def _row_id(row: dict[str, Any]) -> str:
    value = row.get("sample_id", row.get("id"))
    if value is None or not str(value):
        raise ValueError("row is missing a non-empty sample_id/id")
    return str(value)


def _parse_slack_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("slack_grid must contain at least one value")
    return values


if __name__ == "__main__":
    main()
