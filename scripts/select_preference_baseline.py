from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.selectors import (
    assert_selector_rows_are_label_safe,
    select_top_budget,
    select_top_budget_by_group,
)
from mias_dcms.preference_selection_metrics import (
    build_preference_selection_metrics,
    materialize_preference_group_fields,
    utility_retained_from_scores,
)
from mias_dcms.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a top-budget preference batch from selector-safe baseline scores."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--score_field")
    parser.add_argument(
        "--selection_group_field",
        default="",
        help="Optional observable field that may contribute at most one selected row per group.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_path)
    assert_selector_rows_are_label_safe(rows)

    method = str(args.method).strip().lower().replace("-", "_")
    score_field = str(args.score_field or f"{method}_score")
    sample_ids = [str(row[args.id_field]) for row in rows]
    scores = [_score_from_row(row, method=method, score_field=score_field) for row in rows]
    selection_group_field = str(args.selection_group_field).strip()
    if selection_group_field:
        group_ids = [str(row.get(selection_group_field, "")) for row in rows]
        if any(not group_id for group_id in group_ids):
            raise ValueError(f"rows are missing selection group field {selection_group_field!r}")
        selected_ids = select_top_budget_by_group(
            sample_ids=sample_ids,
            scores=scores,
            group_ids=group_ids,
            budget=int(args.budget),
        )
    else:
        selected_ids = select_top_budget(sample_ids=sample_ids, scores=scores, budget=int(args.budget))
    selected_id_set = set(selected_ids)

    membership_rows = [
        {
            **materialize_preference_group_fields(row),
            str(args.id_field): sample_id,
            "sample_id": sample_id,
            "method": method,
            "score_field": score_field,
            "score": score,
            "selected": int(sample_id in selected_id_set),
        }
        for row, sample_id, score in zip(rows, sample_ids, scores, strict=True)
    ]
    summary = {
        "input_path": str(args.input_path),
        "method": method,
        "score_field": score_field,
        "budget": int(args.budget),
        "pool_size": len(rows),
        "selection_group_field": selection_group_field or None,
        "selected_count": len(selected_ids),
        "selected_score_min": min((row["score"] for row in membership_rows if row["selected"]), default=None),
        "selected_score_max": max((row["score"] for row in membership_rows if row["selected"]), default=None),
        "selection_metrics": build_preference_selection_metrics(
            membership_rows,
            method=method,
            score_field=score_field,
            utility_retained=utility_retained_from_scores(
                membership_rows,
                selected_ids=selected_ids,
                score_field="score",
            ),
        ),
    }
    selected_payload = {
        "selected_ids": selected_ids,
        "budget": int(args.budget),
        "selected_count": len(selected_ids),
        "method": method,
        "score_field": score_field,
        "selection_group_field": selection_group_field or None,
    }

    output_dir = args.output_dir
    write_json(selected_payload, output_dir / "selected_ids.json")
    write_jsonl(membership_rows, output_dir / "membership.jsonl")
    write_json(summary, output_dir / "selection_summary.json")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def _score_from_row(row: dict[str, object], *, method: str, score_field: str) -> float:
    if score_field in row and row[score_field] is not None:
        return float(row[score_field])
    selector_scores = row.get("selector_scores")
    if isinstance(selector_scores, dict) and method in selector_scores:
        return float(selector_scores[method])
    sample_id = row.get("sample_id", row.get("id", "<unknown>"))
    raise ValueError(f"row {sample_id!r} is missing score field {score_field!r}")


if __name__ == "__main__":
    main()
