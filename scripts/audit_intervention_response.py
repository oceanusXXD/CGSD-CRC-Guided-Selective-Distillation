from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.interventions import (
    apply_class_intercept,
    apply_length_coefficient,
    entropy_scores_from_logits,
    fixed_budget_response_curve,
)
from mias_dcms.utils import read_jsonl, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit fixed-budget MIAS response curves under controlled score interventions."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--mode",
        choices=("class_intercept_entropy", "length_gamma_margin"),
        required=True,
    )
    parser.add_argument("--values", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--target_group", required=True)
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--group_field", default="group")
    parser.add_argument("--logits_field", default="logits")
    parser.add_argument("--target_class", type=int, default=None)
    parser.add_argument("--margin_field", default="margin")
    parser.add_argument("--length_gap_field", default="length_gap")
    return parser.parse_args(_normalize_value_args(sys.argv[1:] if argv is None else argv))


def _normalize_value_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        current = argv[index]
        if current == "--values" and index + 1 < len(argv):
            normalized.append(f"--values={argv[index + 1]}")
            index += 2
            continue
        normalized.append(current)
        index += 1
    return normalized


def _parse_values(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("values must contain at least one intervention value")
    return parsed


def _class_intercept_scores(
    rows: list[dict[str, Any]],
    *,
    logits_field: str,
    target_class: int,
    values: list[float],
) -> dict[float, list[float]]:
    logits = []
    for row in rows:
        row_logits = row[logits_field]
        if not isinstance(row_logits, list):
            raise ValueError(f"logits field {logits_field!r} must contain a list")
        logits.append([float(value) for value in row_logits])
    return {
        value: entropy_scores_from_logits(
            apply_class_intercept(logits, target_class=target_class, alpha=value)
        )
        for value in values
    }


def _length_gamma_scores(
    rows: list[dict[str, Any]],
    *,
    margin_field: str,
    length_gap_field: str,
    values: list[float],
) -> dict[float, list[float]]:
    margins = [float(row[margin_field]) for row in rows]
    length_gaps = [float(row[length_gap_field]) for row in rows]
    return {
        value: apply_length_coefficient(margins, length_gaps, gamma=value)
        for value in values
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.input_path))
    values = _parse_values(str(args.values))
    sample_ids = [str(row[args.id_field]) for row in rows]
    groups = [str(row[args.group_field]) for row in rows]

    if args.mode == "class_intercept_entropy":
        if args.target_class is None:
            raise ValueError("--target_class is required for class_intercept_entropy")
        score_by_value = _class_intercept_scores(
            rows,
            logits_field=str(args.logits_field),
            target_class=int(args.target_class),
            values=values,
        )
    else:
        score_by_value = _length_gamma_scores(
            rows,
            margin_field=str(args.margin_field),
            length_gap_field=str(args.length_gap_field),
            values=values,
        )

    curve = fixed_budget_response_curve(
        sample_ids=sample_ids,
        groups=groups,
        score_by_value=score_by_value,
        budget=int(args.budget),
        target_group=str(args.target_group),
    )
    payload = {
        "input_path": str(args.input_path),
        "mode": str(args.mode),
        "budget": int(args.budget),
        "target_group": str(args.target_group),
        "values": values,
        **curve.as_dict(),
    }
    write_json(payload, Path(args.output_path))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
