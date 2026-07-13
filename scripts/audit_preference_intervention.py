from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_intervention_audit import (
    audit_ab_position_intervention,
    audit_length_gamma_intervention,
    audit_selector_replacement,
)
from mias_dcms.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit DPO-side preference intervention evidence for MIAS."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("length_gamma", "selector_replacement", "ab_position"),
        required=True,
    )
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--gammas", default="-1,0,1")
    parser.add_argument("--target_length_bin")
    parser.add_argument("--base_margin_field", default="base_margin")
    parser.add_argument("--length_gap_field", default="length_gap")
    parser.add_argument("--length_bin_field", default="length_gap_bin")
    parser.add_argument("--linked_group_fields", default="")
    parser.add_argument("--selector_a_score_field")
    parser.add_argument("--selector_b_score_field")
    parser.add_argument("--group_fields", default="")
    parser.add_argument("--score_field", default="score")
    parser.add_argument("--pair_field", default="swap_pair_id")
    parser.add_argument("--position_field", default="ab_position")
    return parser.parse_args(_normalize_value_args(sys.argv[1:]))


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_path)
    if args.mode == "length_gamma":
        if not args.target_length_bin:
            raise ValueError("--target_length_bin is required for length_gamma")
        summary = audit_length_gamma_intervention(
            rows,
            gammas=_parse_floats(str(args.gammas)),
            budget=int(args.budget),
            target_length_bin=str(args.target_length_bin),
            id_field=str(args.id_field),
            base_margin_field=str(args.base_margin_field),
            length_gap_field=str(args.length_gap_field),
            length_bin_field=str(args.length_bin_field),
            linked_group_fields=_parse_fields(str(args.linked_group_fields)),
        )
    elif args.mode == "selector_replacement":
        if not args.selector_a_score_field or not args.selector_b_score_field:
            raise ValueError("--selector_a_score_field and --selector_b_score_field are required")
        summary = audit_selector_replacement(
            rows,
            selector_a_score_field=str(args.selector_a_score_field),
            selector_b_score_field=str(args.selector_b_score_field),
            budget=int(args.budget),
            group_fields=_parse_fields(str(args.group_fields)),
            id_field=str(args.id_field),
        )
    else:
        summary = audit_ab_position_intervention(
            rows,
            score_field=str(args.score_field),
            budget=int(args.budget),
            id_field=str(args.id_field),
            pair_field=str(args.pair_field),
            position_field=str(args.position_field),
        )
    write_json({"input_path": str(args.input_path), **summary}, args.output_path)


def _parse_floats(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one numeric value")
    return parsed


def _parse_fields(value: str) -> tuple[str, ...]:
    return tuple(field.strip() for field in value.split(",") if field.strip())


def _normalize_value_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        current = argv[index]
        if current == "--gammas" and index + 1 < len(argv):
            normalized.append(f"--gammas={argv[index + 1]}")
            index += 2
            continue
        normalized.append(current)
        index += 1
    return normalized


if __name__ == "__main__":
    main()
