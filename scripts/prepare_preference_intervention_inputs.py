from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_intervention_inputs import build_preference_intervention_rows
from mias_dcms.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare selector-safe CPU-only rows for preference MIAS intervention audits."
    )
    parser.add_argument("--active_pool_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--logprobs_path", type=Path)
    parser.add_argument("--score_path", type=Path)
    parser.add_argument("--summary_path", type=Path)
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--length_bin_edges", default="-0.2,0.2")
    parser.add_argument("--base_margin_field", default="implicit_reward_gap")
    parser.add_argument(
        "--selector_score_fields",
        default="reward_margin_score,apl_score,active_dpo_score",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_preference_intervention_rows(
        active_pool_rows=read_jsonl(args.active_pool_path),
        logprob_rows=read_jsonl(args.logprobs_path) if args.logprobs_path else (),
        score_rows=read_jsonl(args.score_path) if args.score_path else (),
        id_field=str(args.id_field),
        length_bin_edges=_parse_floats(str(args.length_bin_edges)),
        base_margin_field=str(args.base_margin_field),
        selector_score_fields=_parse_fields(str(args.selector_score_fields)),
    )
    write_jsonl(rows, args.output_path)
    summary_path = args.summary_path or args.output_path.with_suffix(".summary.json")
    summary = {
        "active_pool_path": str(args.active_pool_path),
        "logprobs_path": str(args.logprobs_path) if args.logprobs_path else None,
        "score_path": str(args.score_path) if args.score_path else None,
        "output_path": str(args.output_path),
        "row_count": len(rows),
        "length_gap_bins": _counts(row["length_gap_bin"] for row in rows),
        "source_pairs": _counts(row["source_pair"] for row in rows),
        "ab_positions": _counts(row["ab_position"] for row in rows),
        "selector_score_fields": _parse_fields(str(args.selector_score_fields)),
    }
    write_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def _parse_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def _parse_fields(value: str) -> tuple[str, ...]:
    return tuple(field.strip() for field in value.split(",") if field.strip())


def _counts(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
