from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_logprob_audit import audit_preference_logprobs
from mias_dcms.utils import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit preference active-pool policy/reference logprobs and implicit margins."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--summary_path", type=Path)
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--policy_response_1_field", default="policy_logprob_response_1")
    parser.add_argument("--policy_response_2_field", default="policy_logprob_response_2")
    parser.add_argument("--reference_response_1_field", default="reference_logprob_response_1")
    parser.add_argument("--reference_response_2_field", default="reference_logprob_response_2")
    parser.add_argument(
        "--allow_zero_implicit_margin",
        action="store_true",
        help="Write outputs even when policy/reference implicit margins are all zero.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_path)
    audited_rows, summary = audit_preference_logprobs(
        rows,
        id_field=str(args.id_field),
        policy_response_1_field=str(args.policy_response_1_field),
        policy_response_2_field=str(args.policy_response_2_field),
        reference_response_1_field=str(args.reference_response_1_field),
        reference_response_2_field=str(args.reference_response_2_field),
        require_nonzero_implicit_margin=not bool(args.allow_zero_implicit_margin),
    )
    write_jsonl(audited_rows, args.output_path)

    summary_path = args.summary_path or args.output_path.with_suffix(".summary.json")
    summary = {
        **summary,
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
    }
    write_json(summary, summary_path)


if __name__ == "__main__":
    main()
