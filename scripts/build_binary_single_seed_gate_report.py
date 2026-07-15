from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.binary_single_seed_report import (  # noqa: E402
    build_binary_single_seed_gate_summary,
    write_binary_single_seed_gate_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and save one completed binary single-seed feasibility-gate summary."
    )
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config_snapshot_path", type=Path, required=True)
    parser.add_argument(
        "--selection_config_snapshot_path",
        type=Path,
        help="Frozen config whose hash was recorded in selection summaries; defaults to --config_snapshot_path.",
    )
    parser.add_argument("--protocol_manifest_path", type=Path, required=True)
    parser.add_argument("--source_manifest_path", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected_test_size", type=int, default=1000)
    parser.add_argument("--entropy_margin_checkpoint_name", default="entropy_margin")
    parser.add_argument("--additional_limitation", action="append", default=[])
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_binary_single_seed_gate_summary(
        run_root=args.run_root,
        dataset=str(args.dataset),
        config_snapshot_path=args.config_snapshot_path,
        selection_config_snapshot_path=args.selection_config_snapshot_path,
        protocol_manifest_path=args.protocol_manifest_path,
        source_manifest_path=args.source_manifest_path,
        seed=int(args.seed),
        expected_test_size=int(args.expected_test_size),
        entropy_margin_checkpoint_name=str(args.entropy_margin_checkpoint_name),
        additional_limitations=list(args.additional_limitation),
    )
    write_binary_single_seed_gate_summary(summary, output_path=args.output_path)
    print(
        json.dumps(
            {
                "dataset": summary["dataset"],
                "status": summary["status"],
                "output_path": str(args.output_path),
                "entropy_margin_jaccard": summary["entropy_margin_equivalence"]["jaccard"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
