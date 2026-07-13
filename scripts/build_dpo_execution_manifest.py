from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.dpo_execution_manifest import build_dpo_execution_manifest
from mias_dcms.experiment_run_matrix import config_payload_sha256
from mias_dcms.utils import read_json, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an ordered DPO execution manifest from a planned run matrix.")
    parser.add_argument("--run_matrix_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--config_path",
        type=Path,
        help="Optional source config. When supplied, refuse a stale run matrix built from another config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.run_matrix_path)
    manifest = build_dpo_execution_manifest(rows)
    expected_source_config_sha256 = None
    if args.config_path is not None:
        expected_source_config_sha256 = config_payload_sha256(read_json(args.config_path))
        observed_source_config_sha256 = manifest.get("source_config_sha256")
        if observed_source_config_sha256 != expected_source_config_sha256:
            payload = {
                **manifest,
                "is_ready": False,
                "issue_count": int(manifest.get("issue_count", 0)) + 1,
                "issues": [
                    *list(manifest.get("issues", [])),
                    {
                        "code": "source_config_hash_mismatch",
                        "expected_source_config_sha256": expected_source_config_sha256,
                        "observed_source_config_sha256": observed_source_config_sha256,
                    },
                ],
                "run_matrix_path": str(args.run_matrix_path),
                "output_path": str(args.output_path),
                "config_path": str(args.config_path),
            }
            write_json(payload, args.output_path)
            print(json.dumps(_compact_stdout_payload(payload), ensure_ascii=False, sort_keys=True))
            raise SystemExit(1)
    payload = {
        **manifest,
        "run_matrix_path": str(args.run_matrix_path),
        "output_path": str(args.output_path),
        "config_path": str(args.config_path) if args.config_path is not None else None,
        "expected_source_config_sha256": expected_source_config_sha256,
    }
    write_json(payload, args.output_path)
    print(json.dumps(_compact_stdout_payload(payload), ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if payload["is_ready"] else 1)


def _compact_stdout_payload(payload: dict[str, object]) -> dict[str, object]:
    runs = payload.get("runs", [])
    stage_command_counts: dict[str, int] = {}
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            for stage in run.get("stages", []):
                if not isinstance(stage, dict):
                    continue
                stage_name = str(stage.get("stage", "unknown"))
                commands = stage.get("commands", [])
                command_count = len(commands) if isinstance(commands, list) else 0
                stage_command_counts[stage_name] = stage_command_counts.get(stage_name, 0) + command_count
    return {
        "is_ready": payload.get("is_ready"),
        "run_count": payload.get("run_count"),
        "issue_count": payload.get("issue_count"),
        "stage_order": payload.get("stage_order"),
        "stage_command_counts": dict(sorted(stage_command_counts.items())),
        "run_matrix_path": payload.get("run_matrix_path"),
        "output_path": payload.get("output_path"),
    }


if __name__ == "__main__":
    main()
