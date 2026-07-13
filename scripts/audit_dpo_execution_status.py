from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.dpo_execution_status import audit_dpo_execution_status
from mias_dcms.utils import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DPO execution progress from an execution manifest.")
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--print_full",
        action="store_true",
        help="Print the full status payload to stdout. By default only a compact summary is printed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest_path)
    existing_paths = _existing_manifest_paths(
        manifest,
        base_dirs=(PROJECT_ROOT, args.manifest_path.parent),
    )
    report = audit_dpo_execution_status(manifest, existing_paths=existing_paths)
    payload = {
        **report.as_dict(),
        "manifest_path": str(args.manifest_path),
        "output_path": str(args.output_path),
    }
    write_json(payload, args.output_path)
    stdout_payload = payload if args.print_full else _compact_stdout_payload(payload)
    print(json.dumps(stdout_payload, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if report.is_complete else 1)


def _compact_stdout_payload(payload: dict[str, Any]) -> dict[str, Any]:
    runs = payload.get("runs", [])
    next_stage_counts: dict[str, int] = {}
    blocked_method_counts: dict[str, int] = {}
    in_progress_method_counts: dict[str, int] = {}
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            next_stage = str(run.get("next_stage") or "complete")
            next_stage_counts[next_stage] = next_stage_counts.get(next_stage, 0) + 1
            method = str(run.get("method") or "unknown")
            execution_status = str(run.get("execution_status") or "unknown")
            if execution_status == "blocked":
                blocked_method_counts[method] = blocked_method_counts.get(method, 0) + 1
            elif execution_status == "in_progress":
                in_progress_method_counts[method] = in_progress_method_counts.get(method, 0) + 1
    return {
        "is_complete": payload.get("is_complete"),
        "run_count": payload.get("run_count"),
        "completed_run_count": payload.get("completed_run_count"),
        "in_progress_run_count": payload.get("in_progress_run_count"),
        "blocked_run_count": payload.get("blocked_run_count"),
        "failed_run_count": payload.get("failed_run_count"),
        "issue_count": len(payload.get("issues", [])) if isinstance(payload.get("issues"), list) else None,
        "next_stage_counts": dict(sorted(next_stage_counts.items())),
        "in_progress_method_counts": dict(sorted(in_progress_method_counts.items())),
        "blocked_method_counts": dict(sorted(blocked_method_counts.items())),
        "manifest_path": payload.get("manifest_path"),
        "output_path": payload.get("output_path"),
    }


def _existing_manifest_paths(manifest: dict[str, Any], *, base_dirs: tuple[Path, ...]) -> set[str]:
    existing: set[str] = set()
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        return existing
    for run in runs:
        if not isinstance(run, dict):
            continue
        for stage in run.get("stages", []):
            if not isinstance(stage, dict):
                continue
            for field_name in ("inputs", "outputs"):
                paths = stage.get(field_name, {})
                if not isinstance(paths, dict):
                    continue
                for path in paths.values():
                    path_text = str(path)
                    candidate = Path(path_text)
                    if candidate.is_absolute() and candidate.exists():
                        existing.add(path_text)
                    elif any((base_dir / candidate).exists() for base_dir in base_dirs):
                        existing.add(path_text)
    return existing


if __name__ == "__main__":
    main()
