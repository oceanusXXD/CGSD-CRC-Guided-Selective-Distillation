from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.dpo_execution_manifest import DPO_EXECUTION_STAGES
from mias_dcms.utils import read_json, write_json


OUTPUT_PREVIEW_CHARS = 2000
SUMMARY_PREVIEW_ITEMS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one stage from a DPO execution manifest for matching planned runs."
    )
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--stage", choices=DPO_EXECUTION_STAGES, required=True)
    parser.add_argument("--run_id")
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report_path", type=Path)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest_path)
    matched_runs = _matching_runs(
        manifest,
        run_id=args.run_id,
        method=args.method,
        seed=args.seed,
        limit=int(args.limit),
    )
    records: list[dict[str, Any]] = []
    for run in matched_runs:
        stage = _stage_by_name(run, str(args.stage))
        commands = stage.get("commands", [])
        if not commands:
            records.append(
                {
                    "run_id": run.get("run_id"),
                    "stage": str(args.stage),
                    "status": "skipped",
                    "reason": "no_commands",
                }
            )
            continue
        for command_index, command in enumerate(commands):
            command_text = str(command)
            record = {
                "run_id": run.get("run_id"),
                "method": run.get("method"),
                "seed": run.get("seed"),
                "stage": str(args.stage),
                "command_index": command_index,
                "command": command_text,
                "dry_run": bool(args.dry_run),
            }
            if args.dry_run:
                record["status"] = "dry_run"
                records.append(record)
                print(json.dumps(record, ensure_ascii=False, sort_keys=True))
                continue
            completed = subprocess.run(
                shlex.split(command_text),
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            record["returncode"] = completed.returncode
            record["status"] = "complete" if completed.returncode == 0 else "failed"
            _attach_completed_output(record, stdout=completed.stdout, stderr=completed.stderr)
            records.append(record)
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))
            if completed.returncode != 0:
                _write_report(args.report_path, args=args, records=records)
                raise SystemExit(completed.returncode)

    _write_report(args.report_path, args=args, records=records)
    raise SystemExit(0)


def _matching_runs(
    manifest: dict[str, Any],
    *,
    run_id: str | None,
    method: str | None,
    seed: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("manifest runs must be a list")
    matched: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        if run_id is not None and str(run.get("run_id")) != str(run_id):
            continue
        if method is not None and str(run.get("method")) != str(method):
            continue
        if seed is not None and int(run.get("seed", -1)) != int(seed):
            continue
        matched.append(run)
        if limit > 0 and len(matched) >= limit:
            break
    return matched


def _stage_by_name(run: dict[str, Any], stage_name: str) -> dict[str, Any]:
    stages = run.get("stages", [])
    if not isinstance(stages, list):
        raise ValueError(f"run {run.get('run_id')!r} stages must be a list")
    for stage in stages:
        if isinstance(stage, dict) and str(stage.get("stage")) == str(stage_name):
            return stage
    raise ValueError(f"run {run.get('run_id')!r} has no stage {stage_name!r}")


def _attach_completed_output(record: dict[str, Any], *, stdout: str, stderr: str) -> None:
    stdout_text = stdout.strip()
    stderr_text = stderr.strip()
    record.update(_compact_text_field("stdout", stdout_text))
    record.update(_compact_text_field("stderr", stderr_text))
    parsed_stdout = _parse_json_object(stdout_text)
    if parsed_stdout is not None:
        record["stdout_json_summary"] = _summarize_json_object(parsed_stdout)


def _compact_text_field(prefix: str, text: str) -> dict[str, Any]:
    return {
        f"{prefix}_preview": text[:OUTPUT_PREVIEW_CHARS],
        f"{prefix}_truncated": len(text) > OUTPUT_PREVIEW_CHARS,
        f"{prefix}_char_count": len(text),
    }


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _summarize_json_object(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)
            if key.endswith("_preview"):
                summary[key] = value[:SUMMARY_PREVIEW_ITEMS]
            else:
                summary[f"{key}_preview"] = value[:SUMMARY_PREVIEW_ITEMS]
        elif isinstance(value, dict):
            summary[key] = value
        else:
            summary[key] = value
    return summary


def _write_report(report_path: Path | None, *, args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    if report_path is None:
        return
    payload = {
        "manifest_path": str(args.manifest_path),
        "stage": str(args.stage),
        "run_id": args.run_id,
        "method": args.method,
        "seed": args.seed,
        "limit": int(args.limit),
        "dry_run": bool(args.dry_run),
        "command_count": len(records),
        "failed_command_count": sum(1 for record in records if record.get("status") == "failed"),
        "records": records,
    }
    write_json(payload, report_path)


if __name__ == "__main__":
    main()
