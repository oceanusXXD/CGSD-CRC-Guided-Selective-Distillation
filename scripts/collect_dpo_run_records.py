from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.dpo_run_record_collection import SUMMARY_ARTIFACTS, collect_dpo_run_records
from mias_dcms.utils import read_json, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect DPO execution artifacts into run-level records and readiness report."
    )
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--output_records_path", type=Path, required=True)
    parser.add_argument("--output_report_path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest_path)
    artifact_payloads = _read_manifest_artifacts(manifest, base_dir=args.manifest_path.parent)
    report = collect_dpo_run_records(manifest, artifact_payloads=artifact_payloads)
    payload = {
        **report.as_dict(),
        "manifest_path": str(args.manifest_path),
        "output_records_path": str(args.output_records_path),
        "output_report_path": str(args.output_report_path),
    }
    write_jsonl(report.records, args.output_records_path)
    write_json(payload, args.output_report_path)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if report.is_ready else 1)


def _read_manifest_artifacts(manifest: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    payloads: dict[str, Any] = {}
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        return payloads
    for run in runs:
        if not isinstance(run, dict) or str(run.get("run_status")) == "failed":
            continue
        artifacts = run.get("artifacts", {})
        if not isinstance(artifacts, dict):
            continue
        for artifact_name in SUMMARY_ARTIFACTS:
            path_text = str(artifacts.get(artifact_name, ""))
            if not path_text or path_text in payloads:
                continue
            resolved = _resolve_artifact_path(path_text, base_dir=base_dir)
            if resolved is None:
                continue
            payloads[path_text] = read_jsonl(resolved) if resolved.suffix == ".jsonl" else read_json(resolved)
    return payloads


def _resolve_artifact_path(path_text: str, *, base_dir: Path) -> Path | None:
    path = Path(path_text)
    candidates = [path] if path.is_absolute() else [base_dir / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


if __name__ == "__main__":
    main()
