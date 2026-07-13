from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.experiment_gate_readiness import audit_experiment_gate_readiness
from mias_dcms.utils import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit MIAS/DCMS Gate 0-10 readiness from declared experiment evidence."
    )
    parser.add_argument("--evidence_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--require_existing_paths",
        action="store_true",
        help="Treat string evidence values as filesystem paths that must exist.",
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=PROJECT_ROOT,
        help="Base directory for relative evidence paths when --require_existing_paths is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_experiment_gate_readiness(
        read_json(args.evidence_path),
        require_existing_paths=bool(args.require_existing_paths),
        base_dir=args.base_dir,
    )
    payload = {
        **report.as_dict(),
        "evidence_path": str(args.evidence_path),
        "output_path": str(args.output_path),
    }
    write_json(payload, args.output_path)
    print(json.dumps(_compact_stdout_payload(payload), ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if payload["is_ready"] else 1)


def _compact_stdout_payload(payload: dict[str, object]) -> dict[str, object]:
    gates = payload.get("gates", {})
    gate_missing_counts: dict[str, int] = {}
    gate_present_counts: dict[str, int] = {}
    if isinstance(gates, dict):
        for gate_id, gate in gates.items():
            if not isinstance(gate, dict):
                continue
            missing = gate.get("missing_evidence", [])
            present = gate.get("present_evidence", [])
            gate_missing_counts[str(gate_id)] = len(missing) if isinstance(missing, list) else 0
            gate_present_counts[str(gate_id)] = len(present) if isinstance(present, list) else 0
    return {
        "is_ready": payload.get("is_ready"),
        "gate_count": payload.get("gate_count"),
        "ready_gate_count": payload.get("ready_gate_count"),
        "blocked_gate_count": payload.get("blocked_gate_count"),
        "missing_evidence_count": payload.get("missing_evidence_count"),
        "ready_gates": payload.get("ready_gates"),
        "blocked_gates": payload.get("blocked_gates"),
        "gate_present_counts": dict(sorted(gate_present_counts.items())),
        "gate_missing_counts": dict(sorted(gate_missing_counts.items())),
        "issue_count": len(payload.get("issues", [])) if isinstance(payload.get("issues"), list) else 0,
        "evidence_path": payload.get("evidence_path"),
        "output_path": payload.get("output_path"),
    }


if __name__ == "__main__":
    main()
