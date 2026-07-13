from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.checkpoint_registry import register_initial_policy_checkpoint
from mias_dcms.utils import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register a real shared initial DPO policy checkpoint as Gate 4 evidence."
    )
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--output_manifest_path", type=Path, required=True)
    parser.add_argument("--evidence_path", type=Path)
    parser.add_argument("--base_dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--checkpoint_type", default="dpo_initial_policy_adapter")
    parser.add_argument("--evidence_key", default="dpo.initial_policy_checkpoint")
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--training_config_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_config = read_json(args.training_config_path) if args.training_config_path else {}
    report = register_initial_policy_checkpoint(
        checkpoint_path=args.checkpoint_path,
        output_manifest_path=args.output_manifest_path,
        evidence_path=args.evidence_path,
        base_dir=args.base_dir,
        checkpoint_type=str(args.checkpoint_type),
        evidence_key=str(args.evidence_key),
        model_name_or_path=str(args.model_name_or_path),
        training_config=training_config,
    )
    print(json.dumps(_compact_stdout_payload(report.as_dict()), ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if report.is_ready else 1)


def _compact_stdout_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        "is_ready": payload.get("is_ready"),
        "checkpoint_path": payload.get("checkpoint_path"),
        "checkpoint_type": payload.get("checkpoint_type"),
        "evidence_key": payload.get("evidence_key"),
        "present_files": payload.get("present_files"),
        "missing_files": payload.get("missing_files"),
        "issue_count": len(payload.get("issues", [])) if isinstance(payload.get("issues"), list) else 0,
    }


if __name__ == "__main__":
    main()
