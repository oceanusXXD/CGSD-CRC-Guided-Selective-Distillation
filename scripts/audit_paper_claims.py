from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.paper_claim_audit import audit_paper_claim_evidence, audit_paper_text_claims
from mias_dcms.utils import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit paper claim-to-evidence mappings and banned overclaims before result freeze."
    )
    parser.add_argument("--claims_path", required=True)
    parser.add_argument("--evidence_path", required=True)
    parser.add_argument("--requirements_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--paper_text_path")
    parser.add_argument("--minimum_seed_count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    claims = read_json(Path(args.claims_path))
    evidence = read_json(Path(args.evidence_path))
    requirements = read_json(Path(args.requirements_path))
    report = audit_paper_claim_evidence(
        claims,
        evidence=evidence,
        required_evidence_by_claim_type=requirements,
        minimum_seed_count=int(args.minimum_seed_count),
    )
    payload = report.as_dict()
    payload["claims_path"] = str(args.claims_path)
    payload["evidence_path"] = str(args.evidence_path)
    payload["requirements_path"] = str(args.requirements_path)

    if args.paper_text_path:
        text = Path(args.paper_text_path).read_text(encoding="utf-8")
        text_issues = audit_paper_text_claims(text)
        payload["paper_text_path"] = str(args.paper_text_path)
        payload["paper_text_issues"] = text_issues
        payload["issues"] = [*payload["issues"], *text_issues]
        payload["is_ready"] = not payload["issues"]

    write_json(payload, Path(args.output_path))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if payload["is_ready"] else 1)


if __name__ == "__main__":
    main()
