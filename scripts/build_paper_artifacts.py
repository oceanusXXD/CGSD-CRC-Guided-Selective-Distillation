from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.paper_artifacts import build_paper_artifact_pack
from mias_dcms.utils import read_json, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen 3-figure / 3-table paper artifact JSON package from audited run outputs."
    )
    parser.add_argument("--run_records_path", required=True)
    parser.add_argument("--intervention_statistics_path", required=True)
    parser.add_argument("--matched_utility_path", required=True)
    parser.add_argument("--claim_audit_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_baselines", required=True)
    parser.add_argument("--evaluation_metrics", required=True)
    parser.add_argument("--selection_metrics", required=True)
    parser.add_argument("--cost_metrics", required=True)
    parser.add_argument("--judge_version", required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    runs = read_jsonl(Path(args.run_records_path))
    pack = build_paper_artifact_pack(
        runs,
        intervention_statistics=read_json(Path(args.intervention_statistics_path)),
        matched_utility=read_json(Path(args.matched_utility_path)),
        claim_audit=read_json(Path(args.claim_audit_path)),
        output_root=output_dir,
        expected_main_tables=("table1", "table2", "table3"),
        expected_figures=("fig1", "fig2", "fig3"),
        evaluation_metrics=_parse_csv(args.evaluation_metrics),
        selection_metrics=_parse_csv(args.selection_metrics),
        cost_metrics=_parse_csv(args.cost_metrics),
        expected_baselines=_parse_csv(args.expected_baselines),
        judge_version=str(args.judge_version),
        confidence=float(args.confidence),
        resamples=int(args.resamples),
        seed=int(args.seed),
    )
    _write_artifact_map(pack["figure_data"])
    _write_artifact_map(pack["main_tables"])
    _write_artifact_map(pack.get("appendix_tables", {}))
    write_json(pack["claim_evidence_map"], output_dir / "claim_evidence_map.json")
    write_json(pack["results_manifest"], output_dir / "results_manifest.json")
    write_json(pack, output_dir / "freeze_pack.json")
    print(json.dumps(pack, ensure_ascii=False, sort_keys=True))


def _write_artifact_map(artifacts: Mapping[str, Mapping[str, Any]]) -> None:
    for artifact in artifacts.values():
        write_json(dict(artifact), Path(str(artifact["path"])))


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    main()
