from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.dpo_run_pack import DPO_MAIN_METHODS
from mias_dcms.experiment_run_matrix import (
    build_experiment_run_matrix,
    config_payload_sha256,
    validate_experiment_run_matrix,
)
from mias_dcms.utils import read_json, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a planned DPO main-experiment run matrix and readiness summary."
    )
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_matrix_path", required=True)
    parser.add_argument("--output_summary_path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = read_json(Path(args.config_path))
        source_config_sha256 = config_payload_sha256(config)
        rows = build_experiment_run_matrix(
            datasets=_required_sequence(config, "datasets"),
            models=_required_sequence(config, "models"),
            budgets=[int(value) for value in _required_sequence(config, "budgets")],
            seeds=[int(value) for value in _required_sequence(config, "seeds")],
            methods=[str(value) for value in config.get("methods", DPO_MAIN_METHODS)],
            artifact_root=str(config.get("artifact_root", "experiments/runs/dpo_main")),
            training_config=_required_mapping(config, "training_config"),
            judge_config=_required_mapping(config, "judge_config"),
            data_config=dict(config.get("data_config", {})),
            evaluation_config=dict(config.get("evaluation_config", {})),
            source_config_sha256=source_config_sha256,
        )
        report = validate_experiment_run_matrix(
            rows,
            expected_datasets=[str(value) for value in _required_sequence(config, "datasets")],
            expected_models=[str(value) for value in _required_sequence(config, "models")],
            expected_budgets=[int(value) for value in _required_sequence(config, "budgets")],
            expected_seeds=[int(value) for value in _required_sequence(config, "seeds")],
            expected_methods=[str(value) for value in config.get("methods", DPO_MAIN_METHODS)],
            expected_source_config_sha256=source_config_sha256,
        )
        payload = report.as_dict()
        payload["config_path"] = str(args.config_path)
        payload["output_matrix_path"] = str(args.output_matrix_path)
        payload["output_summary_path"] = str(args.output_summary_path)
        payload["source_config_sha256"] = source_config_sha256
        if not report.is_ready:
            write_json(payload, Path(args.output_summary_path))
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            raise SystemExit(1)
        write_jsonl(rows, Path(args.output_matrix_path))
        write_json(payload, Path(args.output_summary_path))
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)


def _required_sequence(config: Mapping[str, Any], key: str) -> list[Any]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    return list(value)


def _required_mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return dict(value)


if __name__ == "__main__":
    main()
