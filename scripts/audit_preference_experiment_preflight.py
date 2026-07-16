from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mias_dcms.preference_experiment_preflight import (
    PreferenceExperimentPreflightInputs,
    audit_preference_experiment_preflight,
)
from mias_dcms.utils import read_json, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit preference/DPO fixed-pool artifacts before running main experiments."
    )
    parser.add_argument("--active_pool_path", type=Path, required=True)
    parser.add_argument("--oracle_store_path", type=Path, required=True)
    parser.add_argument("--logprobs_path", type=Path, required=True)
    parser.add_argument("--split_manifest_path", type=Path, required=True)
    parser.add_argument("--run_matrix_path", type=Path, required=True)
    parser.add_argument("--mias_seed_rows_path", type=Path)
    parser.add_argument("--mias_seed_features_path", type=Path)
    parser.add_argument("--mias_pool_features_path", type=Path)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--expected_methods", default="")
    parser.add_argument("--expected_seeds", default="")
    parser.add_argument("--id_field", default="sample_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oracle_store = _read_oracle_store(args.oracle_store_path, id_field=str(args.id_field))
    report = audit_preference_experiment_preflight(
        PreferenceExperimentPreflightInputs(
            active_pool=read_jsonl(args.active_pool_path),
            oracle_store=oracle_store,
            logprob_rows=read_jsonl(args.logprobs_path),
            split_manifest=read_json(args.split_manifest_path),
            run_matrix=read_jsonl(args.run_matrix_path),
            mias_seed_rows=_read_optional_jsonl(args.mias_seed_rows_path),
            mias_seed_features=_read_optional_jsonl(args.mias_seed_features_path),
            mias_pool_features=_read_optional_jsonl(args.mias_pool_features_path),
            expected_active_pool_path=str(args.active_pool_path),
            expected_oracle_store_path=str(args.oracle_store_path),
            expected_logprobs_path=str(args.logprobs_path),
            expected_mias_seed_rows_path=_optional_path(args.mias_seed_rows_path),
            expected_mias_seed_features_path=_optional_path(args.mias_seed_features_path),
            expected_mias_pool_features_path=_optional_path(args.mias_pool_features_path),
            expected_methods=_parse_csv(args.expected_methods),
            expected_seeds=[int(value) for value in _parse_csv(args.expected_seeds)],
            id_field=str(args.id_field),
        )
    )
    payload = {
        **report.as_dict(),
        "active_pool_path": str(args.active_pool_path),
        "oracle_store_path": str(args.oracle_store_path),
        "logprobs_path": str(args.logprobs_path),
        "split_manifest_path": str(args.split_manifest_path),
        "run_matrix_path": str(args.run_matrix_path),
        "mias_seed_rows_path": _optional_path(args.mias_seed_rows_path),
        "mias_seed_features_path": _optional_path(args.mias_seed_features_path),
        "mias_pool_features_path": _optional_path(args.mias_pool_features_path),
    }
    write_json(payload, args.output_path)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if report.is_ready else 1)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _read_optional_jsonl(path: Path | None) -> list[dict[str, object]] | None:
    return read_jsonl(path) if path is not None else None


def _optional_path(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _read_oracle_store(path: Path, *, id_field: str) -> dict[str, dict[str, object]]:
    if path.suffix == ".json":
        payload = read_json(path)
        if all(isinstance(value, dict) for value in payload.values()):
            return {str(sample_id): dict(row) for sample_id, row in payload.items()}
        raise ValueError("oracle store JSON must be an object keyed by sample id")

    oracle_rows = read_jsonl(path)
    return {
        str(row.get(id_field, row.get("id"))): row
        for row in oracle_rows
        if row.get(id_field, row.get("id")) is not None
    }


if __name__ == "__main__":
    main()
