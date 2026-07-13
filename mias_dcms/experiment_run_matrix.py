from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from itertools import product
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from mias_dcms.dpo_run_pack import DPO_MAIN_METHODS


REQUIRED_TRAINING_CONFIG_FIELDS = (
    "initialization",
    "optimizer",
    "learning_rate",
    "batch_size",
    "update_steps",
    "train_token_budget",
    "data_accumulation",
    "prompt_format",
    "generation_parameters",
)

REQUIRED_JUDGE_CONFIG_FIELDS = (
    "judge_version",
    "judge_prompt_hash",
    "evaluator",
)

REQUIRED_EXPERIMENT_ARTIFACTS = (
    ("active_pool_path", "active_pool.jsonl"),
    ("oracle_store_path", "oracle_store.jsonl"),
    ("logprobs_path", "logprobs.jsonl"),
    ("selection_summary_path", "selection_summary.json"),
    ("selected_ids_path", "selected_ids.json"),
    ("revealed_rows_path", "revealed_rows.jsonl"),
    ("dpo_train_rows_path", "dpo_train_rows.jsonl"),
    ("policy_adapter_path", "policy_adapter"),
    ("training_summary_path", "training_summary.json"),
    ("evaluation_metrics_path", "evaluation_metrics.json"),
    ("cost_report_path", "cost_report.json"),
)


@dataclass(frozen=True)
class ExperimentRunMatrixValidationReport:
    expected_run_count: int
    planned_run_count: int
    covered_methods: list[str]
    covered_datasets: list[str]
    covered_models: list[str]
    covered_budgets: list[int]
    covered_seeds: list[int]
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "expected_run_count": self.expected_run_count,
            "planned_run_count": self.planned_run_count,
            "covered_methods": list(self.covered_methods),
            "covered_datasets": list(self.covered_datasets),
            "covered_models": list(self.covered_models),
            "covered_budgets": list(self.covered_budgets),
            "covered_seeds": list(self.covered_seeds),
            "issues": [dict(issue) for issue in self.issues],
        }


def build_experiment_run_matrix(
    *,
    datasets: Sequence[str],
    models: Sequence[str],
    budgets: Sequence[int],
    seeds: Sequence[int],
    artifact_root: str | PurePosixPath,
    training_config: Mapping[str, Any],
    judge_config: Mapping[str, Any],
    data_config: Mapping[str, Any] | None = None,
    evaluation_config: Mapping[str, Any] | None = None,
    methods: Sequence[str] = DPO_MAIN_METHODS,
    source_config_sha256: str | None = None,
) -> list[dict[str, Any]]:
    _require_non_empty("datasets", datasets)
    _require_non_empty("models", models)
    _require_non_empty("budgets", budgets)
    _require_non_empty("seeds", seeds)
    _require_non_empty("methods", methods)
    _require_fields("training_config", training_config, REQUIRED_TRAINING_CONFIG_FIELDS)
    _require_fields("judge_config", judge_config, REQUIRED_JUDGE_CONFIG_FIELDS)

    training_payload = dict(training_config)
    judge_payload = dict(judge_config)
    data_payload = dict(data_config or {})
    evaluation_payload = dict(evaluation_config or {})
    training_hash = _stable_hash(training_payload)
    judge_hash = _stable_hash(judge_payload)
    data_hash = _stable_hash(data_payload)
    evaluation_hash = _stable_hash(evaluation_payload)
    root = PurePosixPath(str(artifact_root))

    rows: list[dict[str, Any]] = []
    for dataset, model, budget, seed, method in product(
        [str(value) for value in datasets],
        [str(value) for value in models],
        [int(value) for value in budgets],
        [int(value) for value in seeds],
        [str(value) for value in methods],
    ):
        run_dir = root / dataset / model / f"budget_{budget}" / f"seed_{seed}" / method
        run_id = build_experiment_run_id(
            dataset=dataset,
            model=model,
            budget=budget,
            seed=seed,
            method=method,
        )
        row = {
            "run_id": run_id,
            "dataset": dataset,
            "model": model,
            "budget": budget,
            "seed": seed,
            "method": method,
            "run_status": "planned",
            "failure_reason": "pending",
            "artifacts": {
                artifact_name: str(run_dir / filename)
                for artifact_name, filename in REQUIRED_EXPERIMENT_ARTIFACTS
            },
            "training_config": training_payload,
            "judge_config": judge_payload,
            "data_config": data_payload,
            "evaluation_config": evaluation_payload,
            "training_config_hash": training_hash,
            "judge_config_hash": judge_hash,
            "data_config_hash": data_hash,
            "evaluation_config_hash": evaluation_hash,
        }
        if source_config_sha256 is not None:
            row["source_config_sha256"] = str(source_config_sha256)
        row["config_hash"] = _stable_hash(
            {
                "dataset": dataset,
                "model": model,
                "budget": budget,
                "seed": seed,
                "method": method,
                "training_config_hash": training_hash,
                "judge_config_hash": judge_hash,
                "data_config_hash": data_hash,
                "evaluation_config_hash": evaluation_hash,
            }
        )
        rows.append(row)
    return rows


def validate_experiment_run_matrix(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_datasets: Sequence[str],
    expected_models: Sequence[str],
    expected_budgets: Sequence[int],
    expected_seeds: Sequence[int],
    expected_methods: Sequence[str] = DPO_MAIN_METHODS,
    expected_source_config_sha256: str | None = None,
) -> ExperimentRunMatrixValidationReport:
    planned_rows = [dict(row) for row in rows]
    issues: list[dict[str, Any]] = []
    by_run_id: dict[str, list[dict[str, Any]]] = {}

    for row_index, row in enumerate(planned_rows):
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            run_id = build_experiment_run_id(
                dataset=row.get("dataset", ""),
                model=row.get("model", ""),
                budget=int(row.get("budget", 0)),
                seed=int(row.get("seed", 0)),
                method=row.get("method", ""),
            )
            issues.append({"code": "missing_run_id", "row_index": row_index, "run_id": run_id})
        by_run_id.setdefault(run_id, []).append(row)
        if len(by_run_id[run_id]) > 1:
            issues.append({"code": "duplicate_run_id", "row_index": row_index, "run_id": run_id})

        if str(row.get("run_status", "")) != "planned":
            issues.append(
                {
                    "code": "invalid_planned_run_status",
                    "row_index": row_index,
                    "run_id": run_id,
                    "run_status": row.get("run_status"),
                }
            )
        _validate_artifact_paths(row, row_index=row_index, run_id=run_id, issues=issues)
        if expected_source_config_sha256 is not None:
            observed_source_hash = str(row.get("source_config_sha256", ""))
            if observed_source_hash != str(expected_source_config_sha256):
                issues.append(
                    {
                        "code": "source_config_hash_mismatch",
                        "row_index": row_index,
                        "run_id": run_id,
                        "expected_source_config_sha256": str(expected_source_config_sha256),
                        "observed_source_config_sha256": observed_source_hash,
                    }
                )

    expected_run_ids = [
        build_experiment_run_id(
            dataset=dataset,
            model=model,
            budget=int(budget),
            seed=int(seed),
            method=method,
        )
        for dataset, model, budget, seed, method in product(
            [str(value) for value in expected_datasets],
            [str(value) for value in expected_models],
            [int(value) for value in expected_budgets],
            [int(value) for value in expected_seeds],
            [str(value) for value in expected_methods],
        )
    ]
    for run_id in expected_run_ids:
        if run_id not in by_run_id:
            issues.append({"code": "missing_planned_run", "run_id": run_id})

    _validate_shared_hash(
        planned_rows,
        hash_field="training_config_hash",
        issue_code="training_config_hash_drift",
        issues=issues,
    )
    _validate_shared_hash(
        planned_rows,
        hash_field="judge_config_hash",
        issue_code="judge_config_hash_drift",
        issues=issues,
    )
    _validate_shared_hash(
        planned_rows,
        hash_field="evaluation_config_hash",
        issue_code="evaluation_config_hash_drift",
        issues=issues,
    )

    return ExperimentRunMatrixValidationReport(
        expected_run_count=len(expected_run_ids),
        planned_run_count=len(planned_rows),
        covered_methods=_ordered_coverage(planned_rows, "method", expected_methods),
        covered_datasets=_sorted_strings(row.get("dataset") for row in planned_rows),
        covered_models=_sorted_strings(row.get("model") for row in planned_rows),
        covered_budgets=_sorted_ints(row.get("budget") for row in planned_rows),
        covered_seeds=_sorted_ints(row.get("seed") for row in planned_rows),
        issues=issues,
    )


def build_experiment_run_id(
    *,
    dataset: Any,
    model: Any,
    budget: int,
    seed: int,
    method: Any,
) -> str:
    return "__".join(
        [
            _slug(dataset),
            _slug(model),
            f"budget{int(budget)}",
            f"seed{int(seed)}",
            _slug(method),
        ]
    )


def _validate_artifact_paths(
    row: Mapping[str, Any],
    *,
    row_index: int,
    run_id: str,
    issues: list[dict[str, Any]],
) -> None:
    artifacts = row.get("artifacts")
    if not isinstance(artifacts, Mapping):
        issues.append({"code": "missing_artifacts", "row_index": row_index, "run_id": run_id})
        return
    for artifact_name, _filename in REQUIRED_EXPERIMENT_ARTIFACTS:
        value = artifacts.get(artifact_name)
        if not str(value or "").strip():
            issues.append(
                {
                    "code": "missing_artifact_path",
                    "row_index": row_index,
                    "run_id": run_id,
                    "artifact": artifact_name,
                }
            )


def _validate_shared_hash(
    rows: Sequence[Mapping[str, Any]],
    *,
    hash_field: str,
    issue_code: str,
    issues: list[dict[str, Any]],
) -> None:
    hashes_by_setting: dict[tuple[str, str, int, int], set[str]] = {}
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("model", "")),
            int(row.get("budget", 0)),
            int(row.get("seed", 0)),
        )
        hashes_by_setting.setdefault(key, set()).add(str(row.get(hash_field, "")))
    for key, values in sorted(hashes_by_setting.items()):
        if len(values) > 1:
            issues.append(
                {
                    "code": issue_code,
                    "dataset": key[0],
                    "model": key[1],
                    "budget": key[2],
                    "seed": key[3],
                    "hash_values": sorted(values),
                }
            )


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def config_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Return a canonical digest of the source config used to build a run matrix."""
    return _stable_hash(payload)


def _require_non_empty(name: str, values: Sequence[Any]) -> None:
    if not list(values):
        raise ValueError(f"{name} must not be empty")


def _require_fields(
    name: str,
    payload: Mapping[str, Any],
    required_fields: Sequence[str],
) -> None:
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"{name} missing required fields: {', '.join(missing)}")


def _slug(value: Any) -> str:
    text = str(value).strip().lower().replace(".", "_")
    text = re.sub(r"[^a-z0-9-]+", "_", text)
    return text.strip("_-") or "unknown"


def _ordered_coverage(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    expected_order: Sequence[str],
) -> list[str]:
    present = {str(row.get(key)) for row in rows}
    ordered = [str(value) for value in expected_order if str(value) in present]
    extras = sorted(present - set(ordered))
    return [*ordered, *extras]


def _sorted_strings(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if str(value) not in {"", "None"}})


def _sorted_ints(values: Iterable[Any]) -> list[int]:
    parsed: set[int] = set()
    for value in values:
        try:
            parsed.add(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(parsed)
