from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mias_dcms.binary_reaudit import sha256_file
from mias_dcms.utils import read_json, read_jsonl, write_json


DEFAULT_LIMITATIONS = [
    "one training seed only",
    "no confidence interval or permutation test",
    "no causal score-intervention evidence",
    "feasibility gate; not paper-result scale",
]


def build_binary_single_seed_gate_summary(
    *,
    run_root: Path,
    dataset: str,
    config_snapshot_path: Path,
    selection_config_snapshot_path: Path | None,
    protocol_manifest_path: Path,
    source_manifest_path: Path | None,
    seed: int,
    expected_test_size: int,
    entropy_margin_checkpoint_name: str = "entropy_margin",
    additional_limitations: list[str] | None = None,
) -> dict[str, Any]:
    """Validate and collect one binary feasibility-gate result.

    Entropy and absolute binary margin are only represented by one downstream
    adapter when their selected id sets are exactly identical.  The original
    selection records remain embedded so the equivalence is auditable.
    """
    if expected_test_size <= 0:
        raise ValueError("expected_test_size must be positive")
    execution_config_hash = sha256_file(config_snapshot_path)
    selection_config_snapshot_path = selection_config_snapshot_path or config_snapshot_path
    selection_config_hash = sha256_file(selection_config_snapshot_path)
    selection_root = run_root / "selection" / f"seed_{seed}"
    selections = {
        method: _load_selection(
            selection_root / method / "selection_summary.json",
            method=method,
            dataset=dataset,
            seed=seed,
            config_hash=selection_config_hash,
        )
        for method in ("random", "entropy", "margin")
    }
    entropy_ids = set(_selected_ids(selections["entropy"]))
    margin_ids = set(_selected_ids(selections["margin"]))
    overlap = len(entropy_ids & margin_ids)
    union = len(entropy_ids | margin_ids)
    jaccard = float(overlap / union) if union else 1.0
    if jaccard != 1.0:
        raise ValueError("Entropy and Margin selected different ids; they cannot share a downstream checkpoint")

    method_root = run_root / "method_runs" / f"seed_{seed}"
    random_run = _completed_method_run(
        method_root / "random" / "round_1",
        expected_test_size=expected_test_size,
    )
    entropy_margin_run = _completed_method_run(
        method_root / entropy_margin_checkpoint_name / "round_1",
        expected_test_size=expected_test_size,
    )
    random_metrics = random_run["fixed_test_metrics"]
    entropy_metrics = entropy_margin_run["fixed_test_metrics"]
    delta = {
        name: float(random_metrics[name]) - float(entropy_metrics[name])
        for name in sorted(set(random_metrics) & set(entropy_metrics))
        if _is_number(random_metrics[name]) and _is_number(entropy_metrics[name])
    }
    limitations = [*DEFAULT_LIMITATIONS]
    config_provenance_aligned = execution_config_hash == selection_config_hash
    if not config_provenance_aligned:
        limitations.append(
            "selection records use a different frozen config hash than the execution snapshot; "
            "this result has a configuration-provenance exception"
        )
    for limitation in additional_limitations or []:
        cleaned = str(limitation).strip()
        if cleaned and cleaned not in limitations:
            limitations.append(cleaned)

    return {
        "schema_version": "binary-single-seed-gate-summary-v1",
        "status": (
            "completed_single_seed_feasibility_gate"
            if config_provenance_aligned
            else "completed_with_config_provenance_exception"
        ),
        "dataset": str(dataset),
        "seed": int(seed),
        "config_snapshot_path": str(config_snapshot_path),
        "config_snapshot_sha256": execution_config_hash,
        "selection_config_snapshot_path": str(selection_config_snapshot_path),
        "selection_config_snapshot_sha256": selection_config_hash,
        "config_provenance_aligned": config_provenance_aligned,
        "protocol_manifest_path": str(protocol_manifest_path),
        "source_manifest_path": str(source_manifest_path) if source_manifest_path is not None else None,
        "entropy_margin_equivalence": {
            "entropy_selected_count": len(entropy_ids),
            "margin_selected_count": len(margin_ids),
            "selected_id_overlap": overlap,
            "jaccard": jaccard,
            "shared_downstream_checkpoint": str(
                method_root / entropy_margin_checkpoint_name / "round_1" / "model"
            ),
        },
        "selection_methods": {
            "random": _method_summary(
                selections["random"],
                random_run,
                selection_summary_path=selection_root / "random" / "selection_summary.json",
            ),
            "entropy": _method_summary(
                selections["entropy"],
                entropy_margin_run,
                selection_summary_path=selection_root / "entropy" / "selection_summary.json",
            ),
        },
        "random_minus_entropy_margin_fixed_test_delta": delta,
        "limitations": limitations,
    }


def _load_selection(
    path: Path,
    *,
    method: str,
    dataset: str,
    seed: int,
    config_hash: str,
) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("method") != method:
        raise ValueError(f"{path} has method {payload.get('method')!r}, expected {method!r}")
    if payload.get("dataset") != dataset:
        raise ValueError(f"{path} has dataset {payload.get('dataset')!r}, expected {dataset!r}")
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"{path} does not use seed {seed}")
    if payload.get("config_hash") != config_hash:
        raise ValueError(f"{path} config hash does not match the frozen config snapshot")
    selected_ids = _selected_ids(payload)
    if int(payload.get("budget", -1)) != len(selected_ids):
        raise ValueError(f"{path} budget does not equal its unique selected-id count")
    return payload


def _selected_ids(payload: Mapping[str, Any]) -> list[str]:
    selected_ids = [str(sample_id) for sample_id in payload.get("selected_ids", [])]
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selection summary must include non-empty unique selected ids")
    return selected_ids


def _completed_method_run(round_dir: Path, *, expected_test_size: int) -> dict[str, Any]:
    model_dir = round_dir / "model"
    state = read_json(model_dir / "training_state.json")
    if state.get("run_complete") is not True:
        raise ValueError(f"training is incomplete: {model_dir}")
    model_config = read_json(model_dir / "model_config.json")
    if model_config.get("input_format") != "chat_binary":
        raise ValueError(f"checkpoint does not record chat_binary input format: {model_dir}")
    training_summary_path = round_dir / "training_round_summary.json"
    training_summary = read_json(training_summary_path)
    if training_summary.get("input_format") != "chat_binary":
        raise ValueError(f"training summary does not record chat_binary input format: {training_summary_path}")
    predictions_path = round_dir / "fixed_test_predictions.jsonl"
    prediction_count = len(read_jsonl(predictions_path))
    if prediction_count != expected_test_size:
        raise ValueError(
            f"fixed-test prediction count for {round_dir} is {prediction_count}, expected {expected_test_size}"
        )
    return {
        "checkpoint_path": str(model_dir),
        "training_summary_path": str(training_summary_path),
        "fixed_test_metrics_path": str(round_dir / "fixed_test_metrics.json"),
        "fixed_test_predictions_path": str(predictions_path),
        "fixed_test_prediction_count": prediction_count,
        "fixed_test_metrics": read_json(round_dir / "fixed_test_metrics.json"),
    }


def _method_summary(
    selection: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    selection_summary_path: Path,
) -> dict[str, Any]:
    return {
        "checkpoint_path": run["checkpoint_path"],
        "training_summary_path": run["training_summary_path"],
        "fixed_test_metrics_path": run["fixed_test_metrics_path"],
        "fixed_test_predictions_path": run["fixed_test_predictions_path"],
        "fixed_test_prediction_count": run["fixed_test_prediction_count"],
        "fixed_test_metrics": run["fixed_test_metrics"],
        "selection_summary_path": str(selection_summary_path),
        "selection": dict(selection),
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def write_binary_single_seed_gate_summary(
    summary: Mapping[str, Any], *, output_path: Path
) -> None:
    write_json(dict(summary), output_path)
