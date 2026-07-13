from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mias_dcms.records import RunRecord
from mias_dcms.result_aggregation import aggregate_paper_metric_table


DEFAULT_MAIN_TABLES = ("table1", "table2", "table3")
DEFAULT_FIGURES = ("fig1", "fig2", "fig3")


def build_paper_artifact_pack(
    runs: Iterable[RunRecord | Mapping[str, Any]],
    *,
    intervention_statistics: Mapping[str, Any],
    matched_utility: Mapping[str, Any],
    claim_audit: Mapping[str, Any],
    output_root: str | Path,
    expected_main_tables: Sequence[str] = DEFAULT_MAIN_TABLES,
    expected_figures: Sequence[str] = DEFAULT_FIGURES,
    evaluation_metrics: Sequence[str],
    selection_metrics: Sequence[str],
    cost_metrics: Sequence[str],
    expected_baselines: Sequence[str],
    judge_version: str,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    run_list = [_coerce_run(run) for run in runs]
    if not run_list:
        raise ValueError("runs must not be empty")
    if not bool(claim_audit.get("is_ready", False)):
        raise ValueError("claim_audit must be ready before paper artifact generation")
    if not bool(intervention_statistics.get("is_ready", False)):
        raise ValueError("intervention_statistics must be ready before paper artifact generation")
    if not str(judge_version).strip():
        raise ValueError("judge_version must not be empty")

    observed_methods = {run.method for run in run_list}
    missing_baselines = [method for method in expected_baselines if method not in observed_methods]
    if missing_baselines:
        raise ValueError(f"missing expected baselines before artifact freeze: {missing_baselines}")

    root = Path(output_root)
    table_payload = aggregate_paper_metric_table(
        run_list,
        evaluation_metrics=evaluation_metrics,
        selection_metrics=selection_metrics,
        cost_metrics=cost_metrics,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    run_records_path = root / "inputs" / "run_records.jsonl"
    intervention_path = root / "inputs" / "intervention_statistics.json"
    matched_path = root / "inputs" / "matched_utility.json"
    claim_path = root / "inputs" / "claim_audit.json"

    main_tables = _build_main_tables(
        root=root,
        table_payload=table_payload,
        run_list=run_list,
        intervention_statistics=intervention_statistics,
        expected_main_tables=expected_main_tables,
        input_paths=[run_records_path, intervention_path, matched_path, claim_path],
    )
    figure_data = _build_figure_data(
        root=root,
        run_list=run_list,
        intervention_statistics=intervention_statistics,
        matched_utility=matched_utility,
        expected_figures=expected_figures,
        input_paths=[run_records_path, intervention_path, matched_path, claim_path],
    )
    appendix_tables = {
        "appendix_cost": _artifact(
            artifact_type="appendix_table",
            path=root / "appendix_tables" / "appendix_cost.json",
            input_paths=[run_records_path],
            aggregation_rule="cost metrics aggregated from frozen run records",
            seed_count=_seed_count(run_list),
            payload={
                "table": "appendix_cost",
                "cost_metrics": list(cost_metrics),
                "rows": table_payload,
            },
        )
    }
    claim_evidence_map = _artifact(
        artifact_type="claim_evidence_map",
        path=root / "claim_evidence_map.json",
        input_paths=[claim_path],
        aggregation_rule="claim audit readiness copied from Gate 10 claim audit",
        seed_count=_seed_count(run_list),
        payload={
            "claim_audit_ready": bool(claim_audit.get("is_ready", False)),
            "claim_issue_count": len(claim_audit.get("issues", [])),
            "source": dict(claim_audit),
        },
    )
    results_manifest = _artifact(
        artifact_type="manifest",
        path=root / "results_manifest.json",
        input_paths=[run_records_path, intervention_path, matched_path, claim_path],
        aggregation_rule="index all frozen paper artifacts and source inputs",
        seed_count=_seed_count(run_list),
        payload={
            "run_count": len(run_list),
            "datasets": sorted({run.dataset for run in run_list}),
            "models": sorted({run.model for run in run_list}),
            "methods": sorted(observed_methods),
            "main_tables": list(main_tables),
            "figure_data": list(figure_data),
        },
    )

    return {
        "results_manifest": results_manifest,
        "main_tables": main_tables,
        "appendix_tables": appendix_tables,
        "figure_data": figure_data,
        "claim_evidence_map": claim_evidence_map,
        "frozen_protocol": {
            "metrics": [*evaluation_metrics, *selection_metrics, *cost_metrics],
            "evaluation_metrics": list(evaluation_metrics),
            "selection_metrics": list(selection_metrics),
            "cost_metrics": list(cost_metrics),
            "baselines": [str(method) for method in expected_baselines],
            "judge_version": str(judge_version),
            "freeze_policy": "bug-fixes-only",
            "confidence": float(confidence),
            "resamples": int(resamples),
            "seed": int(seed),
        },
    }


def _build_main_tables(
    *,
    root: Path,
    table_payload: list[dict[str, Any]],
    run_list: Sequence[RunRecord],
    intervention_statistics: Mapping[str, Any],
    expected_main_tables: Sequence[str],
    input_paths: Sequence[Path],
) -> dict[str, dict[str, Any]]:
    payloads = {
        "table1": {
            "table": "table1",
            "purpose": "MIAS universality across tasks, models, and selectors",
            "rows": _table1_rows(run_list),
        },
        "table2": {
            "table": "table2",
            "purpose": "DPO main results with average, worst-group, coverage, utility, and cost metrics",
            "rows": table_payload,
        },
        "table3": {
            "table": "table3",
            "purpose": "Key ablation and intervention summaries",
            "intervention_statistics": dict(intervention_statistics),
        },
    }
    return {
        table_name: _artifact(
            artifact_type="main_table",
            path=root / "main_tables" / f"{table_name}.json",
            input_paths=input_paths,
            aggregation_rule=f"frozen protocol aggregation for {table_name}",
            seed_count=_seed_count(run_list),
            payload=payloads.get(str(table_name), {"table": str(table_name), "rows": []}),
        )
        for table_name in expected_main_tables
    }


def _build_figure_data(
    *,
    root: Path,
    run_list: Sequence[RunRecord],
    intervention_statistics: Mapping[str, Any],
    matched_utility: Mapping[str, Any],
    expected_figures: Sequence[str],
    input_paths: Sequence[Path],
) -> dict[str, dict[str, Any]]:
    payloads = {
        "fig1": {
            "figure": "fig1",
            "purpose": "problem and algorithm overview",
            "dcms_stage": "pre_label_acquisition",
            "pathway": ["selector tendency", "rho_g", "P_S(G)", "downstream behavior"],
        },
        "fig2": {
            "figure": "fig2",
            "purpose": "bias intervention response curves",
            "panels": ["class_intercept", "length_coefficient"],
            "intervention_statistics": dict(intervention_statistics),
        },
        "fig3": {
            "figure": "fig3",
            "purpose": "matched-utility composition intervention",
            "x_axis": "coverage_deviation",
            "y_axis": "group_or_capability_change",
            "points": list(matched_utility.get("points", [])),
            "matched_utility": dict(matched_utility),
        },
    }
    return {
        figure_name: _artifact(
            artifact_type="figure_data",
            path=root / "figure_data" / f"{figure_name}.json",
            input_paths=input_paths,
            aggregation_rule=f"frozen protocol figure-data build for {figure_name}",
            seed_count=_seed_count(run_list),
            payload=payloads.get(str(figure_name), {"figure": str(figure_name)}),
        )
        for figure_name in expected_figures
    }


def _table1_rows(runs: Sequence[RunRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[RunRecord]] = {}
    for run in runs:
        grouped.setdefault((run.dataset, run.model, run.method), []).append(run)
    rows = []
    for (dataset, model, method), method_runs in sorted(grouped.items()):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "selector": method,
                "seed_count": len({run.seed for run in method_runs}),
                "acquisition_tv_mean": _mean_metric(method_runs, "selection_metrics", "acquisition_tv"),
                "downstream_worst_group_mean": _mean_metric(
                    method_runs,
                    "evaluation_metrics",
                    "worst_group_preference_accuracy",
                ),
            }
        )
    return rows


def _artifact(
    *,
    artifact_type: str,
    path: Path,
    input_paths: Sequence[Path],
    aggregation_rule: str,
    seed_count: int,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "path": path.as_posix(),
        "input_result_files": [path.as_posix() for path in input_paths],
        "aggregation_rule": aggregation_rule,
        "seed_count": int(seed_count),
        "error_bar": "bootstrap 95% CI",
        "includes_failed_runs": True,
        **dict(payload),
    }


def _seed_count(runs: Sequence[RunRecord]) -> int:
    return len({run.seed for run in runs})


def _mean_metric(runs: Sequence[RunRecord], group_name: str, metric_name: str) -> float | None:
    values = [float(getattr(run, group_name)[metric_name]) for run in runs if metric_name in getattr(run, group_name)]
    if not values:
        return None
    return sum(values) / len(values)


def _coerce_run(run: RunRecord | Mapping[str, Any]) -> RunRecord:
    if isinstance(run, RunRecord):
        return run
    return RunRecord(**dict(run))
