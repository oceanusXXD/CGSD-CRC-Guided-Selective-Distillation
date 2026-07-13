from __future__ import annotations

import unittest

from mias_dcms.result_aggregation import (
    REQUIRED_COST_METRICS,
    aggregate_paper_metric_table,
    validate_run_record_for_paper_table,
)
from mias_dcms.records import RunRecord


def _run_record(
    *,
    method: str,
    seed: int,
    macro_f1: float,
    worst_group: float,
    acquisition_tv: float,
    train_tokens: int = 1000,
) -> RunRecord:
    return RunRecord(
        dataset="toy",
        model="model-a",
        method=method,
        budget=4,
        seed=seed,
        selected_count=4,
        config_hash=f"cfg-{method}-{seed}",
        selection_metrics={"acquisition_tv": acquisition_tv},
        training_metrics={},
        evaluation_metrics={"macro_f1": macro_f1, "worst_group": worst_group},
        cost_metrics={
            "seed_label_count": 2,
            "active_label_count": 4,
            "evaluation_label_count": 6,
            "judge_calls": 0,
            "train_tokens": train_tokens,
            "selector_compute_seconds": 0.25,
        },
    )


class ResultAggregationTest(unittest.TestCase):
    def test_validate_run_record_requires_selection_evaluation_and_cost_metrics(self) -> None:
        run = _run_record(
            method="Random",
            seed=1,
            macro_f1=0.6,
            worst_group=0.5,
            acquisition_tv=0.2,
        )

        validate_run_record_for_paper_table(
            run,
            required_selection_metrics=("acquisition_tv",),
            required_evaluation_metrics=("macro_f1", "worst_group"),
        )

        missing_cost = run.as_dict()
        missing_cost["cost_metrics"] = {
            key: value
            for key, value in run.cost_metrics.items()
            if key != "train_tokens"
        }
        with self.assertRaises(ValueError):
            validate_run_record_for_paper_table(
                RunRecord(**missing_cost),
                required_selection_metrics=("acquisition_tv",),
                required_evaluation_metrics=("macro_f1", "worst_group"),
            )

    def test_aggregate_paper_metric_table_bootstraps_metrics_by_method(self) -> None:
        rows = [
            _run_record(method="Random", seed=1, macro_f1=0.60, worst_group=0.50, acquisition_tv=0.20),
            _run_record(method="Random", seed=2, macro_f1=0.70, worst_group=0.55, acquisition_tv=0.25),
            _run_record(method="Entropy+DCMS", seed=1, macro_f1=0.68, worst_group=0.62, acquisition_tv=0.06),
            _run_record(method="Entropy+DCMS", seed=2, macro_f1=0.72, worst_group=0.66, acquisition_tv=0.08),
        ]

        table = aggregate_paper_metric_table(
            rows,
            evaluation_metrics=("macro_f1", "worst_group"),
            selection_metrics=("acquisition_tv",),
            cost_metrics=("train_tokens",),
            resamples=200,
            seed=9,
        )

        by_method = {row["method"]: row for row in table}
        self.assertEqual({"Entropy+DCMS", "Random"}, set(by_method))
        self.assertAlmostEqual(0.65, by_method["Random"]["evaluation_metrics"]["macro_f1"]["mean"])
        self.assertAlmostEqual(0.07, by_method["Entropy+DCMS"]["selection_metrics"]["acquisition_tv"]["mean"])
        self.assertEqual(2, by_method["Random"]["run_count"])
        self.assertEqual(sorted(REQUIRED_COST_METRICS), sorted(by_method["Random"]["required_cost_metrics"]))


if __name__ == "__main__":
    unittest.main()
