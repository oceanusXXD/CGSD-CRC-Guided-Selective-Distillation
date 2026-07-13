from __future__ import annotations

import unittest

from mias_dcms.records import (
    AcquisitionRecord,
    RunRecord,
    build_acquisition_record,
    build_records_from_dcms,
    build_run_record,
)
from mias_dcms.selection.dcms import solve_dcms_with_slack


class AcquisitionRecordTest(unittest.TestCase):
    def test_build_record_preserves_label_isolation_before_selection(self) -> None:
        record = build_acquisition_record(
            sample_id="p1",
            split="active_pool",
            round_index=0,
            method="entropy",
            model="selector-a",
            seed=7,
            base_score=0.42,
            normalized_score=0.8,
            q_propensity=0.25,
            selected=False,
            observable_groups={"length_bin": "short"},
            oracle_label=None,
            train_tokens=0,
        )

        self.assertIsInstance(record, AcquisitionRecord)
        self.assertEqual(None, record.oracle_label)
        self.assertEqual({"length_bin": "short"}, record.observable_groups)

    def test_rejects_oracle_label_on_unselected_active_pool_record(self) -> None:
        with self.assertRaises(ValueError):
            build_acquisition_record(
                sample_id="p2",
                split="active_pool",
                round_index=0,
                method="entropy",
                model="selector-a",
                seed=7,
                base_score=0.42,
                normalized_score=0.8,
                q_propensity=0.25,
                selected=False,
                observable_groups={},
                oracle_label=1,
                train_tokens=0,
            )

    def test_build_records_from_dcms_preserves_selection_label_boundary(self) -> None:
        result = solve_dcms_with_slack(
            sample_ids=["a", "b", "c"],
            utilities=[0.9, 0.8, 0.1],
            group_membership=[{"short": 1.0}, {"short": 1.0}, {"short": 0.0}],
            budget=2,
            target_moments={"short": 1.0},
            slack_grid=[0.0],
            kappa=0.0,
        )

        records = build_records_from_dcms(
            sample_ids=["a", "b", "c"],
            base_scores=[9.0, 8.0, 1.0],
            normalized_scores=[1.0, 0.5, 0.0],
            observable_groups=[{"length_bin": "short"}, {"length_bin": "short"}, {"length_bin": "long"}],
            dcms_result=result,
            split="active_pool",
            round_index=1,
            method="Entropy+DCMS",
            model="selector-a",
            seed=11,
            revealed_oracle_labels={"a": 1, "b": 0},
            train_tokens={"a": 12, "b": 10},
        )

        by_id = {record.sample_id: record for record in records}
        self.assertEqual({"a", "b", "c"}, set(by_id))
        self.assertTrue(by_id["a"].selected)
        self.assertEqual(1, by_id["a"].oracle_label)
        self.assertEqual(12, by_id["a"].train_tokens)
        self.assertFalse(by_id["c"].selected)
        self.assertIsNone(by_id["c"].oracle_label)
        self.assertEqual(0, by_id["c"].train_tokens)

    def test_build_run_record_summarizes_selection_cost_and_constraints(self) -> None:
        result = solve_dcms_with_slack(
            sample_ids=["a", "b", "c", "d"],
            utilities=[0.9, 0.8, 0.7, 0.1],
            group_membership=[
                {"A": 1.0, "B": 0.0},
                {"A": 1.0, "B": 0.0},
                {"A": 0.0, "B": 1.0},
                {"A": 0.0, "B": 1.0},
            ],
            budget=2,
            target_moments={"A": 0.5, "B": 0.5},
            slack_grid=[0.0, 0.5],
            kappa=0.0,
            rounding_seed=13,
        )

        run = build_run_record(
            dataset="toy",
            model="selector-a",
            method="BADGE+DCMS",
            budget=2,
            seed=13,
            dcms_result=result,
            config_hash="abc123",
            selection_metrics={"acquisition_tv": 0.25},
            training_metrics={"macro_f1": 0.75},
            evaluation_metrics={"worst_group": 0.6},
            cost_metrics={"oracle_label_calls": 2},
        )

        self.assertIsInstance(run, RunRecord)
        as_dict = run.as_dict()
        self.assertEqual("toy", as_dict["dataset"])
        self.assertEqual(2, as_dict["budget"])
        self.assertEqual(2, as_dict["selected_count"])
        self.assertEqual(13, as_dict["rounding_seed"])
        self.assertIn("continuous_moments", as_dict)
        self.assertEqual(0.25, as_dict["selection_metrics"]["acquisition_tv"])


if __name__ == "__main__":
    unittest.main()
