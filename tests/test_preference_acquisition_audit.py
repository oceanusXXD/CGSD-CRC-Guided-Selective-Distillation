from __future__ import annotations

import unittest

from mias_dcms.preference_acquisition_audit import audit_preference_acquisition


class PreferenceAcquisitionAuditTest(unittest.TestCase):
    def test_audits_multiple_preference_attributes_and_random_reference(self) -> None:
        rows = [
            {
                "sample_id": "p1",
                "selected": 1,
                "length_gap_bin": "short",
                "source_pair": "human|model",
                "prompt_cluster": "c1",
            },
            {
                "sample_id": "p2",
                "selected": 1,
                "length_gap_bin": "short",
                "source_pair": "human|model",
                "prompt_cluster": "c1",
            },
            {
                "sample_id": "p3",
                "selected": 0,
                "length_gap_bin": "long",
                "source_pair": "model|human",
                "prompt_cluster": "c2",
            },
            {
                "sample_id": "p4",
                "selected": 0,
                "length_gap_bin": "long",
                "source_pair": "model|human",
                "prompt_cluster": "c2",
            },
        ]
        random_rows = [
            {**rows[0], "selected": 1},
            {**rows[1], "selected": 0},
            {**rows[2], "selected": 1},
            {**rows[3], "selected": 0},
        ]

        summary = audit_preference_acquisition(
            rows,
            method="APL",
            group_fields=("length_gap_bin", "source_pair", "prompt_cluster"),
            random_reference_rows=random_rows,
        )

        self.assertEqual("APL", summary["method"])
        self.assertEqual(4, summary["pool_size"])
        self.assertEqual(2, summary["selected_size"])
        self.assertEqual(["length_gap_bin", "prompt_cluster", "source_pair"], sorted(summary["group_fields"]))
        self.assertTrue(summary["random_reference_present"])
        self.assertAlmostEqual(0.5, summary["by_group_field"]["length_gap_bin"]["acquisition_tv"])
        self.assertAlmostEqual(1.0, summary["by_group_field"]["length_gap_bin"]["maximum_propensity_ratio"])
        self.assertGreater(summary["by_group_field"]["length_gap_bin"]["acquisition_js"], 0.0)
        self.assertAlmostEqual(0.0, summary["random_reference"]["by_group_field"]["length_gap_bin"]["acquisition_tv"])
        self.assertAlmostEqual(0.5, summary["max_acquisition_tv"])
        self.assertAlmostEqual(1.0, summary["max_propensity_ratio"])

    def test_audit_rejects_hidden_labels_before_selection_analysis(self) -> None:
        rows = [
            {
                "sample_id": "p1",
                "selected": 1,
                "length_gap_bin": "short",
                "preference_label": "A",
            }
        ]

        with self.assertRaisesRegex(ValueError, "hidden fields"):
            audit_preference_acquisition(rows, method="APL", group_fields=("length_gap_bin",))

    def test_audit_requires_selected_rows(self) -> None:
        rows = [
            {"sample_id": "p1", "selected": 0, "length_gap_bin": "short"},
            {"sample_id": "p2", "selected": 0, "length_gap_bin": "long"},
        ]

        with self.assertRaisesRegex(ValueError, "at least one row must be selected"):
            audit_preference_acquisition(rows, method="APL", group_fields=("length_gap_bin",))


if __name__ == "__main__":
    unittest.main()
