from __future__ import annotations

import unittest

from mias_dcms.preference_dcms_inputs import build_preference_dcms_candidate_rows


class PreferenceDCMSInputsTest(unittest.TestCase):
    def test_builds_dcms_candidates_from_baseline_scores_and_observable_groups(self) -> None:
        rows = [
            {
                "sample_id": "p1",
                "apl_score": 0.9,
                "length_gap_bin": "short",
                "source_pair": "human|model",
                "ab_position": "original",
            },
            {
                "sample_id": "p2",
                "apl_score": 0.4,
                "length_gap_bin": "long",
                "source_pair": "model|human",
                "ab_position": "swapped",
            },
        ]

        candidates = build_preference_dcms_candidate_rows(
            rows,
            method="apl",
            group_fields=("length_gap_bin", "source_pair", "ab_position"),
        )

        self.assertEqual(
            [
                {
                    "sample_id": "p1",
                    "score": 0.9,
                    "method": "apl",
                    "source_score_field": "apl_score",
                    "groups": {
                        "length_gap_bin=short": 1.0,
                        "source_pair=human|model": 1.0,
                        "ab_position=original": 1.0,
                    },
                },
                {
                    "sample_id": "p2",
                    "score": 0.4,
                    "method": "apl",
                    "source_score_field": "apl_score",
                    "groups": {
                        "length_gap_bin=long": 1.0,
                        "source_pair=model|human": 1.0,
                        "ab_position=swapped": 1.0,
                    },
                },
            ],
            candidates,
        )

    def test_uses_numeric_group_membership_mapping_without_reencoding(self) -> None:
        rows = [
            {
                "sample_id": "p1",
                "active_dpo_score": 1.2,
                "observable_groups": {"cluster_a": 0.8, "cluster_b": 0.2},
            }
        ]

        candidates = build_preference_dcms_candidate_rows(
            rows,
            method="active_dpo",
            group_field="observable_groups",
        )

        self.assertEqual({"cluster_a": 0.8, "cluster_b": 0.2}, candidates[0]["groups"])

    def test_rejects_hidden_preference_labels_before_building_candidates(self) -> None:
        rows = [
            {
                "sample_id": "p1",
                "reward_margin_score": 0.5,
                "preference_label": "A",
                "length_gap_bin": "short",
            }
        ]

        with self.assertRaisesRegex(ValueError, "hidden fields"):
            build_preference_dcms_candidate_rows(
                rows,
                method="reward_margin",
                group_fields=("length_gap_bin",),
            )


if __name__ == "__main__":
    unittest.main()
