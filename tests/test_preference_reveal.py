from __future__ import annotations

import unittest

from mias_dcms.preference_reveal import reveal_selected_preference_labels


class PreferenceRevealTest(unittest.TestCase):
    def test_reveals_only_selected_oracle_labels_as_dpo_training_rows(self) -> None:
        active_pool = [
            {
                "sample_id": "p1",
                "prompt": "Prompt 1",
                "response_a": "chosen answer",
                "response_b": "rejected answer",
                "length_gap": 0.2,
            },
            {
                "sample_id": "p2",
                "prompt": "Prompt 2",
                "response_a": "unselected a",
                "response_b": "unselected b",
                "length_gap": -0.1,
            },
        ]
        oracle_store = {
            "p1": {"sample_id": "p1", "preference_label": "A", "preference_strength": 2},
            "p2": {"sample_id": "p2", "preference_label": "B", "preference_strength": 1},
        }

        result = reveal_selected_preference_labels(
            active_pool,
            oracle_store=oracle_store,
            selected_ids=["p1"],
            round_index=1,
            method="reward_margin",
        )

        self.assertEqual(["p1"], result.revealed_ids)
        self.assertEqual(["p2"], result.unrevealed_ids)
        self.assertEqual(1, len(result.training_rows))
        self.assertEqual(
            {
                "id": "p1",
                "sample_id": "p1",
                "round": 1,
                "method": "reward_margin",
                "prompt": "Prompt 1",
                "response_1": "chosen answer",
                "response_2": "rejected answer",
                "preferred_response": 1,
                "oracle_label": "A",
                "preference_strength": 2,
                "length_gap": 0.2,
            },
            result.training_rows[0],
        )

    def test_tie_labels_are_revealed_but_not_trainable_for_dpo(self) -> None:
        active_pool = [
            {
                "sample_id": "tie_pair",
                "prompt": "Prompt",
                "response_a": "A",
                "response_b": "B",
            }
        ]
        oracle_store = {
            "tie_pair": {"sample_id": "tie_pair", "preference_label": "tie"},
        }

        result = reveal_selected_preference_labels(
            active_pool,
            oracle_store=oracle_store,
            selected_ids=["tie_pair"],
            round_index=0,
            method="random",
        )

        self.assertEqual(["tie_pair"], result.revealed_ids)
        self.assertEqual(0, len(result.training_rows))
        self.assertEqual(1, len(result.revealed_rows))
        self.assertEqual(0, result.revealed_rows[0]["preferred_response"])

    def test_rejects_selected_ids_not_present_in_active_pool(self) -> None:
        with self.assertRaisesRegex(ValueError, "selected id"):
            reveal_selected_preference_labels(
                [],
                oracle_store={},
                selected_ids=["missing"],
                round_index=0,
                method="random",
            )


if __name__ == "__main__":
    unittest.main()
