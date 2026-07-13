from __future__ import annotations

import unittest

from mias_dcms.preference_pool import (
    FORBIDDEN_SELECTOR_FIELDS,
    build_preference_fixed_pool,
    normalized_response_length_gap,
)


class PreferencePoolTest(unittest.TestCase):
    def test_normalized_response_length_gap_is_signed_and_bounded(self) -> None:
        self.assertEqual(0.5, normalized_response_length_gap("one two three", "one"))
        self.assertEqual(-0.5, normalized_response_length_gap("one", "one two three"))
        self.assertEqual(0.0, normalized_response_length_gap("", ""))

    def test_fixed_pool_hides_oracle_labels_and_preserves_oracle_store(self) -> None:
        rows = [
            {
                "id": "p1",
                "prompt": "Prompt 1",
                "response_a": "short answer",
                "response_b": "a much longer answer",
                "chosen": "A",
                "preference_strength": 2,
                "source_a": "model-a",
                "source_b": "model-b",
            },
            {
                "id": "p2",
                "prompt": "Prompt 2",
                "response_a": "alpha beta gamma",
                "response_b": "delta",
                "chosen": "B",
                "preference_strength": 1,
                "source_a": "model-a",
                "source_b": "model-c",
            },
        ]

        fixed_pool = build_preference_fixed_pool(rows, seed=17)

        self.assertEqual(2, len(fixed_pool.active_pool))
        self.assertEqual(2, len(fixed_pool.oracle_store))
        self.assertEqual(2, len(fixed_pool.swap_manifest))
        for row in fixed_pool.active_pool:
            self.assertFalse(FORBIDDEN_SELECTOR_FIELDS.intersection(row))
            self.assertIn(row["ab_position"], {"original", "swapped"})
            self.assertIn("length_gap", row)
            self.assertIn("source_pair", row)
        self.assertEqual({"p1", "p2"}, set(fixed_pool.oracle_store))
        self.assertIn(fixed_pool.oracle_store["p1"]["preference_label"], {"A", "B"})

    def test_ab_swap_updates_oracle_label_consistently(self) -> None:
        rows = [
            {
                "id": "pair",
                "prompt": "Prompt",
                "response_a": "A text",
                "response_b": "B text",
                "chosen": "A",
            }
        ]

        original = build_preference_fixed_pool(rows, seed=2, force_swap=False)
        swapped = build_preference_fixed_pool(rows, seed=2, force_swap=True)

        self.assertEqual("A text", original.active_pool[0]["response_a"])
        self.assertEqual("B text", swapped.active_pool[0]["response_a"])
        self.assertEqual("A", original.oracle_store["pair"]["preference_label"])
        self.assertEqual("B", swapped.oracle_store["pair"]["preference_label"])
        self.assertFalse(original.swap_manifest[0]["swapped"])
        self.assertTrue(swapped.swap_manifest[0]["swapped"])

    def test_paired_swap_emits_pair_ids_for_position_audit(self) -> None:
        fixed_pool = build_preference_fixed_pool(
            [{
                "id": "pair",
                "prompt": "Prompt",
                "response_a": "A text",
                "response_b": "B text",
                "chosen": "A",
            }],
            seed=2,
            include_both_positions=True,
        )

        self.assertEqual(2, len(fixed_pool.active_pool))
        self.assertEqual({"pair"}, {row["swap_pair_id"] for row in fixed_pool.active_pool})
        self.assertEqual({"pair:original", "pair:swapped"}, set(fixed_pool.oracle_store))


if __name__ == "__main__":
    unittest.main()
