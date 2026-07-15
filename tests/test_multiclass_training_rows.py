from __future__ import annotations

import unittest

from mias_dcms.multiclass_training_rows import build_multiclass_training_rows


class MulticlassTrainingRowsTest(unittest.TestCase):
    def test_combines_disjoint_seed_and_selected_rows_with_provenance(self) -> None:
        rows, summary = build_multiclass_training_rows(
            [{"id": "seed-1", "text": "seed text", "label": 0}],
            [{"id": "selected-1", "text": "selected text", "label": 1}],
        )

        self.assertEqual(["seed", "selected"], [row["training_row_source"] for row in rows])
        self.assertEqual(2, summary["training_row_count"])
        self.assertEqual({"0": 1, "1": 1}, summary["label_counts"])

    def test_rejects_overlap_between_seed_and_selected_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_multiclass_training_rows(
                [{"id": "same", "text": "seed text", "label": 0}],
                [{"id": "same", "text": "selected text", "label": 1}],
            )


if __name__ == "__main__":
    unittest.main()
