from __future__ import annotations

import unittest

from mias_dcms.query_binary_benchmark import prepare_query_binary_source


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(8):
        rows.append(
            {
                "id": f"source:{index}",
                "query_id": "2",
                "document_id": str(index),
                "query": "Is this a positive example?",
                "document": f"document {index}",
                "groundtruth": index % 2,
                "parsed_answer": "yes" if index % 2 else "no",
                "parsed_confidence": 1.0,
            }
        )
    rows.append({**rows[0], "id": "source:duplicate", "document_id": "duplicate"})
    return rows


class QueryBinaryBenchmarkTest(unittest.TestCase):
    def test_strips_prediction_fields_and_freezes_document_disjoint_holdout(self) -> None:
        prepared = prepare_query_binary_source(
            _rows(),
            dataset="codebase_q2",
            expected_query_id="2",
            seed=17,
            test_size=2,
        )

        source_rows = [*prepared["source_train_rows"], *prepared["source_test_rows"]]
        self.assertEqual(8, len(source_rows))
        self.assertEqual(1, prepared["source_summary"]["dropped_exact_duplicate_count"])
        self.assertTrue(all("parsed_answer" not in row and "parsed_confidence" not in row for row in source_rows))
        train_documents = {row["document_sha256"] for row in prepared["source_train_rows"]}
        test_documents = {row["document_sha256"] for row in prepared["source_test_rows"]}
        self.assertFalse(train_documents & test_documents)
        self.assertEqual({"0": 4, "1": 4}, prepared["source_summary"]["label_counts_after_deduplication"])

    def test_rejects_mixed_query_ids(self) -> None:
        rows = _rows()
        rows[1]["query_id"] = "3"
        with self.assertRaisesRegex(ValueError, "expected '2'"):
            prepare_query_binary_source(
                rows,
                dataset="codebase_q2",
                expected_query_id="2",
                seed=17,
                test_size=2,
            )

    def test_rejects_conflicting_duplicate_document_labels(self) -> None:
        rows = _rows()
        rows[-1]["groundtruth"] = 1
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            prepare_query_binary_source(
                rows,
                dataset="codebase_q2",
                expected_query_id="2",
                seed=17,
                test_size=2,
            )


if __name__ == "__main__":
    unittest.main()
