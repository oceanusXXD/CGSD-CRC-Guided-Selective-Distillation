from __future__ import annotations

import unittest

from mias_dcms.binary_benchmark_protocol import prepare_binary_benchmark_protocol


def _rows(split: str, count: int, *, offset: int = 0) -> list[dict[str, object]]:
    return [
        {
            "id": f"benchmark:{split}:{offset + index}",
            "query": "Does this example belong to class one?",
            "document": f"{split} document {offset + index}",
            "groundtruth": index % 2,
        }
        for index in range(count)
    ]


class BinaryBenchmarkProtocolTest(unittest.TestCase):
    def test_official_validation_and_test_are_not_resplit_into_training_pool(self) -> None:
        artifacts = prepare_binary_benchmark_protocol(
            _rows("train", 20),
            validation_rows=_rows("validation", 8),
            test_rows=_rows("test", 10),
            dataset="toy",
            seed_label_count=4,
            active_pool_size=8,
            seed=17,
        )

        manifest = artifacts["protocol_manifest"]
        self.assertEqual("official_validation", manifest["development_source"])
        self.assertEqual(8, len(artifacts["development_rows"]))
        self.assertEqual(10, len(artifacts["official_test_rows"]))
        self.assertEqual(8, len(artifacts["selection_pool"]))
        self.assertTrue(all("label" not in row and "groundtruth" not in row for row in artifacts["selection_pool"]))
        self.assertTrue(all("query" in row and "document" in row for row in artifacts["seed_train_rows"]))

        partition_ids = [
            *manifest["development_ids"],
            *manifest["seed_ids"],
            *manifest["active_pool_ids"],
            *manifest["official_test_ids"],
        ]
        self.assertEqual(len(partition_ids), len(set(partition_ids)))

    def test_imdb_style_source_without_validation_gets_fixed_train_holdout(self) -> None:
        artifacts = prepare_binary_benchmark_protocol(
            _rows("train", 20),
            validation_rows=None,
            test_rows=_rows("test", 10),
            dataset="toy",
            seed_label_count=4,
            active_pool_size=8,
            development_size=4,
            seed=17,
            test_row_limit=6,
        )

        manifest = artifacts["protocol_manifest"]
        self.assertEqual("train_derived_fixed_holdout", manifest["development_source"])
        self.assertEqual(4, len(artifacts["development_rows"]))
        self.assertEqual(6, len(artifacts["official_test_rows"]))

    def test_requires_development_holdout_when_no_official_validation_exists(self) -> None:
        with self.assertRaisesRegex(ValueError, "development_size"):
            prepare_binary_benchmark_protocol(
                _rows("train", 20),
                validation_rows=None,
                test_rows=_rows("test", 10),
                dataset="toy",
                seed_label_count=4,
                active_pool_size=8,
                seed=17,
            )


if __name__ == "__main__":
    unittest.main()
