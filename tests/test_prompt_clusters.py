from __future__ import annotations

import unittest

from mias_dcms.prompt_clusters import build_prompt_cluster_assignments


class PromptClustersTest(unittest.TestCase):
    def test_builds_deterministic_prompt_cluster_assignments(self) -> None:
        rows = [
            {"sample_id": "a", "prompt": "alpha"},
            {"sample_id": "b", "prompt": "alpha near"},
            {"sample_id": "c", "prompt": "beta"},
            {"sample_id": "d", "prompt": "beta near"},
        ]
        embeddings = {
            "a": [1.0, 0.0],
            "b": [0.9, 0.1],
            "c": [0.0, 1.0],
            "d": [0.1, 0.9],
        }

        result = build_prompt_cluster_assignments(
            rows=rows,
            embeddings_by_id=embeddings,
            cluster_count=2,
            softmax_temperature=0.25,
        )

        self.assertEqual(4, len(result.rows))
        self.assertEqual(2, result.summary["cluster_count"])
        self.assertEqual({"c0": 2, "c1": 2}, result.summary["cluster_counts"])
        by_id = {row["sample_id"]: row for row in result.rows}
        self.assertEqual(by_id["a"]["prompt_cluster"], by_id["b"]["prompt_cluster"])
        self.assertEqual(by_id["c"]["prompt_cluster"], by_id["d"]["prompt_cluster"])
        self.assertNotEqual(by_id["a"]["prompt_cluster"], by_id["c"]["prompt_cluster"])
        self.assertAlmostEqual(1.0, sum(by_id["a"]["prompt_cluster_probabilities"]))
        self.assertIn("prompt_cluster_membership", by_id["a"])

    def test_rejects_hidden_labels_and_missing_embeddings(self) -> None:
        with self.assertRaisesRegex(ValueError, "hidden fields"):
            build_prompt_cluster_assignments(
                rows=[{"sample_id": "p1", "preference_label": "A"}],
                embeddings_by_id={"p1": [1.0]},
                cluster_count=1,
            )

        with self.assertRaisesRegex(ValueError, "embeddings missing"):
            build_prompt_cluster_assignments(
                rows=[{"sample_id": "p1"}],
                embeddings_by_id={},
                cluster_count=1,
            )


if __name__ == "__main__":
    unittest.main()
