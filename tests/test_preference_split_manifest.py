from __future__ import annotations

import unittest

from mias_dcms.preference_split_manifest import build_preference_split_manifest


class PreferenceSplitManifestTest(unittest.TestCase):
    def test_builds_disjoint_reproducible_preference_splits(self) -> None:
        rows = [{"sample_id": f"p{i}"} for i in range(12)]

        first = build_preference_split_manifest(
            rows,
            seed=7,
            seed_size=2,
            active_size=6,
            heldout_size=2,
            test_size=2,
        )
        second = build_preference_split_manifest(
            rows,
            seed=7,
            seed_size=2,
            active_size=6,
            heldout_size=2,
            test_size=2,
        )

        self.assertEqual(first, second)
        self.assertEqual(2, len(first["seed_ids"]))
        self.assertEqual(6, len(first["active_pool_ids"]))
        self.assertEqual(2, len(first["heldout_ids"]))
        self.assertEqual(2, len(first["test_ids"]))
        all_ids = (
            first["seed_ids"]
            + first["active_pool_ids"]
            + first["heldout_ids"]
            + first["test_ids"]
        )
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_rejects_split_sizes_that_exceed_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "requested split sizes exceed row count"):
            build_preference_split_manifest(
                [{"sample_id": "p1"}],
                seed=1,
                seed_size=1,
                active_size=1,
                heldout_size=0,
                test_size=0,
            )


if __name__ == "__main__":
    unittest.main()
