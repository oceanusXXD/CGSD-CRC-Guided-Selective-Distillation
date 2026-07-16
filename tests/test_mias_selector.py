from __future__ import annotations

import unittest

import numpy as np

from mias_dcms.selection.mias import (
    _logits,
    _nll,
    _single_label_gradient,
    deterministic_stratified_split,
    preference_difference_feature,
    preference_trainability_feature,
    score_expected_validation_influence,
    select_mias_classification,
    select_mias_preference,
)


class MIASSelectorTest(unittest.TestCase):
    def test_seed_split_is_deterministic_disjoint_and_60_20_20(self) -> None:
        sample_ids = [f"s{index}" for index in range(20)]
        labels = [index % 2 for index in range(20)]

        first = deterministic_stratified_split(sample_ids, labels, seed=17)
        second = deterministic_stratified_split(sample_ids, labels, seed=17)

        self.assertEqual(first, second)
        self.assertEqual([12, 4, 4], [len(first[name]) for name in ("fit", "calibration", "meta_validation")])
        flattened = [index for values in first.values() for index in values]
        self.assertEqual(list(range(20)), sorted(flattened))
        self.assertEqual(20, len(set(flattened)))

    def test_small_seed_uses_audited_temperature_fallback(self) -> None:
        result = score_expected_validation_influence(
            seed_ids=["s0", "s1", "s2", "s3"],
            seed_features=[[-2.0], [-1.0], [1.0], [2.0]],
            seed_labels=[0, 0, 1, 1],
            candidate_ids=["c0"],
            candidate_features=[[0.0]],
            bootstrap_heads=20,
            seed=3,
        )

        self.assertEqual(1.0, result.temperature)
        self.assertEqual("insufficient_data", result.temperature_status)
        self.assertEqual("insufficient_data", result.bootstrap_status)

    def test_one_class_seed_is_rejected_instead_of_producing_a_degenerate_ranker(self) -> None:
        with self.assertRaisesRegex(ValueError, "two observed classes"):
            score_expected_validation_influence(
                seed_ids=["s0", "s1", "s2"],
                seed_features=[[0.0], [1.0], [2.0]],
                seed_labels=[1, 1, 1],
                candidate_ids=["c0"],
                candidate_features=[[3.0]],
                class_values=["B", "A"],
            )

    def test_dpo_swap_negates_features_and_complements_posterior(self) -> None:
        row = {
            "sample_id": "a",
            "response_a_embedding": [1.0, 2.0],
            "response_b_embedding": [-1.0, 0.5],
            "response_a_token_count": 10,
            "response_b_token_count": 20,
        }
        swapped = {
            "sample_id": "b",
            "response_a_embedding": row["response_b_embedding"],
            "response_b_embedding": row["response_a_embedding"],
            "response_a_token_count": 20,
            "response_b_token_count": 10,
        }
        feature = preference_difference_feature(row)
        swapped_feature = preference_difference_feature(swapped)
        self.assertTrue(np.allclose(feature, -np.asarray(swapped_feature)))

        result = score_expected_validation_influence(
            seed_ids=[f"s{index}" for index in range(10)],
            seed_features=[[float(index - 5), float(index % 3), float((index % 4) - 2)] for index in range(10)],
            seed_labels=[index % 2 for index in range(10)],
            candidate_ids=["a", "b"],
            candidate_features=[feature, swapped_feature],
            class_values=["B", "A"],
            add_intercept=False,
            center_features=False,
            bootstrap_heads=0,
            seed=5,
        )
        self.assertAlmostEqual(result.posterior["a"][1], result.posterior["b"][0], places=12)
        self.assertAlmostEqual(result.posterior["a"][0], result.posterior["b"][1], places=12)

    def test_dpo_tie_gate_is_order_invariant_and_downweights_untrainable_probability(self) -> None:
        seeds = _preference_seed_rows(20)
        for index in (0, 5, 10, 15):
            seeds[index]["preferred_response"] = 0
        candidate = _preference_candidate(3)
        swapped = {
            **candidate,
            "sample_id": "c-swapped",
            "response_a_embedding": candidate["response_b_embedding"],
            "response_b_embedding": candidate["response_a_embedding"],
            "response_a_token_count": candidate["response_b_token_count"],
            "response_b_token_count": candidate["response_a_token_count"],
        }

        self.assertTrue(
            np.allclose(
                preference_trainability_feature(candidate),
                preference_trainability_feature(swapped),
            )
        )
        result = select_mias_preference(
            seed_rows=seeds,
            candidate_rows=[candidate, swapped],
            budget=1,
            seed=6,
            use_dcms=False,
            bootstrap_heads=0,
        )

        self.assertEqual(["tie", "B", "A"], result.scoring.class_values)
        first = result.scoring.posterior[str(candidate["sample_id"])]
        second = result.scoring.posterior["c-swapped"]
        self.assertAlmostEqual(first[0], second[0], places=12)
        self.assertAlmostEqual(first[1], second[2], places=12)
        self.assertAlmostEqual(first[2], second[1], places=12)
        self.assertIn("preference_trainability", result.scoring.auxiliary_models)

        dcms_result = select_mias_preference(
            seed_rows=seeds,
            candidate_rows=[candidate, swapped],
            budget=1,
            seed=6,
            use_dcms=True,
            bootstrap_heads=0,
            slack_grid=(1.0,),
            kappa=1.0,
        )
        self.assertNotIn("preference=tie", dcms_result.target_moments)

    def test_influence_sign_matches_exact_small_meta_loss_change(self) -> None:
        seed_ids = [f"s{index}" for index in range(10)]
        seed_features = [[float(index - 4), float((index % 3) - 1)] for index in range(10)]
        seed_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        candidate_feature = [2.5, -0.25]
        result = score_expected_validation_influence(
            seed_ids=seed_ids,
            seed_features=seed_features,
            seed_labels=seed_labels,
            candidate_ids=["candidate"],
            candidate_features=[candidate_feature],
            bootstrap_heads=0,
            seed=7,
        )
        label = int(np.argmax(result.per_label_influence["candidate"]))
        predicted_improvement = result.per_label_influence["candidate"][label]
        self.assertGreater(predicted_improvement, 0.0)

        params = np.asarray([*result.weights, *result.bias], dtype=np.float64)
        mean = np.asarray(result.feature_mean)
        scale = np.asarray(result.feature_scale)
        transformed_candidate = (np.asarray(candidate_feature) - mean) / scale
        probability = np.asarray(result.posterior["candidate"])
        gradient = _single_label_gradient(
            transformed_candidate,
            probability,
            label,
            class_count=2,
            add_intercept=True,
            temperature=result.temperature,
        )
        meta_ids = result.split_ids["meta_validation"]
        indexes = [seed_ids.index(sample_id) for sample_id in meta_ids]
        meta_x = (np.asarray([seed_features[index] for index in indexes]) - mean) / scale
        meta_y = np.asarray([seed_labels[index] for index in indexes])
        before = _nll(_logits(meta_x, params, 2, True), meta_y, temperature=result.temperature)
        after = _nll(_logits(meta_x, params - 1e-5 * gradient, 2, True), meta_y, temperature=result.temperature)
        self.assertGreater(before - after, 0.0)

    def test_candidate_oracle_label_is_rejected_before_selection(self) -> None:
        seeds = _preference_seed_rows(10)
        candidate = _preference_candidate(0)
        candidate["oracle_label"] = "A"

        with self.assertRaisesRegex(ValueError, "hidden fields"):
            select_mias_preference(
                seed_rows=seeds,
                candidate_rows=[candidate],
                budget=1,
                seed=1,
                use_dcms=False,
                bootstrap_heads=0,
            )

    def test_classification_dcms_uses_complete_pool_and_full_pool_targets(self) -> None:
        seeds = [
            {
                "id": f"s{index}",
                "label": index % 2,
                "representation_embedding": [float(index), float(index % 3)],
            }
            for index in range(10)
        ]
        candidates = [
            {
                "id": f"c{index}",
                "probabilities": [0.5, 0.5],
                "representation_embedding": [float(index - 3), float(index % 2)],
                "semantic_cluster_membership": {"left": float(index < 3), "right": float(index >= 3)},
                "token_count": 10 + index,
            }
            for index in range(6)
        ]

        result = select_mias_classification(
            seed_rows=seeds,
            candidate_rows=candidates,
            budget=2,
            seed=9,
            use_dcms=True,
            bootstrap_heads=0,
            slack_grid=(1.0,),
            kappa=1.0,
        )

        self.assertEqual(6, len(result.group_membership))
        self.assertEqual({f"c{index}" for index in range(6)}, set(result.group_membership))
        self.assertAlmostEqual(0.5, result.target_moments["semantic_cluster=left"])
        self.assertEqual(2, len(result.selected_ids))
        summary = result.summary_dict(method="mias_dcms", budget=2)
        self.assertIn("continuous_moments", summary)
        self.assertIn("utility_retained", summary)
        self.assertEqual(summary["utility_retained"], summary["dcms"]["utility_retained"])

    def test_sufficient_seed_enables_bootstrap_bounds_for_robust_dcms(self) -> None:
        seeds = [
            {
                "id": f"s{index}",
                "label": index % 2,
                "representation_embedding": [float(index % 5), float(index // 5)],
            }
            for index in range(24)
        ]
        candidates = [
            {
                "id": f"c{index}",
                "probabilities": [0.5, 0.5],
                "representation_embedding": [float(index), float(index % 3)],
                "semantic_cluster_membership": {"all": 1.0},
                "token_count": 8 + index,
            }
            for index in range(6)
        ]

        result = select_mias_classification(
            seed_rows=seeds,
            candidate_rows=candidates,
            budget=2,
            seed=4,
            use_dcms=True,
            bootstrap_heads=3,
            slack_grid=(1.0,),
            kappa=1.0,
        )

        self.assertEqual("fitted", result.scoring.bootstrap_status)
        self.assertEqual(3, result.scoring.bootstrap_heads_fitted)
        sample_id = "c0"
        self.assertTrue(
            any(
                abs(lower - upper) > 1e-12
                for lower, upper in zip(
                    result.scoring.posterior_lower[sample_id],
                    result.scoring.posterior_upper[sample_id],
                    strict=True,
                )
            )
        )
        self.assertIsNotNone(result.dcms)
        self.assertNotEqual(result.dcms.robust_lower_moments, result.dcms.robust_upper_moments)

    def test_multiclass_uses_one_softmax_head_and_per_class_influence(self) -> None:
        seeds = [
            {
                "id": f"s{index}",
                "label": index % 3,
                "representation_embedding": [float(index % 3), float(index // 3), 1.0],
            }
            for index in range(30)
        ]
        candidates = [
            {
                "id": f"c{index}",
                "probabilities": [1.0 / 3.0] * 3,
                "representation_embedding": [float(index), float(index % 3), 1.0],
            }
            for index in range(5)
        ]

        result = select_mias_classification(
            seed_rows=seeds,
            candidate_rows=candidates,
            budget=2,
            seed=2,
            use_dcms=False,
            bootstrap_heads=0,
        )

        self.assertEqual([0, 1, 2], result.scoring.class_values)
        self.assertEqual(3, len(result.scoring.weights))
        for sample_id in result.scoring.posterior:
            self.assertEqual(3, len(result.scoring.posterior[sample_id]))
            self.assertEqual(3, len(result.scoring.per_label_influence[sample_id]))
            self.assertAlmostEqual(1.0, sum(result.scoring.posterior[sample_id]))


def _preference_seed_rows(count: int) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"s{index}",
            "response_a_embedding": [float(index), 1.0],
            "response_b_embedding": [1.0, float(count - index)],
            "response_a_token_count": 10 + index,
            "response_b_token_count": 15,
            "preferred_response": 1 if index % 2 else 2,
        }
        for index in range(count)
    ]


def _preference_candidate(index: int) -> dict[str, object]:
    return {
        "sample_id": f"c{index}",
        "response_a_embedding": [float(index), 1.0],
        "response_b_embedding": [1.0, float(index)],
        "response_a_token_count": 10 + index,
        "response_b_token_count": 12,
        "prompt_cluster": f"p{index % 2}",
        "ab_position": "original" if index % 2 else "swapped",
    }


if __name__ == "__main__":
    unittest.main()
