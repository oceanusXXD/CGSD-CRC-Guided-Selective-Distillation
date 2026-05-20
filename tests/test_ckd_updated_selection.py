import math
import unittest

import numpy as np

from algorithms.cgsd import (
    AdaptiveSamplingPlan,
    DEFAULT_LAMBDA_GRID,
    compute_adaptive_sampling_plan,
    crc_margin_cutoff,
    select_adaptive_distillation_samples,
    select_documented_training_samples,
)
from scripts.cgsd_calibrate import compute_crc_sampling_statistics
from scripts.cgsd_make_fever_ns_difficulty_sets import select_ns_error_mass_split


class CKDUpdatedSelectionTest(unittest.TestCase):
    def test_default_crc_grid_matches_updated_method_range(self) -> None:
        self.assertEqual(0.50, DEFAULT_LAMBDA_GRID[0])
        self.assertEqual(1.00, DEFAULT_LAMBDA_GRID[-1])
        self.assertEqual(51, len(DEFAULT_LAMBDA_GRID))

    def test_sampling_plan_uses_crc_error_concentration(self) -> None:
        calibration_decisions = [
            {"id": "g1", "prediction": 1, "label": 1, "defer": False},
            {"id": "g2", "prediction": 0, "label": 0, "defer": False},
            {"id": "g3", "prediction": 1, "label": 0, "defer": True},
            {"id": "g4", "prediction": 1, "label": 1, "defer": True},
        ]
        pool_decisions = [
            *({"id": f"d{i}", "prediction": 1, "label": 1, "defer": True} for i in range(3)),
            *({"id": f"a{i}", "prediction": 1, "label": 1, "defer": False} for i in range(7)),
        ]

        plan = compute_adaptive_sampling_plan(
            calibration_decisions,
            pool_decisions,
            budget=10,
            temperature=2.0,
            lambda_hat=0.8,
            alpha=0.1,
        )

        self.assertAlmostEqual(2.0 * math.log(0.8 / 0.2), plan.tau_crc)
        self.assertAlmostEqual(0.3, plan.r_U)
        self.assertAlmostEqual(0.5, plan.r_C)
        self.assertAlmostEqual(0.25, plan.e_all)
        self.assertAlmostEqual(0.5, plan.e_defer)
        self.assertAlmostEqual(2.0, plan.c_crc)
        self.assertAlmostEqual(1.0, plan.eta_crc)
        self.assertAlmostEqual(0.79, plan.s_defer)
        self.assertAlmostEqual(0.21, plan.s_accept)
        self.assertEqual(8, plan.B_defer)
        self.assertEqual(2, plan.B_accept)
        plan_payload = plan.to_dict()
        self.assertNotIn("budget", plan_payload)
        self.assertNotIn("B_accept", plan_payload)
        self.assertNotIn("B_defer", plan_payload)

    def test_sampling_plan_disables_boost_when_guide_has_no_defer_errors(self) -> None:
        calibration_decisions = [
            {"id": "g1", "prediction": 1, "label": 1, "defer": True},
            {"id": "g2", "prediction": 0, "label": 1, "defer": False},
            {"id": "g3", "prediction": 0, "label": 0, "defer": False},
            {"id": "g4", "prediction": 1, "label": 1, "defer": False},
        ]
        pool_decisions = [
            {"id": "d1", "prediction": 1, "label": 1, "defer": True},
            {"id": "a1", "prediction": 1, "label": 1, "defer": False},
            {"id": "a2", "prediction": 0, "label": 0, "defer": False},
            {"id": "a3", "prediction": 0, "label": 0, "defer": False},
        ]

        plan = compute_adaptive_sampling_plan(
            calibration_decisions,
            pool_decisions,
            budget=8,
            temperature=10.0,
            lambda_hat=1.01,
        )

        self.assertTrue(math.isinf(plan.tau_crc))
        self.assertEqual(0.0, plan.e_defer)
        self.assertEqual(0.0, plan.eta_crc)
        self.assertAlmostEqual(0.25, plan.s_defer)
        self.assertEqual(2, plan.B_defer)
        self.assertEqual(6, plan.B_accept)

    def test_adaptive_selection_picks_accept_anchors_and_defer_hard_samples(self) -> None:
        rows = [
            {"id": "a_low", "defer": False, "routing_score": 0.70, "prediction": 1, "label": 1},
            {"id": "a_mid", "defer": False, "routing_score": 0.80, "prediction": 1, "label": 1},
            {"id": "a_high", "defer": False, "routing_score": 0.95, "prediction": 1, "label": 1},
            {"id": "d_left", "defer": True, "routing_score": 0.52, "prediction": 0, "label": 1},
            {"id": "d_center", "defer": True, "routing_score": 0.53, "prediction": 0, "label": 1},
            {"id": "d_right", "defer": True, "routing_score": 0.54, "prediction": 0, "label": 1},
        ]
        embeddings = {
            "d_left": np.asarray([-1.0, 0.0], dtype=np.float32),
            "d_center": np.asarray([0.0, 0.0], dtype=np.float32),
            "d_right": np.asarray([1.0, 0.0], dtype=np.float32),
        }
        plan = AdaptiveSamplingPlan(
            temperature=15.0,
            alpha=None,
            lambda_hat=0.75,
            tau_crc=0.0,
            budget=4,
            r_U=0.5,
            r_C=0.5,
            e_all=0.5,
            e_defer=1.0,
            c_crc=2.0,
            eta_crc=1.0,
            s_accept=0.5,
            s_defer=0.5,
            B_accept=2,
            B_defer=2,
            pool_accept_count=3,
            pool_defer_count=3,
            calibration_count=0,
            calibration_defer_count=0,
            calibration_error_count=0,
            calibration_defer_error_count=0,
        )

        selection = select_adaptive_distillation_samples(
            rows,
            sampling_plan=plan,
            already_selected_ids={"a_high"},
            embeddings_by_id=embeddings,
            seed=7,
            accept_strategy="high-confidence",
            defer_strategy="k-center",
        )

        self.assertEqual(4, selection.selected_budget)
        self.assertEqual(["a_mid", "a_low"], selection.accept_ids)
        self.assertEqual(2, len(selection.defer_ids))
        self.assertEqual({"d_left", "d_right"}, set(selection.defer_ids))
        self.assertEqual(
            ["a_mid", "a_low", *selection.defer_ids],
            selection.distillation_ids,
        )
        self.assertEqual("adaptive_accept_defer", selection.selection_method)

    def test_adaptive_selection_keeps_side_budgets_when_one_side_is_short(self) -> None:
        rows = [
            {"id": "a1", "defer": False, "routing_score": 0.90, "prediction": 1, "label": 1},
            {"id": "d1", "defer": True, "routing_score": 0.51, "prediction": 0, "label": 1},
            {"id": "d2", "defer": True, "routing_score": 0.52, "prediction": 0, "label": 1},
            {"id": "d3", "defer": True, "routing_score": 0.53, "prediction": 0, "label": 1},
        ]
        plan = AdaptiveSamplingPlan(
            temperature=15.0,
            alpha=None,
            lambda_hat=0.75,
            tau_crc=0.0,
            budget=4,
            r_U=0.75,
            r_C=0.75,
            e_all=0.5,
            e_defer=0.5,
            c_crc=1.0,
            eta_crc=0.0,
            s_accept=0.5,
            s_defer=0.5,
            B_accept=2,
            B_defer=2,
            pool_accept_count=1,
            pool_defer_count=3,
            calibration_count=0,
            calibration_defer_count=0,
            calibration_error_count=0,
            calibration_defer_error_count=0,
        )

        selection = select_adaptive_distillation_samples(
            rows,
            sampling_plan=plan,
            already_selected_ids=set(),
            seed=3,
            accept_strategy="high-confidence",
            defer_strategy="random",
        )

        self.assertEqual(["a1"], selection.accept_ids)
        self.assertEqual(2, len(selection.defer_ids))
        self.assertEqual(3, selection.selected_budget)
        self.assertTrue(selection.shortfall)

    def test_documented_sampling_methods_match_definitions(self) -> None:
        rows = [
            {"id": "a1", "defer": False, "routing_score": 0.91, "prediction": 1, "label": 1},
            {"id": "a2", "defer": False, "routing_score": 0.92, "prediction": 1, "label": 1},
            {"id": "a3", "defer": False, "routing_score": 0.93, "prediction": 1, "label": 1},
            {"id": "a4", "defer": False, "routing_score": 0.94, "prediction": 1, "label": 1},
            {"id": "d1", "defer": True, "routing_score": 0.51, "prediction": 0, "label": 1},
            {"id": "d2", "defer": True, "routing_score": 0.52, "prediction": 0, "label": 1},
            {"id": "d3", "defer": True, "routing_score": 0.53, "prediction": 0, "label": 1},
            {"id": "test_holdout", "defer": False, "routing_score": 0.99, "prediction": 1, "label": 1},
            {"id": "guide_holdout", "defer": True, "routing_score": 0.50, "prediction": 0, "label": 1},
        ]
        blocked = {"test_holdout", "guide_holdout"}

        pool_random = select_documented_training_samples(
            rows,
            method="pool-random",
            budget=5,
            seed=11,
            blocked_ids=blocked,
        )
        self.assertEqual(5, pool_random.selected_budget)
        self.assertEqual("pool-random", pool_random.selection_method)
        self.assertFalse(set(pool_random.distillation_ids) & blocked)

        pure_accept = select_documented_training_samples(
            rows,
            method="pure-accept",
            budget=5,
            seed=11,
            blocked_ids=blocked,
        )
        self.assertEqual(4, pure_accept.selected_budget)
        self.assertEqual(4, len(pure_accept.accept_ids))
        self.assertEqual([], pure_accept.defer_ids)
        self.assertTrue(pure_accept.shortfall)

        pure_defer = select_documented_training_samples(
            rows,
            method="pure-defer",
            budget=5,
            seed=11,
            blocked_ids=blocked,
        )
        self.assertEqual(3, pure_defer.selected_budget)
        self.assertEqual([], pure_defer.accept_ids)
        self.assertEqual(3, len(pure_defer.defer_ids))
        self.assertTrue(pure_defer.shortfall)

        fixed = select_documented_training_samples(
            rows,
            method="fixed-15-85",
            budget=20,
            seed=11,
            blocked_ids=blocked,
        )
        self.assertEqual(3, len(fixed.accept_ids))
        self.assertEqual(3, len(fixed.defer_ids))
        self.assertEqual(6, fixed.selected_budget)
        self.assertTrue(fixed.shortfall)

    def test_documented_crc_error_mass_uses_adaptive_plan(self) -> None:
        rows = [
            {"id": "a1", "defer": False, "routing_score": 0.90, "prediction": 1, "label": 1},
            {"id": "a2", "defer": False, "routing_score": 0.80, "prediction": 1, "label": 1},
            {"id": "a3", "defer": False, "routing_score": 0.70, "prediction": 1, "label": 1},
            {"id": "d1", "defer": True, "routing_score": 0.51, "prediction": 0, "label": 1},
            {"id": "d2", "defer": True, "routing_score": 0.52, "prediction": 0, "label": 1},
            {"id": "d3", "defer": True, "routing_score": 0.53, "prediction": 0, "label": 1},
        ]
        plan = AdaptiveSamplingPlan(
            temperature=15.0,
            alpha=0.1,
            lambda_hat=0.75,
            tau_crc=0.0,
            budget=4,
            r_U=0.5,
            r_C=0.5,
            e_all=0.5,
            e_defer=1.0,
            c_crc=2.0,
            eta_crc=1.0,
            s_accept=0.25,
            s_defer=0.75,
            B_accept=1,
            B_defer=3,
            pool_accept_count=3,
            pool_defer_count=3,
            calibration_count=0,
            calibration_defer_count=0,
            calibration_error_count=0,
            calibration_defer_error_count=0,
        )

        selection = select_documented_training_samples(
            rows,
            method="crc-error-mass",
            budget=4,
            seed=5,
            sampling_plan=plan,
            accept_strategy="high-confidence",
            defer_strategy="random",
        )

        self.assertEqual("crc-error-mass", selection.selection_method)
        self.assertEqual(["a1"], selection.accept_ids)
        self.assertEqual(3, len(selection.defer_ids))
        self.assertEqual(4, selection.selected_budget)

    def test_ns_crc_split_weights_each_side_by_ns_error_mass(self) -> None:
        rows = [
            {"id": "a_low", "defer": False, "prediction": 1, "label": 1, "ns_p_error": 0.01, "ns_epsilon": 0.001},
            {"id": "a_mid", "defer": False, "prediction": 1, "label": 1, "ns_p_error": 0.20, "ns_epsilon": 0.001},
            {"id": "a_high", "defer": False, "prediction": 1, "label": 1, "ns_p_error": 0.80, "ns_epsilon": 0.001},
        ]
        plan = AdaptiveSamplingPlan(
            temperature=1.0,
            alpha=0.1,
            lambda_hat=0.75,
            tau_crc=0.0,
            budget=2,
            r_U=0.0,
            r_C=0.1,
            e_all=0.1,
            e_defer=0.2,
            c_crc=2.0,
            eta_crc=1.0,
            s_accept=1.0,
            s_defer=0.0,
            B_accept=2,
            B_defer=0,
            pool_accept_count=3,
            pool_defer_count=0,
            calibration_count=1000,
            calibration_defer_count=100,
            calibration_error_count=100,
            calibration_defer_error_count=20,
        )

        _, payload = select_ns_error_mass_split(rows, plan=plan, train_size=2, seed=3)

        weights = {row["id"]: row["ns_sampling_weight"] for row in rows}
        self.assertGreater(weights["a_high"], weights["a_mid"])
        self.assertGreater(weights["a_mid"], weights["a_low"])
        self.assertEqual("ns-error-mass", payload["accept_ns_selection"]["ns_weighting"])

    def test_calibration_statistics_do_not_expose_budget_fields(self) -> None:
        calibration_decisions = [
            {"id": "g1", "prediction": 1, "label": 1, "defer": False, "routing_score": 0.9},
            {"id": "g2", "prediction": 0, "label": 1, "defer": True, "routing_score": 0.6},
            {"id": "g3", "prediction": 0, "label": 0, "defer": False, "routing_score": 0.7},
        ]
        pool_decisions = [
            {"id": "p1", "prediction": 1, "label": 1, "defer": False, "routing_score": 0.8},
            {"id": "p2", "prediction": 0, "label": 1, "defer": True, "routing_score": 0.4},
        ]

        stats = compute_crc_sampling_statistics(
            calibration_decisions,
            pool_decisions,
            temperature=15.0,
            lambda_hat=0.75,
        )

        self.assertIn("tau_crc", stats)
        self.assertIn("r_U", stats)
        self.assertIn("eta_crc", stats)
        self.assertNotIn("B_accept", stats)
        self.assertNotIn("B_defer", stats)


if __name__ == "__main__":
    unittest.main()
