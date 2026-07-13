from __future__ import annotations

import unittest

from mias_dcms.selection.dcms import (
    dcms_utility_coverage_frontier,
    rank_normalize_utilities,
    solve_dcms,
    solve_dcms_with_slack,
)


class DCMSTest(unittest.TestCase):
    def test_rank_normalization_maps_scores_to_unit_interval(self) -> None:
        normalized = rank_normalize_utilities([10.0, 30.0, 20.0])

        self.assertEqual([0.0, 1.0, 0.5], normalized)

    def test_slack_selection_uses_strictest_feasible_utility_retaining_solution(self) -> None:
        result = solve_dcms_with_slack(
            sample_ids=["hi_a0", "hi_a1", "lo_b0", "lo_b1"],
            utilities=[1.0, 0.9, 0.2, 0.1],
            group_membership=[
                {"A": 1.0, "B": 0.0},
                {"A": 1.0, "B": 0.0},
                {"A": 0.0, "B": 1.0},
                {"A": 0.0, "B": 1.0},
            ],
            budget=2,
            target_moments={"A": 0.5, "B": 0.5},
            slack_grid=[0.0, 0.5],
            kappa=0.30,
        )

        self.assertEqual(0.5, result.selected_slack)
        self.assertEqual(["hi_a0", "hi_a1"], result.selected_ids)
        self.assertGreaterEqual(result.utility_retained, 0.70)
        self.assertEqual([0.0, 0.5], [trace.slack for trace in result.slack_trace])
        self.assertFalse(result.slack_trace[0].meets_utility_threshold)
        self.assertTrue(result.slack_trace[1].meets_utility_threshold)

    def test_robust_intervals_use_lower_and_upper_membership_bounds(self) -> None:
        result = solve_dcms(
            sample_ids=["a", "b"],
            utilities=[1.0, 0.9],
            group_membership=[{"A": 0.5}, {"A": 0.5}],
            membership_lower=[{"A": 0.0}, {"A": 0.0}],
            membership_upper=[{"A": 1.0}, {"A": 1.0}],
            budget=1,
            target_moments={"A": 0.5},
            tolerance=0.5,
        )

        self.assertEqual(["a"], result.selected_ids)
        self.assertEqual({"A": 0.0}, result.robust_lower_moments)
        self.assertEqual({"A": 1.0}, result.robust_upper_moments)
        self.assertEqual(0.5, result.max_constraint_violation)

    def test_robust_intervals_reject_target_inside_but_not_covering_interval(self) -> None:
        with self.assertRaises(ValueError):
            solve_dcms(
                sample_ids=["x"],
                utilities=[1.0],
                group_membership=[{"A": 0.7}],
                membership_lower=[{"A": 0.4}],
                membership_upper=[{"A": 1.0}],
                budget=1,
                target_moments={"A": 0.5},
                tolerance=0.0,
            )

    def test_rounding_seed_makes_tie_breaking_reproducible(self) -> None:
        first = solve_dcms(
            sample_ids=["a", "b", "c", "d"],
            utilities=[1.0, 1.0, 1.0, 1.0],
            group_membership=[
                {"A": 1.0},
                {"A": 1.0},
                {"A": 1.0},
                {"A": 1.0},
            ],
            budget=2,
            target_moments={"A": 1.0},
            tolerance=0.0,
            rounding_seed=7,
        )
        second = solve_dcms(
            sample_ids=["a", "b", "c", "d"],
            utilities=[1.0, 1.0, 1.0, 1.0],
            group_membership=[
                {"A": 1.0},
                {"A": 1.0},
                {"A": 1.0},
                {"A": 1.0},
            ],
            budget=2,
            target_moments={"A": 1.0},
            tolerance=0.0,
            rounding_seed=7,
        )

        self.assertEqual(first.selected_ids, second.selected_ids)
        self.assertEqual(7, first.rounding_seed)
        self.assertEqual(2, sum(first.selection_indicator.values()))

    def test_no_constraint_recovers_top_utility_selection(self) -> None:
        result = solve_dcms(
            sample_ids=["a", "b", "c"],
            utilities=[0.1, 0.9, 0.5],
            group_membership=[{"g0": 1.0}, {"g0": 1.0}, {"g0": 1.0}],
            budget=2,
            target_moments={"g0": 1.0},
            tolerance=1.0,
        )

        self.assertEqual(["b", "c"], result.selected_ids)
        self.assertAlmostEqual(1.0, result.utility_retained)
        self.assertEqual("feasible", result.solver_status)

    def test_exact_group_coverage_respects_target_moments(self) -> None:
        result = solve_dcms(
            sample_ids=["a0", "a1", "b0", "b1"],
            utilities=[0.9, 0.8, 0.7, 0.1],
            group_membership=[
                {"A": 1.0, "B": 0.0},
                {"A": 1.0, "B": 0.0},
                {"A": 0.0, "B": 1.0},
                {"A": 0.0, "B": 1.0},
            ],
            budget=2,
            target_moments={"A": 0.5, "B": 0.5},
            tolerance=0.0,
        )

        self.assertEqual(["a0", "b0"], result.selected_ids)
        self.assertEqual({"A": 0.5, "B": 0.5}, result.rounded_moments)
        self.assertEqual(0.0, result.max_constraint_violation)

    def test_infeasible_exact_constraint_raises(self) -> None:
        with self.assertRaises(ValueError):
            solve_dcms(
                sample_ids=["a", "b"],
                utilities=[1.0, 0.0],
                group_membership=[{"A": 1.0}, {"A": 1.0}],
                budget=1,
                target_moments={"A": 0.0},
                tolerance=0.0,
            )

    def test_utility_coverage_frontier_records_slack_tradeoff(self) -> None:
        frontier = dcms_utility_coverage_frontier(
            sample_ids=["hi_a0", "hi_a1", "lo_b0", "lo_b1"],
            utilities=[1.0, 0.9, 0.2, 0.1],
            group_membership=[
                {"A": 1.0, "B": 0.0},
                {"A": 1.0, "B": 0.0},
                {"A": 0.0, "B": 1.0},
                {"A": 0.0, "B": 1.0},
            ],
            budget=2,
            target_moments={"A": 0.5, "B": 0.5},
            slack_grid=[0.0, 0.5],
            kappa=0.30,
        )

        self.assertEqual(0.5, frontier.selected_slack)
        self.assertEqual([0.0, 0.5], [point.slack for point in frontier.points])
        self.assertTrue(frontier.points[0].feasible)
        self.assertAlmostEqual(0.0, frontier.points[0].coverage_deviation)
        self.assertFalse(frontier.points[0].meets_utility_threshold)
        self.assertTrue(frontier.points[1].meets_utility_threshold)
        self.assertAlmostEqual(1.0, frontier.points[1].utility_retained)
        self.assertAlmostEqual(0.5, frontier.points[1].coverage_deviation)

    def test_large_pool_uses_continuous_relaxation_and_exact_batch_rounding(self) -> None:
        sample_ids = [f"x{index}" for index in range(30)]
        memberships = [
            {"A": 1.0 if index < 15 else 0.0, "B": 0.0 if index < 15 else 1.0}
            for index in range(30)
        ]

        result = solve_dcms_with_slack(
            sample_ids=sample_ids,
            utilities=[1.0 - index / 100.0 for index in range(30)],
            group_membership=memberships,
            budget=10,
            target_moments={"A": 0.5, "B": 0.5},
            slack_grid=[0.0, 0.1, 0.5],
            kappa=0.5,
            rounding_seed=7,
        )

        self.assertTrue(result.solver_status.startswith("scalable_"))
        self.assertEqual(10, len(result.selected_ids))
        self.assertEqual(0.5, result.rounded_moments["A"])
        self.assertGreater(sum(0 < value < 1 for value in result.q_propensity.values()), 0)

    def test_large_pool_rounding_respects_robust_endpoint_constraints(self) -> None:
        sample_ids = [f"x{index}" for index in range(20)]
        lower = [{"A": 0.8} for _ in range(10)] + [{"A": 0.0} for _ in range(10)]
        upper = [{"A": 1.0} for _ in range(10)] + [{"A": 0.2} for _ in range(10)]

        result = solve_dcms_with_slack(
            sample_ids=sample_ids,
            utilities=[1.0] * 10 + [0.5] * 10,
            group_membership=[{"A": 0.5}] * 20,
            membership_lower=lower,
            membership_upper=upper,
            budget=10,
            target_moments={"A": 0.5},
            slack_grid=[0.0, 0.1],
            kappa=0.30,
            rounding_seed=7,
        )

        self.assertEqual(0.1, result.selected_slack)
        self.assertGreaterEqual(result.robust_lower_moments["A"], 0.4)
        self.assertLessEqual(result.robust_upper_moments["A"], 0.6)
        self.assertLessEqual(result.max_constraint_violation, 0.1)


if __name__ == "__main__":
    unittest.main()
