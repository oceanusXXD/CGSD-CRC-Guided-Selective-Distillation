from __future__ import annotations

import unittest

from scripts.train_preference_dpo_run import _training_metrics


class TrainPreferenceDPORunTest(unittest.TestCase):
    def test_training_metrics_falls_back_when_no_token_or_step_budget_is_configured(self) -> None:
        rows = [{"prompt": "prompt", "response_1": "chosen", "response_2": "rejected"}]

        metrics = _training_metrics(
            {"train_token_budget": None},
            rows=rows,
            training_config={},
        )

        self.assertEqual(7, metrics["training_token_budget"])
        self.assertIsNone(metrics["update_steps"])


if __name__ == "__main__":
    unittest.main()
