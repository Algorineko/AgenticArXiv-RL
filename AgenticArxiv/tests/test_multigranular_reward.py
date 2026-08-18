"""Unit tests for hierarchical reward shaping."""

import unittest

from rl.reward import RewardCalculator, compute_group_relative_advantages


def _result(history):
    return {"history": history, "iteration_count": len(history)}


class MultiGranularRewardTest(unittest.TestCase):
    def setUp(self):
        self.task = {"id": "composite", "expected_tools": ["search", "download"]}
        self.calculator = RewardCalculator()

    def test_correct_trajectory_scores_all_observed_layers(self):
        history = [
            {"action": '{"name":"search","parameters":{"q":"rl"}}', "observation": "ok"},
            {"action": '{"name":"download","parameters":{"id":"1"}}', "observation": "ok"},
            {"action": "FINISH", "observation": "done"},
        ]
        reward, metrics = self.calculator.compute_reward_breakdown(
            self.task, _result(history), training_step=30
        )
        self.assertEqual(reward.format, 1.0)
        self.assertEqual(reward.tool, 1.0)
        self.assertEqual(reward.process, 1.0)
        self.assertEqual(reward.outcome, 1.0)
        self.assertEqual(reward.total, 1.0)
        self.assertTrue(metrics.task_completed)

    def test_partial_tool_sequence_receives_dense_credit(self):
        history = [
            {"action": '{"name":"search","parameters":{}}', "observation": "ok"},
            {"action": "FINISH", "observation": "done"},
        ]
        reward, _ = self.calculator.compute_reward_breakdown(
            self.task, _result(history), training_step=30
        )
        self.assertGreater(reward.tool, -1.0)
        self.assertLess(reward.tool, 1.0)
        self.assertLess(reward.outcome, 1.0)

    def test_argument_layer_scores_keys_and_values(self):
        task = {
            "id": "search",
            "expected_tools": ["search"],
            "expected_tool_args": [{"category": "cs.AI", "days": 7}],
        }
        history = [
            {"action": '{"name":"search","parameters":{"category":"cs.AI","days":3}}', "observation": "ok"},
            {"action": "FINISH", "observation": "done"},
        ]
        reward, _ = self.calculator.compute_reward_breakdown(task, _result(history))
        self.assertEqual(reward.argument, 0.5)

    def test_curriculum_delays_correctness_weight(self):
        early = self.calculator.schedule(0)
        late = self.calculator.schedule(30)
        self.assertLess(early.tool, late.tool)
        self.assertEqual(early.format, late.format)
        self.assertEqual(early.process, late.process)

    def test_legacy_api_still_returns_scalar_and_metrics(self):
        value, metrics = self.calculator.compute_reward(
            self.task, _result([{"action": "ERROR", "observation": "bad"}])
        )
        self.assertIsInstance(value, float)
        self.assertEqual(metrics.termination_type, "ERROR")

    def test_group_advantages_are_normalized_per_prompt(self):
        advantages = compute_group_relative_advantages(
            [0.0, 1.0, 10.0, 10.0], ["a", "a", "b", "b"]
        )
        self.assertAlmostEqual(advantages[0], -1.0, places=5)
        self.assertAlmostEqual(advantages[1], 1.0, places=5)
        self.assertEqual(advantages[2:], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
