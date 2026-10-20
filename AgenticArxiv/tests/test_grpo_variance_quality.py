"""Distinguish clean full-reward windows from uniform GRPO failure."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("STORE_BACKEND", "memory")

from rl.train_grpo import RewardVarianceGuard  # noqa: E402


def _run_window(rewards, *, start=1, **quality):
    guard = RewardVarianceGuard(patience=5, grace_steps=20)
    control = SimpleNamespace(should_training_stop=False)
    for step, reward in enumerate(rewards, start):
        logs = {
            "frac_reward_zero_std": 1.0,
            "reward_std": 0.0,
            "reward": reward,
            "rollout/finished": 1.0,
            "rollout/parse_error_rate": 0.0,
            "rollout/tool_error_rate": 0.0,
            **quality,
        }
        guard.on_log(
            None, SimpleNamespace(global_step=step), control, logs=logs
        )
    return guard, control


class VarianceQualityTest(unittest.TestCase):
    def test_early_full_score_window_stops_normally(self):
        guard, control = _run_window([1.0] * 5, start=5)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(guard.converged)
        self.assertFalse(guard.tripped)
        self.assertEqual(guard.stop_step, 9)

    def test_early_floor_window_remains_failure(self):
        guard, control = _run_window([-0.75] * 5)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(guard.tripped)
        self.assertFalse(guard.converged)

    def test_late_floor_is_not_mislabeled_saturation(self):
        guard, control = _run_window([-0.75] * 5, start=25)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(guard.tripped)
        self.assertFalse(guard.converged)

    def test_partial_credit_is_not_full_score_saturation(self):
        guard, _ = _run_window([0.5] * 5, start=25)
        self.assertTrue(guard.tripped)
        self.assertFalse(guard.converged)

    def test_last_full_score_cannot_hide_bad_earlier_groups(self):
        guard, _ = _run_window([-0.75, 1.0, 1.0, 1.0, 1.0], start=25)
        self.assertTrue(guard.tripped)
        self.assertFalse(guard.converged)

    def test_full_reward_with_rollout_errors_is_not_saturation(self):
        for quality in (
            {"rollout/finished": 0.8},
            {"rollout/parse_error_rate": 0.1},
            {"rollout/tool_error_rate": 0.1},
        ):
            with self.subTest(quality=quality):
                guard, _ = _run_window([1.0] * 5, start=25, **quality)
                self.assertTrue(guard.tripped)
                self.assertFalse(guard.converged)

    def test_missing_or_nonfinite_reward_cannot_certify_saturation(self):
        for reward in (None, float("nan"), float("inf")):
            with self.subTest(reward=reward):
                guard, _ = _run_window([reward] * 5, start=25)
                self.assertTrue(guard.tripped)
                self.assertFalse(guard.converged)

    def test_informative_group_resets_the_patience_window(self):
        guard, _ = _run_window([1.0] * 4)
        control = SimpleNamespace(should_training_stop=False)
        guard.on_log(
            None,
            SimpleNamespace(global_step=5),
            control,
            logs={"frac_reward_zero_std": 0.0, "reward_std": 0.2, "reward": 0.5},
        )
        guard.on_log(
            None,
            SimpleNamespace(global_step=6),
            control,
            logs={"frac_reward_zero_std": 1.0, "reward_std": 0.0, "reward": 1.0},
        )
        self.assertFalse(control.should_training_stop)
        self.assertFalse(guard.tripped)
        self.assertFalse(guard.converged)

    def test_late_full_score_window_still_stops_normally(self):
        guard, control = _run_window([1.0] * 5, start=25)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(guard.converged)
        self.assertFalse(guard.tripped)


if __name__ == "__main__":
    unittest.main()
