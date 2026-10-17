"""Scoring evidence must survive bounded multi-turn GRPO observations."""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("STORE_BACKEND", "memory")

from rl.grpo_reward import make_multiturn_rollout_func  # noqa: E402
from rl.reward import RewardCalculator  # noqa: E402


class _Tokenizer:
    """Character tokenizer: deterministic, with no model dependencies."""

    def apply_chat_template(self, prompt, **kwargs):
        return self.encode("Task")

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}

    @staticmethod
    def encode(text):
        return [ord(char) for char in text]

    @staticmethod
    def decode(ids, **kwargs):
        return "".join(chr(token) for token in ids)


class _Trainer:
    """Emit one tool action followed by FINISH."""

    processing_class = _Tokenizer()

    def __init__(self, token_budget=5000):
        self.max_completion_length = token_budget
        self.visible_prompts = []
        action = {
            "name": "analyze_figure",
            "args": {"ref": 1, "figure_ref": 2, "question_type": "axes"},
        }
        self.turns = [
            "Thought: analyze\nAction: " + json.dumps(action),
            "Thought: done\nAction: FINISH",
        ]
        self.turn = 0

    def _generate_single_turn(self, prompts, *args):
        self.visible_prompts.extend(
            self.processing_class.decode(prompt) for prompt in prompts
        )
        text = self.turns[self.turn]
        self.turn += 1
        return [self.processing_class.encode(text) for _ in prompts], None, None


def _run_figure(answer, *, tool_error=False, token_budget=5000):
    payload = {"paper_id": "test-paper", "answer": answer, "extra": "x" * 1400}

    class Environment:
        def reset(self, **kwargs):
            pass

        def analyze_figure(self, **kwargs):
            if tool_error:
                raise ValueError("fixture tool failure")
            return payload

    trainer = _Trainer(token_budget)
    with patch("rl.grpo_reward.require_rollout_func_support"):
        rollout = make_multiturn_rollout_func(Environment, max_turns=2)
        output = rollout([[{"role": "user", "content": "Analyze"}]], trainer)
    task = {
        "id": "figure",
        "expected_tools": ["analyze_figure"],
        "expected_tool_args": [
            {"ref": 1, "figure_ref": 2, "question_type": "axes"}
        ],
    }
    trajectory = output["trajectory_results"][0]
    reward, _ = RewardCalculator(curriculum_steps=0).compute_reward_breakdown(
        task, trajectory
    )
    return payload, trainer, output, trajectory, reward


class FullObservationTest(unittest.TestCase):
    def test_long_valid_result_is_not_misclassified_as_tool_failure(self):
        payload, trainer, _, trajectory, reward = _run_figure(
            "a" * 1400 + "ANSWER_END"
        )
        self.assertEqual(reward.total, 1.0)
        self.assertEqual(trajectory["history"][0]["observation"], str(payload))
        self.assertNotIn("ANSWER_END", trainer.visible_prompts[1])

    def test_policy_input_and_environment_mask_keep_the_bounded_view(self):
        payload, trainer, output, trajectory, _ = _run_figure("axis labels")
        model_view = str(payload)[:1000]
        suffix = "\nObservation: " + model_view + "\nThought:"
        first, second = trainer.turns
        encode = trainer.processing_class.encode

        self.assertEqual(
            trajectory["history"][0]["model_observation"], model_view
        )
        self.assertEqual(
            trainer.visible_prompts[1], "Task" + first + suffix
        )
        self.assertEqual(
            output["completion_ids"][0], encode(first + suffix + second)
        )
        self.assertEqual(
            output["env_mask"][0],
            [1] * len(first) + [0] * len(suffix) + [1] * len(second),
        )
        self.assertEqual(
            trajectory["token_usage"]["policy_tokens"], len(first + second)
        )

    def test_empty_answer_remains_a_failure(self):
        _, _, _, _, reward = _run_figure("")
        self.assertLessEqual(reward.total, -0.75)

    def test_tool_exception_remains_a_failure(self):
        _, _, _, trajectory, reward = _run_figure("", tool_error=True)
        step = trajectory["history"][0]
        self.assertIn("fixture tool failure", step["observation"])
        self.assertEqual(step["observation"], step["model_observation"])
        self.assertLessEqual(reward.total, -0.75)

    def test_full_evidence_survives_remaining_token_budget_clipping(self):
        payload, trainer, output, trajectory, _ = _run_figure(
            "a" * 1400 + "ANSWER_END", token_budget=180
        )
        self.assertEqual(trajectory["history"][0]["observation"], str(payload))
        self.assertEqual(
            trajectory["history"][0]["model_observation"], str(payload)[:1000]
        )
        self.assertEqual(len(output["completion_ids"][0]), 180)
        self.assertEqual(len(output["env_mask"][0]), 180)
        self.assertEqual(trainer.turn, 1)
        self.assertTrue(trajectory["clipped"])


if __name__ == "__main__":
    unittest.main()
