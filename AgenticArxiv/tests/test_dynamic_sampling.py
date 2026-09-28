#!/usr/bin/env python3
r"""CPU-only tests for DAPO dynamic sampling.

Run from the repository root with::

    E:\proj\env\python.exe -m pytest AgenticArxiv/tests/test_dynamic_sampling.py -q
"""

from __future__ import annotations

import copy
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
os.environ.setdefault("STORE_BACKEND", "memory")

from rl.dynamic_sampling import (  # noqa: E402
    DeterministicPromptPool,
    DynamicSample,
    DynamicSamplingBuffer,
    DynamicSamplingDistributedError,
    DynamicSamplingError,
    DynamicSamplingExhausted,
    assert_dynamic_sampling_single_process,
    make_dynamic_sampling_rollout_func,
    reward_standard_deviation,
)
from rl.grpo_reward import (  # noqa: E402
    make_grpo_reward_fn,
    make_multiturn_rollout_func,
)


def prompt_row(task_id: str):
    return {
        "prompt": [{"role": "user", "content": f"TASK {task_id}", "_task_id": task_id}],
        "task_id": task_id,
    }


def sample(task_id: str, reward: float, index: int = 0) -> DynamicSample:
    return DynamicSample(
        prompt_ids=[10, index],
        completion_ids=[20, index],
        env_mask=[1, 0],
        task_id=task_id,
        trajectory={"task_id": task_id, "index": index},
        reward=reward,
    )


class DynamicSamplingBufferTest(unittest.TestCase):
    def test_all_zero_and_all_one_groups_are_rejected(self):
        for rewards in ([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]):
            with self.subTest(rewards=rewards):
                buffer = DynamicSamplingBuffer(target_groups=1, group_size=4)
                accepted = buffer.add_group([
                    sample("flat", reward, index) for index, reward in enumerate(rewards)
                ])
                self.assertFalse(accepted)
                self.assertEqual(buffer.accepted_groups, 0)
                self.assertEqual(buffer.rejected_groups, 1)

    def test_group_with_variance_is_accepted(self):
        buffer = DynamicSamplingBuffer(target_groups=1, group_size=4)
        accepted = buffer.add_group([
            sample("useful", reward, index)
            for index, reward in enumerate([0.0, 1.0, 0.5, 1.0])
        ])
        self.assertTrue(accepted)
        self.assertEqual(len(buffer.samples()), 4)

    def test_floating_threshold_is_strict(self):
        rewards = [0.0, 0.2]
        std = reward_standard_deviation(rewards)
        at_boundary = DynamicSamplingBuffer(
            target_groups=1, group_size=2, reward_std_threshold=std
        )
        self.assertFalse(at_boundary.add_group([
            sample("boundary", rewards[0], 0), sample("boundary", rewards[1], 1)
        ]))

        just_below = DynamicSamplingBuffer(
            target_groups=1, group_size=2, reward_std_threshold=std - 1e-12
        )
        self.assertTrue(just_below.add_group([
            sample("boundary", rewards[0], 0), sample("boundary", rewards[1], 1)
        ]))

    def test_alignment_is_validated_before_acceptance(self):
        bad_mask = sample("a", 0.0)
        bad_mask.env_mask = [1]
        with self.assertRaisesRegex(DynamicSamplingError, "env_mask"):
            DynamicSamplingBuffer(target_groups=1, group_size=2).add_group([
                bad_mask, sample("a", 1.0, 1)
            ])

        bad_trajectory = sample("a", 0.0)
        bad_trajectory.trajectory = {"task_id": "b"}
        with self.assertRaisesRegex(DynamicSamplingError, "trajectory"):
            DynamicSamplingBuffer(target_groups=1, group_size=2).add_group([
                bad_trajectory, sample("a", 1.0, 1)
            ])

    def test_incomplete_buffer_never_returns_a_short_batch(self):
        buffer = DynamicSamplingBuffer(target_groups=2, group_size=2)
        buffer.add_group([sample("a", 0.0), sample("a", 1.0, 1)])
        with self.assertRaisesRegex(DynamicSamplingExhausted, "incomplete"):
            buffer.samples()


class FakeRolloutTrainer:
    class Model:
        training = True

    def __init__(self, group_size: int = 2, step: int = 17):
        self.model = self.Model()
        self.num_generations = group_size
        self.num_generations_eval = group_size
        self.state = SimpleNamespace(global_step=step)
        self.accelerator = SimpleNamespace(num_processes=1)


class ScriptedRollout:
    """Minimal public rollout_func with values encoded in every aligned field."""

    def __init__(self, scores_by_task):
        self.scores_by_task = scores_by_task
        self.calls = []

    def __call__(self, prompts, trainer):
        task_ids = [prompt[0]["_task_id"] for prompt in prompts]
        self.calls.append(list(task_ids))
        offsets = {}
        prompt_ids = []
        completion_ids = []
        env_masks = []
        trajectories = []
        for task_id in task_ids:
            index = offsets.get(task_id, 0)
            offsets[task_id] = index + 1
            marker = ord(task_id[-1])
            score = self.scores_by_task[task_id][index]
            prompt_ids.append([marker, index])
            completion_ids.append([marker, 100 + index])
            env_masks.append([1, index % 2])
            trajectories.append({
                "task_id": task_id,
                "marker": marker,
                "generation": index,
                "score": score,
            })
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": None,
            "env_mask": env_masks,
            "trajectory_results": trajectories,
        }


def trajectory_score_reward(
    completions=None, task_id=None, trainer_state=None, trajectory_results=None, **kwargs
):
    return [float(trajectory["score"]) for trajectory in trajectory_results]


class DynamicSamplingRolloutTest(unittest.TestCase):
    def test_rejected_prompt_is_replaced_not_retried_and_size_is_exact(self):
        rows = [prompt_row(task_id) for task_id in ("a", "b", "c")]
        base = ScriptedRollout({
            "a": [0.0, 0.0],
            "b": [0.0, 1.0],
            "c": [0.25, 0.75],
        })
        rollout = make_dynamic_sampling_rollout_func(
            base,
            prompt_rows=rows,
            preview_reward_func=trajectory_score_reward,
            seed=7,
            max_resample_attempts=2,
        )
        original = [copy.deepcopy(rows[0]["prompt"]) for _ in range(2)]

        output = rollout(original, FakeRolloutTrainer())

        self.assertEqual(len(output["prompt_ids"]), len(original))
        self.assertEqual(len(output["completion_ids"]), len(original))
        self.assertEqual(len(output["env_mask"]), len(original))
        self.assertEqual(len(output["trajectory_results"]), len(original))
        self.assertEqual(len(output["task_id"]), len(original))
        self.assertEqual(base.calls[0], ["a", "a"])
        self.assertNotIn("a", base.calls[1])
        self.assertEqual(len(set(output["task_id"])), 1)

        # Atomic records preserve all four identities after filtering/refill.
        for task_id, prompt_ids, completion_ids, env_mask, trajectory in zip(
            output["task_id"],
            output["prompt_ids"],
            output["completion_ids"],
            output["env_mask"],
            output["trajectory_results"],
            strict=True,
        ):
            self.assertEqual(trajectory["task_id"], task_id)
            self.assertEqual(prompt_ids[0], ord(task_id[-1]))
            self.assertEqual(completion_ids[0], ord(task_id[-1]))
            self.assertEqual(len(completion_ids), len(env_mask))

    def test_max_resample_attempts_exhaustion_is_explicit(self):
        rows = [prompt_row(task_id) for task_id in ("a", "b", "c", "d")]
        base = ScriptedRollout({task_id: [1.0, 1.0] for task_id in "abcd"})
        rollout = make_dynamic_sampling_rollout_func(
            base,
            prompt_rows=rows,
            preview_reward_func=trajectory_score_reward,
            seed=3,
            max_resample_attempts=2,
        )
        original = [copy.deepcopy(rows[0]["prompt"]) for _ in range(2)]

        with self.assertRaisesRegex(
            DynamicSamplingExhausted, "max_resample_attempts=2.*partial batch"
        ):
            rollout(original, FakeRolloutTrainer())
        self.assertEqual(len(base.calls), 3)  # initial + exactly two replacements
        self.assertTrue(all("a" not in call for call in base.calls[1:]))

    def test_fixed_seed_reproduces_replacement_choice(self):
        rows = [prompt_row(task_id) for task_id in ("a", "b", "c", "d")]
        scores = {
            "a": [0.0, 0.0],
            "b": [0.0, 1.0],
            "c": [0.0, 1.0],
            "d": [0.0, 1.0],
        }
        results = []
        for _ in range(2):
            rollout = make_dynamic_sampling_rollout_func(
                ScriptedRollout(scores),
                prompt_rows=rows,
                preview_reward_func=trajectory_score_reward,
                seed=2026,
            )
            original = [copy.deepcopy(rows[0]["prompt"]) for _ in range(2)]
            results.append(rollout(original, FakeRolloutTrainer())["task_id"])
        self.assertEqual(results[0], results[1])

    def test_distributed_launch_fails_loudly(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            with self.assertRaisesRegex(
                DynamicSamplingDistributedError, "single-process.*WORLD_SIZE=2"
            ):
                assert_dynamic_sampling_single_process()


class FakeTokenizer:
    """Character tokenizer sufficient for the real multi-turn rollout code."""

    @staticmethod
    def _encode(text):
        return [ord(char) for char in str(text)]

    def apply_chat_template(self, prompt, tokenize=True, add_generation_prompt=True):
        visible = "\n".join(str(message.get("content") or "") for message in prompt)
        return self._encode(visible)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._encode(text)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(token)) for token in ids)

    def batch_decode(self, batch, skip_special_tokens=True):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch]


class FakeEnvironment:
    instances = []

    def __init__(self):
        self.task_id = ""
        self.calls = []
        self.__class__.instances.append(self)

    def reset(self, task_id=""):
        self.task_id = task_id
        return ""

    def search_arxiv_papers(self, **kwargs):
        self.calls.append(dict(kwargs))
        return f"RESULT:{kwargs.get('query')}"


class FullMockTrainer(FakeRolloutTrainer):
    def __init__(self, tokenizer):
        super().__init__(group_size=2, step=23)
        self.processing_class = tokenizer
        self.max_completion_length = 512
        self.generation_calls = 0

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        self.generation_calls += 1
        outputs = []
        for index, sequence in enumerate(prompt_ids):
            text = self.processing_class.decode(sequence)
            if "Observation:" in text:
                response = "Thought: done\nAction: FINISH"
            else:
                # Task a is deliberately flat. Replacement tasks vary across
                # the two generations, producing a useful reward group.
                query = "flat" if "TASK a" in text else f"variant-{index}"
                response = (
                    "Thought: search\nAction: "
                    + json.dumps({
                        "name": "search_arxiv_papers",
                        "args": {"query": query, "max_results": 1},
                    })
                )
            outputs.append(self.processing_class(response)["input_ids"])
        return outputs, None, {}


class FakeCurriculumCalculator:
    def __init__(self):
        self.steps = []

    def compute_reward_breakdown(self, task, trajectory, training_step=0):
        self.steps.append(training_step)
        first_action = trajectory["history"][0]["action"]
        if task["id"] == "a":
            total = 0.0
        else:
            total = 1.0 if "variant-1" in first_action else 0.0
        return SimpleNamespace(total=total), {}


class FakeTracker:
    def __init__(self):
        self.records = []
        self.groups = []

    def record(self, breakdown, trajectory):
        self.records.append((breakdown.total, trajectory["task_id"]))

    def record_group(self, task_ids, rewards):
        self.groups.append((list(task_ids), list(rewards)))


class FakeAuditor:
    def __init__(self):
        self.batches = []

    def record_batch(self, **kwargs):
        self.batches.append(kwargs)


class FullMockRolloutTest(unittest.TestCase):
    def test_fake_tokenizer_trainer_environment_end_to_end(self):
        FakeEnvironment.instances = []
        rows = [prompt_row(task_id) for task_id in ("a", "b", "c")]
        tasks = {task_id: {"id": task_id, "setup": []} for task_id in "abc"}
        tokenizer = FakeTokenizer()
        trainer = FullMockTrainer(tokenizer)
        calculator = FakeCurriculumCalculator()
        tracker = FakeTracker()
        auditor = FakeAuditor()

        base_rollout = make_multiturn_rollout_func(
            FakeEnvironment,
            max_turns=2,
            tasks_by_id=tasks,
            prompt_task_ids={},
        )
        preview_reward = make_grpo_reward_fn(
            tasks, reward_calc=calculator, tracker=None, auditor=None
        )
        formal_reward = make_grpo_reward_fn(
            tasks, reward_calc=calculator, tracker=tracker, auditor=auditor
        )
        rollout = make_dynamic_sampling_rollout_func(
            base_rollout,
            prompt_rows=rows,
            preview_reward_func=preview_reward,
            seed=11,
            max_resample_attempts=2,
        )

        initial = [copy.deepcopy(rows[0]["prompt"]) for _ in range(2)]
        output = rollout(initial, trainer)
        accepted_rewards = formal_reward(
            completions=[None, None],
            task_id=output["task_id"],
            trainer_state=trainer.state,
            trajectory_results=output["trajectory_results"],
        )

        self.assertEqual(len(output["completion_ids"]), len(initial))
        self.assertNotIn("a", output["task_id"])
        self.assertEqual(accepted_rewards, [0.0, 1.0])
        self.assertTrue(all(
            trajectory["task_id"] == task_id
            for task_id, trajectory in zip(
                output["task_id"], output["trajectory_results"], strict=True
            )
        ))
        self.assertTrue(all(
            len(ids) == len(mask)
            for ids, mask in zip(output["completion_ids"], output["env_mask"], strict=True)
        ))
        self.assertTrue(all(0 in mask and 1 in mask for mask in output["env_mask"]))

        # Both preview scoring and the formal accepted batch use the actual
        # trainer state, never a hard-coded step zero.
        self.assertTrue(calculator.steps)
        self.assertEqual(set(calculator.steps), {23})

        # Only the formal accepted batch reaches observability sinks.
        self.assertEqual(len(tracker.records), 2)
        self.assertTrue(all(task_id != "a" for _, task_id in tracker.records))
        self.assertEqual(len(tracker.groups), 1)
        self.assertEqual(len(auditor.batches), 1)
        self.assertNotIn("a", auditor.batches[0]["task_ids"])

        self.assertGreaterEqual(trainer.generation_calls, 4)
        self.assertTrue(FakeEnvironment.instances)
        self.assertTrue(any(env.calls for env in FakeEnvironment.instances))


if __name__ == "__main__":
    unittest.main()
