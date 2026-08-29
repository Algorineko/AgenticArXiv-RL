#!/usr/bin/env python3
"""Unit tests for multi-turn OPD rollout, masking, and environment isolation."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
os.environ.setdefault("STORE_BACKEND", "memory")

from rl.opd_multiturn import (  # noqa: E402
    IGNORE_INDEX,
    MultiTurnOPDCollator,
    _dependency_setup_steps,
    make_multiturn_gkd_trainer,
    pad_multiturn_opd_rollouts,
    run_multiturn_opd_rollouts,
    validate_tokenizer_compatibility,
)


class CharacterTokenizer:
    """Small reversible tokenizer: every Unicode code point is one token."""

    pad_token_id = 0
    eos_token_id = 3
    bos_token_id = 2
    unk_token_id = 1

    def __init__(self, vocab=None):
        self._vocab = vocab or {"a": 4, "b": 5}

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in str(text)]}

    def decode(self, ids, skip_special_tokens=True):
        special = {self.pad_token_id, self.eos_token_id, self.bos_token_id}
        return "".join(chr(token) for token in ids if not skip_special_tokens or token not in special)

    def get_vocab(self):
        return dict(self._vocab)


class FakeEnvironment:
    def __init__(self, identity):
        self.identity = identity
        self.task_id = None
        self.search_calls = 0

    def reset(self, task_id="", **kwargs):
        self.task_id = task_id

    def get_recently_submitted_cs_papers(self, **kwargs):
        self.search_calls += 1
        return [{"env": self.identity, "query": kwargs}]

    def download_arxiv_pdf(self, ref=1):
        return {"env": self.identity, "ref": ref, "status": "READY"}


def encode(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


class MultiTurnCollatorTest(unittest.TestCase):
    def test_left_padding_and_task_ids_are_preserved(self):
        batch = MultiTurnOPDCollator(0)([
            {"input_ids": [7, 8], "task_id": "a"},
            {"input_ids": [9], "task_id": "b"},
        ])
        self.assertEqual(batch["prompts"].tolist(), [[7, 8], [0, 9]])
        self.assertEqual(batch["prompt_attention_mask"].tolist(), [[1, 1], [0, 1]])
        self.assertEqual(batch["task_ids"], ["a", "b"])


class TokenizerCompatibilityTest(unittest.TestCase):
    def test_identical_mapping_is_accepted(self):
        validate_tokenizer_compatibility(CharacterTokenizer(), CharacterTokenizer())

    def test_same_size_but_remapped_vocabulary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "remapped=2"):
            validate_tokenizer_compatibility(
                CharacterTokenizer({"a": 4, "b": 5}),
                CharacterTokenizer({"a": 5, "b": 4}),
            )

    def test_special_token_mismatch_is_rejected(self):
        teacher = CharacterTokenizer()
        teacher.eos_token_id = 99
        with self.assertRaisesRegex(ValueError, "eos_token_id"):
            validate_tokenizer_compatibility(CharacterTokenizer(), teacher)


class EnvironmentSetupTest(unittest.TestCase):
    def test_dependency_steps_are_oldest_first_and_exclude_current_gold(self):
        tasks = {
            "search": {
                "id": "search",
                "expected_tools": ["get_recently_submitted_cs_papers"],
                "expected_tool_args": [{"aspect": "AI"}],
            },
            "download": {
                "id": "download",
                "depends_on": "search",
                "expected_tools": ["download_arxiv_pdf"],
                "expected_tool_args": [{"ref": 1}],
            },
        }
        self.assertEqual(
            _dependency_setup_steps(tasks["download"], tasks),
            [{"name": "get_recently_submitted_cs_papers", "args": {"aspect": "AI"}}],
        )

    def test_explicit_setup_takes_precedence(self):
        task = {
            "id": "x",
            "depends_on": "ignored",
            "setup": [{"name": "download_arxiv_pdf", "args": {"ref": 2}}],
        }
        self.assertEqual(_dependency_setup_steps(task, {}), task["setup"])


class MultiTurnRolloutTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_action_observation_finish_and_noncontiguous_mask(self):
        environments = []

        def factory():
            environment = FakeEnvironment(len(environments))
            environments.append(environment)
            return environment

        scripted = [
            'Thought: search\nAction: {"name":"get_recently_submitted_cs_papers","args":{"aspect":"AI"}}',
            "Thought: done\nAction: FINISH",
        ]
        calls = 0

        def generate_turn(sequences):
            nonlocal calls
            response = encode(self.tokenizer, scripted[calls])
            calls += 1
            return [response for _ in sequences]

        rollouts = run_multiturn_opd_rollouts(
            prompt_ids=[[80]],
            task_ids=["search"],
            tokenizer=self.tokenizer,
            environment_factory=factory,
            tasks_by_id={"search": {"id": "search"}},
            generate_turn=generate_turn,
            max_turns=3,
            max_observation_tokens=32,
        )
        rollout = rollouts[0]
        first_action = encode(self.tokenizer, scripted[0])
        final_action = encode(self.tokenizer, scripted[1])

        self.assertEqual(rollout.termination_type, "FINISH")
        self.assertEqual([step["action"] for step in rollout.history][-1], "FINISH")
        self.assertEqual(rollout.labels[0], IGNORE_INDEX)  # prompt
        self.assertEqual(rollout.labels[1 : 1 + len(first_action)], first_action)
        observation_start = 1 + len(first_action)
        observation_end = observation_start + rollout.observation_tokens
        self.assertTrue(all(label == IGNORE_INDEX for label in rollout.labels[observation_start:observation_end]))
        self.assertEqual(rollout.labels[observation_end:], final_action)
        self.assertGreater(rollout.observation_tokens, 0)
        self.assertEqual(environments[0].search_calls, 1)

    def test_each_sample_receives_an_independent_environment(self):
        environments = []

        def factory():
            environment = FakeEnvironment(len(environments))
            environments.append(environment)
            return environment

        action = encode(
            self.tokenizer,
            'Thought: search\nAction: {"name":"get_recently_submitted_cs_papers","args":{}}',
        )
        rollouts = run_multiturn_opd_rollouts(
            prompt_ids=[[1], [2]],
            task_ids=["a", "b"],
            tokenizer=self.tokenizer,
            environment_factory=factory,
            tasks_by_id={},
            generate_turn=lambda sequences: [action for _ in sequences],
            max_turns=1,
            max_observation_tokens=8,
        )
        self.assertEqual(len({id(environment) for environment in environments}), 2)
        self.assertEqual([environment.task_id for environment in environments], ["a", "b"])
        self.assertEqual([environment.search_calls for environment in environments], [1, 1])
        self.assertTrue(all(rollout.termination_type == "FORCE_STOP" for rollout in rollouts))
        self.assertEqual([rollout.turn_count for rollout in rollouts], [1, 1])

    def test_tool_error_becomes_real_observation_and_rollout_continues(self):
        scripted = [
            'Thought: try tool\nAction: {"name":"missing_tool","args":{}}',
            "Thought: stop\nAction: FINISH",
        ]
        calls = 0

        def generate_turn(sequences):
            nonlocal calls
            response = encode(self.tokenizer, scripted[calls])
            calls += 1
            return [response for _ in sequences]

        rollout = run_multiturn_opd_rollouts(
            prompt_ids=[[1]],
            task_ids=["x"],
            tokenizer=self.tokenizer,
            environment_factory=lambda: FakeEnvironment(0),
            tasks_by_id={},
            generate_turn=generate_turn,
            max_turns=2,
            max_observation_tokens=64,
        )[0]

        self.assertEqual(rollout.termination_type, "FINISH")
        self.assertEqual(rollout.turn_count, 2)
        self.assertIn("Unknown tool: missing_tool", rollout.history[0]["observation"])
        self.assertGreater(rollout.observation_tokens, 0)

    def test_parse_error_is_terminal_but_generated_tokens_remain_targets(self):
        invalid = encode(self.tokenizer, "not a ReAct action")
        rollout = run_multiturn_opd_rollouts(
            prompt_ids=[[10, 11]],
            task_ids=["x"],
            tokenizer=self.tokenizer,
            environment_factory=lambda: FakeEnvironment(0),
            tasks_by_id={},
            generate_turn=lambda sequences: [invalid],
            max_turns=4,
            max_observation_tokens=8,
        )[0]
        self.assertEqual(rollout.termination_type, "PARSE_ERROR")
        self.assertEqual(rollout.labels[:2], [IGNORE_INDEX, IGNORE_INDEX])
        self.assertEqual(rollout.labels[2:], invalid)

    def test_padding_uses_lengths_even_when_pad_equals_a_valid_token(self):
        action = encode(self.tokenizer, "Thought: done\nAction: FINISH")
        rollouts = run_multiturn_opd_rollouts(
            prompt_ids=[[0], [4, 5]],
            task_ids=["a", "b"],
            tokenizer=self.tokenizer,
            environment_factory=lambda: FakeEnvironment(0),
            tasks_by_id={},
            generate_turn=lambda sequences: [action for _ in sequences],
            max_turns=1,
            max_observation_tokens=8,
        )
        batch = pad_multiturn_opd_rollouts(rollouts, pad_token_id=0, device=torch.device("cpu"))
        self.assertEqual(batch.attention_mask[0, 0].item(), 1)
        self.assertEqual(batch.attention_mask[0, -1].item(), 0)
        self.assertEqual(batch.labels[0, -1].item(), IGNORE_INDEX)


class NonContiguousLossMaskTest(unittest.TestCase):
    class FakeGKD:
        @staticmethod
        def generalized_jsd_loss(student_logits, teacher_logits, labels, beta):
            per_position = (student_logits - teacher_logits).pow(2).sum(dim=-1)
            return per_position[labels != IGNORE_INDEX].mean()

    class LogitModel(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

        def forward(self, **kwargs):
            return SimpleNamespace(logits=self.values)

    def test_compute_loss_uses_only_shifted_assistant_positions(self):
        trainer_class = make_multiturn_gkd_trainer(self.FakeGKD)
        trainer = object.__new__(trainer_class)
        trainer.beta = 1.0
        student = self.LogitModel([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        teacher = self.LogitModel([[[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
        teacher.requires_grad_(False)
        trainer.teacher_model = teacher
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            # target positions after the causal shift: assistant, environment, assistant
            "labels": torch.tensor([[IGNORE_INDEX, 2, IGNORE_INDEX, 4]]),
        }
        loss = trainer.compute_loss(student, inputs)
        loss.backward()

        # Logit index 1 predicts token 3, whose label is the masked environment token.
        self.assertEqual(float(student.values.grad[0, 1].abs().sum()), 0.0)
        self.assertGreater(float(student.values.grad[0, 0].abs().sum()), 0.0)
        self.assertGreater(float(student.values.grad[0, 2].abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
