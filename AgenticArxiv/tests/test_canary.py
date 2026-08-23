#!/usr/bin/env python3
"""Canary 评估和回调测试。

不需要 GPU、不需要 LLM、不需要网络（工具执行走离线快照，缺快照时自动跳过相关用例）。

运行：
    cd AgenticArxiv && python tests/test_canary.py
"""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
os.environ.setdefault("STORE_BACKEND", "memory")

import tools.arxiv_tool  # noqa: F401,E402
import tools.cache_status_tool  # noqa: F401,E402
import tools.pdf_download_tool  # noqa: F401,E402
import tools.pdf_translate_tool  # noqa: F401,E402

import torch  # noqa: E402
from rl.canary import (  # noqa: E402
    DEFAULT_CANARY_TASK_IDS,
    CanaryCallback,
    CanaryEvaluator,
    CanaryResult,
)
from rl.reward import RewardCalculator  # noqa: E402

# 正确的 tool call（search_01 的 gold action）
CORRECT_COMPLETION = (
    'Thought: 需要检索AI论文\n'
    'Action: {"name":"get_recently_submitted_cs_papers",'
    '"args":{"aspect":"AI","days":7,"max_results":5}}'
)
# 错误的 tool call（参数不对）
WRONG_TOOL_COMPLETION = (
    'Thought: 下载论文\n'
    'Action: {"name":"download_arxiv_pdf","args":{"ref":1}}'
)
# 不可解析的
BAD_COMPLETION = "Thought: 我先想想\nAction: {invalid json}"


class _FakeTokenizer:
    """模拟 tokenizer：返回确定性的 token id 序列。"""

    def __init__(self, completion_text=CORRECT_COMPLETION):
        self._completion = completion_text
        self.pad_token_id = 0
        self.eos_token_id = 2

    def __call__(self, text, return_tensors="pt", **kwargs):
        # 返回假的 input_ids（长度为 10 的 prompt）
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        class FakeBatch(dict):
            def __getattr__(self, name):
                return self[name]

            def to(self, device):
                return FakeBatch({k: v.to(device) for k, v in self.items()})

        return FakeBatch(input_ids=ids)

    def batch_decode(self, ids, skip_special_tokens=True):
        # 返回预设的 completion 文本
        n = ids.shape[0] if hasattr(ids, 'shape') else len(ids)
        return [self._completion] * n

    @staticmethod
    def apply_chat_template(messages, tokenize=False):
        return "mock prompt"


class _FakeModel:
    """模拟模型：返回确定性的 token id 序列。"""

    def __init__(self, num_return_sequences=4):
        self._num = num_return_sequences
        self._training = True

    def generate(self, **kwargs):
        n = kwargs.get("num_return_sequences", self._num)
        # 返回 n 个假序列（prompt_len + 5 个新 token）
        prompt_len = kwargs.get("input_ids", torch.tensor([[0]])).shape[1]
        return torch.ones((n, prompt_len + 5), dtype=torch.long)

    def eval(self):
        self._training = False
        return self

    def train(self):
        self._training = True
        return self

    def parameters(self):
        return [torch.nn.Parameter(torch.zeros(1))]


class CanaryResultTest(unittest.TestCase):
    def test_defaults(self):
        r = CanaryResult(step=0, mean_reward=0.5, task_completion_rate=0.75,
                         per_task_rewards={"search_01": 0.5})
        self.assertEqual(r.step, 0)
        self.assertEqual(r.mean_reward, 0.5)
        self.assertEqual(r.task_completion_rate, 0.75)
        self.assertEqual(r.num_generations_per_task, 4)


class CanaryEvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.model = _FakeModel()
        self.tokenizer = _FakeTokenizer()
        self.evaluator = CanaryEvaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            canary_task_ids=DEFAULT_CANARY_TASK_IDS,
            reward_calc=RewardCalculator(),
            env=None,
            num_generations=2,
        )

    def test_evaluator_returns_result(self):
        result = self.evaluator.evaluate(step=100)
        self.assertIsInstance(result, CanaryResult)
        self.assertEqual(result.step, 100)
        self.assertIsInstance(result.mean_reward, float)
        self.assertIsInstance(result.task_completion_rate, float)
        self.assertIsInstance(result.per_task_rewards, dict)
        # 两个 canary 任务都应该有结果
        for tid in DEFAULT_CANARY_TASK_IDS:
            self.assertIn(tid, result.per_task_rewards)

    def test_invalid_task_id_raises(self):
        with self.assertRaises(ValueError):
            CanaryEvaluator(
                model=self.model,
                tokenizer=self.tokenizer,
                canary_task_ids=["nonexistent_task"],
            )

    def test_evaluator_restores_train_mode(self):
        self.model.train()  # 模拟训练模式
        self.evaluator.evaluate(step=0)
        self.assertTrue(self.model._training, "evaluate 后应恢复为 train 模式")


class CanaryCallbackTest(unittest.TestCase):
    def _make_cb(self, steps=10, min_reward=-0.5, patience=3):
        evaluator = Mock()
        evaluator.evaluate = Mock(return_value=CanaryResult(
            step=0, mean_reward=-0.8, task_completion_rate=0.0,
            per_task_rewards={"search_01": -0.8},
        ))
        return CanaryCallback(evaluator, steps=steps, min_reward=min_reward,
                              patience=patience)

    def _fire(self, cb, n, step=0, **logs):
        """触发 n 次 on_log。"""
        control = SimpleNamespace(should_training_stop=False)
        for i in range(n):
            state = SimpleNamespace(global_step=step + i * cb.steps)
            cb.on_log(Mock(), state, control, logs=dict(logs))
        return control

    def test_fires_at_correct_intervals(self):
        cb = self._make_cb(steps=10, min_reward=-1.0)
        # step=0 触发
        self._fire(cb, 1, step=0)
        self.assertEqual(len(cb.results), 1)
        # step=5 不触发（不是 10 的倍数）
        cb.on_log(Mock(), SimpleNamespace(global_step=5),
                  SimpleNamespace(should_training_stop=False), logs={})
        self.assertEqual(len(cb.results), 1)
        # step=10 触发
        cb.on_log(Mock(), SimpleNamespace(global_step=10),
                  SimpleNamespace(should_training_stop=False), logs={})
        self.assertEqual(len(cb.results), 2)

    def test_stops_after_patience(self):
        cb = self._make_cb(steps=10, min_reward=0.0, patience=3)
        control = self._fire(cb, 3, step=0)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(cb.tripped)

    def test_does_not_stop_before_patience(self):
        cb = self._make_cb(steps=10, min_reward=0.0, patience=3)
        control = self._fire(cb, 2, step=0)
        self.assertFalse(control.should_training_stop)
        self.assertFalse(cb.tripped)

    def test_streak_resets_on_healthy(self):
        cb = self._make_cb(steps=10, min_reward=0.0, patience=3)
        # 两次低于阈值
        self._fire(cb, 2, step=0)
        self.assertFalse(cb.tripped)
        # 一次健康（切换 evaluator 返回高奖励）
        cb.evaluator.evaluate = Mock(return_value=CanaryResult(
            step=0, mean_reward=0.5, task_completion_rate=1.0,
            per_task_rewards={"search_01": 0.5},
        ))
        self._fire(cb, 1, step=20)
        self.assertFalse(cb.tripped)
        # 再切换回低奖励，重新计数
        cb.evaluator.evaluate = Mock(return_value=CanaryResult(
            step=0, mean_reward=-0.8, task_completion_rate=0.0,
            per_task_rewards={"search_01": -0.8},
        ))
        control = self._fire(cb, 2, step=30)
        self.assertFalse(control.should_training_stop)  # 只 2 次，还没到 3

    def test_min_reward_default_disables_check(self):
        """默认 min_reward=-1.0 时阈值检查被禁用。"""
        cb = self._make_cb(steps=10, min_reward=-1.0, patience=1)
        control = self._fire(cb, 5, step=0)
        self.assertFalse(control.should_training_stop)
        self.assertFalse(cb.tripped)

    def test_results_accumulate(self):
        cb = self._make_cb(steps=10, min_reward=-1.0)
        self._fire(cb, 3, step=0)
        self.assertEqual(len(cb.results), 3)
        for r in cb.results:
            self.assertIsInstance(r, CanaryResult)


if __name__ == "__main__":
    unittest.main(verbosity=2)
