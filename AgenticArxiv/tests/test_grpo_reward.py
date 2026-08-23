#!/usr/bin/env python3
"""GRPO 奖励函数测试。

不需要 GPU、不需要 LLM、不需要网络（工具执行走离线快照，缺快照时自动跳过相关用例）。

运行：
    cd AgenticArxiv && python tests/test_grpo_reward.py
"""

import os
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
os.environ.setdefault("STORE_BACKEND", "memory")

import tools.arxiv_tool  # noqa: F401,E402
import tools.cache_status_tool  # noqa: F401,E402
import tools.pdf_download_tool  # noqa: F401,E402
import tools.pdf_translate_tool  # noqa: F401,E402

from benchmark.tasks import get_all_tasks  # noqa: E402
from rl.grpo_reward import (  # noqa: E402
    build_prompt_dataset,
    build_multiturn_prompt_dataset,
    make_grpo_reward_fn,
    make_multiturn_rollout_func,
    messages_to_trajectory,
    parse_react_action,
    synthesize_trajectory,
)

TASKS = {t["id"]: t for t in get_all_tasks()}
TASK_ID = "search_01"

CORRECT = ('Thought: 需要检索\n'
           'Action: {"name":"get_recently_submitted_cs_papers",'
           '"args":{"aspect":"AI","days":7,"max_results":5}}')
WRONG_ARGS = ('Thought: 需要检索\n'
              'Action: {"name":"get_recently_submitted_cs_papers",'
              '"args":{"aspect":"CR","days":30,"max_results":99}}')
WRONG_TOOL = 'Thought: 下载\nAction: {"name":"download_arxiv_pdf","args":{"ref":1}}'
BARE_FINISH = 'Thought: 完成了\nAction: FINISH'
BAD_JSON = "Thought: t\nAction: {'name': 'get_recently_submitted_cs_papers',}"
NO_ACTION = "Thought: 我先想想该怎么做"


class FakeState:
    def __init__(self, step): self.global_step = step


class TestParseReactAction(unittest.TestCase):
    def test_tool_call(self):
        kind, action = parse_react_action(CORRECT)
        self.assertEqual(kind, "call")
        self.assertEqual(action["name"], "get_recently_submitted_cs_papers")
        self.assertEqual(action["args"]["aspect"], "AI")

    def test_finish(self):
        self.assertEqual(parse_react_action(BARE_FINISH)[0], "finish")

    def test_bad_json_is_parse_error(self):
        """prompt 要求严格 JSON，坏 JSON 不应被降级修复，否则格式惩罚失效。"""
        self.assertEqual(parse_react_action(BAD_JSON)[0], "parse_error")

    def test_missing_action_section(self):
        self.assertEqual(parse_react_action(NO_ACTION)[0], "parse_error")

    def test_empty(self):
        self.assertEqual(parse_react_action("")[0], "parse_error")


class TestSynthesizeTrajectory(unittest.TestCase):
    def test_tool_call_becomes_two_steps(self):
        traj = synthesize_trajectory(CORRECT)
        actions = [s["action"] for s in traj["history"]]
        self.assertEqual(len(actions), 2)
        self.assertIn("get_recently_submitted_cs_papers", actions[0])
        self.assertEqual(actions[1], "FINISH")

    def test_finish_is_single_step(self):
        traj = synthesize_trajectory(BARE_FINISH)
        self.assertEqual([s["action"] for s in traj["history"]], ["FINISH"])

    def test_parse_error_marked(self):
        traj = synthesize_trajectory(BAD_JSON)
        self.assertEqual(traj["history"][0]["action"], "PARSE_ERROR")
        self.assertTrue(traj["history"][0]["parse_failed"])


class TestMultiTurnTrajectory(unittest.TestCase):
    def test_structured_messages_preserve_real_tool_chain(self):
        completion = [
            {
                "role": "assistant",
                "content": "先搜索",
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": "get_recently_submitted_cs_papers",
                        "arguments": {"aspect": "CV", "days": 7, "max_results": 3},
                    },
                }],
            },
            {"role": "tool", "name": "get_recently_submitted_cs_papers", "content": "3 papers"},
            {
                "role": "assistant",
                "content": "再下载",
                "tool_calls": [{
                    "type": "function",
                    "function": {"name": "download_arxiv_pdf", "arguments": {"ref": 1}},
                }],
            },
            {"role": "tool", "name": "download_arxiv_pdf", "content": "READY"},
            {"role": "assistant", "content": "已完成"},
        ]
        traj = messages_to_trajectory(completion)
        actions = [step["action"] for step in traj["history"]]
        self.assertEqual(len(actions), 3)
        self.assertIn("get_recently_submitted_cs_papers", actions[0])
        self.assertIn("download_arxiv_pdf", actions[1])
        self.assertEqual(actions[2], "FINISH")
        self.assertEqual(traj["history"][0]["observation"], "3 papers")
        self.assertEqual(traj["history"][1]["observation"], "READY")

    def test_reward_uses_all_turns(self):
        completion = [
            {
                "role": "assistant", "content": "",
                "tool_calls": [{"type": "function", "function": {
                    "name": "get_recently_submitted_cs_papers",
                    "arguments": {"aspect": "CV", "days": 7, "max_results": 3},
                }}],
            },
            {"role": "tool", "name": "get_recently_submitted_cs_papers", "content": "3 papers"},
            {
                "role": "assistant", "content": "",
                "tool_calls": [{"type": "function", "function": {
                    "name": "download_arxiv_pdf", "arguments": {"ref": 1},
                }}],
            },
            {"role": "tool", "name": "download_arxiv_pdf", "content": "READY"},
            {"role": "assistant", "content": "done"},
        ]
        fn = make_grpo_reward_fn(TASKS)
        reward = fn(completions=[completion], task_id=["composite_01"])[0]
        self.assertEqual(reward, 1.0)

    def test_custom_rollout_masks_environment_tokens(self):
        class Tokenizer:
            def apply_chat_template(self, prompt, tokenize=True, add_generation_prompt=True):
                return [1, 2]

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [200 + i for i, _ in enumerate(text)]}

            def decode(self, ids, skip_special_tokens=True):
                if ids and ids[0] == 101:
                    return CORRECT
                return BARE_FINISH

        class Model:
            training = True

        class Trainer:
            processing_class = Tokenizer()
            num_generations = 1
            num_generations_eval = 1
            max_completion_length = 256
            model = Model()

            def __init__(self): self.calls = 0

            def _generate_single_turn(self, prompt_ids, images, fields):
                self.calls += 1
                token = 101 if self.calls == 1 else 102
                return [[token] for _ in prompt_ids], None, {}

        class Environment:
            def reset(self): return None

            def get_recently_submitted_cs_papers(self, **kwargs):
                return [{"id": "x", "title": "paper"}]

        output = make_multiturn_rollout_func(Environment, max_turns=2)(
            [[{"role": "user", "content": "task"}]], Trainer()
        )
        history = output["trajectory_results"][0]["history"]
        self.assertEqual(len(history), 2)
        self.assertIn("get_recently_submitted_cs_papers", history[0]["action"])
        self.assertEqual(history[1]["action"], "FINISH")
        self.assertIn(0, output["env_mask"][0])
        self.assertIn(1, output["env_mask"][0])
        self.assertEqual(len(output["completion_ids"][0]), len(output["env_mask"][0]))


class TestRewardOrdering(unittest.TestCase):
    def setUp(self):
        self.fn = make_grpo_reward_fn(TASKS, env=None)

    def r(self, completion, step=0):
        return self.fn(completions=[completion], task_id=[TASK_ID],
                       trainer_state=FakeState(step))[0]

    def test_not_a_placeholder(self):
        """回归：原实现是 `return [0.0 for _ in responses]`，奖励恒为 0。"""
        rewards = [self.r(c) for c in (CORRECT, WRONG_TOOL, BAD_JSON)]
        self.assertNotEqual(len(set(rewards)), 1, "奖励不应恒为常数")
        self.assertTrue(any(x != 0.0 for x in rewards))

    def test_correct_beats_wrong_arguments(self):
        self.assertGreater(self.r(CORRECT), self.r(WRONG_ARGS))

    def test_correct_beats_wrong_tool(self):
        self.assertGreater(self.r(CORRECT), self.r(WRONG_TOOL))

    def test_valid_format_beats_unparseable(self):
        self.assertGreater(self.r(WRONG_TOOL), self.r(BAD_JSON))
        self.assertGreater(self.r(BARE_FINISH), self.r(NO_ACTION))

    def test_batch_matches_elementwise(self):
        batch = self.fn(completions=[CORRECT, WRONG_TOOL, BAD_JSON],
                        task_id=[TASK_ID] * 3, trainer_state=FakeState(0))
        self.assertEqual(len(batch), 3)
        self.assertEqual(batch, [self.r(CORRECT), self.r(WRONG_TOOL), self.r(BAD_JSON)])

    def test_unknown_task_id_is_neutral(self):
        self.assertEqual(
            self.fn(completions=[CORRECT], task_id=["nope"], trainer_state=FakeState(0)),
            [0.0],
        )


class TestCurriculum(unittest.TestCase):
    def test_correctness_gap_widens_after_curriculum(self):
        """RewardCalculator 的课程：早期压低语义正确性权重，后期给满。

        接上 trainer_state 后 training_step 才会推进；否则恒为 0、课程从不生效。
        """
        fn = make_grpo_reward_fn(TASKS, env=None)

        def gap(step):
            good = fn(completions=[CORRECT], task_id=[TASK_ID],
                      trainer_state=FakeState(step))[0]
            bad = fn(completions=[WRONG_ARGS], task_id=[TASK_ID],
                     trainer_state=FakeState(step))[0]
            return good - bad

        self.assertGreater(gap(60), gap(0))


class TestPromptDataset(unittest.TestCase):
    def test_prompt_is_full_react_prompt(self):
        rows = build_prompt_dataset(get_all_tasks())
        self.assertEqual(len(rows), len(get_all_tasks()))
        content = rows[0]["prompt"][0]["content"]
        # 训练输入必须与推理时一致：含工具描述与格式约束，而不是裸任务描述
        self.assertIn("Thought:", content)
        self.assertIn("Action:", content)
        self.assertIn("get_recently_submitted_cs_papers", content)
        self.assertIn("task_id", rows[0])

    def test_multiturn_prompt_is_conversational(self):
        rows = build_multiturn_prompt_dataset(get_all_tasks())
        self.assertEqual([m["role"] for m in rows[0]["prompt"]], ["system", "user"])
        self.assertIn("读取每次工具返回", rows[0]["prompt"][0]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
