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
JSON_FINISH = 'Thought: 完成了\nAction: {"name": "FINISH", "args": {}}'
JSON_FORCE_STOP = 'Thought: 放弃\nAction: {"name": "FORCE_STOP", "args": {}}'
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

    def test_json_wrapped_finish_is_not_a_tool_call(self):
        """模型常把结束写成和工具调用一样的 JSON 外壳。

        当成工具调用会连锁出三件事：rollout 拿到「未知工具: FINISH」的
        observation 并继续跑满 max_turns；一条本已正确收尾的轨迹被记上一次
        失败调用；奖励因此低于 benchmark 对同一条轨迹的判定。
        """
        self.assertEqual(parse_react_action(JSON_FINISH), ("finish", None))
        self.assertEqual(parse_react_action(JSON_FORCE_STOP), ("finish", None))

    def test_finish_case_insensitive(self):
        self.assertEqual(parse_react_action('Thought: t\nAction: {"name": "finish"}')[0],
                         "finish")

    def test_real_tool_still_parsed_after_finish_guard(self):
        kind, action = parse_react_action(WRONG_TOOL)
        self.assertEqual(kind, "call")
        self.assertEqual(action["name"], "download_arxiv_pdf")

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


class FakeGenerationTrainer:
    """够 make_multiturn_rollout_func 跑一轮的最小 trainer 替身。"""

    class _Model:
        training = True

    def __init__(self, tokenizer, num_generations=2, max_completion_length=16):
        self.processing_class = tokenizer
        self.model = self._Model()
        self.num_generations = num_generations
        self.num_generations_eval = num_generations
        self.max_completion_length = max_completion_length
        self.seen_batch_sizes = []

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        self.seen_batch_sizes.append(len(prompt_ids))
        # 每条都直接 FINISH，让 rollout 一轮结束
        ids = self.processing_class("Thought: 完成\nAction: FINISH",
                                    add_special_tokens=False)["input_ids"]
        return [list(ids) for _ in prompt_ids], None, None


class MultiTurnRolloutCardinalityTest(unittest.TestCase):
    """rollout_func 返回的条数必须与 TRL 传进来的 prompts 条数一致。

    TRL 交进来的 prompts 已经按 num_generations 重复过（num_generations=2
    时收到 2 条一模一样的 prompt）。rollout 里若再展开一次，返回条数会变成
    N*G*G，TRL 在 shuffle_sequence_dict 处直接抛
    `IndexError: index 3 is out of bounds for dimension 0 with size 2`，
    多轮 GRPO 一步都跑不完。
    """

    def _tokenizer(self):
        class Tok:
            def __call__(self, text, add_special_tokens=True):
                return {"input_ids": [1, 2, 3]}

            def apply_chat_template(self, prompt, tokenize=True, add_generation_prompt=True):
                return [4, 5]

            def decode(self, ids, skip_special_tokens=True):
                return "Thought: 完成\nAction: FINISH"
        return Tok()

    def _rollout(self, num_prompts, num_generations):
        from rl.grpo_reward import make_multiturn_rollout_func

        class _Env:
            def reset(self, *a, **k):
                return ""

        trainer = FakeGenerationTrainer(self._tokenizer(), num_generations=num_generations)
        fn = make_multiturn_rollout_func(lambda: _Env(), max_turns=2)
        prompts = [[{"role": "user", "content": "任务"}]] * num_prompts
        return fn(prompts, trainer), trainer

    def test_output_length_matches_input_length(self):
        for num_prompts, num_generations in ((2, 2), (4, 2), (6, 3), (1, 1)):
            with self.subTest(prompts=num_prompts, generations=num_generations):
                out, _ = self._rollout(num_prompts, num_generations)
                for key in ("prompt_ids", "completion_ids", "env_mask", "trajectory_results"):
                    self.assertEqual(
                        len(out[key]), num_prompts,
                        f"{key} 条数应等于传入的 prompts 条数，不能再乘一次 num_generations",
                    )

    def test_generation_batch_is_not_inflated(self):
        """展开一次会让每步生成量翻 num_generations 倍，显存和耗时都跟着翻。"""
        _, trainer = self._rollout(num_prompts=4, num_generations=2)
        self.assertEqual(trainer.seen_batch_sizes[0], 4)


class TrlVersionGuardTest(unittest.TestCase):
    """TRL 太老时必须启动即失败，而不是让多轮 rollout 静默失效。

    trl < 0.28 只在 use_vllm 且 vllm_mode == "server" 时调用 rollout_func。
    默认配置下多轮采样根本不执行，也不报错 —— 奖励退回
    messages_to_trajectory，任何非空 assistant 文本都被当成一步 FINISH，
    随机初始化的模型输出词沙拉也能拿到正分。
    """

    def test_version_boundary(self):
        from rl.grpo_reward import rollout_func_supported
        for version, supported in (
            ("0.20.0", False), ("0.25.1", False), ("0.27.9", False),
            ("0.28.0", True), ("0.29.1", True), ("1.0.0", True),
        ):
            with self.subTest(version=version):
                self.assertEqual(rollout_func_supported(version), supported)

    def test_old_trl_raises_with_upgrade_hint(self):
        from unittest import mock
        import rl.grpo_reward as gr
        # 伪造一个装了旧版 trl 的环境，别依赖本机实际装的版本
        with mock.patch("importlib.metadata.version", return_value="0.25.1"):
            with self.assertRaises(SystemExit) as ctx:
                gr.require_rollout_func_support()
        message = str(ctx.exception)
        self.assertIn("trl>=0.28.0", message)   # 给出可执行的升级命令
        self.assertIn("不会报错", message)       # 点明这是静默失效
        self.assertIn("0.25.1", message)        # 报出当前实际装的版本

    def test_supported_trl_passes_silently(self):
        from unittest import mock
        import rl.grpo_reward as gr
        with mock.patch("importlib.metadata.version", return_value="0.28.0"):
            gr.require_rollout_func_support()   # 不抛异常即可


if __name__ == "__main__":
    unittest.main(verbosity=2)
