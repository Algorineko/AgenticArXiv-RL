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
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
os.environ.setdefault("STORE_BACKEND", "memory")

import tools.arxiv_tool  # noqa: F401,E402
import tools.cache_status_tool  # noqa: F401,E402
import tools.pdf_download_tool  # noqa: F401,E402
import tools.pdf_translate_tool  # noqa: F401,E402

from benchmark.tasks import get_all_tasks  # noqa: E402
from agents.react_contract import CONTEXT_VERSION, append_react_turn  # noqa: E402
from rl.grpo_reward import (  # noqa: E402
    build_prompt_dataset,
    make_grpo_reward_fn,
    make_multiturn_rollout_func,
    parse_react_action,
)
from rl.reward import RewardCalculator  # noqa: E402
from tests.test_rollout_audit import _clean_trajectory  # noqa: E402

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
        self.assertEqual(parse_react_action(JSON_FORCE_STOP), ("parse_error", None))

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


class TestMultiTurnTrajectory(unittest.TestCase):
    def test_structured_messages_cannot_replace_actual_trajectory(self):
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
        fn = make_grpo_reward_fn(TASKS)
        with self.assertRaisesRegex(ValueError, "缺少实际 trajectory_results"):
            fn(completions=[completion], task_id=["composite_01"])

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
        with self.assertRaisesRegex(ValueError, "缺少实际 trajectory_results"):
            fn(completions=[completion], task_id=["composite_01"])

    @patch("rl.grpo_reward.require_rollout_func_support")
    def test_custom_rollout_masks_environment_tokens(self, _require_trl):
        class Tokenizer:
            def apply_chat_template(self, prompt, tokenize=True, add_generation_prompt=True):
                ids = [1, 2]
                for message in prompt[1:]:
                    if message["role"] == "assistant":
                        ids.append(101 if message["content"] == CORRECT else 102)
                    else:
                        ids.extend(self(message["content"])["input_ids"])
                return ids

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
        trajectory = output["trajectory_results"][0]
        self.assertEqual(trajectory["sampling_protocol"], CONTEXT_VERSION)
        self.assertNotIn("training_protocol", trajectory)
        self.assertIsNone(trajectory["training_protocol_metadata"]["training_protocol"])
        self.assertEqual(len(history), 2)
        self.assertIn("get_recently_submitted_cs_papers", history[0]["action"])
        self.assertEqual(history[1]["action"], "FINISH")
        self.assertIn(0, output["env_mask"][0])
        self.assertIn(1, output["env_mask"][0])
        self.assertEqual(len(output["completion_ids"][0]), len(output["env_mask"][0]))
        self.assertEqual([
            token for token, policy in zip(output["completion_ids"][0], output["env_mask"][0])
            if policy
        ], [101, 102])
        fn = make_grpo_reward_fn(TASKS)
        self.assertEqual(fn(
            completions=[CORRECT], task_id=[TASK_ID],
            trajectory_results=output["trajectory_results"],
        ), [1.0])
        # 复用迁移前已有的固定模板输出 [1, 2] 检查不兼容保护。
        trainer = Trainer()
        with patch.object(trainer.processing_class, "apply_chat_template", return_value=[1, 2]):
            with self.assertRaisesRegex(ValueError, "无法保留原始采样 token 前缀"):
                make_multiturn_rollout_func(Environment, max_turns=2)(
                    [[{"role": "user", "content": "task"}]], trainer,
                )

    @patch("rl.grpo_reward.require_rollout_func_support")
    def test_custom_rollout_excludes_generated_observations(self, _require_trl):
        fabricated = "FABRICATED_RESULT\nThought: pretend done\nAction: FINISH"
        cases = [
            ("clean", list(CORRECT), len(CORRECT), 1024),
            ("split_marker", list(CORRECT + "\nObservation: " + fabricated),
             len(CORRECT) + 1, 1024),
            ("merged_marker", list(CORRECT) + ["\nOb", "servation: " + fabricated],
             len(CORRECT), 1024),
            ("budget_inside_marker", list(CORRECT + "\nObservation: " + fabricated),
             len(CORRECT) + 1, len(CORRECT) + 5),
            ("overlapping_action", list(CORRECT[:-1]) + ["}\nObservation: " + fabricated],
             len(CORRECT) - 1, 1024),
            ("marker_at_start", list("Observation: " + fabricated), 1, 1024),
        ]

        for name, pieces, retained, budget in cases:
            with self.subTest(name=name):
                class Tokenizer:
                    # Sampled IDs deliberately differ from re-encoding the text.
                    tokens = {1000000 + i: piece for i, piece in enumerate(pieces)}

                    def __call__(self, text, add_special_tokens=False):
                        return {"input_ids": [ord(char) for char in text]}

                    def apply_chat_template(self, prompt, tokenize=True, add_generation_prompt=True):
                        ids = self(prompt[0]["content"])["input_ids"]
                        for message in prompt[1:]:
                            content = message["content"]
                            if message["role"] == "assistant" and content.startswith(CORRECT):
                                prefix = ""
                                for token, piece in self.tokens.items():
                                    if not content.startswith(prefix + piece):
                                        break
                                    ids.append(token)
                                    prefix += piece
                                    if prefix == content:
                                        break
                            else:
                                ids.extend(self(content)["input_ids"])
                        return ids

                    def decode(self, ids, skip_special_tokens=True):
                        return "".join(self.tokens.get(t, chr(t)) for t in ids)

                class Trainer:
                    processing_class = Tokenizer()
                    max_completion_length = budget

                    def __init__(self):
                        self.inputs = []

                    def _generate_single_turn(self, prompt_ids, images, fields):
                        self.inputs.append([list(ids) for ids in prompt_ids])
                        ids = (list(self.processing_class.tokens) if len(self.inputs) == 1
                               else [ord(char) for char in BARE_FINISH])
                        return [ids], None, {}

                class Environment:
                    def reset(self):
                        pass

                    def get_recently_submitted_cs_papers(self, **kwargs):
                        return "REAL_RESULT"

                trainer = Trainer()
                if name == "marker_at_start":
                    with self.assertRaisesRegex(ValueError, "首轮没有合法策略 token"):
                        make_multiturn_rollout_func(Environment, max_turns=2)(["task"], trainer)
                    self.assertEqual(len(trainer.inputs), 1)
                    continue
                output = make_multiturn_rollout_func(Environment, max_turns=2)(
                    ["task"], trainer,
                )
                ids = output["completion_ids"][0]
                mask = output["env_mask"][0]
                tokenizer = trainer.processing_class
                expected = list(tokenizer.tokens)[:retained]
                invalid = name == "overlapping_action"
                if name != "budget_inside_marker" and not invalid:
                    expected += [ord(char) for char in BARE_FINISH]
                self.assertEqual([t for t, m in zip(ids, mask) if m], expected)
                self.assertTrue(ids)  # TRL inspects completion_ids[-1].
                self.assertEqual(len(ids), len(mask))
                self.assertLessEqual(len(ids), budget)

                trajectory = output["trajectory_results"][0]
                self.assertEqual(trajectory["raw_assistant_turns"][0], "".join(pieces))
                self.assertNotIn("FABRICATED_RESULT", tokenizer.decode(ids))
                if invalid:
                    self.assertEqual(trajectory["history"][0]["action"], "PARSE_ERROR")
                    self.assertEqual(len(trainer.inputs), 1)
                    continue

                suffix = append_react_turn(
                    [{"role": "user", "content": "task"}], CORRECT, "REAL_RESULT",
                )[-1]["content"]
                env_text = tokenizer.decode([t for t, m in zip(ids, mask) if not m])
                self.assertEqual(env_text, suffix[:budget - retained])
                self.assertEqual(trajectory["history"][0]["observation"], "REAL_RESULT")
                if name == "budget_inside_marker":
                    self.assertEqual(trajectory["termination_reason"], "TOKEN_LIMIT")
                    self.assertEqual(trajectory["history"][-1]["action"], "FORCE_STOP")
                    self.assertTrue(trajectory["history"][-1]["framework_generated"])
                    self.assertEqual(trajectory["iteration_count"], len(trainer.inputs))
                if name != "budget_inside_marker":
                    self.assertEqual(trainer.inputs[1][0],
                                     output["prompt_ids"][0] + ids[:retained + len(suffix)])
                    self.assertNotIn("FABRICATED_RESULT", tokenizer.decode(trainer.inputs[1][0]))
                    self.assertEqual(trajectory["history"][-1]["action"], "FINISH")


class TestActualTrajectoryReward(unittest.TestCase):
    def setUp(self):
        self.fn = make_grpo_reward_fn(TASKS)
        # 复用已有审计测试的完整 history，不另建单轮合成助手。
        self.trajectory = _clean_trajectory()

    def r(self, completion, step=0):
        return self.fn(completions=[completion], task_id=[TASK_ID],
                       trajectory_results=[self.trajectory],
                       trainer_state=FakeState(step))[0]

    def test_not_a_placeholder(self):
        """回归：原实现是 `return [0.0 for _ in responses]`，奖励恒为 0。"""
        expected, _ = RewardCalculator().compute_reward_breakdown(
            TASKS[TASK_ID], self.trajectory,
        )
        self.assertNotEqual(expected.total, 0.0)
        # completion 文本不能覆写实际 history，也不能触发额外执行。
        with patch("rl.grpo_reward._dispatch_environment_tool", side_effect=AssertionError):
            rewards = [self.r(c) for c in (CORRECT, WRONG_TOOL, BAD_JSON)]
        self.assertEqual(rewards, [expected.total] * 3)

    def test_batch_matches_elementwise(self):
        batch = self.fn(completions=[CORRECT, WRONG_TOOL, BAD_JSON],
                        trajectory_results=[self.trajectory] * 3,
                        task_id=[TASK_ID] * 3, trainer_state=FakeState(0))
        self.assertEqual(len(batch), 3)
        self.assertEqual(batch, [self.r(CORRECT), self.r(WRONG_TOOL), self.r(BAD_JSON)])

    def test_unknown_task_id_is_neutral(self):
        self.assertEqual(
            self.fn(completions=[CORRECT], task_id=["nope"],
                    trajectory_results=[self.trajectory], trainer_state=FakeState(0)),
            [0.0],
        )

    def test_missing_trajectory_is_an_integration_error(self):
        with self.assertRaisesRegex(ValueError, "缺少实际 trajectory_results"):
            self.fn(completions=[CORRECT], task_id=[TASK_ID])

    def test_trajectory_length_must_match_completions(self):
        with self.assertRaisesRegex(ValueError, "输入长度不一致"):
            self.fn(completions=[CORRECT, WRONG_TOOL, BAD_JSON],
                    task_id=[TASK_ID] * 3, trajectory_results=[self.trajectory])


class TestCurriculum(unittest.TestCase):
    def test_callback_forwards_curriculum_step(self):
        """RewardCalculator 的课程：早期压低语义正确性权重，后期给满。

        接上 trainer_state 后 training_step 才会推进；否则恒为 0、课程从不生效。
        """
        calc = RewardCalculator()
        fn = make_grpo_reward_fn(TASKS, reward_calc=calc)
        with patch.object(calc, "compute_reward_breakdown", wraps=calc.compute_reward_breakdown) as score:
            for step in (0, 60):
                fn(completions=[CORRECT], task_id=[TASK_ID],
                   trajectory_results=[_clean_trajectory()], trainer_state=FakeState(step))
        self.assertEqual([call.kwargs["training_step"] for call in score.call_args_list], [0, 60])


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
        self.assertEqual(rows[0]["prompt"][0]["_task_id"], rows[0]["task_id"])

    def test_same_user_text_keeps_hidden_ids_without_invented_setup_state(self):
        tasks = [
            {"id": "empty_session", "task": "把刚才那篇论文翻译一下"},
            {
                "id": "seeded_session",
                "task": "把刚才那篇论文翻译一下",
                "setup": [
                    {
                        "name": "get_recently_submitted_cs_papers",
                        "args": {"aspect": "AI", "days": 7, "max_results": 5},
                    },
                    {"name": "download_arxiv_pdf", "args": {"ref": 2}},
                ],
                "expected_tools": ["translate_arxiv_pdf"],
                "expected_tool_args": [{"ref": None}],
            },
        ]
        rows = build_prompt_dataset(tasks)
        empty_prompt = rows[0]["prompt"][0]["content"]
        seeded_prompt = rows[1]["prompt"][0]["content"]
        self.assertEqual(empty_prompt, seeded_prompt)
        self.assertNotIn("没有既有论文列表", empty_prompt)
        self.assertNotIn("此前已下载 ref=2", seeded_prompt)
        self.assertNotIn("ref=null 会指向它", seeded_prompt)
        # 环境状态可以看见，但任务标签和当前任务标准步骤不能混入文本。
        self.assertNotIn("empty_session", empty_prompt)
        self.assertNotIn("seeded_session", seeded_prompt)
        self.assertNotIn("expected_tools", seeded_prompt)
        self.assertNotIn("expected_tool_args", seeded_prompt)
        self.assertEqual(
            [row["prompt"][0]["_task_id"] for row in rows],
            ["empty_session", "seeded_session"],
        )

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
        with patch("rl.grpo_reward.require_rollout_func_support"):
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

    def test_visible_state_and_hidden_task_id_reach_the_right_consumers(self):
        class RecordingTokenizer:
            def __init__(self):
                self.seen_prompts = []

            def __call__(self, text, add_special_tokens=True):
                return {"input_ids": [1, 2, 3]}

            def apply_chat_template(self, prompt, tokenize=True, add_generation_prompt=True):
                self.seen_prompts.append(prompt)
                return [4, 5]

            def decode(self, ids, skip_special_tokens=True):
                return "Thought: 完成\nAction: FINISH"

        class Environment:
            def reset(self, task_id=""):
                self.task_id = task_id

            def get_recently_submitted_cs_papers(self, **kwargs):
                return [{"id": "paper-1"}]

        task_rows = [
            {
                "id": "empty_session",
                "task": "把刚才那篇论文翻译一下",
                "setup": [],
            },
            {
                "id": "seeded_session",
                "task": "把刚才那篇论文翻译一下",
                "setup": [{
                    "name": "get_recently_submitted_cs_papers",
                    "args": {"aspect": "AI", "days": 7, "max_results": 5},
                }],
            },
        ]
        tasks = {
            "empty_session": {"id": "empty_session", "setup": []},
            "seeded_session": {
                "id": "seeded_session",
                "setup": [{
                    "name": "get_recently_submitted_cs_papers",
                    "args": {"aspect": "AI", "days": 7, "max_results": 5},
                }],
            },
        }
        prompts = [row["prompt"] for row in build_prompt_dataset(task_rows)]
        tokenizer = RecordingTokenizer()
        trainer = FakeGenerationTrainer(
            tokenizer, num_generations=1, max_completion_length=16
        )
        with patch("rl.grpo_reward.require_rollout_func_support"):
            rollout = make_multiturn_rollout_func(
                Environment,
                max_turns=1,
                tasks_by_id=tasks,
                prompt_task_ids={},
            )

        output = rollout(prompts, trainer)

        self.assertEqual(
            [item["task_id"] for item in output["trajectory_results"]],
            ["empty_session", "seeded_session"],
        )
        self.assertEqual(output["trajectory_results"][0]["setup_actions"], [])
        self.assertEqual(
            output["trajectory_results"][1]["setup_actions"],
            tasks["seeded_session"]["setup"],
        )
        self.assertTrue(tokenizer.seen_prompts)
        self.assertEqual(
            tokenizer.seen_prompts[0][0]["content"],
            tokenizer.seen_prompts[1][0]["content"],
        )
        self.assertNotIn(
            "没有既有论文列表",
            tokenizer.seen_prompts[0][0]["content"],
        )
        self.assertNotIn(
            "此前已检索最近 7 天",
            tokenizer.seen_prompts[1][0]["content"],
        )
        self.assertTrue(all(
            "_task_id" not in message
            for prompt in tokenizer.seen_prompts
            for message in prompt
        ))


class TrlVersionGuardTest(unittest.TestCase):
    """TRL 太老时必须启动即失败，而不是让多轮 rollout 静默失效。

    trl < 0.28 只在 use_vllm 且 vllm_mode == "server" 时调用 rollout_func。
    默认配置下多轮采样根本不执行，也不报错 —— 奖励退回
    旧单轮降级会将任意非空 assistant 文本当成 FINISH；当前回调禁止
    缺少实际轨迹，因此仍须提前拒绝不调用 rollout_func 的旧版 TRL。
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
