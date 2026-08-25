"""Unit tests for hierarchical reward shaping."""

import json
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from benchmark.tasks import BENCHMARK_TASKS, get_task_by_id
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

    def test_argument_layer_scores_values_not_key_presence(self):
        """键填齐不算分，只有值对了才算。

        原实现是 (键覆盖率 + 取值正确率) / 2，这里 days 填错、键齐全，
        旧口径给 0.75 → 分量 0.5。键覆盖率几乎不携带独立信息（键缺失时
        取值同样判错），它唯一的作用是把一半参数分白送给「照着工具签名
        把键填齐」，而那是任何能吐出合法 JSON 的模型免费拿到的。
        现在 2 个键对 1 个 → 0.5 → 分量 0.0。
        """
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
        self.assertEqual(reward.argument, 0.0)

    def test_complete_args_receive_maximum_argument_score(self):
        task = {
            "id": "search",
            "expected_tools": ["get_recently_submitted_cs_papers"],
            "expected_tool_args": [
                {"aspect": "AI", "days": 7, "max_results": 5}
            ],
        }
        history = [
            {
                "action": '{"name":"get_recently_submitted_cs_papers",'
                '"args":{"aspect":"AI","days":7,"max_results":5}}',
                "observation": "ok",
            },
            {"action": "FINISH", "observation": "done"},
        ]
        reward, _ = self.calculator.compute_reward_breakdown(task, _result(history))
        self.assertEqual(reward.argument, 1.0)

    def test_wrong_argument_values_receive_lower_score(self):
        task = {
            "id": "search",
            "expected_tools": ["get_recently_submitted_cs_papers"],
            "expected_tool_args": [
                {"aspect": "AI", "days": 7, "max_results": 5}
            ],
        }
        history = [
            {
                "action": '{"name":"get_recently_submitted_cs_papers",'
                '"args":{"aspect":"CV","days":30,"max_results":20}}',
                "observation": "ok",
            },
            {"action": "FINISH", "observation": "done"},
        ]
        reward, _ = self.calculator.compute_reward_breakdown(task, _result(history))
        # 三个值全错 → 取值正确率 0 → 分量 -1.0。
        # 旧口径因为键齐全还能拿 0.0（不罚），等于「搜错了方向」零成本。
        self.assertEqual(reward.argument, -1.0)

    def test_missing_arguments_receive_lower_score(self):
        task = {
            "id": "search",
            "expected_tools": ["get_recently_submitted_cs_papers"],
            "expected_tool_args": [
                {"aspect": "AI", "days": 7, "max_results": 5}
            ],
        }
        history = [
            {
                "action": '{"name":"get_recently_submitted_cs_papers",'
                '"args":{"aspect":"AI"}}',
                "observation": "ok",
            },
            {"action": "FINISH", "observation": "done"},
        ]
        reward, _ = self.calculator.compute_reward_breakdown(task, _result(history))
        self.assertAlmostEqual(reward.argument, -1.0 / 3.0, places=6)

    def test_task_without_expected_args_keeps_argument_layer_inactive(self):
        history = [
            {"action": '{"name":"search","args":{}}', "observation": "ok"},
            {"action": '{"name":"download","args":{}}', "observation": "ok"},
            {"action": "FINISH", "observation": "done"},
        ]
        reward, _ = self.calculator.compute_reward_breakdown(
            self.task, _result(history), training_step=30
        )
        self.assertEqual(reward.argument, 0.0)
        self.assertEqual(reward.total, 1.0)

    def test_composite_download_step_has_a_static_argument_oracle(self):
        """composite_01 第二步曾经写成 None（免检），前提是错的。

        任务原文是「然后下载第1篇」，而 download_arxiv_pdf 的 ref 就是
        1-based 序号（tools/pdf_download_tool.py: ref: Union[str,int,None] = 1），
        「第1篇」= ref 1 是静态可判的，不存在动态性。免检的代价是这一步
        随便填什么都拿满参数分。
        """
        task = get_task_by_id("composite_01")
        self.assertEqual(
            task["expected_tool_args"],
            [{"aspect": "CV", "days": 7, "max_results": 3}, {"ref": 1}],
        )
        head = {
            "action": '{"name":"get_recently_submitted_cs_papers",'
            '"args":{"aspect":"CV","days":7,"max_results":3}}',
            "observation": "ok",
        }
        finish = {"action": "FINISH", "observation": "done"}

        def download(ref):
            return {
                "action": '{"name":"download_arxiv_pdf","args":{"ref":%s}}' % json.dumps(ref),
                "observation": "ok",
            }

        right, _ = self.calculator.compute_reward_breakdown(
            task, _result([head, download(1), finish])
        )
        wrong, _ = self.calculator.compute_reward_breakdown(
            task, _result([head, download("dynamic-search-result"), finish])
        )
        self.assertEqual(right.argument, 1.0)
        self.assertEqual(wrong.argument, 0.0)

    def test_benchmark_search_tasks_define_static_argument_oracles(self):
        expected = {
            "search_01": {"aspect": "AI", "days": 7, "max_results": 5},
            "search_02": {"aspect": "LG", "days": 3, "max_results": 10},
            "search_03": {"aspect": "CL", "days": 7, "max_results": 5},
        }
        for task_id, args in expected.items():
            with self.subTest(task_id=task_id):
                self.assertEqual(get_task_by_id(task_id)["expected_tool_args"], [args])

    def test_benchmark_aspect_oracles_match_arxiv_tool_schema(self):
        utils_module = ModuleType("utils")
        utils_module.__path__ = []
        file_writer_module = ModuleType("utils.file_writer")
        file_writer_module.save_papers_to_file = Mock()
        modules = {
            "arxiv": ModuleType("arxiv"),
            "utils": utils_module,
            "utils.file_writer": file_writer_module,
        }
        with patch.dict("sys.modules", modules):
            from tools.arxiv_tool import ARXIV_TOOL_SCHEMA

        valid_aspects = ARXIV_TOOL_SCHEMA["properties"]["aspect"]["enum"]
        for task in BENCHMARK_TASKS:
            for expected_args in task.get("expected_tool_args", []):
                if expected_args is not None and "aspect" in expected_args:
                    with self.subTest(task_id=task["id"]):
                        self.assertIn(expected_args["aspect"], valid_aspects)

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
