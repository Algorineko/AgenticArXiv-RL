"""训练可观测性的测试。

重点不在「能不能记」，而在两件容易静默出错的事：
  1. --report_to 指了个没装的后端时必须报错，不能安静地什么都不记；
  2. 五个奖励分量必须真的进到 TRL 的指标缓冲区，否则曲线里只有一个
     被课程权重污染过的 total。
"""

import sys
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.grpo_reward import TOOL_ERROR_PREFIX, make_grpo_reward_fn  # noqa: E402
from rl.observability import (  # noqa: E402
    RewardComponentTracker,
    describe_logging,
    resolve_report_to,
    trajectory_health,
)
from rl.reward import RewardCalculator  # noqa: E402


class FakeTrainer:
    """只暴露 TRL GRPOTrainer 用来汇总自定义指标的那个缓冲区。"""

    def __init__(self):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}


class ResolveReportToTest(unittest.TestCase):
    def test_none_variants_disable(self):
        for value in (None, "", "none", "NONE", " off ", "no"):
            self.assertEqual(resolve_report_to(value), [], value)

    def test_known_backend_passes_when_installed(self):
        with mock.patch("rl.observability._backend_available", return_value=True):
            self.assertEqual(resolve_report_to("tensorboard"), ["tensorboard"])
            self.assertEqual(resolve_report_to("wandb,tensorboard"),
                             ["wandb", "tensorboard"])

    def test_missing_backend_fails_loudly(self):
        """静默降级是这个仓库反复踩过的坑：训练跑完才发现没有任何曲线。"""
        with mock.patch("rl.observability._backend_available", return_value=False):
            with self.assertRaises(SystemExit) as ctx:
                resolve_report_to("tensorboard")
        self.assertIn("pip install tensorboard", str(ctx.exception))

    def test_unknown_backend_fails(self):
        with self.assertRaises(SystemExit):
            resolve_report_to("tensorbored")

    def test_auto_falls_back_without_raising(self):
        """auto 是唯一允许降级的取值，因为它表达的就是「有就用」。"""
        with mock.patch("rl.observability._backend_available", return_value=False):
            self.assertEqual(resolve_report_to("auto"), [])
        with mock.patch("rl.observability._backend_available", return_value=True):
            self.assertEqual(resolve_report_to("auto"), ["tensorboard"])


class TrajectoryHealthTest(unittest.TestCase):
    def test_finished_trajectory(self):
        history = [
            {"action": '{"name": "get_recently_submitted_cs_papers"}', "observation": "成功获取 5 篇论文"},
            {"action": "FINISH", "observation": "任务完成"},
        ]
        stats = trajectory_health(history)
        self.assertEqual(stats["turns"], 2.0)
        self.assertEqual(stats["finished"], 1.0)
        self.assertEqual(stats["parse_error_rate"], 0.0)
        self.assertEqual(stats["tool_error_rate"], 0.0)

    def test_ran_out_of_turns_is_not_finished(self):
        """跑满 max_turns 和主动收尾必须区分开。

        两者 reward 都可能低，但前者是「没学会结束」，后者是「工具用错了」，
        处理方式完全不同 —— 只看 reward 曲线分不出来。
        """
        history = [{"action": '{"name": "download_arxiv_pdf"}', "observation": "ok"}] * 4
        self.assertEqual(trajectory_health(history)["finished"], 0.0)

    def test_parse_error_rate(self):
        history = [
            {"action": "PARSE_ERROR", "observation": "无法解析 Action", "parse_failed": True},
            {"action": "FINISH", "observation": "任务完成"},
        ]
        self.assertEqual(trajectory_health(history)["parse_error_rate"], 0.5)

    def test_tool_error_rate_uses_shared_prefix(self):
        """用 grpo_reward 导出的常量判定，避免两处各写一份会漂的中文文案。"""
        history = [
            {"action": '{"name": "download_arxiv_pdf"}',
             "observation": f"{TOOL_ERROR_PREFIX}未找到论文"},
            {"action": '{"name": "download_arxiv_pdf"}', "observation": "ok"},
            {"action": "FINISH", "observation": "任务完成"},
        ]
        stats = trajectory_health(history)
        self.assertEqual(stats["tool_error_rate"], 0.5)   # FINISH 不算进分母
        self.assertEqual(stats["parse_error_rate"], 0.0)


class TrackerTest(unittest.TestCase):
    def test_bind_and_record(self):
        tracker = RewardComponentTracker()
        trainer = FakeTrainer()
        self.assertTrue(tracker.bind(trainer))

        calc = RewardCalculator()
        task = {"id": "t", "task": "x", "expected_tools": ["get_recently_submitted_cs_papers"],
                "expected_termination": "FINISH"}
        result = {"history": [
            {"action": '{"name": "get_recently_submitted_cs_papers", "args": {}}',
             "observation": "成功获取 5 篇论文"},
            {"action": "FINISH", "observation": "任务完成"},
        ], "timing": {}, "token_usage": {}, "iteration_count": 2}
        breakdown, _ = calc.compute_reward_breakdown(task, result, training_step=0)
        tracker.record(breakdown, result)

        keys = trainer._metrics["train"]
        for name in RewardComponentTracker.COMPONENTS:
            self.assertIn(f"reward_components/{name}", keys, name)
            self.assertIn(f"reward_weights/{name}", keys, name)
        for name in ("turns", "finished", "parse_error_rate", "tool_error_rate"):
            self.assertIn(f"rollout/{name}", keys, name)

    def test_records_before_bind_are_not_lost(self):
        """reward fn 在 trainer 构造前就造好了，先 record 后 bind 是常态。"""
        tracker = RewardComponentTracker()
        calc = RewardCalculator()
        task = {"id": "t", "task": "x", "expected_tools": [], "expected_termination": "FINISH"}
        result = {"history": [{"action": "FINISH", "observation": "任务完成"}],
                  "timing": {}, "token_usage": {}, "iteration_count": 1}
        breakdown, _ = calc.compute_reward_breakdown(task, result)
        tracker.record(breakdown, result)

        trainer = FakeTrainer()
        tracker.bind(trainer)
        self.assertIn("reward_components/format", trainer._metrics["train"])

    def test_bind_failure_degrades_to_noop(self):
        """TRL 换了内部结构时，日志记不上不该把训练带崩。"""
        tracker = RewardComponentTracker()
        self.assertFalse(tracker.bind(object()))
        calc = RewardCalculator()
        task = {"id": "t", "task": "x", "expected_tools": [], "expected_termination": "FINISH"}
        result = {"history": [{"action": "FINISH", "observation": "ok"}],
                  "timing": {}, "token_usage": {}, "iteration_count": 1}
        breakdown, _ = calc.compute_reward_breakdown(task, result)
        tracker.record(breakdown, result)     # 不抛异常即可

    def test_curriculum_moves_total_while_components_hold_still(self):
        """这是「必须单独记分量」的核心论据，用一条固定轨迹钉住。

        同一条轨迹（策略完全没变），跨过第 30 步的课程边界后：
            total  +0.3125  ->  -0.03125
        而五个分量一模一样。原因是课程把 tool/argument/outcome 的权重从
        1/3 恢复到满权重，这条轨迹的 tool 分量是 -1，权重放开后被放大。

        也就是说 total reward 在策略没有任何变化时**下跌**了。只看
        total 曲线会把它误读成策略退化。
        """
        calc = RewardCalculator()
        task = {"id": "t", "task": "x",
                "expected_tools": ["get_recently_submitted_cs_papers"],
                "expected_termination": "FINISH"}
        # 故意调错工具：让各分量取值不同，否则加权平均对权重不敏感
        result = {"history": [
            {"action": '{"name": "download_arxiv_pdf", "args": {"ref": 1}}',
             "observation": "ok"},
            {"action": "FINISH", "observation": "任务完成"},
        ], "timing": {}, "token_usage": {}, "iteration_count": 2}

        early, _ = calc.compute_reward_breakdown(task, result, training_step=0)
        late, _ = calc.compute_reward_breakdown(task, result, training_step=100)

        for name in RewardComponentTracker.COMPONENTS:
            self.assertAlmostEqual(getattr(early, name), getattr(late, name), places=6, msg=name)
        self.assertAlmostEqual(early.total, 0.3125, places=4)
        self.assertAlmostEqual(late.total, -0.03125, places=5)
        self.assertLess(late.total, early.total)


class RewardFnTrackerWiringTest(unittest.TestCase):
    def test_reward_fn_feeds_tracker(self):
        tracker = RewardComponentTracker()
        trainer = FakeTrainer()
        tracker.bind(trainer)
        tasks = {"t": {"id": "t", "task": "x",
                       "expected_tools": ["get_recently_submitted_cs_papers"],
                       "expected_termination": "FINISH"}}
        fn = make_grpo_reward_fn(tasks, env=None, tracker=tracker)
        rewards = fn(
            completions=['Thought: 检索\nAction: {"name": "get_recently_submitted_cs_papers", "args": {}}'],
            task_id=["t"],
        )
        self.assertEqual(len(rewards), 1)
        self.assertIn("reward_components/tool", trainer._metrics["train"])

    def test_reward_fn_without_tracker_still_works(self):
        tasks = {"t": {"id": "t", "task": "x", "expected_tools": [],
                       "expected_termination": "FINISH"}}
        fn = make_grpo_reward_fn(tasks, env=None)
        self.assertEqual(len(fn(completions=["Thought: 完成\nAction: FINISH"], task_id=["t"])), 1)


class DescribeLoggingTest(unittest.TestCase):
    def test_disabled_message_tells_you_how_to_enable(self):
        text = describe_logging([], None)
        self.assertIn("--report_to tensorboard", text)

    def test_tensorboard_message_includes_command(self):
        text = describe_logging(["tensorboard"], "/tmp/run/logs")
        self.assertIn("tensorboard --logdir /tmp/run/logs", text)


if __name__ == "__main__":
    unittest.main()
