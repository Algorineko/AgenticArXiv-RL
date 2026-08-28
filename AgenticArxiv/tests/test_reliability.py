"""pass^k 与难度分档。

单次成功率会掩盖不稳定：实测 state_ref_ordinal_cn 在 8 次试验里
regex 5/8、mcp 4/8 —— 看着「一半能对」，但没有一次能连续三次做对。
pass^k 把这件事量出来。
"""

import unittest

from benchmark.metrics import TaskMetrics
from benchmark.report import BenchmarkReport, SUCCESS_CRITERIA, pass_hat_k


def _m(task_id, agent_type, trial, ok, completed=None):
    completed = ok if completed is None else completed
    return TaskMetrics(
        task_id=task_id, agent_type=agent_type, trial=trial,
        task_completed=completed, tool_call_accurate=ok,
        termination_type="FINISH" if completed else "FORCE_STOP",
    )


def _report(plan, agent="regex"):
    """plan: {task_id: [是否成功, ...]}"""
    metrics = [
        _m(task, agent, i, bool(ok))
        for task, outcomes in plan.items()
        for i, ok in enumerate(outcomes)
    ]
    return BenchmarkReport(metrics, model="test")


class PassHatKTest(unittest.TestCase):
    def test_pass_1_equals_naive_success_rate(self):
        # 自检：k=1 必须退化成朴素成功率
        for n, c in ((8, 5), (4, 1), (10, 10), (3, 0)):
            self.assertAlmostEqual(pass_hat_k(n, c, 1), c / n, places=9)

    def test_uses_the_unbiased_combinatorial_estimator(self):
        # C(5,3)/C(8,3) = 10/56
        self.assertAlmostEqual(pass_hat_k(8, 5, 3), 10 / 56, places=9)

    def test_all_successes_gives_one(self):
        self.assertEqual(pass_hat_k(8, 8, 3), 1.0)

    def test_fewer_successes_than_k_gives_zero(self):
        self.assertEqual(pass_hat_k(8, 2, 3), 0.0)

    def test_is_non_increasing_in_k(self):
        # pass^k 衡量可靠性，k 越大只会越低（与 HumanEval 的 pass@k 相反）
        values = [pass_hat_k(8, 5, k) for k in (1, 2, 3, 4)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_too_few_trials_returns_none_not_zero(self):
        # 「样本不够」不能和「真的做不到」混为一谈
        self.assertIsNone(pass_hat_k(2, 2, 3))

    def test_rejects_non_positive_k(self):
        with self.assertRaises(ValueError):
            pass_hat_k(8, 5, 0)


class ReliabilityByAgentTest(unittest.TestCase):
    def test_perfectly_stable_agent_scores_one_everywhere(self):
        row = _report({"a": [1, 1, 1, 1]}).reliability_by_agent()["regex"]
        self.assertEqual((row["pass^1"], row["pass^2"], row["pass^3"]), (1.0, 1.0, 1.0))

    def test_coin_flip_task_collapses_at_higher_k(self):
        row = _report({"a": [1, 0, 1, 0]}).reliability_by_agent()["regex"]
        self.assertAlmostEqual(row["pass^1"], 0.5)
        self.assertAlmostEqual(row["pass^2"], 1 / 6)
        self.assertEqual(row["pass^3"], 0.0)

    def test_averages_per_task_not_over_pooled_runs(self):
        # 「一稳成一稳败」与「两个各一半」的朴素成功率都是 50%，
        # 但可靠性完全不同 —— 必须先按任务估计再平均
        split = _report({"ok": [1, 1, 1, 1], "bad": [0, 0, 0, 0]}).reliability_by_agent()["regex"]
        mixed = _report({"x": [1, 0, 1, 0], "y": [1, 0, 1, 0]}).reliability_by_agent()["regex"]
        self.assertAlmostEqual(split["pass^1"], mixed["pass^1"])   # 朴素成功率相同
        self.assertGreater(split["pass^3"], mixed["pass^3"])       # 可靠性不同

    def test_tasks_with_too_few_trials_are_counted_not_averaged_in(self):
        row = _report({"short": [1, 1], "long": [1, 1, 1, 1]}).reliability_by_agent()["regex"]
        self.assertEqual(row["pass^3_skipped"], 1)
        self.assertEqual(row["pass^3"], 1.0)      # 只由 long 贡献，不被 short 拉低
        self.assertEqual(row["min_trials"], 2)

    def test_criterion_accurate_requires_both_completed_and_accurate(self):
        # 声称完成但工具序列不对，不算成功
        metrics = [_m("a", "regex", i, ok=False, completed=True) for i in range(4)]
        report = BenchmarkReport(metrics, model="test")
        self.assertEqual(report.reliability_by_agent(criterion="accurate")["regex"]["pass^1"], 0.0)
        self.assertEqual(report.reliability_by_agent(criterion="completed")["regex"]["pass^1"], 1.0)

    def test_rejects_unknown_criterion(self):
        with self.assertRaises(ValueError):
            _report({"a": [1]}).reliability_by_agent(criterion="nope")

    def test_every_declared_criterion_is_callable(self):
        sample = _m("a", "regex", 0, True)
        for name, fn in SUCCESS_CRITERIA.items():
            self.assertIsInstance(fn(sample), bool, name)


class DifficultyBandsTest(unittest.TestCase):
    """GRPO 的梯度来自组内奖励方差，成功率贴近 0 或 1 的任务不产生梯度。"""

    def test_splits_into_floor_middle_ceiling(self):
        bands = _report({
            "never": [0, 0, 0, 0],
            "half": [1, 0, 1, 0],
            "always": [1, 1, 1, 1],
        }).difficulty_bands()
        self.assertEqual(bands["floor"], ["never"])
        self.assertEqual(bands["middle"], ["half"])
        self.assertEqual(bands["ceiling"], ["always"])

    def test_boundaries_are_inclusive_of_the_middle_band(self):
        bands = _report({"low": [1, 0, 0, 0, 0], "high": [1, 1, 1, 1, 0]}).difficulty_bands()
        self.assertEqual(bands["middle"], ["high", "low"])   # 20% 与 80% 都算中间带

    def test_merges_agents_so_a_task_lands_in_one_band(self):
        metrics = ([_m("t", "regex", i, True) for i in range(4)]
                   + [_m("t", "mcp", i, False) for i in range(4)])
        bands = BenchmarkReport(metrics, model="test").difficulty_bands()
        self.assertEqual(bands["middle"], ["t"])             # 合并后 50%


if __name__ == "__main__":
    unittest.main()
