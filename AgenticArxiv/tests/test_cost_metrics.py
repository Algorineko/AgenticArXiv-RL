"""按成功归一化的代价指标。

对全部运行取平均会双向污染：撞上限的失败跑满迭代（拉高），
提前放弃的失败只跑一两步（拉低）。混在一起得到的数字既不代表
成功时的效率，也不代表失败的代价。
"""

import unittest

from benchmark.metrics import TaskMetrics
from benchmark.report import BenchmarkReport, _median


def _m(agent, ok, tokens, iters, tools=(), termination=None):
    if termination is None:
        termination = "FINISH" if ok else "FORCE_STOP"
    return TaskMetrics(
        task_id="t", agent_type=agent, trial=0,
        task_completed=termination == "FINISH", tool_call_accurate=ok,
        termination_type=termination, total_tokens=tokens,
        iteration_count=iters, tool_call_sequence=list(tools),
        total_time_ms=tokens,          # 让耗时可预测，便于断言
    )


class MedianTest(unittest.TestCase):
    def test_odd_length(self):
        self.assertEqual(_median([3, 1, 2]), 2.0)

    def test_even_length_averages_the_middle_pair(self):
        self.assertEqual(_median([1, 2, 3, 4]), 2.5)

    def test_empty_is_none_not_zero(self):
        self.assertIsNone(_median([]))

    def test_resists_the_long_tail(self):
        # 均值会被撞上限的样本拉走，中位不会
        values = [2, 2, 2, 2, 50]
        self.assertEqual(_median(values), 2.0)
        self.assertNotEqual(_median(values), sum(values) / len(values))


class CostByAgentTest(unittest.TestCase):
    def _cost(self, metrics, agent="regex"):
        return BenchmarkReport(metrics, model="test").cost_by_agent()[agent]

    def test_failed_runs_count_toward_the_numerator(self):
        # 两次运行共 300 token，只有一次成功 -> 300，不是 100
        row = self._cost([_m("regex", True, 100, 2), _m("regex", False, 200, 5)])
        self.assertEqual(row["tokens_per_success"], 300.0)
        self.assertEqual(row["successes"], 1)
        self.assertEqual(row["runs"], 2)

    def test_cheap_but_useless_agent_is_not_flattered(self):
        # 失败得快 != 便宜。这正是不能只统计成功运行的原因。
        good = self._cost([_m("regex", True, 500, 3)] * 2)
        fast_fail = self._cost([_m("regex", True, 500, 3)] + [_m("regex", False, 50, 1)] * 8)
        self.assertLess(good["tokens_per_success"], fast_fail["tokens_per_success"])

    def test_zero_successes_gives_none_not_zero_or_infinity(self):
        row = self._cost([_m("regex", False, 100, 5)] * 3)
        self.assertIsNone(row["tokens_per_success"])
        self.assertIsNone(row["calls_per_success"])
        self.assertIsNone(row["ms_per_success"])

    def test_calls_per_success_counts_tool_calls_not_steps(self):
        row = self._cost([_m("regex", True, 10, 4, tools=("a", "b", "c"))])
        self.assertEqual(row["calls_per_success"], 3.0)

    def test_iterations_are_reported_separately_for_wins_and_losses(self):
        metrics = [_m("regex", True, 10, 2), _m("regex", True, 10, 2),
                   _m("regex", False, 10, 9)]
        row = self._cost(metrics)
        self.assertEqual(row["median_iterations_success"], 2.0)
        self.assertEqual(row["median_iterations_failure"], 9.0)

    def test_no_failures_leaves_failure_median_as_none(self):
        row = self._cost([_m("regex", True, 10, 2)])
        self.assertIsNone(row["median_iterations_failure"])

    def test_failure_modes_are_split(self):
        metrics = [
            _m("regex", False, 10, 9, termination="FORCE_STOP"),
            _m("regex", False, 10, 1, termination="FINISH"),
            _m("regex", False, 10, 1, termination="ERROR"),
            _m("regex", True, 10, 2),
        ]
        row = self._cost(metrics)
        self.assertEqual(row["failed_at_limit"], 1)
        self.assertEqual(row["failed_claiming_done"], 1)   # 声称完成但工具不对
        self.assertEqual(row["failed_with_error"], 1)

    def test_appears_in_the_markdown_report(self):
        md = BenchmarkReport([_m("regex", True, 10, 2)], model="test").comparison_table_md()
        self.assertIn("Token / 成功", md)
        self.assertIn("失败形态", md)


if __name__ == "__main__":
    unittest.main()
