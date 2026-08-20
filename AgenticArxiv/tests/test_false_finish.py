"""假完成：声称完成，但期望的工具调用没做全。

抓的是「不做事也能拿分」这条奖励漏洞。实测 state_cache_before_dl
24 次运行里 18 次属于此类（75%）—— 查完缓存直接 FINISH，
task_completed=True，但论文根本没下载。只看 task_completed 这一栏，它是满分。
"""

import json
import unittest

from benchmark.metrics import extract_metrics, is_false_finish, lcs_length
from benchmark.report import BenchmarkReport

SEARCH, DOWNLOAD, CACHE = "get_recently_submitted_cs_papers", "download_arxiv_pdf", "get_paper_cache_status"


def _step(name, args=None):
    return {"thought": "t", "action": json.dumps({"name": name, "args": args or {}}), "observation": ""}


def _finish():
    return {"thought": "done", "action": "FINISH", "observation": "任务完成"}


class LcsLengthTest(unittest.TestCase):
    """与 rl/reward.py 共用一份实现，改动必须两边同时生效。"""

    def test_matches_known_values(self):
        self.assertEqual(lcs_length(["a", "b", "c"], ["a", "c"]), 2)
        self.assertEqual(lcs_length([], ["a"]), 0)
        self.assertEqual(lcs_length(["a"], []), 0)

    def test_respects_order(self):
        self.assertEqual(lcs_length(["b", "a"], ["a", "b"]), 1)

    def test_reward_module_uses_the_same_function(self):
        from rl import reward
        self.assertIs(reward.lcs_length, lcs_length)


class IsFalseFinishTest(unittest.TestCase):
    def test_all_expected_tools_called_is_not_a_false_finish(self):
        self.assertFalse(is_false_finish("FINISH", [SEARCH, DOWNLOAD], [SEARCH, DOWNLOAD]))

    def test_missing_a_required_tool_is_a_false_finish(self):
        # 实测样本：查完缓存就 FINISH，本该还要下载
        self.assertTrue(is_false_finish("FINISH", [CACHE], [CACHE, DOWNLOAD]))

    def test_extra_tools_are_not_a_false_finish(self):
        # 与 tool_call_accurate 的关键区别：严格序列比对会判 False，这里不算
        self.assertFalse(is_false_finish("FINISH", [SEARCH, CACHE, DOWNLOAD], [SEARCH, DOWNLOAD]))

    def test_doing_nothing_at_all_is_a_false_finish(self):
        self.assertTrue(is_false_finish("FINISH", [], [DOWNLOAD]))

    def test_wrong_order_counts_as_a_false_finish(self):
        # 先下载后检索，说明下载时会话里还没有论文列表 —— 工作没能真正完成。
        # 这是刻意的取舍：LCS 计顺序，所以顺序错等于该做的没做成。
        self.assertTrue(is_false_finish("FINISH", [DOWNLOAD, SEARCH], [SEARCH, DOWNLOAD]))

    def test_force_stop_is_never_a_false_finish(self):
        # 撞上限不算「声称完成」，它已经被 termination_type 记下了
        self.assertFalse(is_false_finish("FORCE_STOP", [], [DOWNLOAD]))

    def test_error_is_never_a_false_finish(self):
        self.assertFalse(is_false_finish("ERROR", [], [DOWNLOAD]))

    def test_tasks_expecting_no_tools_are_never_false_finishes(self):
        # 「什么都不该做」的任务，FINISH 就是正确行为
        self.assertFalse(is_false_finish("FINISH", [], []))


class ExtractMetricsIntegrationTest(unittest.TestCase):
    def _metrics(self, history, expected):
        return extract_metrics({"id": "t", "expected_tools": expected},
                               {"history": history}, "regex", 0)

    def test_the_measured_failure_mode_is_flagged(self):
        m = self._metrics([_step(CACHE, {"ref": 2}), _finish()], [CACHE, DOWNLOAD])
        self.assertTrue(m.task_completed)        # 只看这一栏是满分
        self.assertTrue(m.false_finish)          # 这一栏才暴露问题

    def test_a_clean_run_is_not_flagged(self):
        m = self._metrics([_step(SEARCH), _step(DOWNLOAD), _finish()], [SEARCH, DOWNLOAD])
        self.assertTrue(m.task_completed)
        self.assertFalse(m.false_finish)

    def test_detour_is_inaccurate_but_not_a_false_finish(self):
        # 两个指标必须能区分「绕路」和「没做」
        m = self._metrics([_step(SEARCH), _step(CACHE), _step(DOWNLOAD), _finish()], [SEARCH, DOWNLOAD])
        self.assertFalse(m.tool_call_accurate)   # 严格序列不符
        self.assertFalse(m.false_finish)         # 但活确实干了

    def test_json_wrapped_finish_does_not_break_the_check(self):
        history = [_step(DOWNLOAD), _step("FINISH")]
        self.assertFalse(self._metrics(history, [DOWNLOAD]).false_finish)


class ReportAggregationTest(unittest.TestCase):
    def _report(self, rows):
        metrics = [
            extract_metrics({"id": tid, "expected_tools": exp}, {"history": hist}, agent, i)
            for i, (tid, exp, hist, agent) in enumerate(rows)
        ]
        return BenchmarkReport(metrics, model="test")

    def test_reports_both_denominators(self):
        rows = [
            ("t1", [DOWNLOAD], [_step(CACHE), _finish()], "regex"),          # 假完成
            ("t2", [DOWNLOAD], [_step(DOWNLOAD), _finish()], "regex"),       # 正常
            ("t3", [DOWNLOAD], [{"thought": "x", "action": "FORCE_STOP", "observation": ""}], "regex"),
        ]
        row = self._report(rows).summary_by_agent()["regex"]
        self.assertAlmostEqual(row["false_finish_rate"], 1 / 3)            # 占全部运行
        self.assertAlmostEqual(row["false_finish_of_completed"], 1 / 2)    # 占 FINISH 运行

    def test_no_completed_runs_does_not_divide_by_zero(self):
        rows = [("t", [DOWNLOAD], [{"thought": "x", "action": "ERROR", "observation": ""}], "regex")]
        self.assertEqual(self._report(rows).summary_by_agent()["regex"]["false_finish_of_completed"], 0.0)

    def test_appears_in_the_markdown_report(self):
        rows = [("t", [DOWNLOAD], [_step(CACHE), _finish()], "regex")]
        self.assertIn("假完成率", self._report(rows).comparison_table_md())


if __name__ == "__main__":
    unittest.main()
