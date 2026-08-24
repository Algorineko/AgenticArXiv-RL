"""_check_tool_sequence 严格匹配的单元测试

修复前：子序列匹配，中间插入错误工具或重复调用也会被判 accurate；
修复后：实际工具序列必须与预期序列完全一致。
"""

import unittest

from benchmark.metrics import _check_tool_sequence


class StrictToolSequenceTest(unittest.TestCase):
    EXPECTED = ["get_recently_submitted_cs_papers", "download_arxiv_pdf"]

    def test_exact_match_is_accepted(self):
        self.assertTrue(_check_tool_sequence(list(self.EXPECTED), self.EXPECTED))

    def test_inserted_wrong_tool_is_rejected(self):
        actual = [
            "get_recently_submitted_cs_papers",
            "get_paper_cache_status",
            "download_arxiv_pdf",
        ]
        self.assertFalse(_check_tool_sequence(actual, self.EXPECTED))

    def test_duplicate_call_is_rejected(self):
        actual = [
            "get_recently_submitted_cs_papers",
            "get_recently_submitted_cs_papers",
            "download_arxiv_pdf",
        ]
        self.assertFalse(_check_tool_sequence(actual, self.EXPECTED))

    def test_reversed_order_is_rejected(self):
        actual = ["download_arxiv_pdf", "get_recently_submitted_cs_papers"]
        self.assertFalse(_check_tool_sequence(actual, self.EXPECTED))

    def test_missing_tool_is_rejected(self):
        actual = ["get_recently_submitted_cs_papers"]
        self.assertFalse(_check_tool_sequence(actual, self.EXPECTED))

    def test_empty_expected_means_no_tool_call(self):
        """expected 为空 = 正确行为是一次工具都不调，不是「没声明标准答案」。

        这条原本断言的是 `_check_tool_sequence(["anything"], []) is True`，
        当时没有任何任务的 expected_tools 为空，那个返回值只是个安全默认。
        引入 category="infeasible"（请求超出能力边界或指向不存在的对象）后，
        空 expected 有了确切含义：无条件返回 True 会让幻觉调用也算「准确」，
        `_outcome_score` 因 task_completed and tool_call_accurate 给出满分——
        硬调不存在的第 20 篇反而拿到 outcome=+1.0。

        rl/reward.py 的 `_tool_score` 在同样输入下返回 0.0（既不算对也不算错），
        与这里的 True 本就不一致；现在两边统一成「调了就是错」。
        """
        self.assertTrue(_check_tool_sequence([], []))
        self.assertFalse(_check_tool_sequence(["anything"], []))


if __name__ == "__main__":
    unittest.main()