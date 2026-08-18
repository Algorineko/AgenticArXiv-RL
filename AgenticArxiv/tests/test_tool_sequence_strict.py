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

    def test_empty_expected_always_ok(self):
        self.assertTrue(_check_tool_sequence(["anything"], []))


if __name__ == "__main__":
    unittest.main()