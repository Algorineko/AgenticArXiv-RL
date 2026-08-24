"""负向约束 benchmark case 的回归测试。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.metrics import _check_tool_sequence, argument_match_score  # noqa: E402
from benchmark.tasks_expanded import EXPANDED_TASKS  # noqa: E402


class ConstraintTaskTest(unittest.TestCase):
    def setUp(self):
        self.tasks = {
            task["id"]: task
            for task in EXPANDED_TASKS
            if task["category"] == "constraint"
        }

    def test_six_constraint_cases_are_registered(self):
        self.assertEqual(len(self.tasks), 6)

    def test_cases_do_not_expand_the_tool_space(self):
        existing_tools = {
            "get_recently_submitted_cs_papers",
            "download_arxiv_pdf",
            "translate_arxiv_pdf",
            "get_paper_cache_status",
        }
        used_tools = {
            name
            for task in self.tasks.values()
            for name in task["expected_tools"]
        }
        self.assertLessEqual(used_tools, existing_tools)

    def test_no_file_case_declares_the_side_effect_switch(self):
        task = self.tasks["constraint_search_no_file"]
        self.assertEqual(task["expected_tool_args"], [{
            "aspect": "*",
            "days": 7,
            "max_results": 5,
            "save_to_file": False,
        }])

        history = [{
            "action": {
                "name": "get_recently_submitted_cs_papers",
                "args": task["expected_tool_args"][0],
            }
        }]
        self.assertEqual(argument_match_score(history, task["expected_tool_args"]), 1.0)

    def test_extra_helpful_call_is_rejected(self):
        task = self.tasks["constraint_download_only"]
        expected = task["expected_tools"]

        self.assertTrue(_check_tool_sequence(["download_arxiv_pdf"], expected))
        self.assertFalse(_check_tool_sequence([
            "get_paper_cache_status",
            "download_arxiv_pdf",
        ], expected))
        self.assertFalse(_check_tool_sequence([
            "download_arxiv_pdf",
            "translate_arxiv_pdf",
        ], expected))


if __name__ == "__main__":
    unittest.main()
