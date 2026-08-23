import os
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
os.environ.setdefault("STORE_BACKEND", "memory")

from rl.multiturn_env import AgenticArxivMultiTurnEnv  # noqa: E402


PAPER = {
    "id": "2601.00001v1",
    "title": "A Test Paper",
    "authors": ["A"],
    "summary": "test",
    "published": "2026-01-01 00:00:00",
    "updated": "2026-01-01 00:00:00",
    "pdf_url": "https://arxiv.org/pdf/2601.00001v1",
    "primary_category": "cs.AI",
    "categories": ["cs.AI"],
    "comment": None,
    "links": [],
}


class FakeBackend:
    def execute_tool(self, name, args):
        if name == "get_recently_submitted_cs_papers":
            return [PAPER]
        raise AssertionError(name)


class MultiTurnEnvTest(unittest.TestCase):
    def setUp(self):
        self.env = AgenticArxivMultiTurnEnv()
        self.env.backend = FakeBackend()
        self.env.reset(task_id="composite_01")

    def test_search_then_download_uses_same_session_state(self):
        papers = self.env.get_recently_submitted_cs_papers("AI", 7, 3)
        downloaded = self.env.download_arxiv_pdf(1)
        self.assertEqual(len(papers), 1)
        self.assertEqual(downloaded["paper_id"], PAPER["id"])
        self.assertTrue(downloaded["offline"])

    def test_reset_clears_paper_memory(self):
        self.env.get_recently_submitted_cs_papers("AI", 7, 3)
        self.env.reset(task_id="next")
        with self.assertRaises(ValueError):
            self.env.download_arxiv_pdf(1)


if __name__ == "__main__":
    unittest.main()
