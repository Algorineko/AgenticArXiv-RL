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
        if name == "search_arxiv_papers":
            return [{**PAPER, "_mock_env": {"offline_fallback": True}}]
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

    def test_keyword_search_then_download_uses_same_session_state(self):
        papers = self.env.search_arxiv_papers("all:agentic reinforcement learning", 3, 30)
        downloaded = self.env.download_arxiv_pdf(1)
        self.assertEqual(len(papers), 1)
        self.assertEqual(downloaded["paper_id"], PAPER["id"])

    def test_optional_arguments_match_production_tool_contract(self):
        self.env.get_recently_submitted_cs_papers(
            "AI", 7, 3, output_path=None, save_to_file=False
        )
        downloaded = self.env.download_arxiv_pdf(1, force=True)
        translated = self.env.translate_arxiv_pdf(
            None, force=True, service="bing", threads=8, keep_dual=True
        )
        self.assertEqual(downloaded["paper_id"], PAPER["id"])
        self.assertEqual(translated["paper_id"], PAPER["id"])
        self.assertTrue(translated["keep_dual"])
        self.assertEqual(translated["threads"], 8)

    def test_cache_status_reflects_rollout_state(self):
        self.env.get_recently_submitted_cs_papers("AI", 7, 3)
        before = self.env.get_paper_cache_status(1)
        self.assertFalse(before["pdf_ready"])
        self.env.download_arxiv_pdf(None)
        after = self.env.get_paper_cache_status(paper_id=PAPER["id"])
        self.assertTrue(after["pdf_ready"])

    def test_null_reference_without_active_paper_fails(self):
        self.env.get_recently_submitted_cs_papers("AI", 7, 3)
        with self.assertRaises(ValueError):
            self.env.translate_arxiv_pdf(None)


if __name__ == "__main__":
    unittest.main()
