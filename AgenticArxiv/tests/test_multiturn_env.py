import os
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
os.environ.setdefault("STORE_BACKEND", "memory")

from rl.multiturn_env import AgenticArxivMultiTurnEnv  # noqa: E402
from agents.react_contract import (  # noqa: E402
    ParsedAction, ParseKind, make_history_step, render_observation,
)
from benchmark.runner import BenchmarkResult  # noqa: E402
from benchmark.run_benchmark import _trace_record  # noqa: E402
from rl.trajectory import create_trajectory, load_trajectories, save_trajectory  # noqa: E402


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

    # RolloutSandbox 契约：capture/restore 与 env 其余状态一起快照。
    def capture_state(self):
        return self.__dict__.copy()

    def restore_state(self, state):
        self.__dict__.update(state)


class MultiTurnEnvTest(unittest.TestCase):
    def setUp(self):
        self.env = AgenticArxivMultiTurnEnv(mode="replay", backend=FakeBackend())
        self.addCleanup(self.env.close)
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

    def test_keyword_fallback_does_not_replace_session_state(self):
        papers = self.env.get_recently_submitted_cs_papers("AI", 7, 3)
        before = self.env.store.get_last_papers(self.env.session_id)
        active = self.env.store.get_last_active_paper_id(self.env.session_id)
        with self.assertRaisesRegex(ValueError, "未命中离线快照"):
            self.env.search_arxiv_papers("all:agentic reinforcement learning", 3, 30)
        self.assertEqual(self.env.store.get_last_papers(self.env.session_id), before)
        self.assertEqual(self.env.store.get_last_active_paper_id(self.env.session_id), active)
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

        # 复用上面的真实缓存结果，检查两条持久化路径，不补模型响应。
        step = make_history_step(ParsedAction(
            ParseKind.CALL, action={"name": "get_paper_cache_status", "args": {"paper_id": PAPER["id"]}},
        ), result=after)
        for field in ("downloaded_at", "updated_at"):
            self.assertIsInstance(after["pdf"][field], datetime)
            self.assertEqual(step["tool_result"]["pdf"][field], after["pdf"][field].isoformat())
        self.assertEqual(step["visible_observation"], render_observation(after).text)
        self.assertEqual(step["scoring_observation"], render_observation(
            after, body_limit=None, include_reference_map=False,
        ).text)
        raw = {"history": [step]}
        record = _trace_record(BenchmarkResult("composite_01", "regex", 0, raw))
        restored = json.loads(json.dumps(record, ensure_ascii=False))
        self.assertEqual(restored["history"][0]["tool_result"], step["tool_result"])
        trajectory = create_trajectory(
            task_id="composite_01", task="", session_id=self.env.session_id,
            history=raw["history"], final_reward=0.0, metrics={},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.jsonl"
            save_trajectory(trajectory, path)
            loaded = load_trajectories(path)
        self.assertEqual(loaded[0].steps[0].tool_result, step["tool_result"])

    def test_null_reference_without_active_paper_fails(self):
        self.env.get_recently_submitted_cs_papers("AI", 7, 3)
        with self.assertRaises(ValueError):
            self.env.translate_arxiv_pdf(None)


if __name__ == "__main__":
    unittest.main()
