"""Regression coverage for the P0 keyword-search tool and offline replay."""

from datetime import datetime
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
os.environ.setdefault("STORE_BACKEND", "memory")

from benchmark.tasks_expanded import get_by_category  # noqa: E402
from rl.env import MockArxivEnv  # noqa: E402
from tools.arxiv_tool import (  # noqa: E402
    ARXIV_NETWORK_TIMEOUT_S,
    _keyword_query_expression,
    _network_timeout,
    search_arxiv_papers,
)


def _paper(index: int) -> dict:
    return {
        "id": f"2601.0000{index}v1",
        "title": f"Paper {index}",
        "authors": ["Test Author"],
        "summary": "test summary",
        "published": "2026-01-01 00:00:00",
        "updated": "2026-01-01 00:00:00",
        "pdf_url": f"https://arxiv.org/pdf/2601.0000{index}v1",
        "primary_category": "cs.AI",
        "categories": ["cs.AI"],
        "comment": None,
        "links": [],
    }


class KeywordQueryTest(unittest.TestCase):
    def test_maps_bare_text_and_supported_prefixes(self):
        self.assertEqual(_keyword_query_expression("agentic RL"), 'all:"agentic RL"')
        self.assertEqual(_keyword_query_expression("ti: Agentic RL"), 'ti:"Agentic RL"')
        self.assertEqual(_keyword_query_expression("au: Jane Doe"), 'au:"Jane Doe"')

    def test_rejects_empty_query(self):
        with self.assertRaises(ValueError):
            _keyword_query_expression("   ")

    def test_search_uses_the_normalized_arxiv_expression(self):
        class Result:
            def get_short_id(self):
                return "2601.00001v1"

            title = "Agentic RL"
            authors = []
            summary = "summary"
            published = datetime(2026, 1, 1)
            updated = None
            pdf_url = "https://arxiv.org/pdf/2601.00001v1"
            primary_category = "cs.AI"
            categories = ["cs.AI"]
            comment = None
            links = []

        with mock.patch("tools.arxiv_tool.arxiv.Client") as client_cls, \
             mock.patch("tools.arxiv_tool.arxiv.Search") as search_cls:
            client_cls.return_value.results.return_value = [Result()]
            papers = search_arxiv_papers("ti: Agentic RL", max_results=3)

        self.assertEqual(papers[0]["id"], "2601.00001v1")
        self.assertEqual(search_cls.call_args.kwargs["query"], 'ti:"Agentic RL"')


class KeywordSnapshotReplayTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.snapshot_path = Path(self.tmpdir.name) / "snapshot.json"
        self.known_args = {
            "query": "all:agentic reinforcement learning",
            "max_results": 3,
            "days": 30,
        }

        recorder = MockArxivEnv(self.snapshot_path, mode="record")
        with mock.patch("rl.env.registry.execute_tool", return_value=[_paper(1), _paper(2), _paper(3)]):
            recorder.execute_tool("search_arxiv_papers", self.known_args)
        recorder.save_snapshot()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_query_hash_replay_reuses_the_pool_with_a_new_limit(self):
        env = MockArxivEnv(self.snapshot_path, mode="replay")
        result = env.execute_tool(
            "search_arxiv_papers",
            {**self.known_args, "max_results": 1},
        )
        self.assertEqual([paper["id"] for paper in result], ["2601.00001v1"])
        self.assertEqual(env.stats["real_calls"], 0)
        self.assertEqual(env.stats["fallback"], 0)

    def test_unseen_query_uses_a_marked_deterministic_fallback(self):
        args = {"query": "all:unseen query", "max_results": 2, "days": 30}
        first = MockArxivEnv(self.snapshot_path, mode="replay").execute_tool(
            "search_arxiv_papers", args
        )
        second_env = MockArxivEnv(self.snapshot_path, mode="replay")
        second = second_env.execute_tool("search_arxiv_papers", args)

        self.assertEqual([paper["id"] for paper in first], [paper["id"] for paper in second])
        self.assertTrue(first[0]["_mock_env"]["offline_fallback"])
        self.assertEqual(second_env.stats["real_calls"], 0)
        self.assertEqual(second_env.stats["fallback"], 1)

    def test_fallback_does_not_pollute_session_papers(self):
        """回退结果带显式标记，且不得写入会话论文列表。

        BaseAgent 检测到 offline_fallback 会向模型报告"工具执行失败"；
        若 env 在回退分支同步 store，模型后续用 ref 操作会解析到这些
        无关论文——观测说失败、状态里却有"论文"，两者矛盾。
        """
        from models.store import store, use_memory_store

        use_memory_store(reset=True)
        env = MockArxivEnv(self.snapshot_path, mode="replay")
        args = {
            "query": "all:unseen query",
            "max_results": 2,
            "days": 30,
            "session_id": "sess-fallback-guard",
        }
        result = env.execute_tool("search_arxiv_papers", args)

        self.assertTrue(result[0]["_mock_env"]["offline_fallback"])
        self.assertEqual(env.stats["hit"], 0, "回退不是快照命中，不应计入 hit")
        self.assertEqual(
            store.get_last_papers("sess-fallback-guard"),
            [],
            "回退论文不应写入会话论文列表",
        )


class ArxivNetworkTimeoutTest(unittest.TestCase):
    """arXiv API 调用必须有时间上界。

    `arxiv.Client` 自建 requests.Session 且从不传 timeout，export.arxiv.org
    一旦卡住就会把调用方永久挂起——而 build_snapshot 是整条流水线唯一联网的
    一步，挂在那里等于整个离线训练链起不来。
    """

    def test_scopes_and_restores_the_socket_default(self):
        import socket

        original = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(None)
            with _network_timeout(7):
                self.assertEqual(socket.getdefaulttimeout(), 7)
            self.assertIsNone(socket.getdefaulttimeout())
        finally:
            socket.setdefaulttimeout(original)

    def test_restores_the_previous_value_on_error(self):
        import socket

        original = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(11)
            with self.assertRaises(RuntimeError):
                with _network_timeout(3):
                    raise RuntimeError("boom")
            self.assertEqual(socket.getdefaulttimeout(), 11)
        finally:
            socket.setdefaulttimeout(original)

    def test_default_is_a_finite_positive_number(self):
        self.assertGreater(ARXIV_NETWORK_TIMEOUT_S, 0)

    def test_search_never_leaves_an_unbounded_socket_behind(self):
        import socket

        original = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(None)
            with mock.patch(
                "tools.arxiv_tool.arxiv.Client",
                side_effect=RuntimeError("network down"),
            ):
                with self.assertRaises(RuntimeError):
                    search_arxiv_papers("all:anything")
            self.assertIsNone(socket.getdefaulttimeout())
        finally:
            socket.setdefaulttimeout(original)


class KeywordTaskSpecTest(unittest.TestCase):
    def test_keyword_tasks_have_a_complete_argument_oracle(self):
        tasks = get_by_category("keyword_search")
        self.assertEqual(len(tasks), 3)
        for task in tasks:
            with self.subTest(task=task["id"]):
                self.assertEqual(task["expected_tools"], ["search_arxiv_papers"])
                self.assertEqual(len(task["expected_tool_args"]), 1)
                self.assertTrue(task["expected_tool_args"][0]["query"])


if __name__ == "__main__":
    unittest.main()
