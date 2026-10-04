import json
import os
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("STORE_BACKEND", "memory")

import pymupdf

from benchmark.metrics import _match_arg_value
from models.schemas import Paper, TranslateAsset
from models.store import store, use_memory_store
from rl.build_snapshot import _snapshot_translated_content
from rl.env import MockArxivEnv
from rl.grpo_reward import _dispatch_environment_tool
from rl.multiturn_env import AgenticArxivMultiTurnEnv
from rl.reward import RewardCalculator
from tools.tool_registry import registry
from tools.translated_content_tool import get_translated_content


PAPER = Paper(
    id="2601.00002v1",
    title="Deterministic Translation Reading",
    authors=["A. Tester"],
    summary="A fixture paper.",
    pdf_url="https://arxiv.org/pdf/2601.00002v1.pdf",
)


class TranslatedContentToolTest(unittest.TestCase):
    def setUp(self):
        use_memory_store(reset=True)

        self.tmp = tempfile.TemporaryDirectory()
        self.mono_path = Path(self.tmp.name) / f"{PAPER.id}-mono.pdf"

        # The reader is language-agnostic; ASCII keeps the fixture free of
        # CJK font requirements.  Chinese text is covered by the replay tests.
        doc = pymupdf.open()
        for text in ("Translated page one.", "Translated page two."):
            doc.new_page().insert_text((50, 50), text, fontsize=11)
        doc.save(self.mono_path)
        doc.close()

        store.set_last_papers("s", [PAPER])
        self._set_translation("READY")

    def tearDown(self):
        self.tmp.cleanup()

    def _set_translation(self, status):
        store.upsert_translate_asset(
            TranslateAsset(
                paper_id=PAPER.id,
                output_mono_path=str(self.mono_path),
                status=status,
            )
        )

    def test_reads_one_page_of_the_translation(self):
        first = get_translated_content(session_id="s", ref=1)

        self.assertEqual(
            set(first),
            {"paper_id", "title", "page", "total_pages", "content"},
        )
        self.assertEqual(first["paper_id"], PAPER.id)
        self.assertEqual(first["page"], 1)
        self.assertEqual(first["total_pages"], 2)
        self.assertIn("page one", first["content"])
        self.assertNotIn("page two", first["content"])

        second = get_translated_content(session_id="s", ref=1, page=2)

        self.assertEqual(second["page"], 2)
        self.assertIn("page two", second["content"])

    def test_requires_ready_translation(self):
        self._set_translation("TRANSLATING")

        with self.assertRaisesRegex(ValueError, "call translate_arxiv_pdf first"):
            get_translated_content(session_id="s", ref=1)

        use_memory_store(reset=True)
        store.set_last_papers("s", [PAPER])

        with self.assertRaisesRegex(ValueError, "call translate_arxiv_pdf first"):
            get_translated_content(session_id="s", ref=1)

    def test_missing_translated_file_asks_for_retranslation(self):
        self.mono_path.unlink()

        with self.assertRaisesRegex(ValueError, "file is missing; call translate_arxiv_pdf again"):
            get_translated_content(session_id="s", ref=1)

    def test_rejects_invalid_pages(self):
        for page in (0, True):
            with self.subTest(page=page):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    get_translated_content(session_id="s", ref=1, page=page)

        with self.assertRaisesRegex(ValueError, "between 1 and 2"):
            get_translated_content(session_id="s", ref=1, page=3)

    def test_snapshot_replay_is_offline_and_deterministic(self):
        snapshot_path = Path(self.tmp.name) / "snapshot.json"
        record_env = MockArxivEnv(
            snapshot_path=snapshot_path,
            mode="record",
            snapshot_tools={"get_translated_content"},
        )
        first = record_env.execute_tool(
            "get_translated_content", {"session_id": "s", "ref": 1, "page": 2}
        )
        record_env.save_snapshot()

        # A fresh store with the paper but no TranslateAsset: success proves
        # replay neither opens the PDF nor needs a finished translation.
        use_memory_store(reset=True)
        store.set_last_papers("other-session", [PAPER])
        replay_env = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")

        second = replay_env.execute_tool(
            "get_translated_content",
            {"session_id": "other-session", "ref": 1, "page": 2},
        )

        self.assertEqual(first, second)
        self.assertEqual(replay_env.stats["real_calls"], 0)
        self.assertEqual(replay_env.stats["hit"], 1)

        keys = list(json.loads(snapshot_path.read_text(encoding="utf-8"))[
            "get_translated_content"
        ])
        self.assertEqual(keys, [json.dumps({"page": 2, "paper_id": PAPER.id})])

        # A malformed page is rejected before the lookup, not reported as a
        # missing snapshot entry.
        with self.assertRaisesRegex(ValueError, "positive integer"):
            replay_env.execute_tool(
                "get_translated_content",
                {"session_id": "other-session", "ref": 1, "page": 0},
            )
        self.assertEqual(replay_env.stats["miss"], 0)

        # A page past the recorded translation's end is the live tool's range
        # error, not a snapshot gap.
        with self.assertRaisesRegex(ValueError, "between 1 and 2"):
            replay_env.execute_tool(
                "get_translated_content",
                {"session_id": "other-session", "ref": 1, "page": 3},
            )
        self.assertEqual(replay_env.stats["miss"], 0)

        # An in-range page that was simply not recorded is still a snapshot gap.
        with self.assertRaisesRegex(KeyError, "快照缺失"):
            replay_env.execute_tool(
                "get_translated_content",
                {"session_id": "other-session", "ref": 1, "page": 1},
            )

    def test_multiturn_replay_returns_translated_text(self):
        paper_dict = PAPER.model_dump(mode="json")
        search_key = json.dumps(
            {"days": 7, "query_sha256": sha256(b"agent").hexdigest()},
            sort_keys=True,
        )
        page_key = json.dumps({"paper_id": PAPER.id, "page": 1}, sort_keys=True)
        recorded = {
            "paper_id": PAPER.id,
            "title": PAPER.title,
            "page": 1,
            "total_pages": 1,
            "content": "确定性的译文：第一页。",
        }
        snapshot_path = Path(self.tmp.name) / "multiturn_snapshot.json"
        snapshot_path.write_text(
            json.dumps(
                {
                    "search_arxiv_papers": {
                        search_key: {
                            "args": {"query": "agent", "max_results": 5, "days": 7},
                            "result": [paper_dict],
                        }
                    },
                    "get_translated_content": {
                        page_key: {"args": {"ref": 1, "page": 1}, "result": recorded}
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        original_get_tool = registry.get_tool

        def fake_search(query: str, max_results: int = 10, days=None):
            raise AssertionError("snapshot replay unexpectedly executed search")

        def lookup(name):
            if name == "search_arxiv_papers":
                return {"func": fake_search}
            return original_get_tool(name)

        env = AgenticArxivMultiTurnEnv(snapshot_path=snapshot_path)
        env.reset(task_id="translated")
        with patch.object(registry, "get_tool", side_effect=lookup):
            env.search_arxiv_papers("agent", max_results=5, days=7)
            env.download_arxiv_pdf(ref=1)
            env.translate_arxiv_pdf(ref=1)
            # Go through the GRPO dispatcher so its allow-list is covered too.
            result = _dispatch_environment_tool(
                env, "get_translated_content", {"ref": 1}
            )

        self.assertEqual(result, recorded)
        self.assertEqual(env.backend.stats["real_calls"], 0)

    def test_result_quality_grades_the_translated_text(self):
        task = {
            "id": "read_translation",
            "expected_tools": ["get_translated_content"],
            "expected_tool_args": [{"ref": 1, "page": 1}],
        }
        action = '{"name":"get_translated_content","args":{"ref":1,"page":1}}'
        page = {"paper_id": PAPER.id, "page": 1, "total_pages": 9}
        cases = (
            # Failure words inside the translation are text, not failures.
            (str({**page, "content": "当命令失败时，代理无法解析输出；工具执行失败: 超时。"}), 1.0, 0),
            # A payload without text is not useful work; unknown tool names
            # would score 1.0 here, so this pins the reading-tool rule.
            (str(page), 0.0, 0),
            # A framework failure still starts the observation and is caught.
            ("工具执行失败: Translated PDF is not ready; call translate_arxiv_pdf first.", -1.0, 1),
        )
        for observation, quality, failures in cases:
            with self.subTest(observation=observation):
                breakdown, metrics = RewardCalculator().compute_reward_breakdown(task, {
                    "history": [
                        {"action": action, "observation": observation},
                        {"action": "FINISH", "observation": "任务完成"},
                    ],
                    "iteration_count": 2,
                })
                self.assertEqual(breakdown.result_quality, quality)
                self.assertEqual(metrics.parse_failures, 0)
                self.assertEqual(metrics.tool_exec_failures, failures)

    def test_argument_match_treats_missing_page_as_first_page(self):
        for predicted, expected, matches in (
            (None, 1, True),
            (1, None, True),
            (None, 2, False),
            (2, 1, False),
        ):
            with self.subTest(predicted=predicted, expected=expected):
                self.assertEqual(_match_arg_value(predicted, expected, "page"), matches)

    def test_build_snapshot_records_every_translated_page(self):
        class RecordingEnv:
            snapshot = {
                "search_arxiv_papers": {
                    "pool": {"result": [PAPER.model_dump(mode="json")]}
                }
            }
            calls = []

            def execute_tool(self, name, args):
                self.calls.append((name, args.get("page")))
                return {"total_pages": 3} if name == "get_translated_content" else {}

        env = RecordingEnv()

        self.assertEqual(_snapshot_translated_content(env, max_ref=1), (1, 0))
        self.assertEqual(
            env.calls,
            [
                ("translate_arxiv_pdf", None),
                ("get_translated_content", 1),
                ("get_translated_content", 2),
                ("get_translated_content", 3),
            ],
        )


if __name__ == "__main__":
    unittest.main()
