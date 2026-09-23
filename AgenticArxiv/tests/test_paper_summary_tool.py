import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("STORE_BACKEND", "memory")

import pymupdf

import tools.arxiv_tool  # noqa: F401
import tools.paper_summary_tool  # noqa: F401
from models.schemas import Paper, PdfAsset
from models.store import store, use_memory_store
from rl.env import MockArxivEnv
from rl.multiturn_env import AgenticArxivMultiTurnEnv
from tools.paper_summary_tool import (
    MAX_WORDS_BUCKETS,
    SUMMARY_STYLES,
    bucket_max_words,
    normalize_style,
    summarize_paper,
)
from tools.tool_registry import registry


PAPER = Paper(
    id="2601.00002v1",
    title="Deterministic Summarisation for Testing",
    authors=["A. Tester"],
    summary="A fixture paper.",
    pdf_url="https://arxiv.org/pdf/2601.00002v1.pdf",
)

_PDF_BODY = (
    "Deterministic Summarisation for Testing\n\n"
    "Abstract\n"
    "This paper studies deterministic summarisation of research papers. "
    "Summaries are extracted from text without any sampling. "
    "The same paper always produces the same summary text.\n\n"
    "1 Introduction\n"
    "Summarisation is normally produced by a language model.\n\n"
    "2 Method\n"
    "We select whole sentences from each section in reading order. "
    "The selection never calls an LLM and never uses randomness. "
    "Budgets are enforced by counting words.\n\n"
    "3 Results\n"
    "Repeated runs returned byte-identical summaries. "
    "Section coverage improved when the abstract was preserved.\n\n"
    "4 Conclusion\n"
    "Deterministic extraction keeps offline evaluation reproducible.\n\n"
    "References\n"
    "[1] Example."
)


class PaperSummaryToolTest(unittest.TestCase):
    def setUp(self):
        use_memory_store(reset=True)

        self.tmp = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self.tmp.name) / "paper.pdf"

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 550, 780),
            _PDF_BODY,
            fontsize=11,
        )
        doc.save(self.pdf_path)
        doc.close()

        store.set_last_papers("s", [PAPER])
        store.upsert_pdf_asset(
            PdfAsset(
                paper_id=PAPER.id,
                pdf_url=PAPER.pdf_url or "",
                local_path=str(self.pdf_path),
                status="READY",
                size_bytes=self.pdf_path.stat().st_size,
                downloaded_at=datetime(2000, 1, 1),
            )
        )

    def tearDown(self):
        self.tmp.cleanup()

    # ---------- argument contract ----------

    def test_registered_in_tool_registry(self):
        tool = registry.get_tool("summarize_paper")

        self.assertIsNotNone(tool)
        self.assertEqual(
            set(tool["parameters"]["properties"]) >= {"ref", "style", "max_words"},
            True,
        )

    def test_style_normalisation(self):
        self.assertEqual(normalize_style(None), "tldr")
        self.assertEqual(normalize_style("  Structured "), "structured")
        with self.assertRaisesRegex(ValueError, "style must be one of"):
            normalize_style("poem")

    def test_max_words_buckets_up(self):
        self.assertEqual(bucket_max_words(None), 120)
        self.assertEqual(bucket_max_words(1), 60)
        self.assertEqual(bucket_max_words(60), 60)
        self.assertEqual(bucket_max_words(61), 120)
        self.assertEqual(bucket_max_words(500), MAX_WORDS_BUCKETS[-1])

    def test_invalid_max_words_is_a_tool_error(self):
        for bad in (0, -5, "many"):
            with self.subTest(max_words=bad):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    bucket_max_words(bad)

    # ---------- summarisation ----------

    def test_tldr_stays_inside_budget(self):
        result = summarize_paper(session_id="s", ref=1, style="tldr", max_words=60)

        self.assertEqual(result["paper_id"], PAPER.id)
        self.assertEqual(result["style"], "tldr")
        self.assertEqual(result["max_words"], 60)
        self.assertLessEqual(len(result["summary"].split()), 60)
        self.assertIn("deterministic summarisation", result["summary"].lower())
        self.assertEqual(result["sections_used"], ["abstract"])
        self.assertEqual(result["backend"], "extractive")

    def test_structured_covers_several_sections(self):
        result = summarize_paper(session_id="s", ref=1, style="structured", max_words=250)

        self.assertEqual(result["sections_used"], ["abstract", "method", "result", "conclusion"])
        for label in ("[abstract]", "[method]", "[result]", "[conclusion]"):
            self.assertIn(label, result["summary"])
        self.assertLessEqual(len(result["summary"].split()), 250)

    def test_bullet_style_emits_bullets(self):
        result = summarize_paper(session_id="s", ref=1, style="bullet", max_words=250)

        lines = [line for line in result["summary"].splitlines() if line.strip()]
        self.assertTrue(lines)
        self.assertTrue(all(line.startswith("- ") for line in lines))
        self.assertLessEqual(len(result["summary"].split()), 250)

    def test_summaries_are_deterministic(self):
        for style in SUMMARY_STYLES:
            with self.subTest(style=style):
                first = summarize_paper(session_id="s", ref=1, style=style)
                second = summarize_paper(session_id="s", ref=1, style=style)
                self.assertEqual(first["summary"], second["summary"])

    def test_smaller_budget_is_not_longer(self):
        short = summarize_paper(session_id="s", ref=1, style="tldr", max_words=60)
        long = summarize_paper(session_id="s", ref=1, style="tldr", max_words=250)

        self.assertLessEqual(
            len(short["summary"].split()),
            len(long["summary"].split()),
        )

    def test_ref_resolves_like_the_other_paper_tools(self):
        by_id = summarize_paper(session_id="s", ref=PAPER.id)
        by_title = summarize_paper(session_id="s", ref="Deterministic Summarisation")
        self.assertEqual(by_id["paper_id"], PAPER.id)
        self.assertEqual(by_title["paper_id"], PAPER.id)

    def test_invalid_ref_fails(self):
        with self.assertRaisesRegex(ValueError, "Paper not found"):
            summarize_paper(session_id="s", ref=99)

    def test_pdf_must_be_downloaded(self):
        store.reset()
        store.set_last_papers("s", [PAPER])

        with self.assertRaisesRegex(ValueError, "PDF is not downloaded"):
            summarize_paper(session_id="s", ref=1)

    def test_unknown_style_fails_before_reading_the_pdf(self):
        store.reset()
        store.set_last_papers("s", [PAPER])

        with self.assertRaisesRegex(ValueError, "style must be one of"):
            summarize_paper(session_id="s", ref=1, style="haiku")

    def test_paper_without_usable_sections_fails(self):
        path = Path(self.tmp.name) / "bare.pdf"

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), "1 Introduction\nOnly an introduction exists here.")
        doc.save(path)
        doc.close()

        store.upsert_pdf_asset(
            PdfAsset(
                paper_id=PAPER.id,
                local_path=str(path),
                status="READY",
                size_bytes=path.stat().st_size,
            )
        )

        with self.assertRaises(ValueError):
            summarize_paper(session_id="s", ref=1, style="tldr")

    # ---------- local-model backend ----------

    def test_local_model_backend_requires_a_model_path(self):
        with patch.dict(os.environ, {"SUMMARY_BACKEND": "local_model"}):
            os.environ.pop("SUMMARY_MODEL_PATH", None)
            with self.assertRaisesRegex(RuntimeError, "SUMMARY_MODEL_PATH"):
                summarize_paper(session_id="s", ref=1)

    def test_unknown_backend_is_rejected(self):
        with patch.dict(os.environ, {"SUMMARY_BACKEND": "magic"}):
            with self.assertRaisesRegex(RuntimeError, "SUMMARY_BACKEND must be"):
                summarize_paper(session_id="s", ref=1)

    def test_local_model_backend_uses_greedy_decoding(self):
        import torch

        captured = {}

        class FakeBatch(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return messages[0]["content"]

            def __call__(self, text, return_tensors="pt"):
                return FakeBatch(input_ids=torch.zeros((1, 2), dtype=torch.long))

            def decode(self, tokens, skip_special_tokens=True):
                return "A generated summary."

        class FakeModel:
            device = "cpu"

            def generate(self, **kwargs):
                captured.update(kwargs)
                return torch.zeros((1, 4), dtype=torch.long)

        with patch.dict(
            os.environ,
            {"SUMMARY_BACKEND": "local_model", "SUMMARY_MODEL_PATH": "/tmp/fake"},
        ), patch(
            "tools.paper_summary_tool._load_local_model",
            return_value=(FakeTokenizer(), FakeModel()),
        ):
            result = summarize_paper(session_id="s", ref=1, style="tldr", max_words=60)

        self.assertEqual(result["summary"], "A generated summary.")
        self.assertEqual(result["backend"], "local_model")
        self.assertIs(captured["do_sample"], False)
        self.assertEqual(captured["num_beams"], 1)

    # ---------- snapshot replay ----------

    def _record_snapshot(self, path: Path) -> None:
        env = MockArxivEnv(
            snapshot_path=path,
            mode="record",
            snapshot_tools={"summarize_paper"},
        )
        for style in SUMMARY_STYLES:
            for max_words in MAX_WORDS_BUCKETS:
                env.execute_tool(
                    "summarize_paper",
                    {
                        "session_id": "s",
                        "ref": 1,
                        "style": style,
                        "max_words": max_words,
                    },
                )
        env.save_snapshot()

    def test_snapshot_replay_is_offline_and_keyed_by_paper(self):
        snapshot_path = Path(self.tmp.name) / "summary_snapshot.json"
        self._record_snapshot(snapshot_path)

        original = summarize_paper(session_id="s", ref=1, style="structured")

        # Replay with a fresh store that has paper identity but deliberately no
        # PDF asset: a hit proves replay neither parses a file nor hits network.
        use_memory_store(reset=True)
        store.set_last_papers("other-session", [PAPER])

        replay_env = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")
        replayed = replay_env.execute_tool(
            "summarize_paper",
            {
                "session_id": "other-session",
                "ref": 1,
                "style": "structured",
                "max_words": 250,
            },
        )

        self.assertEqual(original["summary"], replayed["summary"])
        self.assertEqual(replay_env.stats["real_calls"], 0)
        self.assertEqual(replay_env.stats["hit"], 1)

        keys = list(json.loads(snapshot_path.read_text(encoding="utf-8"))["summarize_paper"])
        self.assertIn(PAPER.id, keys[0])
        self.assertNotIn('"ref"', keys[0])
        self.assertNotIn("session_id", keys[0])

    def test_replay_clamps_the_budget_onto_the_recorded_grid(self):
        snapshot_path = Path(self.tmp.name) / "summary_snapshot.json"
        self._record_snapshot(snapshot_path)

        replay_env = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")
        result = replay_env.execute_tool(
            "summarize_paper",
            {
                "session_id": "s",
                "ref": 1,
                "style": "tldr",
                # A policy-chosen value between buckets still maps to the
                # recorded 120-word entry instead of missing the snapshot.
                "max_words": 100,
            },
        )

        self.assertEqual(result["max_words"], 120)
        self.assertEqual(replay_env.stats["hit"], 1)

    def test_replay_rejects_invalid_arguments_without_a_key_error(self):
        snapshot_path = Path(self.tmp.name) / "summary_snapshot.json"
        self._record_snapshot(snapshot_path)

        replay_env = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")

        with self.assertRaisesRegex(ValueError, "style must be one of"):
            replay_env.execute_tool(
                "summarize_paper",
                {"session_id": "s", "ref": 1, "style": "haiku"},
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            replay_env.execute_tool(
                "summarize_paper",
                {"session_id": "s", "ref": 1, "style": "tldr", "max_words": 0},
            )

        self.assertEqual(replay_env.stats["miss"], 0)

    def test_multiturn_summarize_hits_snapshot(self):
        snapshot_path = Path(self.tmp.name) / "multiturn_summary.json"
        self._record_snapshot(snapshot_path)

        paper_dict = (
            PAPER.model_dump(mode="json")
            if hasattr(PAPER, "model_dump")
            else PAPER.dict()
        )
        import hashlib

        search_key = json.dumps(
            {
                "days": 7,
                "query_sha256": hashlib.sha256(b"agent").hexdigest(),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        payload["search_arxiv_papers"] = {
            search_key: {
                "args": {"query": "agent", "max_results": 5, "days": 7},
                "result": [paper_dict],
            }
        }
        snapshot_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        env = AgenticArxivMultiTurnEnv(snapshot_path=snapshot_path)
        env.reset(task_id="summary")

        papers = env.search_arxiv_papers("agent", max_results=5, days=7)
        self.assertEqual(papers[0]["id"], PAPER.id)

        download = env.download_arxiv_pdf(ref=1)
        self.assertEqual(download["status"], "READY")

        result = env.summarize_paper(ref=1, style="tldr", max_words=120)

        self.assertEqual(result["paper_id"], PAPER.id)
        self.assertTrue(result["summary"])
        self.assertEqual(env.backend.stats["real_calls"], 0)


if __name__ == "__main__":
    unittest.main()
