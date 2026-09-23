import json
from hashlib import sha256
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("STORE_BACKEND", "memory")

import pymupdf

import tools.paper_content_tool  # noqa: F401
from models.schemas import Paper, PdfAsset
from models.store import store, use_memory_store
from rl.env import MockArxivEnv
from rl.multiturn_env import AgenticArxivMultiTurnEnv
from tools.paper_content_tool import _heading_name, get_paper_content
from tools.tool_registry import registry


PAPER = Paper(
    id="2601.00001v1",
    title="Deterministic Agents for Testing",
    authors=["A. Tester"],
    summary="A fixture paper.",
    pdf_url="https://arxiv.org/pdf/2601.00001v1.pdf",
)


class PaperContentToolTest(unittest.TestCase):
    def setUp(self):
        use_memory_store(reset=True)

        self.tmp = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self.tmp.name) / "paper.pdf"

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_textbox(
            pymupdf.Rect(50, 50, 550, 780),
            "Deterministic Agents for Testing\n\n"
            "Abstract\n"
            "This paper studies deterministic agent environments.\n\n"
            "1 Introduction\n"
            "Introductory material.\n\n"
            "2 Method\n"
            "We use a deterministic parser and a replayable tool environment.\n"
            "The method never calls an LLM.\n\n"
            "3 Results\n"
            "The replay result is identical across repeated executions.\n\n"
            "4 Conclusion\n"
            "Deterministic extraction makes offline evaluation reproducible.\n\n"
            "References\n"
            "[1] Example.",
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

    def test_heading_name_preserves_plain_headings(self):
        expected = {
            "Introduction": "_other",
            "Conclusion": "conclusion",
            "I. Introduction": "_other",
            "IV Conclusion": "conclusion",
            "1 Introduction": "_other",
            "4. Conclusion": "conclusion",
        }

        for heading, canonical in expected.items():
            with self.subTest(heading=heading):
                self.assertEqual(_heading_name(heading), canonical)

    def test_registered_in_tool_registry(self):
        tool = registry.get_tool("get_paper_content")

        self.assertIsNotNone(tool)
        self.assertIn(
            "section",
            tool["parameters"]["properties"],
        )

    def test_section_none_returns_title_and_abstract(self):
        result = get_paper_content(
            session_id="s",
            ref=1,
        )

        self.assertEqual(result["paper_id"], PAPER.id)
        self.assertIsNone(result["section"])
        self.assertTrue(result["content"].startswith(PAPER.title))
        self.assertIn("Abstract", result["content"])
        self.assertIn(
            "studies deterministic agent environments",
            result["content"],
        )
        self.assertNotIn(
            "deterministic parser",
            result["content"],
        )
        self.assertNotIn(
            "References",
            result["content"],
        )
        self.assertNotIn(
            "session_id",
            result,
        )

    def test_extracts_supported_sections(self):
        expected = {
            "abstract": "studies deterministic agent environments",
            "method": "never calls an LLM",
            "result": "identical across repeated executions",
            "conclusion": "offline evaluation reproducible",
        }

        for section, phrase in expected.items():
            with self.subTest(section=section):
                result = get_paper_content(
                    session_id="s",
                    ref=1,
                    section=section,
                )

                self.assertEqual(
                    result["section"],
                    section,
                )
                self.assertIn(
                    phrase,
                    result["content"],
                )

    def test_invalid_ref_fails(self):
        with self.assertRaisesRegex(
            ValueError,
            "Paper not found",
        ):
            get_paper_content(
                session_id="s",
                ref=99,
            )

    def test_pdf_must_be_downloaded(self):
        store.reset()
        store.set_last_papers("s", [PAPER])

        with self.assertRaisesRegex(
            ValueError,
            "PDF is not downloaded",
        ):
            get_paper_content(
                session_id="s",
                ref=1,
            )

    def test_invalid_section_fails(self):
        with self.assertRaisesRegex(
            ValueError,
            "section must be one of",
        ):
            get_paper_content(
                session_id="s",
                ref=1,
                section="introduction",
            )

    def test_missing_section_fails(self):
        path = Path(self.tmp.name) / "missing.pdf"

        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text(
            (50, 50),
            "Abstract\n"
            "Only an abstract is present.\n"
            "References\n"
            "[1] X",
        )
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

        with self.assertRaisesRegex(
            ValueError,
            "was not found",
        ):
            get_paper_content(
                session_id="s",
                ref=1,
                section="conclusion",
            )

    def _pdf_with_text(self, name: str, body: str) -> Path:
        path = Path(self.tmp.name) / name
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), body, fontsize=11)
        doc.save(path)
        doc.close()
        return path

    def _point_at(self, path: Path) -> None:
        store.upsert_pdf_asset(
            PdfAsset(
                paper_id=PAPER.id,
                local_path=str(path),
                status="READY",
                size_bytes=path.stat().st_size,
            )
        )

    def test_abstract_is_recovered_when_the_template_has_no_heading(self):
        """CVPR 一类模板不排 "Abstract" 标题，纯靠标题行会整篇读不出摘要。"""
        path = self._pdf_with_text(
            "noheading.pdf",
            "A Study of Headingless Abstracts\n"
            "A. Author, B. Author\n"
            "Institute of Examples, Somewhere\n\n"
            "This paper studies what happens when a template typesets the "
            "abstract without any label at all. We show that the longest prose "
            "block before the first section heading is exactly the abstract "
            "paragraph, because title and author lines are short by "
            "construction and the abstract is not.\n\n"
            "1 Introduction\n"
            "Introductory material.",
        )
        self._point_at(path)

        result = get_paper_content(session_id="s", ref=1)

        self.assertIn("studies what happens when a template", result["content"])
        self.assertIn("longest prose block", result["content"])
        self.assertNotIn("Introductory material", result["content"])
        self.assertNotIn("Institute of Examples", result["content"])

    def test_inline_abstract_label_is_stripped(self):
        path = self._pdf_with_text(
            "inline.pdf",
            "A Study of Inline Labels\n"
            "A. Author\n\n"
            "Abstract—We propose a method that keeps the label glued to the "
            "text, which is what several conference templates actually do, and "
            "which defeats a heading-only parser completely.\n\n"
            "1 Introduction\n"
            "Introductory material.",
        )
        self._point_at(path)

        result = get_paper_content(session_id="s", ref=1)
        abstract = result["content"].split("Abstract", 1)[-1]

        self.assertIn("We propose a method", abstract)
        self.assertNotIn("—We propose", result["content"])

    def test_other_sections_keep_failing_without_a_heading(self):
        """兜底只给摘要：正文小节确实可能没有，显式要它就该报错而不是猜。"""
        path = self._pdf_with_text(
            "noconclusion.pdf",
            "A Study Without Sections\n"
            "A. Author\n\n"
            "This paper has a reasonably long abstract paragraph but no "
            "conclusion section at all, so asking for one must fail rather "
            "than fall back to whatever text happens to be longest.\n\n"
            "1 Introduction\n"
            "Introductory material.",
        )
        self._point_at(path)

        with self.assertRaisesRegex(ValueError, "was not found"):
            get_paper_content(session_id="s", ref=1, section="conclusion")

    def test_short_front_matter_does_not_become_an_abstract(self):
        """没有摘要、只有标题与作者时不能把作者行当成摘要交出去。"""
        path = self._pdf_with_text(
            "noprose.pdf",
            "Short Paper\n"
            "A. Author, B. Author\n"
            "Institute of Examples\n\n"
            "1 Introduction\n"
            "Introductory material.",
        )
        self._point_at(path)

        with self.assertRaisesRegex(ValueError, "was not found"):
            get_paper_content(session_id="s", ref=1)

    def test_snapshot_replay_is_offline_and_deterministic(self):
        record_path = Path(self.tmp.name) / "snapshot.json"

        record_env = MockArxivEnv(
            snapshot_path=record_path,
            mode="record",
            snapshot_tools={"get_paper_content"},
        )

        first = record_env.execute_tool(
            "get_paper_content",
            {
                "session_id": "s",
                "ref": 1,
                "section": "method",
            },
        )

        record_env.save_snapshot()

        # Replay with a fresh store that has paper identity but deliberately no
        # PDF asset: success proves replay does not parse a file or hit network.
        use_memory_store(reset=True)
        store.set_last_papers(
            "other-session",
            [PAPER],
        )

        replay_env = MockArxivEnv(
            snapshot_path=record_path,
            mode="replay",
        )

        second = replay_env.execute_tool(
            "get_paper_content",
            {
                "session_id": "other-session",
                "ref": 1,
                "section": "method",
            },
        )

        self.assertEqual(
            first["content"],
            second["content"],
        )
        self.assertEqual(
            replay_env.stats["real_calls"],
            0,
        )
        self.assertEqual(
            replay_env.stats["hit"],
            1,
        )

        payload = json.loads(
            record_path.read_text(encoding="utf-8")
        )
        keys = list(payload["get_paper_content"])

        self.assertIn(
            PAPER.id,
            keys[0],
        )
        self.assertNotIn(
            '"ref"',
            keys[0],
        )

    def _search_tool_lookup(self):
        original_get_tool = registry.get_tool

        def fake_search(
            query: str,
            max_results: int = 10,
            days=None,
        ):
            raise AssertionError(
                "snapshot replay unexpectedly executed search"
            )

        def lookup(name):
            if name == "search_arxiv_papers":
                return {
                    "func": fake_search,
                }

            return original_get_tool(name)

        return lookup

    def _write_multiturn_snapshot(
        self,
        path: Path,
    ) -> None:
        paper_dict = (
            PAPER.model_dump(mode="json")
            if hasattr(PAPER, "model_dump")
            else PAPER.dict()
        )

        search_key = json.dumps(
            {
                "days": 7,
                "query_sha256": sha256(b"agent").hexdigest(),
            },
            sort_keys=True,
            ensure_ascii=False,
        )

        content_key = json.dumps(
            {
                "paper_id": PAPER.id,
                "section": "method",
            },
            sort_keys=True,
            ensure_ascii=False,
        )

        payload = {
            "search_arxiv_papers": {
                search_key: {
                    "args": {
                        "query": "agent",
                        "max_results": 5,
                        "days": 7,
                    },
                    "result": [
                        paper_dict,
                    ],
                }
            },
            "get_paper_content": {
                content_key: {
                    "args": {
                        "ref": 1,
                        "section": "method",
                    },
                    "result": {
                        "paper_id": PAPER.id,
                        "title": PAPER.title,
                        "section": "method",
                        "content": (
                            "We use a deterministic parser and "
                            "a replayable tool environment."
                        ),
                    },
                }
            },
        }

        path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def test_multiturn_search_download_read_hits_snapshot(self):
        snapshot_path = (
            Path(self.tmp.name)
            / "multiturn_snapshot.json"
        )

        self._write_multiturn_snapshot(
            snapshot_path
        )

        env = AgenticArxivMultiTurnEnv(
            snapshot_path=snapshot_path
        )
        env.reset(
            task_id="integration"
        )

        with patch.object(
            registry,
            "get_tool",
            side_effect=self._search_tool_lookup(),
        ):
            papers = env.search_arxiv_papers(
                "agent",
                max_results=5,
                days=7,
            )

            self.assertEqual(
                papers[0]["id"],
                PAPER.id,
            )

            download = env.download_arxiv_pdf(
                ref=1
            )

            self.assertEqual(
                download["status"],
                "READY",
            )

            result = env.get_paper_content(
                ref=1,
                section="method",
            )

        self.assertEqual(
            result["paper_id"],
            PAPER.id,
        )
        self.assertIn(
            "deterministic parser",
            result["content"],
        )
        self.assertEqual(
            env.backend.stats["real_calls"],
            0,
        )
        self.assertEqual(
            env.backend.stats["hit"],
            2,
        )

    def test_cross_session_multiturn_replay_is_deterministic(self):
        snapshot_path = (
            Path(self.tmp.name)
            / "cross_session_snapshot.json"
        )

        self._write_multiturn_snapshot(
            snapshot_path
        )

        outputs = []
        session_ids = []

        for task_id in (
            "rollout-a",
            "rollout-b",
        ):
            env = AgenticArxivMultiTurnEnv(
                snapshot_path=snapshot_path
            )
            env.reset(
                task_id=task_id
            )

            session_ids.append(
                env.session_id
            )

            with patch.object(
                registry,
                "get_tool",
                side_effect=self._search_tool_lookup(),
            ):
                env.search_arxiv_papers(
                    "agent",
                    max_results=5,
                    days=7,
                )
                env.download_arxiv_pdf(
                    ref=1
                )

                outputs.append(
                    env.get_paper_content(
                        ref=1,
                        section="method",
                    )
                )

                self.assertEqual(
                    env.backend.stats["real_calls"],
                    0,
                )

        self.assertNotEqual(
            session_ids[0],
            session_ids[1],
        )
        self.assertEqual(
            outputs[0],
            outputs[1],
        )


if __name__ == "__main__":
    unittest.main()
