import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

os.environ.setdefault("STORE_BACKEND", "memory")

import pymupdf

import tools.arxiv_tool  # noqa: F401
import tools.paper_figures_tool  # noqa: F401
from models.schemas import Paper, PdfAsset
from models.store import store, use_memory_store
from rl.env import MockArxivEnv
from rl.multiturn_env import AgenticArxivMultiTurnEnv
from tools.paper_figures_tool import MIN_FIGURE_PIXELS, extract_paper_figures
from tools.tool_registry import registry


PAPER = Paper(
    id="2601.00003v1",
    title="Figures for Testing",
    authors=["A. Tester"],
    summary="A fixture paper.",
    pdf_url="https://arxiv.org/pdf/2601.00003v1.pdf",
)


def _figure_pixmap(width: int = 200, height: int = 160) -> pymupdf.Pixmap:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height))
    pixmap.set_rect(pixmap.irect, (200, 60, 60))
    return pixmap


def _write_pdf(path: Path, *, with_figure: bool = True) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Figures for Testing", fontsize=14)
    if with_figure:
        page.insert_image(pymupdf.Rect(50, 80, 250, 240), pixmap=_figure_pixmap())
        page.insert_text((50, 260), "Figure 1: A deterministic test figure.")
    page.insert_text((50, 290), "1 Method")
    page.insert_text((50, 310), "The method is described here.")
    doc.save(path)
    doc.close()


class PaperFiguresToolTest(unittest.TestCase):
    def setUp(self):
        use_memory_store(reset=True)
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.figure_root = self.tmp_path / "figures"

        # ``settings`` is a frozen dataclass, so point the tool at a scratch
        # directory by swapping its module-level reference instead of mutating
        # process-wide config.
        self._settings_patch = mock.patch(
            "tools.paper_figures_tool.settings",
            mock.Mock(figures_path=str(self.figure_root)),
        )
        self._settings_patch.start()

        self.pdf_path = self.tmp_path / "paper.pdf"
        _write_pdf(self.pdf_path)
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
        self._settings_patch.stop()
        self.tmp.cleanup()

    def test_registered_in_tool_registry(self):
        tool = registry.get_tool("extract_paper_figures")
        self.assertIsNotNone(tool)
        self.assertIn("ref", tool["parameters"]["properties"])

    def test_extracts_figure_files_with_captions(self):
        result = extract_paper_figures(session_id="s", ref=1)

        self.assertEqual(result["paper_id"], PAPER.id)
        self.assertEqual(result["count"], 1)
        figure = result["figures"][0]
        self.assertEqual(figure["figure_no"], 1)
        self.assertEqual(figure["page"], 1)
        self.assertIn("Figure 1", figure["caption"])
        self.assertTrue(Path(figure["path"]).is_file())
        self.assertGreater(Path(figure["path"]).stat().st_size, 0)
        self.assertGreaterEqual(figure["width"], MIN_FIGURE_PIXELS)

    def test_repeated_extraction_is_deterministic(self):
        first = extract_paper_figures(session_id="s", ref=1)
        second = extract_paper_figures(session_id="s", ref=1)

        self.assertEqual(first["count"], second["count"])
        self.assertEqual(
            [f["path"] for f in first["figures"]],
            [f["path"] for f in second["figures"]],
        )

    def test_paper_without_figures_reports_zero_instead_of_raising(self):
        bare = self.tmp_path / "bare.pdf"
        _write_pdf(bare, with_figure=False)
        store.upsert_pdf_asset(
            PdfAsset(
                paper_id=PAPER.id,
                local_path=str(bare),
                status="READY",
                size_bytes=bare.stat().st_size,
            )
        )

        result = extract_paper_figures(session_id="s", ref=1)

        self.assertEqual(result["count"], 0)
        self.assertEqual(result["figures"], [])

    def test_invalid_ref_fails(self):
        with self.assertRaisesRegex(ValueError, "Paper not found"):
            extract_paper_figures(session_id="s", ref=99)

    def test_pdf_must_be_downloaded(self):
        store.reset()
        store.set_last_papers("s", [PAPER])

        with self.assertRaisesRegex(ValueError, "PDF is not downloaded"):
            extract_paper_figures(session_id="s", ref=1)


class PaperFiguresSnapshotTest(unittest.TestCase):
    def setUp(self):
        use_memory_store(reset=True)
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

        self._settings_patch = mock.patch(
            "tools.paper_figures_tool.settings",
            mock.Mock(figures_path=str(self.tmp_path / "figures")),
        )
        self._settings_patch.start()

        self.pdf_path = self.tmp_path / "paper.pdf"
        _write_pdf(self.pdf_path)
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
        self._settings_patch.stop()
        self.tmp.cleanup()

    def test_snapshot_replay_is_offline_and_keyed_by_paper(self):
        snapshot_path = self.tmp_path / "figures_snapshot.json"
        record_env = MockArxivEnv(
            snapshot_path=snapshot_path,
            mode="record",
            snapshot_tools={"extract_paper_figures"},
        )
        original = record_env.execute_tool(
            "extract_paper_figures", {"session_id": "s", "ref": 1}
        )
        record_env.save_snapshot()

        # Replay with paper identity but no PDF asset: a hit proves replay
        # neither parses a file nor touches the network.
        use_memory_store(reset=True)
        store.set_last_papers("other-session", [PAPER])

        replay_env = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")
        replayed = replay_env.execute_tool(
            "extract_paper_figures", {"session_id": "other-session", "ref": 1}
        )

        self.assertEqual(original, replayed)
        self.assertEqual(replay_env.stats["real_calls"], 0)
        self.assertEqual(replay_env.stats["hit"], 1)

        keys = list(
            json.loads(snapshot_path.read_text(encoding="utf-8"))["extract_paper_figures"]
        )
        self.assertIn(PAPER.id, keys[0])
        self.assertNotIn('"ref"', keys[0])
        self.assertNotIn("session_id", keys[0])

    def test_multiturn_extraction_hits_snapshot(self):
        import hashlib

        snapshot_path = self.tmp_path / "multiturn_figures.json"
        record_env = MockArxivEnv(
            snapshot_path=snapshot_path,
            mode="record",
            snapshot_tools={"extract_paper_figures"},
        )
        record_env.execute_tool(
            "extract_paper_figures", {"session_id": "s", "ref": 1}
        )
        record_env.save_snapshot()

        paper_dict = (
            PAPER.model_dump(mode="json")
            if hasattr(PAPER, "model_dump")
            else PAPER.dict()
        )
        search_key = json.dumps(
            {"days": 7, "query_sha256": hashlib.sha256(b"agent").hexdigest()},
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
        snapshot_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

        env = AgenticArxivMultiTurnEnv(snapshot_path=snapshot_path)
        env.reset(task_id="figures")
        env.search_arxiv_papers("agent", max_results=5, days=7)
        env.download_arxiv_pdf(ref=1)

        result = env.extract_paper_figures(ref=1)

        self.assertEqual(result["paper_id"], PAPER.id)
        self.assertEqual(result["count"], 1)
        self.assertEqual(env.backend.stats["real_calls"], 0)


if __name__ == "__main__":
    unittest.main()
