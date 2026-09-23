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
import tools.figure_analysis_tool  # noqa: F401
from models.schemas import Paper, PdfAsset
from models.store import store, use_memory_store
from rl.env import MockArxivEnv
from rl.multiturn_env import AgenticArxivMultiTurnEnv
from tools.figure_analysis_tool import (
    FIGURE_QUESTIONS,
    MAX_FIGURE_NO,
    analyze_figure,
    normalize_question,
    validate_figure_no,
)
from tools.tool_registry import registry


PAPER = Paper(
    id="2601.00004v1",
    title="Figures for Analysis",
    authors=["A. Tester"],
    summary="A fixture paper.",
    pdf_url="https://arxiv.org/pdf/2601.00004v1.pdf",
)

_CAPTION = (
    "Figure 1: Accuracy versus latency for the three configurations. "
    "The proposed method is faster than the baseline at equal accuracy. "
    "Accuracy increases as the batch size grows."
)


def _figure_pixmap(width: int = 200, height: int = 160) -> pymupdf.Pixmap:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height))
    pixmap.set_rect(pixmap.irect, (60, 120, 200))
    return pixmap


def _write_pdf(path: Path, *, with_figure: bool = True, caption: str = _CAPTION) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Figures for Analysis", fontsize=14)
    if with_figure:
        page.insert_image(pymupdf.Rect(50, 80, 250, 240), pixmap=_figure_pixmap())
        page.insert_text((50, 262), caption, fontsize=9)
    doc.save(path)
    doc.close()


class FigureAnalysisToolTest(unittest.TestCase):
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

    # ---------- argument contract ----------

    def test_registered_in_tool_registry(self):
        tool = registry.get_tool("analyze_figure")

        self.assertIsNotNone(tool)
        self.assertEqual(
            set(tool["parameters"]["properties"]) >= {"ref", "figure_no", "question"},
            True,
        )

    def test_question_normalisation(self):
        self.assertEqual(normalize_question(None), "describe")
        self.assertEqual(normalize_question(" Axes "), "axes")
        with self.assertRaisesRegex(ValueError, "question must be one of"):
            normalize_question("summarize it")

    def test_figure_no_bounds(self):
        self.assertEqual(validate_figure_no(1), 1)
        self.assertEqual(validate_figure_no(MAX_FIGURE_NO), MAX_FIGURE_NO)
        for bad in (0, -1, MAX_FIGURE_NO + 1, "first", True):
            with self.subTest(figure_no=bad):
                with self.assertRaisesRegex(ValueError, "figure_no"):
                    validate_figure_no(bad)

    # ---------- extractive backend ----------

    def test_describe_returns_the_caption(self):
        result = analyze_figure(session_id="s", ref=1, figure_no=1)

        self.assertEqual(result["paper_id"], PAPER.id)
        self.assertEqual(result["figure_no"], 1)
        self.assertEqual(result["question"], "describe")
        self.assertEqual(result["backend"], "extractive")
        self.assertIn("Accuracy versus latency", result["answer"])
        self.assertTrue(Path(result["image_path"]).is_file())

    def test_axes_and_trend_pick_the_relevant_sentences(self):
        axes = analyze_figure(session_id="s", ref=1, figure_no=1, question="axes")
        trend = analyze_figure(session_id="s", ref=1, figure_no=1, question="trend")

        self.assertIn("latency", axes["answer"])
        self.assertNotIn("batch size grows", axes["answer"])
        self.assertIn("faster than the baseline", trend["answer"])

    def test_unanswerable_question_says_so_instead_of_guessing(self):
        capture = self.tmp_path / "capture.pdf"
        _write_pdf(capture, caption="Figure 1: A photograph of the apparatus.")
        store.upsert_pdf_asset(
            PdfAsset(
                paper_id=PAPER.id,
                local_path=str(capture),
                status="READY",
                size_bytes=capture.stat().st_size,
            )
        )

        result = analyze_figure(session_id="s", ref=1, figure_no=1, question="trend")

        self.assertIn("does not state this", result["answer"])

    def test_repeated_analysis_is_deterministic(self):
        first = analyze_figure(session_id="s", ref=1, figure_no=1, question="trend")
        second = analyze_figure(session_id="s", ref=1, figure_no=1, question="trend")

        self.assertEqual(first["answer"], second["answer"])

    def test_out_of_range_figure_number_fails(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            analyze_figure(session_id="s", ref=1, figure_no=3)

    def test_paper_without_figures_fails_on_any_figure(self):
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

        with self.assertRaisesRegex(ValueError, "out of range"):
            analyze_figure(session_id="s", ref=1, figure_no=1)

    def test_invalid_ref_fails(self):
        with self.assertRaisesRegex(ValueError, "Paper not found"):
            analyze_figure(session_id="s", ref=99)

    def test_pdf_must_be_downloaded(self):
        store.reset()
        store.set_last_papers("s", [PAPER])

        with self.assertRaisesRegex(ValueError, "PDF is not downloaded"):
            analyze_figure(session_id="s", ref=1)

    # ---------- vlm backend ----------

    def test_vlm_backend_requires_a_model_path(self):
        with mock.patch.dict(os.environ, {"FIGURE_ANALYSIS_BACKEND": "vlm"}, clear=False):
            os.environ.pop("VLM_MODEL_PATH", None)
            with self.assertRaisesRegex(RuntimeError, "VLM_MODEL_PATH"):
                analyze_figure(session_id="s", ref=1)

    def test_unknown_backend_is_rejected(self):
        with mock.patch.dict(os.environ, {"FIGURE_ANALYSIS_BACKEND": "magic"}):
            with self.assertRaisesRegex(RuntimeError, "FIGURE_ANALYSIS_BACKEND must be"):
                analyze_figure(session_id="s", ref=1)

    def test_vlm_backend_uses_greedy_decoding(self):
        import torch

        captured = {}

        class FakeBatch(dict):
            def to(self, device):
                return self

        class FakeProcessor:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "rendered"

            def __call__(self, text, images=None, return_tensors="pt"):
                return FakeBatch(input_ids=torch.zeros((1, 2), dtype=torch.long))

            def batch_decode(self, tokens, skip_special_tokens=True):
                return ["The figure plots accuracy against latency."]

        class FakeModel:
            device = "cpu"

            def generate(self, **kwargs):
                captured.update(kwargs)
                return torch.zeros((1, 5), dtype=torch.long)

        with mock.patch.dict(
            os.environ,
            {"FIGURE_ANALYSIS_BACKEND": "vlm", "VLM_MODEL_PATH": "/tmp/fake-vlm"},
        ), mock.patch(
            "tools.figure_analysis_tool._load_vlm",
            return_value=(FakeProcessor(), FakeModel()),
        ):
            result = analyze_figure(session_id="s", ref=1, figure_no=1)

        self.assertEqual(result["backend"], "vlm")
        self.assertIn("accuracy against latency", result["answer"])
        self.assertIs(captured["do_sample"], False)

    def test_vlm_downscales_large_images(self):
        """实测：4404×2351 的原图喂给 VLM 要 ~27s，缩放后才是可用的环境步。"""
        import torch
        from PIL import Image as _Image

        captured = {}

        class FakeBatch(dict):
            def to(self, device):
                return self

        class FakeProcessor:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                captured["image_size"] = messages[0]["content"][0]["image"].size
                return "rendered"

            def __call__(self, text, images=None, return_tensors="pt"):
                return FakeBatch(input_ids=torch.zeros((1, 2), dtype=torch.long))

            def batch_decode(self, tokens, skip_special_tokens=True):
                return ["ok"]

        class FakeModel:
            device = "cpu"

            def generate(self, **kwargs):
                return torch.zeros((1, 3), dtype=torch.long)

        big = self.tmp_path / "big.pdf"
        doc = pymupdf.open()
        page = doc.new_page(width=2000, height=1200)
        page.insert_image(
            pymupdf.Rect(20, 20, 1980, 1180),
            pixmap=pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 2400, 1400)),
        )
        page.insert_text((20, 1195), _CAPTION, fontsize=8)
        doc.save(big)
        doc.close()
        store.upsert_pdf_asset(
            PdfAsset(
                paper_id=PAPER.id,
                local_path=str(big),
                status="READY",
                size_bytes=big.stat().st_size,
            )
        )

        from tools.figure_analysis_tool import VLM_MAX_IMAGE_SIDE

        with mock.patch.dict(
            os.environ,
            {"FIGURE_ANALYSIS_BACKEND": "vlm", "VLM_MODEL_PATH": "/tmp/fake-vlm"},
        ), mock.patch(
            "tools.figure_analysis_tool._load_vlm",
            return_value=(FakeProcessor(), FakeModel()),
        ):
            analyze_figure(session_id="s", ref=1, figure_no=1)

        del _Image  # 仅为让 PIL 被显式引用，避免静态检查误判
        self.assertLessEqual(max(captured["image_size"]), VLM_MAX_IMAGE_SIDE)


class FigureAnalysisSnapshotTest(unittest.TestCase):
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

    def _record(self, path: Path, tools=("analyze_figure",)) -> None:
        env = MockArxivEnv(
            snapshot_path=path,
            mode="record",
            snapshot_tools=set(tools),
        )
        if "extract_paper_figures" in tools:
            env.execute_tool(
                "extract_paper_figures", {"session_id": "s", "ref": 1}
            )
        for question in FIGURE_QUESTIONS:
            env.execute_tool(
                "analyze_figure",
                {"session_id": "s", "ref": 1, "figure_no": 1, "question": question},
            )
        env.save_snapshot()

    def test_replay_is_offline_and_keyed_by_paper_figure_and_question(self):
        snapshot_path = self.tmp_path / "analysis_snapshot.json"
        self._record(snapshot_path)
        original = analyze_figure(session_id="s", ref=1, figure_no=1, question="trend")

        # Replay with paper identity but no PDF asset: a hit proves replay
        # neither reads the image nor calls a model.
        use_memory_store(reset=True)
        store.set_last_papers("other-session", [PAPER])

        replay_env = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")
        replayed = replay_env.execute_tool(
            "analyze_figure",
            {
                "session_id": "other-session",
                "ref": 1,
                "figure_no": 1,
                "question": "trend",
            },
        )

        self.assertEqual(original["answer"], replayed["answer"])
        self.assertEqual(replay_env.stats["real_calls"], 0)
        self.assertEqual(replay_env.stats["hit"], 1)

        keys = list(json.loads(snapshot_path.read_text(encoding="utf-8"))["analyze_figure"])
        trend_keys = [k for k in keys if '"question": "trend"' in k]
        self.assertEqual(len(trend_keys), 1)
        self.assertIn(PAPER.id, trend_keys[0])
        self.assertNotIn('"ref"', trend_keys[0])
        self.assertNotIn("session_id", trend_keys[0])

    def test_different_questions_have_different_keys(self):
        snapshot_path = self.tmp_path / "analysis_snapshot.json"
        self._record(snapshot_path)

        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        keys = list(payload["analyze_figure"])
        self.assertEqual(len(keys), len(FIGURE_QUESTIONS))

    def test_replay_rejects_invalid_arguments_without_a_key_error(self):
        snapshot_path = self.tmp_path / "analysis_snapshot.json"
        self._record(snapshot_path)

        replay_env = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")

        with self.assertRaisesRegex(ValueError, "question must be one of"):
            replay_env.execute_tool(
                "analyze_figure",
                {"session_id": "s", "ref": 1, "figure_no": 1, "question": "why"},
            )
        with self.assertRaisesRegex(ValueError, "figure_no"):
            replay_env.execute_tool(
                "analyze_figure",
                {"session_id": "s", "ref": 1, "figure_no": 0, "question": "describe"},
            )

        self.assertEqual(replay_env.stats["miss"], 0)

    def test_multiturn_analysis_hits_snapshot(self):
        import hashlib

        snapshot_path = self.tmp_path / "multiturn_analysis.json"
        # 多轮路径要先抽图（T4）再分析（T5），两者都得在快照里
        self._record(snapshot_path, tools=("analyze_figure", "extract_paper_figures"))

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
        snapshot_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        env = AgenticArxivMultiTurnEnv(snapshot_path=snapshot_path)
        env.reset(task_id="analysis")
        env.search_arxiv_papers("agent", max_results=5, days=7)
        env.download_arxiv_pdf(ref=1)
        env.extract_paper_figures(ref=1)

        result = env.analyze_figure(ref=1, figure_no=1, question="describe")

        self.assertEqual(result["paper_id"], PAPER.id)
        self.assertTrue(result["answer"])
        self.assertEqual(env.backend.stats["real_calls"], 0)


if __name__ == "__main__":
    unittest.main()
