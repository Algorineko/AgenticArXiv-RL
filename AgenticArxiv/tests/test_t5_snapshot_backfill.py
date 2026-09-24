"""T5 caption 补录和回放回归测试，无需 PDF 或模型。"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from models.schemas import Paper
from models.store import store, use_memory_store
from rl.backfill_figure_analysis import backfill_entries, backfill_snapshot
from rl.env import MockArxivEnv
from tools.figure_analysis_tool import FIGURE_QUESTIONS


FIXTURE = Path(__file__).parent / "fixtures" / "t4_snapshot_for_t5.json"
PAPER_ID = "2601.00004v1"


class T5SnapshotBackfillTest(unittest.TestCase):
    def setUp(self):
        use_memory_store(reset=True)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.snapshot_path = Path(self.tmpdir.name) / "snapshot.json"
        shutil.copyfile(FIXTURE, self.snapshot_path)

    def test_backfill_replays_all_questions_without_image_files(self):
        with mock.patch.dict(os.environ, {"FIGURE_ANALYSIS_BACKEND": "extractive"}):
            stats = backfill_snapshot(self.snapshot_path)

        self.assertEqual((stats.figures, stats.added, stats.existing), (1, 3, 0))
        payload = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["unrelated_tool"], {"kept": {"result": "unchanged"}})
        self.assertEqual(len(payload["analyze_figure"]), len(FIGURE_QUESTIONS))

        store.set_last_papers("replay", [Paper(
            id=PAPER_ID,
            title="Figures for Analysis",
            authors=["A. Tester"],
            summary="A fixture paper.",
            pdf_url=f"https://arxiv.org/pdf/{PAPER_ID}.pdf",
        )])
        env = MockArxivEnv(snapshot_path=self.snapshot_path, mode="replay")
        for question in FIGURE_QUESTIONS:
            with self.subTest(question=question):
                observation = env.execute_tool("analyze_figure", {
                    "session_id": "replay", "ref": 1,
                    "figure_no": 1, "question": question,
                })
                self.assertEqual(observation["paper_id"], PAPER_ID)
                self.assertEqual(observation["question"], question)
                self.assertEqual(observation["backend"], "extractive")
                self.assertTrue(observation["answer"])
        self.assertEqual(env.stats["real_calls"], 0)
        self.assertEqual(env.stats["hit"], len(FIGURE_QUESTIONS))

    def test_existing_results_survive_and_second_run_is_byte_identical(self):
        with mock.patch.dict(os.environ, {"FIGURE_ANALYSIS_BACKEND": "extractive"}):
            backfill_snapshot(self.snapshot_path)
            payload = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            key = next(key for key in payload["analyze_figure"] if '"trend"' in key)
            payload["analyze_figure"][key]["result"]["answer"] = "existing answer"
            self.snapshot_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            before = self.snapshot_path.read_bytes()
            stats = backfill_snapshot(self.snapshot_path)
            after = self.snapshot_path.read_bytes()

        self.assertEqual((stats.added, stats.existing), (0, 3))
        self.assertEqual(before, after)

    def test_invalid_t4_record_does_not_rewrite_snapshot(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        record = next(iter(payload["extract_paper_figures"].values()))
        record["result"]["count"] = 2
        self.snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
        before = self.snapshot_path.read_bytes()

        with mock.patch.dict(os.environ, {"FIGURE_ANALYSIS_BACKEND": "extractive"}):
            with self.assertRaisesRegex(ValueError, "figure count"):
                backfill_snapshot(self.snapshot_path)
        self.assertEqual(self.snapshot_path.read_bytes(), before)

    def test_missing_t4_data_and_vlm_mode_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "extract_paper_figures"):
            backfill_entries({})
        with mock.patch.dict(os.environ, {"FIGURE_ANALYSIS_BACKEND": "vlm"}):
            with self.assertRaisesRegex(ValueError, "extractive only"):
                backfill_snapshot(self.snapshot_path)


if __name__ == "__main__":
    unittest.main()
