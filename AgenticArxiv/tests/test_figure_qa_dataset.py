#!/usr/bin/env python3
"""build_figure_qa_dataset 的目标抽取与切分测试（纯 CPU，无网络/模型）。"""

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT.parent))

from scripts.build_figure_qa_dataset import (  # noqa: E402
    build_rows,
    build_target,
    clean_text,
    describe_target,
    evidence_target,
    iter_usable_figures,
    split_papers,
)
from tools.figure_analysis_tool import _QUESTION_PROMPTS  # noqa: E402


class TargetRulesTest(unittest.TestCase):
    def test_describe_strips_figure_prefix_and_collapses_whitespace(self):
        caption = "Figure 2: A pipeline with\n  three stages."
        self.assertEqual(describe_target(caption), "A pipeline with three stages.")
        self.assertEqual(describe_target("Fig. 3  -  Bar chart."), "Bar chart.")

    def test_describe_without_prefix_kept(self):
        self.assertEqual(describe_target("Plain caption."), "Plain caption.")

    def test_evidence_joins_matching_sentences_and_strips_prefix(self):
        caption = "Figure 6: Accuracy (%) of successful trials. The method is unrelated."
        self.assertEqual(
            evidence_target(caption, "axes"),
            "Accuracy (%) of successful trials.",
        )

    def test_evidence_empty_without_hint(self):
        caption = "A photograph of a robot arm."
        self.assertEqual(evidence_target(caption, "axes"), "")
        self.assertEqual(evidence_target(caption, "trend"), "")

    def test_build_target_dispatch(self):
        caption = "Figure 1: Loss decreases over time."
        self.assertIn("Loss decreases", build_target(caption, "describe"))
        self.assertIn("decreases", build_target(caption, "trend"))

    def test_targets_respect_sentence_caps(self):
        three = "Figure 1: One. Two. Three."
        self.assertEqual(describe_target(three), "One. Two.")
        two_matches = "Figure 2: Accuracy (%) is reported. Accuracy grows with depth."
        self.assertEqual(evidence_target(two_matches, "axes"), "Accuracy (%) is reported.")

    def test_clean_text(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(" a \n b\t"), "a b")


class SplitTest(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        papers = [f"paper-{i}" for i in range(20)]
        train1, val1 = split_papers(papers, seed=42, val_fraction=0.15)
        train2, val2 = split_papers(list(reversed(papers)), seed=42, val_fraction=0.15)
        self.assertEqual(val1, val2)
        self.assertEqual(train1, train2)
        self.assertEqual(set(train1) & set(val1), set())
        self.assertEqual(set(train1) | set(val1), set(papers))
        self.assertEqual(len(val1), 3)

    def test_val_fraction_bounds(self):
        with self.assertRaises(ValueError):
            split_papers(["a"], seed=1, val_fraction=0.0)
        with self.assertRaises(ValueError):
            split_papers(["a"], seed=1, val_fraction=1.0)


def _figure(path: Path, figure_no: int, caption: str):
    return {
        "figure_no": figure_no,
        "page": 1,
        "path": str(path),
        "width": 100,
        "height": 100,
        "caption": caption,
    }


class BuildRowsTest(unittest.TestCase):
    def test_rows_use_relative_paths_and_paper_level_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_root = Path(tmp)
            papers = []
            for index in range(4):
                paper_dir = image_root / "output" / "pdf_figures" / f"paper{index}"
                paper_dir.mkdir(parents=True)
                fig = paper_dir / "fig_1.jpg"
                Image.new("RGB", (4, 4), color=(index, index, index)).save(fig)
                caption = "Figure 1: Accuracy increases with scale." if index % 2 else None
                papers.append(
                    {
                        "result": {
                            "paper_id": f"paper{index}",
                            "figures": [_figure(fig, 1, caption)] if caption else [],
                        }
                    }
                )
            missing = _figure(image_root / "nope.jpg", 1, "Figure 1: ghost")
            empty = image_root / "output" / "pdf_figures" / "paper0" / "fig_2.jpg"
            empty.write_bytes(b"")
            papers.append(
                {
                    "result": {
                        "paper_id": "ghost",
                        "figures": [
                            missing,
                            _figure(empty, 2, "Figure 2: Accuracy decreases."),
                        ],
                    }
                }
            )
            snapshot = {"extract_paper_figures": {f"k{i}": p for i, p in enumerate(papers)}}

            rows, summary = build_rows(
                snapshot, seed=42, val_fraction=0.34, image_root=image_root
            )
            usable = [pid for pid, _ in iter_usable_figures(snapshot)]
            self.assertEqual(sorted(usable), ["paper1", "paper3"])
            self.assertNotIn("ghost", usable)

            per_paper_split = {}
            for row in rows:
                self.assertFalse(Path(row["image_relpath"]).is_absolute())
                self.assertEqual(row["prompt"], _QUESTION_PROMPTS[row["question"]])
                per_paper_split.setdefault(row["paper_id"], set()).add(row["split"])
            for splits in per_paper_split.values():
                self.assertEqual(len(splits), 1, "同一论文的图不能跨 train/val")

            # The caption matches describe/axes/trend, so 2 figures yield 6 rows;
            # the missing and the zero-byte files are counted as unreadable.
            self.assertEqual(summary["figures_usable"], 2)
            self.assertEqual(summary["figures_unreadable"], 2)
            self.assertEqual(summary["rows"], 6)
            self.assertEqual(
                summary["counts"]["train"]["describe"] + summary["counts"]["val"]["describe"], 2
            )


if __name__ == "__main__":
    unittest.main()
