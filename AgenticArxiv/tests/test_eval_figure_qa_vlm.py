#!/usr/bin/env python3
"""eval_figure_qa_vlm 的指标函数测试（纯 CPU）。"""

import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT.parent))

from scripts.eval_figure_qa_vlm import aggregate, rouge_1, rouge_l, tokenize  # noqa: E402


class RougeTest(unittest.TestCase):
    def test_tokenize_lowercases_and_strips_punctuation(self):
        self.assertEqual(tokenize("Accuracy (%) rises, 5x!"), ["accuracy", "rises", "5x"])

    def test_rouge1_identical(self):
        self.assertEqual(rouge_1("a b c", "a b c"), 1.0)

    def test_rouge1_disjoint(self):
        self.assertEqual(rouge_1("x y", "a b"), 0.0)

    def test_rouge1_partial(self):
        # pred {a,b,c}, ref {a,b,d}: overlap 2, p=2/3, r=2/3, f=2/3
        self.assertAlmostEqual(rouge_1("a b c", "a b d"), 2 / 3, places=6)

    def test_rouge_l_rewards_order(self):
        ref = "the accuracy increases with depth"
        ordered = rouge_l("accuracy increases with depth", ref)
        shuffled = rouge_l("depth with increases accuracy", ref)
        self.assertGreater(ordered, shuffled)

    def test_rouge_l_identical(self):
        self.assertEqual(rouge_l("a b c", "a b c"), 1.0)

    def test_empty_side_is_zero(self):
        self.assertEqual(rouge_1("", "a b"), 0.0)
        self.assertEqual(rouge_l("a b", ""), 0.0)


class AggregateTest(unittest.TestCase):
    def test_aggregate_per_question(self):
        results = [
            {"question": "describe", "rouge1_f": 0.5, "rougeL_f": 0.4, "answer": "x", "error": None},
            {"question": "describe", "rouge1_f": 0.7, "rougeL_f": 0.6, "answer": "", "error": None},
            {"question": "trend", "rouge1_f": 0.1, "rougeL_f": 0.1, "answer": "y", "error": "boom"},
        ]
        aggregate_result = aggregate(results)
        self.assertEqual(aggregate_result["n"], 3)
        self.assertEqual(aggregate_result["per_question"]["describe"]["n"], 2)
        self.assertAlmostEqual(aggregate_result["per_question"]["describe"]["rougeL_f"], 0.5)
        self.assertEqual(aggregate_result["per_question"]["describe"]["empty_or_error"], 1)
        self.assertEqual(aggregate_result["per_question"]["trend"]["empty_or_error"], 1)


if __name__ == "__main__":
    unittest.main()
