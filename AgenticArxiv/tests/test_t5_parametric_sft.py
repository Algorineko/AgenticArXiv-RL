"""检查 T5 SFT 派生任务的训练集血缘与专家样本唯一性。"""

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_parametric_sft_data import (  # noqa: E402
    build_parametric_tasks,
    canonical_hash,
    tool_names,
    validate_derived_tasks,
)
from generate_sft_data import generate_deterministic_trajectories  # noqa: E402
from benchmark.tasks_expanded import EXPANDED_SPECS  # noqa: E402


V3_PATH = REPO_ROOT / "data" / "splits" / "v3_81.json"
T5_PARENTS = {
    "analyze_cv5_ref1_desc",
    "analyze_cv5_ref3_trend",
    "analyze_ro5_ref2_axes",
}


class _OfflineT5Env:
    """无需 PDF、网络或模型，返回确定性的观察结果。"""

    def reset_runtime_state(self):
        pass

    def execute_tool(self, tool_name, args):
        if tool_name == "get_recently_submitted_cs_papers":
            return [
                {
                    "id": f"2601.{index:05d}v1",
                    "title": f"Fixture Paper {index}",
                    "authors": ["A. Tester"],
                    "pdf_url": f"https://arxiv.org/pdf/2601.{index:05d}v1.pdf",
                }
                for index in range(1, 6)
            ]
        paper_id = f"2601.{int(args['ref']):05d}v1"
        if tool_name == "download_arxiv_pdf":
            return {"paper_id": paper_id, "status": "READY"}
        if tool_name == "extract_paper_figures":
            return {"paper_id": paper_id, "count": 2, "figures": [
                {"figure_no": 1, "caption": "First figure"},
                {"figure_no": 2, "caption": "Second figure"},
            ]}
        if tool_name == "analyze_figure":
            return {
                "paper_id": paper_id,
                "figure_no": args["figure_no"],
                "question": args["question"],
                "answer": "Answer from the recorded environment",
                "backend": "extractive",
            }
        raise AssertionError(tool_name)


class T5ParametricSftTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = json.loads(V3_PATH.read_text(encoding="utf-8"))
        cls.all_variants = build_parametric_tasks(include_t5=True)
        cls.t5 = [
            item for item in cls.all_variants if item.parent_task_id in T5_PARENTS
        ]

    def test_v2_default_remains_unchanged(self):
        default = build_parametric_tasks()
        self.assertEqual(len(default), 80)
        self.assertFalse({item.parent_task_id for item in default} & T5_PARENTS)

    def test_t5_variants_are_train_only_and_keep_parent_topology(self):
        validate_derived_tasks(self.all_variants, self.split)
        self.assertEqual(len(self.t5), 6)
        self.assertEqual({item.parent_task_id for item in self.t5}, T5_PARENTS)

        by_id = {spec.id: spec for spec in EXPANDED_SPECS}
        heldout = set().union(*(
            self.split["split"][name]
            for name in ("dev", "iid_test", "ood_test")
        ))
        for item in self.t5:
            parent = by_id[item.parent_task_id]
            with self.subTest(task=item.spec.id):
                self.assertNotIn(item.parent_task_id, heldout)
                self.assertEqual(tool_names(item.spec.steps), tool_names(parent.steps))
                self.assertEqual(item.spec.steps[-1].args["figure_no"],
                                 parent.steps[-1].args["figure_no"])
                self.assertNotEqual(item.spec.steps[-1].args["question"],
                                    parent.steps[-1].args["question"])

    def test_offline_expert_rows_have_unique_messages_and_lineage(self):
        rows = generate_deterministic_trajectories(
            [item.spec for item in self.t5],
            _OfflineT5Env(),
            "fixture tools",
            source_split="v3_81.json:train:parametric_v1_t5",
        )
        self.assertEqual(len(rows), 6 * 5)  # 每个任务四步工具调用和一步 FINISH
        self.assertEqual(
            {row["source_task_id"] for row in rows},
            {item.spec.id for item in self.t5},
        )
        self.assertTrue(all(
            row["source_split"] == "v3_81.json:train:parametric_v1_t5"
            for row in rows
        ))
        fingerprints = [canonical_hash(row["messages"]) for row in rows]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))


if __name__ == "__main__":
    unittest.main()
