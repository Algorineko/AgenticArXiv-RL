#!/usr/bin/env python3
"""参数化语言扩增审计与最终数据混合测试。"""

import sys
import unittest
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from augment_parametric_sft_data import validate_parametric_rows  # noqa: E402
from augment_sft_data import canonical_hash  # noqa: E402
from build_sft_train_mix import build_mix  # noqa: E402


def prompt(task: str) -> str:
    return (
        "tools\n当前任务：" + task
        + "\n请按照ReAct框架的格式思考和行动:\nhistory"
    )


def row(task_id: str, parent: str, task: str, trajectory_step: int):
    messages = [
        {"role": "user", "content": prompt(task)},
        {"role": "assistant", "content": "Thought: x\nAction: FINISH"},
    ]
    return {
        "source_task_id": task_id,
        "source_split": "v2_62.json:train:parametric_v1",
        "trajectory_step": trajectory_step,
        "derived_task_id": task_id,
        "parent_task_id": parent,
        "generation_parameters": {"ref": 1},
        "dataset_stage": "parametric_v1_expert_seed",
        "sample_sha256": canonical_hash(messages),
        "messages": messages,
    }


class ParametricRowValidationTest(unittest.TestCase):
    def setUp(self):
        self.task_id = "augp1_parent_ref1"
        self.parent = "parent"
        self.task = "下载第1篇"
        self.rows = [
            row(self.task_id, self.parent, self.task, 0),
            row(self.task_id, self.parent, self.task, 1),
        ]
        # 两个决策必须拥有不同 messages/指纹。
        self.rows[1]["messages"][0]["content"] += "\nObservation: done"
        self.rows[1]["sample_sha256"] = canonical_hash(self.rows[1]["messages"])
        self.payload = {
            "version": 2,
            "split": {
                "train": [self.parent], "dev": [], "iid_test": [], "ood_test": [],
            },
        }
        self.manifest = {
            "kind": "train_only_parametric_expert_seed",
            "sample_rows": 2,
            "derived_tasks": 1,
            "tasks": [{
                "derived_task_id": self.task_id,
                "parent_task_id": self.parent,
                "task": self.task,
                "steps": [{"name": "download_arxiv_pdf", "args": {"ref": 1}}],
                "parameters": {"ref": 1},
            }],
        }

    def test_valid_rows_pass(self):
        self.assertEqual(
            len(validate_parametric_rows(self.rows, self.payload, self.manifest)), 2
        )

    def test_changed_messages_without_new_hash_are_rejected(self):
        bad = deepcopy(self.rows)
        bad[0]["messages"][1]["content"] += " corrupted"
        with self.assertRaisesRegex(ValueError, "sample_sha256"):
            validate_parametric_rows(bad, self.payload, self.manifest)

    def test_heldout_parent_is_rejected(self):
        payload = deepcopy(self.payload)
        payload["split"]["train"] = []
        payload["split"]["iid_test"] = [self.parent]
        with self.assertRaisesRegex(ValueError, "纯 train"):
            validate_parametric_rows(self.rows, payload, self.manifest)


class MixTest(unittest.TestCase):
    @staticmethod
    def _mix_row(label: str):
        messages = [
            {"role": "user", "content": label},
            {"role": "assistant", "content": "answer"},
        ]
        return {"messages": messages, "sample_sha256": canonical_hash(messages)}

    def test_mix_is_deterministic_and_tags_sources(self):
        original = [self._mix_row("o1"), self._mix_row("o2")]
        parametric = [self._mix_row("p1"), self._mix_row("p2")]
        first = build_mix(original, parametric, seed=42)
        second = build_mix(original, parametric, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(
            {row["mixture_source"] for row in first},
            {"original_train_linguistic", "parametric_v1_linguistic"},
        )

    def test_cross_source_duplicate_is_rejected(self):
        duplicate = self._mix_row("same")
        with self.assertRaisesRegex(ValueError, "重复"):
            build_mix([duplicate], [deepcopy(duplicate)], seed=42)


if __name__ == "__main__":
    unittest.main()
