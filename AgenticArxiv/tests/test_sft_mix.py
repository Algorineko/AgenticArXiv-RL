#!/usr/bin/env python3
"""参数化语言扩增审计与最终数据混合测试。"""

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from augment_parametric_sft_data import validate_parametric_rows  # noqa: E402
from augment_sft_data import canonical_hash, sha256_file  # noqa: E402
from build_sft_train_mix import build_mix, validate_source_manifest  # noqa: E402


SPLIT_NAME = "v2_62.json"


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
        "source_split": f"{SPLIT_NAME}:train:parametric_v1",
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
            len(validate_parametric_rows(
                self.rows, self.payload, self.manifest,
                split_name=SPLIT_NAME,
            )),
            2,
        )

    def test_changed_messages_without_new_hash_are_rejected(self):
        bad = deepcopy(self.rows)
        bad[0]["messages"][1]["content"] += " corrupted"
        with self.assertRaisesRegex(ValueError, "sample_sha256"):
            validate_parametric_rows(
                bad, self.payload, self.manifest, split_name=SPLIT_NAME,
            )

    def test_heldout_parent_is_rejected(self):
        payload = deepcopy(self.payload)
        payload["split"]["train"] = []
        payload["split"]["iid_test"] = [self.parent]
        with self.assertRaisesRegex(ValueError, "纯 train"):
            validate_parametric_rows(
                self.rows, payload, self.manifest, split_name=SPLIT_NAME,
            )

    def test_source_split_must_match_the_split_actually_used(self):
        """种子记录的是生成它的那份切分，校验必须按它比对而不是按版本号猜。

        早先这里由 version 拼出 `*_62.json`，于是任何更新的切分（比如 v3_73）
        都会被判成不匹配，参数化流水线根本无法往前走。
        """
        with self.assertRaisesRegex(ValueError, "source_split"):
            validate_parametric_rows(
                self.rows, self.payload, self.manifest,
                split_name="v3_73.json",
            )


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


class SourceManifestValidationTest(unittest.TestCase):
    """来源行数以各自 manifest 为准，不再写死某一代数据的 1020 / 1908。"""

    def _source(self, directory: Path, rows: int, *, kind: str, manifest_rows=None):
        path = directory / "source.jsonl"
        path.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(rows)), encoding="utf-8")
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest = {
            "augmentation_kind" if kind == "original_train_linguistic" else "kind": (
                "linguistic_semantics_preserving" if kind == "original_train_linguistic" else kind
            ),
            "output_sha256": sha256_file(path),
            "output_rows": rows if manifest_rows is None else manifest_rows,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, manifest_path

    def test_any_row_count_passes_when_file_and_manifest_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rows in (3, 1020, 2628):
                with self.subTest(rows=rows):
                    path, manifest_path = self._source(Path(tmp), rows, kind="parametric_v1_linguistic")
                    manifest = validate_source_manifest(
                        path, manifest_path, expected_kind="parametric_v1_linguistic", expected_rows=rows
                    )
                    self.assertEqual(manifest["output_rows"], rows)

    def test_manifest_row_count_mismatch_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, manifest_path = self._source(Path(tmp), 5, kind="parametric_v1_linguistic", manifest_rows=4)
            with self.assertRaisesRegex(ValueError, "行数"):
                validate_source_manifest(path, manifest_path, expected_kind="parametric_v1_linguistic", expected_rows=5)

    def test_wrong_kind_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, manifest_path = self._source(Path(tmp), 2, kind="original_train_linguistic")
            with self.assertRaisesRegex(ValueError, "kind"):
                validate_source_manifest(path, manifest_path, expected_kind="parametric_v1_linguistic", expected_rows=2)


if __name__ == "__main__":
    unittest.main()
