#!/usr/bin/env python3
"""参数化 SFT 任务的血缘、拓扑和 heldout 边界测试。"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_parametric_sft_data import (  # noqa: E402
    build_parametric_tasks,
    tool_names,
    validate_derived_tasks,
)
from benchmark.tasks_expanded import EXPANDED_SPECS  # noqa: E402


SPLIT_PATH = REPO_ROOT / "data" / "splits" / "v3_81.json"


class ParametricSftTaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        cls.derived = build_parametric_tasks()
        cls.by_id = {spec.id: spec for spec in EXPANDED_SPECS}

    def test_expected_count_and_unique_ids(self):
        self.assertEqual(len(self.derived), 80)
        ids = [item.spec.id for item in self.derived]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_parents_are_train_only(self):
        train = set(self.payload["split"]["train"])
        heldout = set(sum(
            (self.payload["split"][name] for name in ("dev", "iid_test", "ood_test")),
            [],
        ))
        parents = {item.parent_task_id for item in self.derived}
        self.assertTrue(parents <= train)
        self.assertFalse(parents & heldout)

    def test_every_variant_preserves_parent_topology(self):
        for item in self.derived:
            parent = self.by_id[item.parent_task_id]
            with self.subTest(task=item.spec.id):
                self.assertEqual(tool_names(item.spec.steps), tool_names(parent.steps))
                self.assertEqual(tool_names(item.spec.setup), tool_names(parent.setup))

    def test_no_exact_benchmark_text_or_id_is_reused(self):
        benchmark_ids = set(self.by_id)
        benchmark_texts = {spec.task.strip() for spec in EXPANDED_SPECS}
        self.assertFalse({item.spec.id for item in self.derived} & benchmark_ids)
        self.assertFalse({item.spec.task.strip() for item in self.derived} & benchmark_texts)

    def test_full_validation_passes(self):
        validate_derived_tasks(self.derived, self.payload)

    def test_no_heldout_composite_chain_length_is_created(self):
        # OOD keys are composite/3 and composite/4. Derived composite stays at length 2.
        for item in self.derived:
            if item.spec.category == "composite":
                self.assertEqual(len(item.spec.steps), 2, item.spec.id)

    def test_infeasible_variants_preserve_terminal_contract(self):
        blocked = [
            item for item in self.derived
            if item.parent_task_id.startswith("infeasible_")
        ]
        self.assertTrue(blocked)
        for item in blocked:
            parent = self.by_id[item.parent_task_id]
            with self.subTest(task=item.spec.id):
                self.assertEqual(item.spec.terminal_mode, "blocked")
                self.assertEqual(item.spec.terminal_reason, parent.terminal_reason)


if __name__ == "__main__":
    unittest.main()
