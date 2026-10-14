"""S0 资产关口：核对完整定义，不能以相同数量代替业务契约不变。"""

import importlib.util
from itertools import combinations
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "freeze_mainline_assets", ROOT / "scripts/freeze_mainline_assets.py",
)
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)


class MainlineAssetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = json.loads(freeze.BASELINE.read_text(encoding="utf-8"))
        cls.frozen = cls.baseline["assets"]
        cls.current = freeze.fresh_assets()
        cls.task_ids = {task["id"] for task in cls.current["expanded_tasks"]}

    def split_document(self, filename):
        return self.current["split_files"][f"data/splits/{filename}"]["definition"]

    def assert_disjoint(self, groups):
        for name, ids in groups.items():
            self.assertEqual(len(ids), len(set(ids)), f"{name} 内部重复")
            self.assertLessEqual(set(ids), self.task_ids, f"{name} 含未知任务")
        for left, right in combinations(groups, 2):
            self.assertFalse(set(groups[left]) & set(groups[right]), f"{left}/{right} 交叉污染")

    def test_frozen_content_and_section_hashes_are_self_consistent(self):
        self.assertEqual(self.baseline["format_version"], "mainline-assets-s0-v1")
        self.assertEqual(freeze.sha256(self.frozen), self.baseline["assets_sha256"])
        self.assertEqual(
            {key: freeze.sha256(value) for key, value in self.frozen.items()},
            self.baseline["section_sha256"],
        )

    def test_ten_tool_business_schemas_and_registration_order_are_unchanged(self):
        tools = self.current["tools"]
        self.assertEqual(len(tools), 10)
        self.assertEqual(len({tool["name"] for tool in tools}), 10)
        self.assertEqual(tools, self.frozen["tools"])
        for tool in tools:
            self.assertNotIn("session_id", tool["parameters"]["properties"])
        # 即使工具函数实现变化，业务 ref/选项都必须保留在完整 schema 中。
        self.assertIn("input_pdf_path", tools[7]["parameters"]["properties"])

    def test_all_86_tasks_and_full_taskspec_definitions_are_unchanged(self):
        self.assertEqual(len(self.current["expanded_task_specs"]), 86)
        self.assertEqual(len(self.current["expanded_tasks"]), 86)
        self.assertEqual(len(self.task_ids), 86)
        for section in ("expanded_task_specs", "expanded_tasks"):
            self.assertEqual(self.current[section], self.frozen[section], section)
        self.assertEqual(
            [task["id"] for task in self.current["expanded_task_specs"]],
            [task["id"] for task in self.current["expanded_tasks"]],
        )

    def test_eight_smoke_tasks_include_original_dependency_and_setup_contracts(self):
        for section in ("smoke_task_specs", "smoke_tasks"):
            self.assertEqual(len(self.current[section]), 8)
            self.assertEqual(self.current[section], self.frozen[section], section)
        ids = [task["id"] for task in self.current["smoke_tasks"]]
        self.assertEqual(len(set(ids)), 8)

    def test_every_existing_split_content_and_original_file_hash_is_unchanged(self):
        self.assertEqual(len(self.current["split_files"]), 9)
        self.assertEqual(self.current["split_files"], self.frozen["split_files"])
        for path, record in self.current["split_files"].items():
            groups = record["definition"].get("split", {})
            self.assert_disjoint(groups)

    def test_v6_33_selected_tasks_derive_only_from_v3_train(self):
        v3 = self.split_document("v3_81.json")
        v6 = self.split_document("v6_grpo_train.json")
        selected = v6["split"]["rl_train"]
        self.assertEqual(len(selected), 33)
        self.assertIn("data/splits/v3_81.json:train", v6["derived_from"])
        self.assertLessEqual(set(selected), set(v3["split"]["train"]))
        for name in ("dev", "iid_test", "ood_test"):
            self.assertFalse(set(selected) & set(v3["split"][name]), name)
        self.assertEqual(v6["split"], self.frozen["split_files"][
            "data/splits/v6_grpo_train.json"
        ]["definition"]["split"])

    def test_v7_55_8_19_4_partition_is_complete_and_disjoint(self):
        groups = self.split_document("v7_86.json")["split"]
        self.assertEqual(
            {name: len(ids) for name, ids in groups.items()},
            {"train": 55, "dev": 8, "iid_test": 19, "ood_test": 4},
        )
        self.assert_disjoint(groups)
        self.assertEqual({tid for ids in groups.values() for tid in ids}, self.task_ids)
        for name, ids in self.split_document("v3_81.json")["split"].items():
            self.assertLessEqual(set(ids), set(groups[name]), f"v3 的 {name} 被重新分配")
        self.assertEqual(groups, self.frozen["split_files"][
            "data/splits/v7_86.json"
        ]["definition"]["split"])

    def test_v6_selection_is_not_mistaken_for_v7_full_training_split(self):
        selected = set(self.split_document("v6_grpo_train.json")["split"]["rl_train"])
        groups = self.split_document("v7_86.json")["split"]
        self.assertLessEqual(selected, set(groups["train"]))
        self.assertNotEqual(selected, set(groups["train"]))
        for name in ("dev", "iid_test", "ood_test"):
            self.assertFalse(selected & set(groups[name]), name)


if __name__ == "__main__":
    unittest.main()
