#!/usr/bin/env python3
"""SFT 语言扩增的数据不变性与泄漏回归测试。"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from augment_sft_data import (  # noqa: E402
    TASK_END,
    TASK_START,
    augment_rows,
    augment_validated_rows,
    replace_prompt_task,
    split_assistant_action,
    split_prompt_task,
)
from benchmark.metrics import classify_blocked_terminal_semantics  # noqa: E402
from benchmark.task_spec import reference_terminal_thought  # noqa: E402
from benchmark.tasks_expanded import get_expanded_tasks  # noqa: E402


def _prompt(task="训练任务", history="历史保持不变"):
    return f"工具描述\n{TASK_START}{task}{TASK_END}\n{history}"


class PromptBoundaryTest(unittest.TestCase):
    def test_only_the_task_span_is_replaced(self):
        original = _prompt(history="Observation: immutable")
        replaced = replace_prompt_task(original, "改写后的训练任务")
        before, task, after = split_prompt_task(replaced)

        self.assertEqual(task, "改写后的训练任务")
        self.assertIn("Observation: immutable", after)
        self.assertTrue(before.startswith("工具描述"))

    def test_ambiguous_or_missing_boundary_is_rejected(self):
        for bad in ("no markers", _prompt() + TASK_END):
            with self.subTest(prompt=bad), self.assertRaises(ValueError):
                split_prompt_task(bad)


class ActionInvariantTest(unittest.TestCase):
    def test_action_text_is_byte_for_byte_unchanged(self):
        action = ' {"name":"download_arxiv_pdf","args":{"ref":1}}'
        parent = {
            "source_task_id": "train_1",
            "source_split": "v2_62.json:train",
            "trajectory_step": 0,
            "messages": [
                {"role": "user", "content": _prompt()},
                {"role": "assistant", "content": f"Thought: 原始思考\nAction:{action}"},
            ],
        }
        payload = {
            "version": 2,
            "split": {"train": ["train_1"], "dev": [], "iid_test": [], "ood_test": []},
        }

        # 单测使用合成 task id，因此绕过真实任务文本审计，只验证核心扩增函数。
        import augment_sft_data as module
        original_get_tasks = module.get_expanded_tasks
        module.get_expanded_tasks = lambda: [{"id": "train_1", "task": "训练任务"}]
        try:
            rows = augment_rows([parent], payload)
        finally:
            module.get_expanded_tasks = original_get_tasks

        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["sample_sha256"] for row in rows}), 12)
        for row in rows:
            _, actual_action = split_assistant_action(row["messages"][1]["content"])
            self.assertEqual(actual_action, action)
            self.assertEqual(row["parent_sample_sha256"], rows[0]["parent_sample_sha256"])

    def test_blocked_finish_augmentation_preserves_specific_reason(self):
        task = next(
            task for task in get_expanded_tasks()
            if task["id"] == "infeasible_no_session"
        )
        parent = {
            "source_task_id": task["id"],
            "source_split": "v2_62.json:train",
            "trajectory_step": 0,
            "messages": [
                {"role": "user", "content": _prompt(task["task"])},
                {"role": "assistant", "content": (
                    f"Thought: {reference_terminal_thought(task)}\nAction: FINISH"
                )},
            ],
        }
        rows = augment_validated_rows([parent])
        self.assertEqual(len(rows), 12)
        for row in rows:
            thought, action = split_assistant_action(row["messages"][1]["content"])
            semantic = classify_blocked_terminal_semantics(
                task, [{"thought": thought, "action": action.strip()}]
            )
            self.assertEqual(semantic, "explained_block")

    def test_old_generic_blocked_seed_fails_fast(self):
        task = next(
            task for task in get_expanded_tasks()
            if task["id"] == "infeasible_no_session"
        )
        parent = {
            "source_task_id": task["id"],
            "messages": [
                {"role": "user", "content": _prompt(task["task"])},
                {"role": "assistant", "content": (
                    "Thought: 该任务无法通过现有工具完成或参数无效\nAction: FINISH"
                )},
            ],
        }
        with self.assertRaisesRegex(ValueError, "blocked FINISH seed"):
            augment_validated_rows([parent])


class LeakageGuardTest(unittest.TestCase):
    def test_heldout_source_id_is_rejected(self):
        row = {
            "source_task_id": "heldout",
            "source_split": "v2_62.json:train",
            "messages": [
                {"role": "user", "content": _prompt("测试任务")},
                {"role": "assistant", "content": "Thought: 完成\nAction: FINISH"},
            ],
        }
        payload = {
            "version": 2,
            "split": {
                "train": ["train_1"], "dev": ["heldout"],
                "iid_test": [], "ood_test": [],
            },
        }
        with self.assertRaisesRegex(ValueError, "来源不等于完整 train"):
            augment_rows([row], payload)


class WrapperSetTest(unittest.TestCase):
    """定向补数据的措辞框架必须与基础集不相交，否则自进化闭环只会生成重复行。"""

    @staticmethod
    def _parent():
        return {
            "source_task_id": "search_AI_1d_3",
            "messages": [
                {"role": "user", "content": _prompt("查最近一天 AI 论文")},
                {"role": "assistant", "content": (
                    "Thought: 需要先检索\nAction: "
                    '{"name": "get_recently_submitted_cs_papers", "args": {"aspect": "AI", "days": 1, "max_results": 3}}'
                )},
            ],
        }

    def test_targeted_set_is_disjoint_from_base_and_tagged(self):
        from augment_sft_data import WRAPPER_SETS

        base = augment_validated_rows([self._parent()])
        targeted = augment_validated_rows([self._parent()], wrapper_set="targeted")
        self.assertEqual(len(base), 12)
        self.assertEqual(len(targeted), 12)
        self.assertTrue({r["sample_sha256"] for r in base}.isdisjoint({r["sample_sha256"] for r in targeted}))
        self.assertEqual({r["augmentation"]["wrapper_set"] for r in base}, {"base"})
        self.assertEqual({r["augmentation"]["wrapper_set"] for r in targeted}, {"targeted"})
        self.assertTrue(set(WRAPPER_SETS["base"]).isdisjoint(WRAPPER_SETS["targeted"]))
        # same decisions, different phrasing: the assistant turns are identical across the two sets
        self.assertEqual(
            sorted(r["messages"][1]["content"] for r in base),
            sorted(r["messages"][1]["content"] for r in targeted),
        )

    def test_unknown_wrapper_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "wrapper_set"):
            augment_validated_rows([self._parent()], wrapper_set="nope")


if __name__ == "__main__":
    unittest.main()
