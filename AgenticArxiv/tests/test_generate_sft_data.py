#!/usr/bin/env python3
"""SFT 数据生成的防泄漏、血缘和 fail-fast 回归测试。"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_sft_data import (  # noqa: E402
    build_sft_samples_from_history,
    generate_deterministic_trajectories,
    select_task_specs,
)
from benchmark.task_spec import Step, TaskSpec  # noqa: E402
from benchmark.tasks_expanded import EXPANDED_SPECS, EXPANDED_TASKS  # noqa: E402

V2_PATH = REPO_ROOT / "data" / "splits" / "v2_62.json"


class SplitLeakageGuardTest(unittest.TestCase):
    def test_expanded_specs_and_benchmark_dicts_have_one_source_of_truth(self):
        self.assertEqual(
            [spec.to_task() for spec in EXPANDED_SPECS],
            EXPANDED_TASKS,
        )

    def test_expanded_requires_an_explicit_versioned_train(self):
        for bad in (None, "train", f"{V2_PATH}:dev", f"{V2_PATH}:iid_test",
                    f"{V2_PATH}:ood_test"):
            with self.subTest(split=bad), self.assertRaises(SystemExit):
                select_task_specs("expanded", bad)

    def test_v2_train_selects_exactly_the_36_training_ids(self):
        payload = json.loads(V2_PATH.read_text(encoding="utf-8"))
        specs, source_split = select_task_specs("expanded", f"{V2_PATH}:train")
        selected = {spec.id for spec in specs}

        self.assertEqual(source_split, "v2_62.json:train")
        self.assertEqual(selected, set(payload["split"]["train"]))
        self.assertEqual(len(selected), 36)
        for held_out in ("dev", "iid_test", "ood_test"):
            self.assertFalse(selected & set(payload["split"][held_out]))

    def test_basic_rejects_a_split_instead_of_silently_ignoring_it(self):
        with self.assertRaises(SystemExit):
            select_task_specs("basic", f"{V2_PATH}:train")


class DataLineageTest(unittest.TestCase):
    def test_each_step_records_source_and_position(self):
        history = [
            {"thought": "检索", "action": '{"name":"search","args":{}}',
             "observation": "ok"},
            {"thought": "完成", "action": "FINISH", "observation": "done"},
        ]
        rows = build_sft_samples_from_history(
            "task", "tools", history,
            source_task_id="source_1", source_split="train",
        )

        self.assertEqual([row["source_task_id"] for row in rows], ["source_1"] * 2)
        self.assertEqual([row["source_split"] for row in rows], ["train"] * 2)
        self.assertEqual([row["trajectory_step"] for row in rows], [0, 1])


class _FailingEnv:
    def reset_runtime_state(self):
        pass

    def execute_tool(self, tool_name, args):
        raise ValueError("intentional tool failure")


class _StateProbeEnv:
    def reset_runtime_state(self):
        pass

    def execute_tool(self, tool_name, args):
        if tool_name == "get_recently_submitted_cs_papers":
            return [{
                "id": "2601.00001v1",
                "title": "State Probe Paper",
                "authors": ["Tester"],
                "pdf_url": "https://arxiv.org/pdf/2601.00001v1",
            }]
        if tool_name == "download_arxiv_pdf":
            from models.store import store

            paper = store.resolve_paper("sft_spec_state_probe", args["ref"])
            if paper is None:
                raise ValueError("search result was not persisted")
            return {"paper_id": paper.id, "status": "READY"}
        raise AssertionError(tool_name)


class ExpertFailFastTest(unittest.TestCase):
    def test_failed_expert_tool_never_becomes_a_finish_training_sample(self):
        spec = TaskSpec(
            id="broken_expert",
            task="检索论文",
            steps=(Step("get_recently_submitted_cs_papers", {"aspect": "AI"}),),
        )
        with self.assertRaisesRegex(RuntimeError, "SFT 专家步骤执行失败"):
            generate_deterministic_trajectories([spec], _FailingEnv(), "tools")

    def test_search_result_becomes_session_memory_for_the_next_step(self):
        spec = TaskSpec(
            id="state_probe",
            task="先搜索再下载第一篇",
            steps=(
                Step("get_recently_submitted_cs_papers", {
                    "aspect": "AI", "days": 7, "max_results": 1,
                }),
                Step("download_arxiv_pdf", {"ref": 1}),
            ),
        )
        rows = generate_deterministic_trajectories(
            [spec], _StateProbeEnv(), "tools", source_split="v2_62.json:train"
        )

        self.assertEqual(len(rows), 3)  # search、download、FINISH
        self.assertTrue(all(row["source_task_id"] == "state_probe" for row in rows))
        self.assertTrue(all(
            "session_id" not in row["messages"][-1]["content"]
            for row in rows
        ))

    def test_blocked_expert_names_the_declared_reason(self):
        spec = TaskSpec(
            id="missing_context_probe",
            task="翻译刚才那篇论文",
            terminal_mode="blocked",
            terminal_reason="missing_context",
        )
        rows = generate_deterministic_trajectories(
            [spec], _FailingEnv(), "tools", source_split="train"
        )
        self.assertEqual(len(rows), 1)
        answer = rows[0]["messages"][1]["content"]
        self.assertIn("会话", answer)
        self.assertIn("无法解析", answer)
        self.assertIn("Action: FINISH", answer)


if __name__ == "__main__":
    unittest.main()
