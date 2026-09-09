#!/usr/bin/env python3
"""Raw GRPO rollout audit tests (CPU-only, no network or model required)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT.parent))

from rl.reward import RewardBreakdown, RewardCalculator, RewardSchedule  # noqa: E402
from rl.grpo_reward import (  # noqa: E402
    _prepare_rollout_environment,
    _resolve_prompt_task_id,
    make_grpo_reward_fn,
    make_multiturn_rollout_func,
)
from rl.rollout_audit import (  # noqa: E402
    RolloutAuditWriter,
    detect_rollout_anomalies,
)
from scripts.audit_grpo_rollouts import summarize  # noqa: E402


ALLOWED_TOOLS = {"search_arxiv_papers", "download_arxiv_pdf"}


def _breakdown(total=0.5):
    weights = RewardSchedule(1.0, 3.0, 2.0, 1.0, 3.0)
    return RewardBreakdown(total, 1.0, 0.5, 0.5, 0.5, 0.0, weights)


def _clean_trajectory():
    turn = (
        "Thought: search first\n"
        'Action: {"name":"search_arxiv_papers","args":{"query":"agent"}}'
    )
    return {
        "history": [
            {
                "thought": "search first",
                "action": '{"name":"search_arxiv_papers","args":{"query":"agent"}}',
                "observation": "成功获取 3 条记录",
            },
            {"thought": "done", "action": "FINISH", "observation": "任务完成"},
        ],
        "raw_assistant_turns": [turn, "Thought: done\nAction: FINISH"],
        "clipped": False,
        "reached_max_turns": False,
    }


class RolloutAnomalyTest(unittest.TestCase):
    def test_clean_multiturn_rollout_has_no_anomalies(self):
        trajectory = _clean_trajectory()
        anomalies = detect_rollout_anomalies(
            "completion contains environment suffixes",
            trajectory,
            allowed_tools=ALLOWED_TOOLS,
        )
        self.assertFalse(any(anomalies.values()), anomalies)

    def test_checks_model_authored_turns_not_environment_observations(self):
        trajectory = _clean_trajectory()
        # The normalized history legitimately contains environment observations;
        # those must not be mistaken for text hallucinated by the policy.
        self.assertIn("observation", trajectory["history"][0])
        anomalies = detect_rollout_anomalies("ignored", trajectory)
        self.assertFalse(anomalies["hallucinated_observation"])

    def test_flags_malformed_think_and_invented_observation(self):
        trajectory = {
            "history": [{
                "thought": "",
                "action": "PARSE_ERROR",
                "observation": "无法解析 Action",
                "parse_failed": True,
            }],
            "raw_assistant_turns": [
                '<think>unfinished\nAction: {"name":"invented_tool","args":{}}\n'
                "Observation: fake\nAction: FINISH"
            ],
            "clipped": True,
            "reached_max_turns": True,
        }
        anomalies = detect_rollout_anomalies(
            "ignored", trajectory, allowed_tools=ALLOWED_TOOLS
        )
        self.assertTrue(anomalies["invalid_action"])
        self.assertTrue(anomalies["unbalanced_think_tags"])
        self.assertTrue(anomalies["multiple_actions_in_turn"])
        self.assertTrue(anomalies["hallucinated_observation"])
        self.assertTrue(anomalies["clipped"])
        self.assertTrue(anomalies["reached_max_turns"])


class RolloutSetupTest(unittest.TestCase):
    def test_prompt_task_id_resolves_before_and_after_chat_rendering(self):
        content = "full react prompt for task A"
        lookup = {content: "task_a"}
        raw = [[{"role": "user", "content": content}]][0]
        rendered = f"<|user|>\n{content}\n<|assistant|>"
        self.assertEqual(_resolve_prompt_task_id(raw, lookup), "task_a")
        self.assertEqual(_resolve_prompt_task_id(rendered, lookup), "task_a")

    def test_task_setup_seeds_state_without_becoming_policy_history(self):
        class Environment:
            def __init__(self):
                self.calls = []
                self.seeded = False

            def reset(self, task_id=""):
                self.calls.append(("reset", task_id, {}))

            def get_recently_submitted_cs_papers(self, **args):
                self.calls.append(("get_recently_submitted_cs_papers", None, args))
                self.seeded = True
                return [{"id": "paper-1"}]

            def download_arxiv_pdf(self, **args):
                if not self.seeded:
                    raise ValueError("not seeded")
                return {"paper_id": "paper-1"}

        task = {
            "id": "ref_task",
            "setup": [{
                "name": "get_recently_submitted_cs_papers",
                "args": {"aspect": "CV", "days": 7, "max_results": 5},
            }],
        }
        environment = Environment()
        applied = _prepare_rollout_environment(
            environment, "ref_task", {"ref_task": task}
        )
        result = environment.download_arxiv_pdf(ref="Image Restoration")

        self.assertEqual(result["paper_id"], "paper-1")
        self.assertEqual(applied, task["setup"])
        self.assertEqual(environment.calls[0][:2], ("reset", "ref_task"))
        self.assertEqual(environment.calls[1][0], "get_recently_submitted_cs_papers")

    def test_setup_mapping_failure_is_fail_fast(self):
        class Environment:
            def reset(self, task_id=""):
                return None

        with self.assertRaisesRegex(ValueError, "task_id"):
            _prepare_rollout_environment(Environment(), None, {"task_a": {}})

    def test_multiturn_rollout_applies_setup_before_first_model_action(self):
        prompt_text = "download the paper titled Image Restoration"
        task = {
            "id": "ref_task",
            "setup": [{
                "name": "get_recently_submitted_cs_papers",
                "args": {"aspect": "CV", "days": 7, "max_results": 5},
            }],
        }

        class Tokenizer:
            def apply_chat_template(self, prompt, tokenize=True, add_generation_prompt=True):
                return [1, 2]

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [200 + i for i, _ in enumerate(text)]}

            def decode(self, ids, skip_special_tokens=True):
                if ids and ids[0] == 101:
                    return (
                        "Thought: use the prepared search state\n"
                        'Action: {"name":"download_arxiv_pdf",'
                        '"args":{"ref":"Image Restoration"}}'
                    )
                return "Thought: done\nAction: FINISH"

        class Trainer:
            processing_class = Tokenizer()
            num_generations = 1
            num_generations_eval = 1
            max_completion_length = 256
            model = type("Model", (), {"training": True})()

            def __init__(self):
                self.calls = 0

            def _generate_single_turn(self, prompt_ids, images, fields):
                self.calls += 1
                token = 101 if self.calls == 1 else 102
                return [[token] for _ in prompt_ids], None, {}

        class Environment:
            def __init__(self):
                self.seeded = False

            def reset(self, task_id=""):
                self.seeded = False

            def get_recently_submitted_cs_papers(self, **args):
                self.seeded = True
                return [{"id": "paper-1"}]

            def download_arxiv_pdf(self, **args):
                if not self.seeded:
                    raise ValueError("not seeded")
                return {"paper_id": "paper-1", "status": "READY"}

        with patch("rl.grpo_reward.require_rollout_func_support"):
            rollout = make_multiturn_rollout_func(
                Environment,
                max_turns=2,
                tasks_by_id={"ref_task": task},
                prompt_task_ids={prompt_text: "ref_task"},
            )
        output = rollout(
            [[{"role": "user", "content": prompt_text}]],
            Trainer(),
        )
        trajectory = output["trajectory_results"][0]
        self.assertEqual(trajectory["task_id"], "ref_task")
        self.assertEqual(trajectory["setup_actions"], task["setup"])
        self.assertIn("paper-1", trajectory["history"][0]["observation"])
        self.assertNotIn("工具执行失败", trajectory["history"][0]["observation"])

    def test_real_snapshot_setup_makes_reference_gold_actions_executable(self):
        import tools.arxiv_tool  # noqa: F401
        import tools.cache_status_tool  # noqa: F401
        import tools.pdf_download_tool  # noqa: F401
        import tools.pdf_translate_tool  # noqa: F401
        from benchmark.tasks_expanded import get_expanded_tasks
        from rl.multiturn_env import AgenticArxivMultiTurnEnv

        snapshot = PACKAGE_ROOT.parent / "data" / "mock_arxiv_snapshot.json"
        if not snapshot.exists():
            self.skipTest("offline snapshot not available")
        wanted = {
            "ref_ctrl_word_handover",
            "ref_stress_image_restoration",
            "ref_stress_state_across",
        }
        tasks = {
            task["id"]: task
            for task in get_expanded_tasks()
            if task["id"] in wanted
        }
        self.assertEqual(set(tasks), wanted)

        for task_id, task in tasks.items():
            with self.subTest(task_id=task_id):
                environment = AgenticArxivMultiTurnEnv(snapshot)
                applied = _prepare_rollout_environment(
                    environment, task_id, tasks
                )
                action = task["expected_tools"][0]
                args = task["expected_tool_args"][0]
                result = getattr(environment, action)(**args)
                self.assertEqual(applied, task["setup"])
                self.assertTrue(result.get("paper_id"))


class ToolFailureRewardTest(unittest.TestCase):
    def test_finish_after_tool_failure_has_negative_outcome(self):
        task = {
            "id": "download",
            "expected_tools": ["download_arxiv_pdf"],
            "expected_tool_args": [{"ref": 1}],
        }
        failed = {
            "history": [
                {
                    "thought": "download",
                    "action": '{"name":"download_arxiv_pdf","args":{"ref":1}}',
                    "observation": "工具执行失败: 未找到论文",
                },
                {"thought": "done", "action": "FINISH", "observation": "任务完成"},
            ]
        }
        clean = {
            "history": [
                {
                    "thought": "download",
                    "action": '{"name":"download_arxiv_pdf","args":{"ref":1}}',
                    "observation": '{"paper_id":"paper-1","status":"READY"}',
                },
                {"thought": "done", "action": "FINISH", "observation": "任务完成"},
            ]
        }
        calculator = RewardCalculator(curriculum_steps=0)
        bad_reward, bad_metrics = calculator.compute_reward_breakdown(task, failed)
        good_reward, _ = calculator.compute_reward_breakdown(task, clean)

        self.assertEqual(bad_metrics.tool_exec_failures, 1)
        self.assertEqual(bad_reward.outcome, -1.0)
        self.assertLess(bad_reward.total, good_reward.total)


class RolloutAuditWriterTest(unittest.TestCase):
    def test_reward_function_writes_the_exact_scored_trajectories(self):
        class FakeCalculator:
            def compute_reward_breakdown(self, task, trajectory, training_step=0):
                return _breakdown(float(task["score"])), None

        class State:
            global_step = 7

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "rollout_traces.jsonl"
            writer = RolloutAuditWriter(trace_path, num_generations=2)
            reward_fn = make_grpo_reward_fn(
                {"task_a": {"id": "task_a", "score": 0.75}},
                reward_calc=FakeCalculator(),
                auditor=writer,
            )
            rewards = reward_fn(
                completions=["first", "second"],
                task_id=["task_a", "task_a"],
                trajectory_results=[_clean_trajectory(), _clean_trajectory()],
                trainer_state=State(),
            )
            records = [json.loads(line) for line in trace_path.read_text(
                encoding="utf-8"
            ).splitlines()]
            self.assertEqual(rewards, [0.75, 0.75])
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["training_step"], 7)
            self.assertEqual(records[0]["reward"], 0.75)

    def test_writes_generation_records_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "rollout_traces.jsonl"
            writer = RolloutAuditWriter(
                trace_path,
                num_generations=2,
                allowed_tools=ALLOWED_TOOLS,
            )
            writer.record_batch(
                completions=["a", "b"],
                task_ids=["task_a", "task_a"],
                trajectories=[_clean_trajectory(), _clean_trajectory()],
                breakdowns=[_breakdown(0.25), _breakdown(0.75)],
                rewards=[0.25, 0.75],
                training_step=3,
            )

            records = [json.loads(line) for line in trace_path.read_text(
                encoding="utf-8"
            ).splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual([r["generation_index"] for r in records], [0, 1])
            self.assertEqual(records[0]["group_reward_mean"], 0.5)
            self.assertEqual(records[0]["group_reward_std"], 0.25)
            self.assertEqual(records[1]["reward_breakdown"]["outcome"], 0.0)
            recomputed = summarize(records)
            self.assertEqual(recomputed["zero_reward_std_group_count"], 0)
            self.assertEqual(recomputed["exact_duplicate_group_count"], 1)

            summary_path = writer.save_summary(
                Path(tmpdir) / "rollout_traces.summary.json"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["group_count"], 1)
            self.assertEqual(summary["seen_samples"], 2)
            self.assertEqual(summary["saved_samples"], 2)
            self.assertEqual(summary["anomalous_samples"], 0)

    def test_recomputed_summary_audits_blocked_terminal_semantics(self):
        def row(thought, reward, generation):
            return {
                "task_id": "infeasible_no_session",
                "batch_index": 0,
                "group_index": 0,
                "generation_index": generation,
                "reward": reward,
                "group_reward_std": 0.2,
                "trajectory": {"history": [{
                    "thought": thought,
                    "action": "FINISH",
                    "observation": "任务结束",
                }]},
                "raw_assistant_turns": [thought],
                "active_anomalies": [],
            }

        summary = summarize([
            row("当前会话没有最近操作的论文，无法解析指代", 1.0, 0),
            row("任务已完成", 0.625, 1),
        ])
        semantics = summary["terminal_semantics"]
        self.assertEqual(semantics["applicable_sample_count"], 2)
        self.assertEqual(semantics["counts"]["explained_block"], 1)
        self.assertEqual(semantics["counts"]["false_completion"], 1)
        self.assertEqual(semantics["mean_reward_by_class"]["explained_block"], 1.0)

    def test_rejects_mixed_tasks_inside_one_grpo_group(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = RolloutAuditWriter(
                Path(tmpdir) / "rollout_traces.jsonl", num_generations=2
            )
            with self.assertRaisesRegex(ValueError, "mixed task ids"):
                writer.record_batch(
                    completions=["a", "b"],
                    task_ids=["task_a", "task_b"],
                    trajectories=[_clean_trajectory(), _clean_trajectory()],
                    breakdowns=[_breakdown(), _breakdown()],
                    rewards=[0.5, 0.5],
                    training_step=0,
                )

    def test_sample_limit_only_limits_jsonl_not_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "rollout_traces.jsonl"
            writer = RolloutAuditWriter(
                trace_path, num_generations=2, max_samples=1
            )
            writer.record_batch(
                completions=["a", "b"],
                task_ids=["task_a", "task_a"],
                trajectories=[_clean_trajectory(), _clean_trajectory()],
                breakdowns=[_breakdown(), _breakdown()],
                rewards=[0.5, 0.5],
                training_step=0,
            )
            self.assertEqual(len(trace_path.read_text(encoding="utf-8").splitlines()), 1)
            self.assertEqual(writer.summary()["seen_samples"], 2)
            self.assertEqual(writer.summary()["saved_samples"], 1)


if __name__ == "__main__":
    unittest.main()
