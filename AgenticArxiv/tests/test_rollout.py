"""Tests for rollout metadata helpers.

These tests do not load a real local model.  A lightweight fake preserves the
important ``TransformersLLMClient`` contract: ``model_name`` is the identifier
while ``model`` is the in-memory Hugging Face model object.
"""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.agent_engine import ReActAgent  # noqa: E402
from agents.side_effects import NoOpSideEffectManager  # noqa: E402
from rl.rollout import _get_model_name, _history_with_timings, rollout_single_task  # noqa: E402
from rl.trajectory import create_trajectory  # noqa: E402


class TestModelMetadata(unittest.TestCase):
    def test_local_client_uses_serializable_model_name(self):
        class LocalClient:
            model_name = "outputs/sft/final"
            model = object()

        model_name = _get_model_name(LocalClient())

        self.assertEqual(model_name, "outputs/sft/final")
        self.assertEqual(
            json.loads(json.dumps({"model": model_name})),
            {"model": "outputs/sft/final"},
        )

    def test_remote_client_keeps_legacy_string_model(self):
        class RemoteClient:
            model = "gpt-4"

        self.assertEqual(_get_model_name(RemoteClient()), "gpt-4")

    def test_non_serializable_model_object_is_not_recorded(self):
        class UnknownClient:
            model = object()

        self.assertEqual(_get_model_name(UnknownClient()), "")


class TestStepLatencyMetadata(unittest.TestCase):
    def test_agent_timings_are_saved_on_corresponding_trajectory_steps(self):
        result = {
            "history": [
                {"thought": "search", "action": '{"name":"search"}', "observation": "ok"},
                {"thought": "done", "action": "FINISH", "observation": "done"},
            ],
            "timing": {"steps": [
                {"llm_ms": 410, "tool_ms": 23},
                {"llm_ms": 275, "tool_ms": 0},
            ]},
        }
        traj = create_trajectory(
            task_id="search_01", task="search", session_id="test",
            history=_history_with_timings(result), final_reward=1.0, metrics={},
        )

        self.assertEqual(
            [(step.llm_latency_ms, step.tool_latency_ms) for step in traj.steps],
            [(410, 23), (275, 0)],
        )

    def test_rollout_jsonl_contains_measured_step_latencies(self):
        class Client:
            calls = 0

            def chat_completions(self, **kwargs):
                time.sleep(0.02)
                self.calls += 1
                action = (
                    'Action: {"name":"get_recently_submitted_cs_papers",'
                    '"args":{"aspect":"AI","days":7,"max_results":5}}'
                    if self.calls == 1 else "Action: FINISH"
                )
                return {"choices": [{"message": {"content": "Thought: test\n" + action}}]}

        class Environment:
            def execute_tool(self, name, args):
                time.sleep(0.02)
                return [{"id": "paper-1", "title": "Paper one", "authors": [], "categories": ["cs.AI"]}]

        client = Client()
        agent = ReActAgent(client, side_effect_mgr=NoOpSideEffectManager(), env=Environment())
        with tempfile.TemporaryDirectory() as output_dir:
            rollout_single_task("search_01", output_dir=output_dir, agent=agent, llm_client=client)
            trajectory = json.loads(next(Path(output_dir).glob("*.jsonl")).read_text())

        steps = trajectory["steps"]
        self.assertEqual(len(steps), 2)
        self.assertGreater(steps[0]["llm_latency_ms"], 0)
        self.assertGreater(steps[0]["tool_latency_ms"], 0)
        self.assertGreater(steps[1]["llm_latency_ms"], 0)
        self.assertEqual(steps[1]["tool_latency_ms"], 0)

        forced = ReActAgent(
            Client(), side_effect_mgr=NoOpSideEffectManager(),
            env=Environment(), max_iterations=1,
        ).run("检索论文", session_id="forced-stop")
        forced_history = _history_with_timings(forced)
        self.assertEqual(forced_history[-1]["action"], "FORCE_STOP")
        self.assertEqual(forced_history[-1]["llm_latency_ms"], 0)
        self.assertEqual(forced_history[-1]["tool_latency_ms"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
