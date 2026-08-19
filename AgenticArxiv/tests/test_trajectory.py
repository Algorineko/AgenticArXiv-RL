"""rl/trajectory.py 的单元测试

Trajectory 是 Agent 执行记录，以 JSONL 落盘供后续 RL 训练使用。
本文件覆盖：
- TrajectoryStep / Trajectory 数据类的构造与默认值
- save_trajectory / load_trajectories 的 JSONL 读写、追加、容错、自动建目录
- create_trajectory 从 Agent history 构造轨迹的正确性

运行：python -m unittest tests.test_trajectory -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.trajectory import TrajectoryStep, Trajectory, save_trajectory, load_trajectories, create_trajectory


class TestTrajectoryStep(unittest.TestCase):
    def test_default_values(self):
        step = TrajectoryStep(step=1, thought="thinking", action="FINISH", observation="done")
        self.assertEqual(step.llm_latency_ms, 0)
        self.assertEqual(step.tool_latency_ms, 0)
        self.assertFalse(step.parse_failed)

    def test_custom_values(self):
        step = TrajectoryStep(step=2, thought="t", action='{"name":"search"}', observation="ok",
                              llm_latency_ms=100, tool_latency_ms=200, parse_failed=True)
        self.assertEqual(step.step, 2)
        self.assertTrue(step.parse_failed)
        self.assertEqual(step.llm_latency_ms, 100)


class TestTrajectory(unittest.TestCase):
    def _make_traj(self, **overrides):
        defaults = dict(
            task_id="test_01", task="test task", session_id="s1",
            steps=[TrajectoryStep(step=1, thought="t", action="FINISH", observation="ok")],
            final_reward=1.0, metrics={"task_completed": True}, timestamp="2026-01-01T00:00:00",
        )
        defaults.update(overrides)
        return Trajectory(**defaults)

    def test_basic_construction(self):
        traj = self._make_traj()
        self.assertEqual(traj.task_id, "test_01")
        self.assertEqual(len(traj.steps), 1)
        self.assertEqual(traj.final_reward, 1.0)

    def test_default_optional_fields(self):
        traj = self._make_traj()
        self.assertEqual(traj.model, "")
        self.assertEqual(traj.termination_type, "")
        self.assertEqual(traj.reward_components, {})

    def test_reward_components_not_shared(self):
        """field(default_factory=dict) 不应跨实例共享"""
        t1 = self._make_traj()
        t2 = self._make_traj()
        t1.reward_components["x"] = 1
        self.assertNotIn("x", t2.reward_components)


class TestSaveLoadTrajectory(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.filepath = Path(self.tmpdir) / "test.jsonl"

    def _make_traj(self, task_id="t1", reward=1.5):
        return Trajectory(
            task_id=task_id, task="test task", session_id="s1",
            steps=[
                TrajectoryStep(step=1, thought="search", action='{"name":"search"}', observation="ok"),
                TrajectoryStep(step=2, thought="done", action="FINISH", observation="completed"),
            ],
            final_reward=reward, metrics={"task_completed": True},
            timestamp="2026-01-01T00:00:00", model="test-model",
            termination_type="FINISH", reward_components={"format": 1.0},
        )

    def test_save_and_load_single(self):
        traj = self._make_traj()
        save_trajectory(traj, self.filepath)
        loaded = load_trajectories(self.filepath)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].task_id, "t1")
        self.assertEqual(len(loaded[0].steps), 2)
        self.assertEqual(loaded[0].steps[0].thought, "search")
        self.assertEqual(loaded[0].final_reward, 1.5)
        self.assertEqual(loaded[0].model, "test-model")
        self.assertEqual(loaded[0].reward_components["format"], 1.0)

    def test_save_appends_to_existing(self):
        save_trajectory(self._make_traj(task_id="a"), self.filepath)
        save_trajectory(self._make_traj(task_id="b"), self.filepath)
        loaded = load_trajectories(self.filepath)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].task_id, "a")
        self.assertEqual(loaded[1].task_id, "b")

    def test_load_nonexistent_file_returns_empty(self):
        loaded = load_trajectories(Path("/nonexistent/path.jsonl"))
        self.assertEqual(loaded, [])

    def test_load_empty_file_returns_empty(self):
        self.filepath.touch()
        loaded = load_trajectories(self.filepath)
        self.assertEqual(loaded, [])

    def test_load_skips_blank_lines(self):
        with open(self.filepath, "w") as f:
            f.write("\n\n")
        save_trajectory(self._make_traj(), self.filepath)
        loaded = load_trajectories(self.filepath)
        self.assertEqual(len(loaded), 1)

    def test_creates_parent_directories(self):
        nested = Path(self.tmpdir) / "a" / "b" / "c" / "test.jsonl"
        save_trajectory(self._make_traj(), nested)
        self.assertTrue(nested.exists())
        loaded = load_trajectories(nested)
        self.assertEqual(len(loaded), 1)


class TestCreateTrajectory(unittest.TestCase):
    def test_basic_creation(self):
        history = [
            {"thought": "search", "action": '{"name":"search"}', "observation": "ok"},
            {"thought": "done", "action": "FINISH", "observation": "completed"},
        ]
        traj = create_trajectory(
            task_id="search_01", task="检索论文", session_id="s1",
            history=history, final_reward=1.5, metrics={"task_completed": True},
            model="gpt-4", termination_type="FINISH",
        )
        self.assertEqual(traj.task_id, "search_01")
        self.assertEqual(len(traj.steps), 2)
        self.assertEqual(traj.steps[0].step, 1)
        self.assertEqual(traj.steps[1].step, 2)
        self.assertEqual(traj.steps[0].thought, "search")
        self.assertNotEqual(traj.timestamp, "")

    def test_missing_optional_fields_default(self):
        history = [{"action": "FINISH", "observation": "done"}]
        traj = create_trajectory(
            task_id="t1", task="t", session_id="s1",
            history=history, final_reward=0.0, metrics={},
        )
        self.assertEqual(traj.steps[0].thought, "")
        self.assertEqual(traj.steps[0].llm_latency_ms, 0)
        self.assertFalse(traj.steps[0].parse_failed)
        self.assertEqual(traj.model, "")
        self.assertEqual(traj.termination_type, "FINISH")
        self.assertEqual(traj.reward_components, {})

    def test_empty_history(self):
        traj = create_trajectory(
            task_id="t1", task="t", session_id="s1",
            history=[], final_reward=-1.0, metrics={},
        )
        self.assertEqual(len(traj.steps), 0)

    def test_reward_components_stored(self):
        traj = create_trajectory(
            task_id="t1", task="t", session_id="s1",
            history=[{"thought": "x", "action": "FINISH", "observation": ""}],
            final_reward=1.0, metrics={},
            reward_components={"format": 1.0, "tool": 0.5},
        )
        self.assertEqual(traj.reward_components["format"], 1.0)
        self.assertEqual(traj.reward_components["tool"], 0.5)


if __name__ == "__main__":
    unittest.main()