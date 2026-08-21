#!/usr/bin/env python3
"""DPO 偏好对构造的回归测试。

不需要 LLM API、不需要网络、不需要 TRL —— 用合成轨迹直接验证配对逻辑。

运行：
    cd AgenticArxiv && python tests/test_dpo_pairs.py
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_dpo_data import build_preference_pair, first_tool_action  # noqa: E402

SEARCH = '{"name": "get_recently_submitted_cs_papers", "args": {"aspect": "AI"}}'
DOWNLOAD = '{"name": "download_arxiv_pdf", "args": {"ref": 1}}'
TASK = {"id": "search_01", "task": "检索最近7天内人工智能(cs.AI)方向的论文"}


def traj(*actions):
    """把动作序列包成 agent.run() 的返回结构。"""
    return {"history": [{"thought": "", "action": a, "observation": ""} for a in actions]}


def rollout(reward, *actions):
    return {"reward": reward, "result": traj(*actions)}


class TestFirstToolAction(unittest.TestCase):
    def test_skips_terminal_markers(self):
        self.assertEqual(first_tool_action(traj(SEARCH, "FINISH")), SEARCH)

    def test_returns_first_of_several(self):
        self.assertEqual(first_tool_action(traj(SEARCH, DOWNLOAD, "FINISH")), SEARCH)

    def test_empty_when_no_tool_call(self):
        self.assertEqual(first_tool_action(traj("FINISH")), "")
        self.assertEqual(first_tool_action(traj("FORCE_STOP")), "")
        self.assertEqual(first_tool_action(traj("ERROR")), "")
        self.assertEqual(first_tool_action({"history": []}), "")


class TestBuildPreferencePair(unittest.TestCase):
    def test_regression_last_step_is_always_finish(self):
        """回归：两条轨迹末步都是 FINISH，旧实现会因 chosen==rejected 全部跳过。"""
        rollouts = [rollout(1.5, SEARCH, "FINISH"), rollout(0.5, DOWNLOAD, "FINISH")]

        # 旧实现取 history[-1]，两边都是 "FINISH"
        last_actions = {r["result"]["history"][-1]["action"] for r in rollouts}
        self.assertEqual(last_actions, {"FINISH"})

        # 新实现取首个工具调用，能构造出有效偏好对
        pair = build_preference_pair(rollouts, TASK)
        self.assertIsNotNone(pair)
        self.assertEqual(pair["chosen"], SEARCH)
        self.assertEqual(pair["rejected"], DOWNLOAD)
        self.assertEqual(pair["prompt"], TASK["task"])

    def test_picks_extremes_not_adjacent(self):
        rollouts = [rollout(0.5, DOWNLOAD, "FINISH"),
                    rollout(2.0, SEARCH, "FINISH"),
                    rollout(1.0, SEARCH, DOWNLOAD, "FINISH")]
        pair = build_preference_pair(rollouts, TASK)
        self.assertEqual(pair["chosen"], SEARCH)
        self.assertEqual(pair["rejected"], DOWNLOAD)

    def test_finds_distinct_middle_action_when_extremes_match(self):
        rollouts = [rollout(2.0, SEARCH), rollout(1.0, DOWNLOAD), rollout(0.0, SEARCH)]
        pair = build_preference_pair(rollouts, TASK)
        self.assertEqual(pair["chosen"], SEARCH)
        self.assertEqual(pair["rejected"], DOWNLOAD)

    def test_respects_minimum_reward_gap(self):
        rollouts = [rollout(1.01, SEARCH), rollout(1.0, DOWNLOAD)]
        self.assertIsNone(build_preference_pair(rollouts, TASK, min_reward_gap=0.05))

    def test_none_when_actions_identical(self):
        rollouts = [rollout(1.5, SEARCH, "FINISH"), rollout(0.5, SEARCH, "FINISH")]
        self.assertIsNone(build_preference_pair(rollouts, TASK))

    def test_none_when_rewards_tie(self):
        rollouts = [rollout(1.0, SEARCH, "FINISH"), rollout(1.0, DOWNLOAD, "FINISH")]
        self.assertIsNone(build_preference_pair(rollouts, TASK))

    def test_none_when_a_trajectory_has_no_tool_call(self):
        rollouts = [rollout(1.5, SEARCH, "FINISH"), rollout(0.0, "FINISH")]
        self.assertIsNone(build_preference_pair(rollouts, TASK))

    def test_none_when_fewer_than_two_rollouts(self):
        self.assertIsNone(build_preference_pair([rollout(1.5, SEARCH, "FINISH")], TASK))


if __name__ == "__main__":
    unittest.main(verbosity=2)
