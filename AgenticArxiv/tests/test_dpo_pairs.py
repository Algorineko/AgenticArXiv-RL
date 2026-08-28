#!/usr/bin/env python3
"""DPO 偏好对构造的回归测试。

不需要 LLM API、不需要网络、不需要 TRL —— 用合成轨迹直接验证配对逻辑。

配对口径在「多轮 RL 数据生成」那次重写里换过一次：原来是各取轨迹的首个工具
调用（`first_tool_action`）配成一对，现在是找两条轨迹在**相同前缀**下的第一个
分歧动作（`find_divergence_step`），prompt 也随之对齐到那一步之前的历史。
换句话说偏好信号从「整条轨迹谁好」收紧成了「同一状态下这一步该怎么走」，
复合任务里前几步相同、后面才走岔的情形因此才有对可配。

运行：
    cd AgenticArxiv && python tests/test_dpo_pairs.py
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_dpo_data import (  # noqa: E402
    _canonical_action_str,
    build_preference_pair,
    find_divergence_step,
)

SEARCH = '{"name": "get_recently_submitted_cs_papers", "args": {"aspect": "AI"}}'
DOWNLOAD = '{"name": "download_arxiv_pdf", "args": {"ref": 1}}'
TRANSLATE = '{"name": "translate_arxiv_pdf", "args": {"ref": 1}}'
TASK = {"id": "search_01", "task": "检索最近7天内人工智能(cs.AI)方向的论文"}
TOOLS = "TOOLS_DESCRIPTION_PLACEHOLDER"


def step(action, thought=""):
    return {"thought": thought, "action": action, "observation": "obs"}


def traj(*actions):
    """把动作序列包成 agent.run() 的返回结构。"""
    return {"history": [step(a) for a in actions]}


def rollout(reward, *actions):
    return {"reward": reward, "result": traj(*actions)}


def pair(rollouts, **kwargs):
    return build_preference_pair(rollouts, TASK, TOOLS, **kwargs)


class CanonicalActionTest(unittest.TestCase):
    """动作归一化决定了「两步算不算同一步」，也就决定了分歧点落在哪。"""

    def test_dict_and_json_string_are_the_same_action(self):
        self.assertEqual(
            _canonical_action_str({"name": "download_arxiv_pdf", "args": {"ref": 1}}),
            _canonical_action_str(DOWNLOAD),
        )

    def test_key_order_does_not_make_two_actions_differ(self):
        self.assertEqual(
            _canonical_action_str('{"args": {"ref": 1}, "name": "download_arxiv_pdf"}'),
            _canonical_action_str(DOWNLOAD),
        )

    def test_terminal_markers_pass_through(self):
        self.assertEqual(_canonical_action_str("  FINISH  "), "FINISH")


class FindDivergenceStepTest(unittest.TestCase):
    """分歧点：前缀相同、当前动作不同的第一步。"""

    def test_first_step_divergence(self):
        found = find_divergence_step([step(SEARCH), step("FINISH")],
                                     [step(DOWNLOAD), step("FINISH")])
        self.assertIsNotNone(found)
        index, chosen, rejected = found
        self.assertEqual(index, 0)
        self.assertEqual(chosen["action"], SEARCH)
        self.assertEqual(rejected["action"], DOWNLOAD)

    def test_divergence_after_a_shared_prefix(self):
        """复合任务的核心场景：都先搜索，第二步才走岔。

        旧口径各取首个工具调用，两边都是 SEARCH，判为相同、丢弃；
        真正有信息量的第 2 步决策拿不到任何监督信号。
        """
        found = find_divergence_step([step(SEARCH), step(DOWNLOAD)],
                                     [step(SEARCH), step(TRANSLATE)])
        self.assertIsNotNone(found)
        index, chosen, rejected = found
        self.assertEqual(index, 1)
        self.assertEqual(chosen["action"], DOWNLOAD)
        self.assertEqual(rejected["action"], TRANSLATE)

    def test_identical_trajectories_have_no_divergence(self):
        self.assertIsNone(find_divergence_step([step(SEARCH)], [step(SEARCH)]))

    def test_one_trajectory_being_a_prefix_diverges_against_finish(self):
        """短的那条等价于「到此为止」，拿 FINISH 补齐再比。"""
        found = find_divergence_step([step(SEARCH), step(DOWNLOAD)], [step(SEARCH)])
        self.assertIsNotNone(found)
        index, chosen, rejected = found
        self.assertEqual(index, 1)
        self.assertEqual(chosen["action"], DOWNLOAD)
        self.assertEqual(rejected["action"], "FINISH")

    def test_an_empty_action_is_not_a_usable_divergence(self):
        self.assertIsNone(find_divergence_step([step(SEARCH)], [step("")]))


class BuildPreferencePairTest(unittest.TestCase):
    def test_regression_last_step_is_always_finish(self):
        """回归：两条轨迹末步都是 FINISH，取 history[-1] 的实现会全部跳过。"""
        rollouts = [rollout(1.5, SEARCH, "FINISH"), rollout(0.5, DOWNLOAD, "FINISH")]

        last_actions = {r["result"]["history"][-1]["action"] for r in rollouts}
        self.assertEqual(last_actions, {"FINISH"})

        result = pair(rollouts)
        self.assertIsNotNone(result)
        self.assertEqual(result["divergence_step"], 0)
        self.assertIn(SEARCH, result["chosen"])
        self.assertIn(DOWNLOAD, result["rejected"])

    def test_picks_the_widest_reward_gap(self):
        rollouts = [rollout(0.5, DOWNLOAD, "FINISH"),
                    rollout(2.0, SEARCH, "FINISH"),
                    rollout(1.0, SEARCH, DOWNLOAD, "FINISH")]
        result = pair(rollouts)
        self.assertEqual(result["reward_gap"], 1.5)
        self.assertIn(SEARCH, result["chosen"])
        self.assertIn(DOWNLOAD, result["rejected"])

    def test_finds_a_distinct_action_when_the_extremes_match(self):
        rollouts = [rollout(2.0, SEARCH), rollout(1.0, DOWNLOAD), rollout(0.0, SEARCH)]
        result = pair(rollouts)
        self.assertIn(SEARCH, result["chosen"])
        self.assertIn(DOWNLOAD, result["rejected"])

    def test_prompt_is_aligned_to_the_shared_prefix(self):
        """prompt 必须停在分歧点之前：既要带上共同历史，又不能剧透该走哪步。"""
        rollouts = [rollout(2.0, SEARCH, DOWNLOAD), rollout(0.5, SEARCH, TRANSLATE)]
        result = pair(rollouts)

        self.assertEqual(result["divergence_step"], 1)
        self.assertIn(SEARCH, result["prompt"])          # 共同前缀在
        self.assertNotIn(DOWNLOAD, result["prompt"])     # 答案不在
        self.assertIn(TOOLS, result["prompt"])           # 与推理期同一套工具说明
        self.assertIn(TASK["task"], result["prompt"])

    def test_thought_travels_with_the_action(self):
        rollouts = [
            {"reward": 2.0, "result": {"history": [step(SEARCH, "先搜一下")]}},
            {"reward": 0.0, "result": {"history": [step(DOWNLOAD, "直接下载")]}},
        ]
        result = pair(rollouts)
        self.assertTrue(result["chosen"].startswith("Thought: 先搜一下\nAction: "))
        self.assertTrue(result["rejected"].startswith("Thought: 直接下载\nAction: "))

    def test_task_id_is_carried_through(self):
        result = pair([rollout(1.5, SEARCH, "FINISH"), rollout(0.5, DOWNLOAD, "FINISH")])
        self.assertEqual(result["task_id"], TASK["id"])

    def test_respects_minimum_reward_gap(self):
        rollouts = [rollout(1.01, SEARCH), rollout(1.0, DOWNLOAD)]
        self.assertIsNone(pair(rollouts, min_reward_gap=0.05))
        self.assertIsNotNone(pair(rollouts, min_reward_gap=0.0))

    def test_none_when_actions_identical(self):
        self.assertIsNone(pair([rollout(1.5, SEARCH, "FINISH"),
                                rollout(0.5, SEARCH, "FINISH")]))

    def test_none_when_rewards_tie(self):
        self.assertIsNone(pair([rollout(1.0, SEARCH, "FINISH"),
                                rollout(1.0, DOWNLOAD, "FINISH")]))

    def test_giving_up_immediately_is_a_valid_rejected_sample(self):
        """口径变更：一条只有 FINISH 的轨迹现在会配成对，而不是被丢掉。

        旧实现要求两边都有工具调用，「该动手却直接收工」这种典型失败因此
        进不了偏好数据。分歧点口径下它就是第 0 步 SEARCH vs FINISH。
        """
        result = pair([rollout(1.5, SEARCH, "FINISH"), rollout(0.0, "FINISH")])
        self.assertIsNotNone(result)
        self.assertIn(SEARCH, result["chosen"])
        self.assertEqual(result["rejected"], "Action: FINISH")

    def test_none_when_a_trajectory_has_no_history(self):
        self.assertIsNone(pair([rollout(1.5, SEARCH, "FINISH"),
                                {"reward": 0.0, "result": {"history": []}}]))

    def test_none_when_fewer_than_two_rollouts(self):
        self.assertIsNone(pair([rollout(1.5, SEARCH, "FINISH")]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
