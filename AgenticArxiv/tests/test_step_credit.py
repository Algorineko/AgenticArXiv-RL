"""Prefix-tree step credit: conservation, gating, labels and anti-hacking checks.

Pure CPU tests: no torch, no TRL, no snapshot.
"""

import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import patch  # noqa: E402

from benchmark.task_io import load_tasks  # noqa: E402
from rl.grpo_reward import make_grpo_reward_fn, make_multiturn_rollout_func  # noqa: E402
from rl.reward import RewardCalculator  # noqa: E402
from rl.step_credit import (  # noqa: E402
    CORRECT,
    OFF_PREFIX,
    WRONG,
    StepCreditCollector,
    align_step_advantages,
    canonical_action_key,
    compute_batch_step_credit,
    compute_group_step_credit,
    label_trajectory,
    prefix_tree_deltas,
    summarize,
    token_advantages,
)

TASKS = {t["id"]: t for t in load_tasks("expanded", None)}
CHAIN = TASKS["chain_ro5_dl_three"]          # 检索 → 下 1 → 下 2 → 下 3
CALC = RewardCalculator(curriculum_steps=0)


def call(name, args, obs, thought="t"):
    return {"thought": thought, "action": json.dumps({"name": name, "args": args}), "observation": obs}


SEARCH = call("get_recently_submitted_cs_papers", {"aspect": "RO", "days": 7, "max_results": 5},
              "[{'id': '2601.00001v1'}]")
DL = {i: call("download_arxiv_pdf", {"ref": i}, f"{{'paper_id': '2601.0000{i}v1', 'status': 'READY'}}")
      for i in (1, 2, 3, 4)}
FAIL = call("download_arxiv_pdf", {"ref": 9}, "工具执行失败: 索引越界")
FIN = {"thought": "done", "action": "FINISH", "observation": "任务完成"}


def rewards_of(task, histories):
    return [CALC.compute_reward_breakdown(task, {"history": h})[0].total for h in histories]


class PrefixTreeTest(unittest.TestCase):
    def test_deltas_conserve_to_value_minus_mean(self):
        rng = random.Random(0)
        alphabet = ["a", "b", "c"]
        for _ in range(200):
            seqs = [[rng.choice(alphabet) for _ in range(rng.randint(1, 5))] for _ in range(4)]
            values = [rng.uniform(-1, 1) for _ in range(4)]
            mean = sum(values) / 4
            for row, value in zip(prefix_tree_deltas(seqs, values), values):
                self.assertAlmostEqual(sum(row), value - mean, places=9)

    def test_same_action_same_state_gets_same_credit_before_the_end(self):
        seqs = [["s", "d1", "F"], ["s", "d1", "d2", "F"], ["s", "F"], ["s", "d1", "d2", "F"]]
        deltas = prefix_tree_deltas(seqs, [-0.25, 1.0, -0.25, 1.0])
        self.assertAlmostEqual(deltas[0][1], deltas[1][1])   # 两条轨迹在 s 之后都做了 d1
        self.assertAlmostEqual(deltas[1][1], deltas[3][1])


class GroupCreditTest(unittest.TestCase):
    def test_reward_mode_conserves_grpo_advantage(self):
        histories = [[SEARCH, FIN], [SEARCH, DL[1], DL[2], FIN],
                     [SEARCH, DL[1], DL[2], DL[3], FIN], [SEARCH, DL[1], DL[2], DL[3], FIN]]
        group = compute_group_step_credit(CHAIN, histories, rewards_of(CHAIN, histories))
        self.assertEqual(group.mode, "reward")
        for row, a_traj in zip(group.steps, group.trajectory_advantages):
            self.assertAlmostEqual(sum(s.advantage for s in row), a_traj, places=6)
        # 早停那条的过错落在 FINISH 上，它前面做对的下载被鼓励
        b = group.steps[1]
        self.assertGreater(b[1].advantage, 0)
        self.assertLess(b[3].advantage, 0)
        self.assertEqual([s.label for s in b], [CORRECT, CORRECT, CORRECT, WRONG])

    def test_progress_only_when_reward_is_flat(self):
        histories = [[SEARCH, FIN], [SEARCH, DL[1], FIN], [SEARCH, DL[1], DL[2], FIN], [SEARCH, FIN]]
        rewards = rewards_of(CHAIN, histories)
        self.assertEqual(len(set(rewards)), 1)                      # 全部被封顶压到 -0.25
        group = compute_group_step_credit(CHAIN, histories, rewards)
        self.assertEqual(group.mode, "progress")
        self.assertEqual(group.trajectory_advantages, [0.0] * 4)
        self.assertLess(group.steps[0][1].advantage, 0)             # 检索后就停
        self.assertGreater(group.steps[1][1].advantage, 0)          # 继续下载

        off = compute_group_step_credit(CHAIN, histories, rewards, use_progress=False)
        self.assertEqual(off.mode, "none")
        self.assertEqual(off.category, "plateau_recoverable")

    def test_progress_never_runs_when_rewards_differ(self):
        histories = [[SEARCH, FIN], [SEARCH, FAIL, FIN], [SEARCH, DL[1], FIN], [SEARCH, FIN]]
        group = compute_group_step_credit(CHAIN, histories, rewards_of(CHAIN, histories))
        self.assertEqual(group.mode, "reward")

    def test_identical_group_is_reported_as_identical(self):
        histories = [[SEARCH, FIN]] * 4
        group = compute_group_step_credit(CHAIN, histories, rewards_of(CHAIN, histories))
        self.assertEqual((group.mode, group.category), ("none", "identical"))


class LabelTest(unittest.TestCase):
    def test_repeating_a_correct_step_freezes_progress(self):
        once = label_trajectory(CHAIN, [SEARCH, DL[1], FIN])
        twice = label_trajectory(CHAIN, [SEARCH, DL[1], DL[1], DL[2], DL[3], FIN])
        self.assertEqual(twice.progress_count, once.progress_count)
        self.assertEqual(twice.first_error_kind, "wrong_args")
        self.assertEqual(twice.labels[3:], [OFF_PREFIX, OFF_PREFIX, OFF_PREFIX])

    def test_finishing_beats_not_finishing(self):
        done = label_trajectory(CHAIN, [SEARCH, DL[1], DL[2], DL[3], FIN])
        truncated = label_trajectory(CHAIN, [SEARCH, DL[1], DL[2], DL[3]])
        extra = label_trajectory(CHAIN, [SEARCH, DL[1], DL[2], DL[3], DL[4], FIN])
        self.assertEqual(done.phi, 1.0)
        self.assertLess(truncated.phi, done.phi)
        self.assertEqual(truncated.first_error_kind, "unfinished")
        self.assertLess(extra.phi, done.phi)
        self.assertEqual(extra.first_error_kind, "extra_call")

    def test_tool_failure_is_not_progress(self):
        lab = label_trajectory(CHAIN, [SEARCH, FAIL, FIN])
        self.assertEqual((lab.progress_count, lab.first_error_kind), (1, "tool_fail"))

    def test_blocked_task_needs_explained_finish(self):
        task = TASKS["infeasible_unknown_id"]
        good = {"thought": "快照中不存在指定的 arXiv 论文 ID，因此找不到该论文，不能下载",
                "action": "FINISH", "observation": "任务完成"}
        bad = {"thought": "任务已完成", "action": "FINISH", "observation": "任务完成"}
        self.assertEqual(label_trajectory(task, [good]).phi, 1.0)
        self.assertEqual(label_trajectory(task, [bad]).phi, 0.0)


class CanonicalKeyTest(unittest.TestCase):
    def test_thought_and_aspect_spelling_do_not_change_the_key(self):
        a = call("get_recently_submitted_cs_papers", {"aspect": "cs.ro", "days": 7}, "[]", thought="x")
        b = call("get_recently_submitted_cs_papers", {"aspect": "RO", "days": 7}, "[]", thought="y")
        self.assertEqual(canonical_action_key(a), canonical_action_key(b))

    def test_refs_resolving_to_the_same_paper_share_a_key(self):
        by_index = call("download_arxiv_pdf", {"ref": 1}, "{'paper_id': '2601.00001v1'}")
        by_id = call("download_arxiv_pdf", {"ref": "2601.00001"}, "{'paper_id': '2601.00001v2'}")
        self.assertEqual(canonical_action_key(by_index), canonical_action_key(by_id))

    def test_terminal_and_parse_errors(self):
        self.assertEqual(canonical_action_key(FIN), "FINISH")
        self.assertEqual(canonical_action_key({"action": '{"name": "FINISH", "args": {}}'}), "FINISH")
        self.assertEqual(canonical_action_key({"action": "PARSE_ERROR", "parse_failed": True}), "PARSE_ERROR")


class BatchAndTokenTest(unittest.TestCase):
    def test_batch_rejects_mixed_groups(self):
        trajs = [{"history": [SEARCH, FIN]}] * 4
        with self.assertRaises(ValueError):
            compute_batch_step_credit(["chain_ro5_dl_three"] * 3 + ["search_01"], trajs, [0.0] * 4, TASKS, 4)

    def test_zero_weight_reproduces_grpo(self):
        out = token_advantages(0.7, [1.0, -2.0], [(0, 3), (5, 8)], 10, weight=0.0)
        self.assertEqual(out, [0.7] * 10)

    def test_spans_receive_step_advantage(self):
        out = token_advantages(0.5, [1.0, -1.0], [(0, 2), (4, 6)], 7, weight=1.0)
        self.assertEqual(out, [1.0, 1.0, 0.5, 0.5, -1.0, -1.0, 0.5])

    def test_collector_round_trip_and_summary(self):
        histories = [[SEARCH, FIN], [SEARCH, DL[1], FIN], [SEARCH, DL[1], DL[2], FIN], [SEARCH, FIN]]
        collector = StepCreditCollector(TASKS, num_generations=4)
        collector.record_batch(["chain_ro5_dl_three"] * 4, [{"history": h} for h in histories],
                               rewards_of(CHAIN, histories))
        groups, spans = collector.pop()
        self.assertEqual(spans, [[], [], [], []])
        report = summarize(groups)
        self.assertEqual(report["informative_fraction_grpo"], 0.0)
        self.assertEqual(report["informative_fraction_with_progress"], 1.0)
        self.assertEqual(report["credit_accuracy"], 1.0)
        with self.assertRaises(RuntimeError):
            collector.pop()

    def test_alignment_follows_trainer_scale(self):
        histories = [[SEARCH, FIN], [SEARCH, DL[1], DL[2], FIN],
                     [SEARCH, DL[1], DL[2], DL[3], FIN], [SEARCH, DL[1], DL[2], DL[3], FIN]]
        rewards = rewards_of(CHAIN, histories)
        group = compute_group_step_credit(CHAIN, histories, rewards)
        mean = sum(rewards) / 4
        unscaled = [r - mean for r in rewards]          # 例如 Dr.GRPO 的 scale_rewards=none
        aligned, error = align_step_advantages([group], unscaled, 4)
        self.assertLess(error, 1e-9)
        for row, target in zip(aligned, unscaled):
            self.assertAlmostEqual(sum(row), target, places=9)


class _CharTokenizer:
    """字符级假分词器，与 test_grpo_full_observation 同一写法。"""

    def apply_chat_template(self, prompt, **kwargs):
        return self.encode("Task")

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}

    @staticmethod
    def encode(text):
        return [ord(ch) for ch in text]

    @staticmethod
    def decode(ids, **kwargs):
        return "".join(chr(t) for t in ids)


class _ScriptedTrainer:
    processing_class = _CharTokenizer()

    def __init__(self, turns, budget=5000):
        self.max_completion_length = budget
        self.turns = list(turns)
        self.turn = 0

    def _generate_single_turn(self, prompts, *args):
        text = self.turns[self.turn]
        self.turn += 1
        return [self.processing_class.encode(text) for _ in prompts], None, None


class RolloutSpanTest(unittest.TestCase):
    def _rollout(self, budget=5000):
        class Environment:
            def reset(self, **kwargs):
                pass

            def download_arxiv_pdf(self, **kwargs):
                return {"paper_id": "2601.00001v1", "status": "READY"}

        turns = [
            "Thought: get it\nAction: " + json.dumps({"name": "download_arxiv_pdf", "args": {"ref": 1}}),
            "Thought: done\nAction: FINISH",
        ]
        trainer = _ScriptedTrainer(turns, budget)
        with patch("rl.grpo_reward.require_rollout_func_support"):
            rollout = make_multiturn_rollout_func(Environment, max_turns=3)
            output = rollout([[{"role": "user", "content": "Download"}]], trainer)
        return output

    def test_spans_cover_exactly_the_policy_tokens(self):
        output = self._rollout()
        trajectory = output["trajectory_results"][0]
        spans = trajectory["turn_token_spans"]
        self.assertEqual(len(spans), len(trajectory["history"]))
        covered = [0] * len(output["completion_ids"][0])
        for start, end in spans:
            for j in range(start, end):
                covered[j] = 1
        self.assertEqual(covered, output["env_mask"][0])

    def test_reward_fn_feeds_the_collector(self):
        output = self._rollout()
        trajectory = output["trajectory_results"][0]
        task = {"id": "dl", "expected_tools": ["download_arxiv_pdf"], "expected_tool_args": [{"ref": 1}]}
        collector = StepCreditCollector({"dl": task}, num_generations=2)
        reward_fn = make_grpo_reward_fn({"dl": task}, step_credit=collector)
        rewards = reward_fn(completions=["c", "c"], task_id=["dl", "dl"],
                            trajectory_results=[trajectory, trajectory])
        groups, spans = collector.pop()
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].category, "identical")
        self.assertEqual(spans[0], [tuple(s) for s in trajectory["turn_token_spans"]])
        self.assertEqual(len(rewards), 2)

    def test_collector_does_not_change_rewards(self):
        # 默认路径不变的保证：挂上收集器前后，返回给 TRL 的标量奖励逐位相同。
        output = self._rollout()
        trajectory = output["trajectory_results"][0]
        task = {"id": "dl", "expected_tools": ["download_arxiv_pdf"], "expected_tool_args": [{"ref": 1}]}
        kwargs = dict(completions=["c", "c"], task_id=["dl", "dl"],
                      trajectory_results=[trajectory, trajectory])
        plain = make_grpo_reward_fn({"dl": task})(**kwargs)
        collector = StepCreditCollector({"dl": task}, num_generations=2)
        hooked = make_grpo_reward_fn({"dl": task}, step_credit=collector)(**kwargs)
        self.assertEqual(plain, hooked)


if __name__ == "__main__":
    unittest.main()
