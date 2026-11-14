"""Pure parts of the step-credit helper scripts (no torch, no transformers, no snapshot)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "AgenticArxiv"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark.task_io import load_tasks  # noqa: E402
from chain_length_report import chain_length_report  # noqa: E402
from check_completion_budget import (  # noqa: E402
    MODEL_OBSERVATION_CHARS,
    estimate_task_budget,
    recommend_budget,
)
from rl.reward import RewardCalculator  # noqa: E402
from step_entropy_probe import (  # noqa: E402
    attach_credit,
    rebuild_segments,
    split_assistant_turn,
    summarize_entropy,
)

TASKS = {t["id"]: t for t in load_tasks("expanded", None)}
CHAIN = TASKS["chain_ro5_dl_three"]


def call(name, args, obs):
    return {"thought": "t", "action": json.dumps({"name": name, "args": args}), "observation": obs,
            "model_observation": obs[:MODEL_OBSERVATION_CHARS]}


SEARCH = call("get_recently_submitted_cs_papers", {"aspect": "RO", "days": 7, "max_results": 5},
              "[{'id': '2601.00001v1'}]")
DL = {i: call("download_arxiv_pdf", {"ref": i}, f"{{'paper_id': '2601.0000{i}v1'}}") for i in (1, 2, 3)}
FIN = {"thought": "done", "action": "FINISH", "observation": "任务完成"}


class _FakeEnvironment:
    """记录调用；检索返回一段很长的结果，用来模拟截断。"""

    def __init__(self):
        self.calls = []

    def reset(self, **kwargs):
        self.calls.append(("reset", kwargs))

    def get_recently_submitted_cs_papers(self, **kwargs):
        self.calls.append(("search", kwargs))
        return [{"id": f"2601.0000{i}v1", "summary": "x" * 600} for i in range(1, 6)]

    def download_arxiv_pdf(self, **kwargs):
        self.calls.append(("download", kwargs))
        return {"paper_id": "2601.00001v1", "status": "READY"}


class CompletionBudgetTest(unittest.TestCase):
    def test_replays_gold_steps_and_counts_bounded_observations(self):
        env = _FakeEnvironment()
        row = estimate_task_budget(CHAIN, env, len, TASKS)
        self.assertEqual(row["chain_length"], 4)
        self.assertEqual([turn["tool"] for turn in row["turns"]],
                         CHAIN["expected_tools"] + ["FINISH"])
        # 检索结果被截到 1000 字符，再加前后缀
        self.assertEqual(row["turns"][0]["observation_tokens"],
                         len("\nObservation: ") + MODEL_OBSERVATION_CHARS + len("\nThought:"))
        self.assertEqual(row["total_tokens"],
                         sum(t["policy_tokens"] + t["observation_tokens"] for t in row["turns"]))
        self.assertGreater(row["first_turn_tokens"], 1000)
        self.assertEqual([c[0] for c in env.calls].count("download"), 3)

    def test_recommendation_rounds_up_with_margin(self):
        self.assertEqual(recommend_budget([700, 800]), 1024)
        self.assertEqual(recommend_budget([]), 0)


class ChainLengthReportTest(unittest.TestCase):
    def test_counts_by_chain_length_and_first_error(self):
        records = [
            {"task_id": "chain_ro5_dl_three", "history": [SEARCH, DL[1], DL[2], DL[3], FIN]},
            {"task_id": "chain_ro5_dl_three", "history": [SEARCH, DL[1], FIN]},
            {"task_id": "not_a_task", "history": []},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traces.jsonl"
            path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
            report = chain_length_report([str(path)])
        self.assertEqual(report["by_length"][4], [1, 2])
        self.assertEqual(report["failures"][(4, "early_finish", 3)], 1)
        self.assertEqual(report["skipped"], 1)


class StepEntropyProbeTest(unittest.TestCase):
    def test_split_assistant_turn(self):
        thought, action = split_assistant_turn('Thought: go\nAction: {"name": "x"}\nObservation: fake')
        self.assertEqual(thought, "Thought: go\nAction:")
        self.assertEqual(action, ' {"name": "x"}\n')
        self.assertEqual(split_assistant_turn("no action here"), ("no action here", ""))

    def test_rebuild_segments_follows_rollout_order(self):
        trajectory = {
            "raw_assistant_turns": ['Thought: a\nAction: {"name": "download_arxiv_pdf"}', "Thought: b\nAction: FINISH"],
            "history": [DL[1], FIN],
        }
        segments = rebuild_segments(trajectory)
        self.assertEqual([(i, role) for i, role, _ in segments],
                         [(0, "thought"), (0, "action"), (0, "observation"), (1, "thought"), (1, "action")])
        self.assertTrue(segments[2][2].startswith("\nObservation: "))
        self.assertTrue(segments[2][2].endswith("\nThought:"))

    def test_attach_credit_and_summary(self):
        calc = RewardCalculator(curriculum_steps=0)
        histories = [[SEARCH, FIN], [SEARCH, DL[1], FIN], [SEARCH, FIN], [SEARCH, FIN]]
        records = []
        for g, history in enumerate(histories):
            reward = calc.compute_reward_breakdown(CHAIN, {"history": history})[0].total
            records.append({"batch_index": 0, "group_index": 0, "generation_index": g,
                            "task_id": "chain_ro5_dl_three", "reward": reward,
                            "trajectory": {"history": history}})
        entropies = {(0, 0, g): {t: {"thought": 1.0, "action": 0.1 * (t + 1) + g}
                                 for t in range(len(h))} for g, h in enumerate(histories)}
        rows = attach_credit(records, entropies, TASKS)
        self.assertEqual(len(rows), sum(len(h) for h in histories))
        first = [r for r in rows if r["is_first_error"]]
        self.assertEqual(len(first), 4)                      # 每条轨迹都提前 FINISH
        summary = summarize_entropy(rows)
        # 检索之后的状态有两种备选（FINISH 与下载），三条轨迹在这里出错；
        # 「下 1 篇后 FINISH」所在节点只有它一条，没有备选。
        self.assertEqual(summary["first_error_with_alternatives"], 3)
        self.assertEqual(summary["first_error_without_alternatives"], 1)
        self.assertEqual(summary["first_error_kinds"], {"early_finish": 4})


if __name__ == "__main__":
    unittest.main()
