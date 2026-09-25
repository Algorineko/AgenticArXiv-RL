#!/usr/bin/env python3
"""build_grpo_signal_split 的选任务逻辑测试（纯 CPU，无网络/模型）。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT.parent))

from scripts.build_grpo_signal_split import (  # noqa: E402
    build_document,
    informative_group_count,
    select_tasks,
)


def _entry(groups, informative, reward=0.5):
    fraction = (informative / groups) if groups else 0.0
    return {
        "sample_count": groups * 4,
        "group_count": groups,
        "mean_reward": reward,
        "min_reward": reward,
        "max_reward": reward,
        "mean_group_std": 0.1,
        "zero_std_fraction": 1 - fraction,
        "informative_group_fraction": fraction,
    }


class SelectTasksTest(unittest.TestCase):
    def test_buckets_by_informative_group_count(self):
        tasks = {
            "rich": _entry(8, 4),
            "borderline": _entry(8, 2),
            "single": _entry(8, 1),
            "ceiling": _entry(8, 0, reward=1.0),
        }
        buckets = select_tasks(
            tasks, ["rich", "borderline", "single", "ceiling"], 2
        )
        self.assertEqual(buckets["rl_train"], ["rich", "borderline"])
        self.assertEqual(buckets["dropped_low_signal"], ["single"])
        self.assertEqual(buckets["ceiling_control"], ["ceiling"])
        self.assertEqual(buckets["missing"], [])

    def test_missing_and_unsampled_tasks_are_reported(self):
        buckets = select_tasks({"a": _entry(0, 0)}, ["a", "b"], 2)
        self.assertEqual(buckets["missing"], ["a", "b"])
        self.assertEqual(buckets["rl_train"], [])

    def test_fraction_rounds_to_group_count(self):
        entry = _entry(8, 0)
        entry["informative_group_fraction"] = 0.25
        self.assertEqual(informative_group_count(entry), 2)


class BuildDocumentTest(unittest.TestCase):
    def test_summary_matches_selection(self):
        probe = {"tasks": {"a": _entry(8, 3), "b": _entry(8, 0, reward=1.0)}}
        with tempfile.TemporaryDirectory() as tmp:
            probe_path = Path(tmp) / "probe.json"
            probe_path.write_text(json.dumps(probe), encoding="utf-8")
            buckets = select_tasks(probe["tasks"], ["a", "b"], 2)
            document = build_document(
                probe_path=probe_path,
                probe=probe,
                source_ref="data/splits/x.json:train",
                model="outputs/m",
                num_generations=4,
                seed=43,
                min_informative_groups=2,
                buckets=buckets,
            )
        self.assertEqual(document["split"]["rl_train"], ["a"])
        summary = document["summary"]
        self.assertEqual(summary["selected_task_count"], 1)
        self.assertEqual(summary["selected_prompt_group_count"], 8)
        self.assertEqual(summary["selected_rollout_count"], 32)
        self.assertEqual(summary["informative_group_count"], 3)
        self.assertAlmostEqual(summary["informative_group_fraction"], 3 / 8)
        self.assertEqual(document["audit"]["a"]["informative_group_count"], 3)
        self.assertEqual(document["split"]["ceiling_control"], ["b"])


if __name__ == "__main__":
    unittest.main()
