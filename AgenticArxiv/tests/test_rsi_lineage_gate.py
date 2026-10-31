"""Prevent live lineage calls on failed, saturated or unaudited baselines."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from queue_rsi_lineage_connectivity import eligible


class LineageGateTests(unittest.TestCase):
    def setUp(self):
        self.status = {"state": "complete", "requested_model": "fixture"}
        self.summary = {"complete": True, "episodes": [
            {"task_id_sha256": "first", "success": False}, {"task_id_sha256": "second", "success": True}],
            "protocol": {"task_id_sha256": ["first", "second"], "model": "fixture",
                         "training_pool": {"selection": "Frozen meta_train prefix, inner_train only"}}}
        self.replay = {"all_observations_match": True, "all_goal_verdicts_match": True,
                       "source_digests_match": {"proxy": True}}

    def test_single_success_and_matching_replay_is_eligible(self):
        self.assertTrue(eligible(self.status, self.summary, self.replay))

    def test_floor_and_saturation_do_not_spend_model_budget(self):
        for values in ((False, False), (True, True)):
            summary = deepcopy(self.summary)
            for row, value in zip(summary["episodes"], values):
                row["success"] = value
            self.assertFalse(eligible(self.status, summary, self.replay))

    def test_missing_audit_or_model_or_task_mismatch_is_ineligible(self):
        for modification in ("missing_sources", "mismatch", "model", "task"):
            summary, replay = deepcopy(self.summary), deepcopy(self.replay)
            if modification == "missing_sources": replay["source_digests_match"] = {}
            if modification == "mismatch": replay["all_goal_verdicts_match"] = False
            if modification == "model": summary["protocol"]["model"] = "different"
            if modification == "task": summary["episodes"][0]["task_id_sha256"] = "outside"
            self.assertFalse(eligible(self.status, summary, replay))


if __name__ == "__main__":
    unittest.main()
