"""Prevent GPU task runs without a verified neutral model probe."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from rsi_qwen_appworld_probe import neutral_ready
from queue_rsi_a40_local_calibration import eligible


class LocalCalibrationGateTests(unittest.TestCase):
    def setUp(self):
        self.status = {"state": "complete", "checks": {"source": True, "gpu": True}}
        self.probe = {"native_tool_call_valid": True}
        self.manifest = {"model_id": "Qwen/Qwen3-8B"}

    def test_verified_neutral_probe_is_eligible(self):
        self.assertTrue(neutral_ready(self.status, self.probe, self.manifest))

    def test_missing_checks_or_bad_tool_output_never_launch_gpu_tasks(self):
        for case in ("missing", "failed", "format", "model"):
            status, probe, manifest = map(deepcopy, (self.status, self.probe, self.manifest))
            if case == "missing": status["checks"] = {}
            if case == "failed": status["checks"]["gpu"] = False
            if case == "format": probe["native_tool_call_valid"] = False
            if case == "model": manifest["model_id"] = "other"
            self.assertFalse(neutral_ready(status, probe, manifest))

    def test_gpu_task_run_requires_complete_replayed_zero_success_pair(self):
        previous = {"complete": True, "episodes": [{"success": False}, {"success": False}]}
        replay = {"all_observations_match": True, "all_goal_verdicts_match": True,
                  "source_digests_match": {"proxy": True}}
        self.assertTrue(eligible(previous, replay, self.status, self.probe, self.manifest))
        for condition in ("partial", "success", "replay", "no_sources"):
            case, audit = deepcopy(previous), deepcopy(replay)
            if condition == "partial": case["complete"] = False
            if condition == "success": case["episodes"][0]["success"] = True
            if condition == "replay": audit["all_observations_match"] = False
            if condition == "no_sources": audit["source_digests_match"] = {}
            self.assertFalse(eligible(case, audit, self.status, self.probe, self.manifest))


if __name__ == "__main__":
    unittest.main()
