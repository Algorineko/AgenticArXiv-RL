"""Guard the conditional experiment against unintended API expenditure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class QueueGateTests(unittest.TestCase):
    def run_skip(self, complete, success_values):
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = root / "previous.json"
            previous.write_text(json.dumps({"complete": complete,
                                            "episodes": [{"success": value} for value in success_values]}))
            findings = root / "previous-findings.md"
            findings.write_text("Replay completed.\n")
            public = root / "public"
            private = root / "private"
            env = dict(os.environ, RSI_API_KEY="fixture-key")
            command = [sys.executable, str(repo / "scripts/queue_rsi_same_api_development.py"),
                       "--root", str(root), "--previous-summary", str(previous),
                       "--previous-findings", str(findings), "--probe-summary", str(root / "probe.json"),
                       "--private-pool", str(root / "private-pool.json"), "--public-pool", str(root / "public-pool.json"),
                       "--private-output", str(private), "--public-dir", str(public),
                       "--findings", str(root / "findings.md"), "--local-findings", str(root / "local.md"),
                       "--wait-seconds", "1"]
            subprocess.run(command, cwd=repo, env=env, check=True, capture_output=True)
            status = json.loads((public / "queue_status.json").read_text())
            self.assertEqual(status["state"], "skipped_preregistered_condition_not_met")
            self.assertFalse(private.exists())
            self.assertFalse((public / "summary.json").exists())

    def test_successful_baseline_skips_extra_development_calls(self):
        self.run_skip(True, [False, True, False, False])

    def test_incomplete_baseline_skips_extra_development_calls(self):
        self.run_skip(False, [False, False])


if __name__ == "__main__":
    unittest.main()
