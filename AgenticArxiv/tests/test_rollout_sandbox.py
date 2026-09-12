"""Rollout sandbox isolation tests (CPU-only, no network or model required)."""

import tempfile
import unittest
from pathlib import Path

from rl.sandbox import RolloutSandbox


class _Component:
    def __init__(self):
        self.state = {"papers": [], "active": None}

    def capture_state(self):
        return dict(self.state)

    def restore_state(self, state):
        self.state = dict(state)


class RolloutSandboxTest(unittest.TestCase):
    def test_reset_restores_component_and_explicit_artifact_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.txt"
            baseline.write_text("before", encoding="utf-8")
            component = _Component()
            sandbox = RolloutSandbox(component, file_roots=(root,))

            component.state["papers"].append("paper-1")
            component.state["active"] = "paper-1"
            baseline.write_text("mutated", encoding="utf-8")
            (root / "leaked.txt").write_text("should disappear", encoding="utf-8")

            sandbox.reset()

            self.assertEqual(component.state, {"papers": [], "active": None})
            self.assertEqual(baseline.read_text(encoding="utf-8"), "before")
            self.assertFalse((root / "leaked.txt").exists())

    def test_repeated_reset_is_idempotent(self):
        component = _Component()
        sandbox = RolloutSandbox(component)
        component.state["active"] = "x"
        sandbox.reset()
        sandbox.reset()
        self.assertEqual(component.state, {"papers": [], "active": None})


if __name__ == "__main__":
    unittest.main()
