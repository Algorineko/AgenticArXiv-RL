"""State and budget invariants for the shared fixed tool interface."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from rsi_cpu_tasks import Environment, catalog
from AgenticArxiv.rsi.batch_environment import BoundedBatchEnvironment
from AgenticArxiv.rsi.tool_environment import RegisteredToolEnvironment


class BatchEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.task = catalog()[0]
        self.raw = Environment(self.task)
        self.batch = BoundedBatchEnvironment(self.raw, primitive_budget=5)
        self.env = RegisteredToolEnvironment(self.batch)

    def tearDown(self):
        self.env.close()

    @staticmethod
    def reserve(sku, n):
        return {"name": "reserve_item", "arguments": {"sku": sku, "quantity": n}}

    def test_batch_and_serial_have_identical_observations_and_goal(self):
        direct = Environment(self.task)
        try:
            calls = [self.reserve(k, v) for k, v in self.task["lines"].items()]
            expected = [{"name": c["name"], "result": direct.execute(c["name"], c["arguments"])} for c in calls]
            self.assertEqual(self.env.execute("batch_tools", {"calls": calls}), {"results": expected})
            self.assertEqual(self.env.execute("dispatch_order", {}), direct.execute("dispatch_order", {}))
            self.assertTrue(self.env.check()["success"])
            self.assertEqual(self.env.check(), direct.check())
            self.assertEqual(self.batch.primitive_attempts, 3)
        finally:
            direct.close()

    def test_budget_rejects_whole_batch_without_side_effects(self):
        self.batch.primitive_budget = 1
        calls = [self.reserve(k, v) for k, v in self.task["lines"].items()]
        self.assertIn("error", self.env.execute("batch_tools", {"calls": calls}))
        self.assertEqual(self.batch.primitive_attempts, 0)
        self.assertEqual(self.raw.execute("inspect_order", {})["reserved"], {})

    def test_no_rollback_and_no_calls_after_first_error(self):
        calls = [self.reserve("item_A", 1), self.reserve("unknown", 1), self.reserve("item_B", 1)]
        result = self.env.execute("batch_tools", {"calls": calls})
        self.assertIn("error", result)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(self.raw.execute("inspect_order", {})["reserved"], {"item_A": 1})
        self.assertEqual(self.batch.primitive_attempts, 2)

    def test_invalid_structure_cannot_dispatch_and_nested_batch_is_rejected(self):
        for calls in ([], [self.reserve("item_A", 1)] * 17,
                      [{"name": "batch_tools", "arguments": {"calls": []}}],
                      [self.reserve("item_A", 1), {"name": "unknown", "arguments": {}}]):
            self.assertIn("error", self.env.execute("batch_tools", {"calls": calls}))
        self.assertEqual(self.batch.primitive_attempts, 0)
        self.assertEqual(self.raw.execute("inspect_order", {})["reserved"], {})

    def test_direct_calls_cannot_escape_shared_budget(self):
        self.batch.primitive_budget = 1
        self.assertNotIn("error", self.env.execute("inspect_order", {}))
        self.assertIn("error", self.env.execute("reserve_item", {"sku": "item_A", "quantity": 1}))
        self.assertEqual(self.batch.primitive_attempts, 1)
        self.assertEqual(self.raw.execute("inspect_order", {})["reserved"], {})


if __name__ == "__main__":
    unittest.main()
