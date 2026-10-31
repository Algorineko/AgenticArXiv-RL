import unittest

from AgenticArxiv.rsi.api_batch_environment import SameAPIBatchEnvironment
from AgenticArxiv.rsi.tool_environment import RegisteredToolEnvironment


class FixtureEnvironment:
    def __init__(self):
        self.calls = []

    def tools(self):
        return [{"type": "function", "function": {"name": "call_api", "description": "fixture",
                 "parameters": {"type": "object", "properties": {}}}}]

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        if arguments["arguments"].get("fail"):
            return {"error": "fixture failure"}
        return {"observation": arguments["arguments"].get("id")}

    def check(self):
        return {"success": False}

    def close(self):
        pass


class SameAPIBatchTests(unittest.TestCase):
    def setUp(self):
        self.raw = FixtureEnvironment()
        self.budget = SameAPIBatchEnvironment(self.raw, primitive_budget=3)
        self.env = RegisteredToolEnvironment(self.budget)

    @staticmethod
    def args(items):
        return {"app_name": "store", "api_name": "update_item", "arguments_list": items}

    def test_order_and_shared_budget_across_api_and_generic_batches(self):
        result = self.env.execute("batch_api_calls", self.args([{"id": 1}, {"id": 2}]))
        self.assertEqual(result, {"results": [{"observation": 1}, {"observation": 2}]})
        self.assertEqual(self.budget.primitive_attempts, 2)
        result = self.env.execute("batch_tools", {"calls": [
            {"name": "call_api", "arguments": {"app_name": "store", "api_name": "update_item",
                                                  "arguments": {"id": 3}}}]})
        self.assertEqual(result["results"][0]["result"], {"observation": 3})
        self.assertEqual(self.budget.primitive_attempts, 3)
        self.assertIn("error", self.env.execute("batch_api_calls", self.args([{"id": 4}])))
        self.assertEqual(len(self.raw.calls), 3)

    def test_bad_structure_and_oversized_or_overbudget_batches_do_not_dispatch(self):
        for args in (self.args([]), self.args([{}] * 17), self.args([{}] * 4),
                     self.args([{}, "not a mapping"]),
                     {"app_name": "bad.name", "api_name": "x", "arguments_list": [{}]}):
            self.assertIn("error", self.env.execute("batch_api_calls", args))
        self.assertEqual(self.raw.calls, [])
        self.assertEqual(self.budget.primitive_attempts, 0)

    def test_partial_error_keeps_prior_effect_and_stops(self):
        result = self.env.execute("batch_api_calls", self.args([{"id": 1}, {"fail": True}, {"id": 3}]))
        self.assertIn("error", result)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(len(self.raw.calls), 2)
        self.assertEqual(self.budget.primitive_attempts, 2)


if __name__ == "__main__":
    unittest.main()
