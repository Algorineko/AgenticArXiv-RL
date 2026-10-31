"""Test the experimental distinction between shared-history control arms."""
import json
import unittest

from AgenticArxiv.rsi.improver import make_request, propose, text_digest


class FakeAPI:
    def __init__(self, text="candidate", finish="tool_calls", count=1):
        self.text, self.finish, self.count = text, finish, count
        self.calls = 0

    def chat(self, messages, tools, identifier):
        self.calls += 1
        return {"choices": [{"finish_reason": self.finish, "message": {"tool_calls": [
            {"function": {"name": "submit_revision", "arguments": json.dumps({"text": self.text})}}
            for _ in range(self.count)]}}]}


class ImproverControlTests(unittest.TestCase):
    def request(self, arm, target="improver", active="evolved"):
        return make_request(arm, target, "solver", active, "seed", [{"training_success": False}])

    def test_meta_update_uses_active_policy_only_for_recursive_arm(self):
        fixed, recursive = self.request("fixed_outer"), self.request("recursive")
        self.assertEqual(fixed.messages[1], recursive.messages[1])
        self.assertEqual(fixed.metadata["experience_sha256"], recursive.metadata["experience_sha256"])
        self.assertEqual(fixed.metadata["proposer_policy_sha256"], text_digest("seed"))
        self.assertEqual(recursive.metadata["proposer_policy_sha256"], text_digest("evolved"))

    def test_first_meta_generation_is_identical_at_same_start(self):
        fixed = self.request("fixed_outer", active="seed")
        recursive = self.request("recursive", active="seed")
        self.assertEqual(fixed.messages, recursive.messages)
        self.assertEqual(fixed.metadata["request_content_sha256"], recursive.metadata["request_content_sha256"])

    def test_outer_and_recursive_both_use_learned_improver_for_solver(self):
        fixed = self.request("fixed_outer", target="solver")
        recursive = self.request("recursive", target="solver")
        self.assertEqual(fixed.messages, recursive.messages)
        experience = self.request("fixed_experience", target="solver")
        self.assertEqual(experience.metadata["proposer_policy_sha256"], text_digest("seed"))
        with self.assertRaises(ValueError):
            self.request("fixed_experience", target="improver")

    def test_validity_is_not_acceptance_and_noop_is_recorded(self):
        api = FakeAPI(text="evolved")
        record, text = propose(api, self.request("recursive"), "test")
        self.assertTrue(record["valid_syntax"])
        self.assertTrue(record["unchanged"])
        self.assertNotIn("accepted", record)
        self.assertEqual(text, "evolved")
        self.assertEqual(api.calls, 1)

    def test_truncated_empty_or_multiple_submissions_cannot_be_candidates(self):
        for api in (FakeAPI(finish="length"), FakeAPI(text=""), FakeAPI(count=2)):
            record, text = propose(api, self.request("recursive"), "test")
            self.assertFalse(record["valid_syntax"])
            self.assertIsNone(text)


if __name__ == "__main__":
    unittest.main()
