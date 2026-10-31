"""No-network request format and retry-accounting checks for shared API calls."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from rsi_api_pilot import API


def response():
    return io.StringIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                                  "usage": {"prompt_tokens": 10, "completion_tokens": 5}}))


class APIAccountingTests(unittest.TestCase):
    def make_api(self, root):
        return API("https://example.invalid/v1", "fixture", Path(root),
                   {"calls": 2, "input_tokens": 100000, "output_tokens": 10000}, min_interval=0)

    def test_plain_reflection_omits_tool_fields_and_solver_keeps_them(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"RSI_API_KEY": "fixture-key"}):
            api = self.make_api(root)
            bodies = []

            def fake_transport(request, timeout):
                bodies.append(json.loads(request.data))
                return response()

            with patch("urllib.request.urlopen", side_effect=fake_transport):
                api.chat([{"role": "user", "content": "reflect"}], [], "reflection")
                tools = [{"type": "function", "function": {"name": "fixture"}}]
                api.chat([{"role": "user", "content": "solve"}], tools, "solver")
            self.assertNotIn("tools", bodies[0])
            self.assertNotIn("tool_choice", bodies[0])
            self.assertEqual(bodies[1]["tools"], tools)
            self.assertEqual(bodies[1]["tool_choice"], "auto")
            self.assertEqual(api.totals()["calls"], 2)

    def test_unknown_usage_retry_is_charged_in_addition_to_success(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"RSI_API_KEY": "fixture-key"}):
            api = self.make_api(root)
            error = urllib.error.HTTPError("https://example.invalid", 429, "limited", {}, io.BytesIO(b"limited"))
            with patch("urllib.request.urlopen", side_effect=[error, response()]), patch("time.sleep"):
                api.chat([{"role": "user", "content": "fixture"}], [], "fixture")
            rows = [json.loads(line) for line in (Path(root) / "api_calls.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(api.totals()["calls_without_provider_usage"], 1)
            self.assertEqual(api.totals()["accounted_input_tokens"], rows[0]["reserved_input_upper_estimate"] + 10)
            self.assertEqual(api.totals()["accounted_output_tokens"], 512 + 5)


if __name__ == "__main__":
    unittest.main()
