"""Exercise native-tool parsing and local-model budget/error transport offline."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from rsi_api_pilot import BudgetExceeded, episode
from rsi_qwen_stdio_client import LocalQwenClient
from rsi_qwen_stdio_server import normalize_messages, parse_generation


FAKE_SERVER = '''import json,os,sys
print(json.dumps({"status":"ready","model_id":"fixture"}),flush=True)
for line in sys.stdin:
 request=json.loads(line)
 if request.get("shutdown"):
  print(json.dumps({"status":"shutdown"}),flush=True);break
 text=request["messages"][0]["content"]
 if text=="crash":os._exit(2)
 if text=="context":
  print(json.dumps({"status":"context_limit","usage":{"prompt_tokens":20,"completion_tokens":0}}),flush=True);continue
 print(json.dumps({"status":"ok","response":{"choices":[{"finish_reason":"stop","message":{"content":"done"}}],"usage":{"prompt_tokens":10,"completion_tokens":3}}}),flush=True)
'''


class LocalTransportTests(unittest.TestCase):
    def test_tool_parser_preserves_json_arguments_and_rejects_incomplete_calls(self):
        raw = '<tool_call>{"name":"probe_echo","arguments":{"value":7}}</tool_call>'
        calls = parse_generation(raw, 2)
        self.assertEqual(calls[0]["id"], "qwen_2_0")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"value": 7})
        self.assertIsNone(parse_generation('<tool_call>{"name":"x"}', 2))
        self.assertIsNone(parse_generation('<tool_call>{"name":"x","arguments":[]}</tool_call>', 2))

    def test_message_normalization_preserves_malformed_previous_arguments(self):
        messages = [{"role": "assistant", "tool_calls": [{"function": {"name": "x", "arguments": "{"}}]}]
        self.assertEqual(normalize_messages(messages)[0]["tool_calls"][0]["function"]["arguments"], "{")

    def test_known_usage_and_context_limit_are_counted(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "fake.py"
            script.write_text(FAKE_SERVER)
            client = LocalQwenClient([sys.executable, "-u", str(script)], Path(root) / "output",
                                     {"calls": 2, "input_tokens": 10000, "output_tokens": 100},
                                     {"model_id": "fixture"}, max_completion_tokens=32)
            try:
                response = client.chat([{"role": "user", "content": "hello"}], [], "a")
                self.assertEqual(response["choices"][0]["finish_reason"], "stop")
                response = client.chat([{"role": "user", "content": "context"}], [], "b")
                self.assertEqual(response["choices"][0]["finish_reason"], "model_context_limit")
                self.assertEqual(client.totals()["accounted_input_tokens"], 30)
                self.assertEqual(client.totals()["accounted_output_tokens"], 3)
                with self.assertRaises(BudgetExceeded):
                    client.chat([{"role": "user", "content": "hello"}], [], "c")
            finally:
                client.close()

    def test_unknown_usage_after_crash_is_charged_conservatively(self):
        with tempfile.TemporaryDirectory() as root:
            script = Path(root) / "fake.py"
            script.write_text(FAKE_SERVER)
            client = LocalQwenClient([sys.executable, "-u", str(script)], Path(root) / "output",
                                     {"calls": 2, "input_tokens": 10000, "output_tokens": 100},
                                     {"model_id": "fixture"}, max_completion_tokens=32)
            try:
                with self.assertRaises(RuntimeError):
                    client.chat([{"role": "user", "content": "crash"}], [], "a")
                totals = client.totals()
                self.assertEqual(totals["calls_without_provider_usage"], 1)
                self.assertGreater(totals["accounted_input_tokens"], 4096)
                self.assertEqual(totals["accounted_output_tokens"], 32)
            finally:
                client.close()

    def test_context_limit_is_reported_separately_from_assistant_final(self):
        class FixtureEnvironment:
            def tools(self): return []
            def check(self): return {"success": False}
            def close(self): pass

        class FixtureAPI:
            def chat(self, messages, tools, identifier):
                return {"choices": [{"finish_reason": "model_context_limit", "message": {"content": ""}}]}
            def append(self, filename, value): pass

        trace = episode(FixtureAPI(), {"id": "fixture", "domain": "fixture", "prompt": "fixture"},
                        repeat=0, max_steps=1, environment=FixtureEnvironment())
        self.assertEqual(trace["stop"], "model_context_limit")


if __name__ == "__main__":
    unittest.main()
