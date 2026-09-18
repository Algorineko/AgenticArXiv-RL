"""API rollout records the model actually sent to chat completions."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.agent_engine import ReActAgent  # noqa: E402
from agents.side_effects import NoOpSideEffectManager  # noqa: E402
from rl.rollout import rollout_single_task  # noqa: E402
from utils.llm_client import LLMClient  # noqa: E402


class TestApiRolloutModel(unittest.TestCase):
    def test_rollout_persists_the_model_sent_to_chat_completions(self):
        model = "nvidia/nemotron-3-ultra-550b-a55b:free"
        responses = iter((
            'Thought: search\nAction: {"name":"get_recently_submitted_cs_papers",'
            '"args":{"aspect":"AI","days":7,"max_results":5}}',
            "Thought: done\nAction: FINISH",
        ))

        def post(url, headers, json, timeout):
            self.assertEqual(json["model"], model)
            return Mock(status_code=200, json=lambda: {
                "choices": [{"message": {"content": next(responses)}}]
            })

        class Environment:
            def execute_tool(self, name, args):
                return [{"id": "paper-1", "title": "Paper one", "authors": [], "categories": ["cs.AI"]}]

        client = LLMClient(base_url="https://example.test", api_key="test")
        agent = ReActAgent(client, side_effect_mgr=NoOpSideEffectManager(), env=Environment())
        configured = SimpleNamespace(models=SimpleNamespace(agent_model=model))
        with tempfile.TemporaryDirectory() as output_dir:
            with patch("agents.base_agent.settings", configured), patch("utils.llm_client.requests.post", side_effect=post):
                rollout_single_task("search_01", output_dir=output_dir, agent=agent, llm_client=client)
            trajectory = json.loads(next(Path(output_dir).glob("*.jsonl")).read_text())

        self.assertEqual(trajectory["model"], model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
