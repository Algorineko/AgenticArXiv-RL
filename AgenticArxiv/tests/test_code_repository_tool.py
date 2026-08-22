"""Offline unit tests for repository search/download tools."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("STORE_BACKEND", "memory")

import tools.code_repository_tool as repo_tool  # noqa: E402
from agents.agent_engine import ReActAgent  # noqa: E402
from agents.side_effects import LocalSideEffectManager  # noqa: E402
from rl.env import MockArxivEnv  # noqa: E402


class RepositoryToolTests(unittest.TestCase):
    def setUp(self):
        repo_tool.clear_repository_memory()

    @staticmethod
    def _response(payload):
        response = Mock()
        response.status_code = 200
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        return response

    @patch("tools.code_repository_tool.requests.get")
    def test_github_search_normalises_and_remembers(self, get):
        get.return_value = self._response({"items": [{
            "id": 7, "full_name": "acme/rag", "name": "rag",
            "owner": {"login": "acme"}, "stargazers_count": 9,
            "forks_count": 2, "default_branch": "main", "language": "Python",
        }]})
        result = repo_tool.search_github_repositories("rag", language="Python", session_id="s")
        self.assertEqual(result[0]["full_name"], "acme/rag")
        self.assertEqual(repo_tool.get_repository_results("s", "github")[0]["stars"], 9)
        self.assertIn("language:Python", get.call_args.kwargs["params"]["q"])

    @patch("tools.code_repository_tool.requests.get")
    def test_gitee_search_supports_list_response(self, get):
        get.return_value = self._response([{
            "id": 8, "full_name": "acme/workflow", "name": "workflow",
            "owner": {"login": "acme"}, "stars_count": 5,
            "forks_count": 1, "default_branch": "master", "language": "Java",
        }])
        result = repo_tool.search_gitee_repositories("工作流", session_id="s")
        self.assertEqual(result[0]["platform"], "gitee")
        self.assertEqual(result[0]["stars"], 5)

    def test_offline_search_to_download_by_index(self):
        repos = [{"platform": "github", "full_name": "acme/demo", "default_branch": "main"}]
        repo_tool.remember_repository_results("s", "github", repos)
        with tempfile.TemporaryDirectory() as directory:
            fake_settings = SimpleNamespace(repository_download_path=directory)
            with patch("config.settings", fake_settings):
                env = MockArxivEnv(mode="replay")
                result = env.execute_tool(
                    "download_github_repository", {"session_id": "s", "repository": 1}
                )
            self.assertEqual(result["status"], "READY")
            self.assertFalse(result["extracted"])
            self.assertTrue(Path(result["local_path"]).is_file())

    def test_agent_formats_and_remembers_mock_search_results(self):
        class FakeEnv:
            def execute_tool(self, name, args):
                return [{"platform": "gitee", "full_name": "acme/demo", "stars": 3, "language": "Python"}]

        agent = ReActAgent(None, side_effect_mgr=LocalSideEffectManager(), env=FakeEnv())
        agent.session_id = "agent-session"
        observation = agent._execute_with_side_effects({
            "name": "search_gitee_repositories", "args": {"query": "demo"}
        })
        self.assertIn("acme/demo", observation)
        self.assertEqual(repo_tool.get_repository_results("agent-session", "gitee")[0]["full_name"], "acme/demo")

    def test_rejects_wrong_host_and_path_like_repository(self):
        for value in ("https://evil.example/acme/demo", "../demo", "a/b/c"):
            with self.assertRaises(ValueError):
                repo_tool._split_full_name(value, "github")


if __name__ == "__main__":
    unittest.main()
