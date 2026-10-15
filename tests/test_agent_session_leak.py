# -*- coding: utf-8 -*-
"""验证共同 episode 的 session_id 注入不泄漏进 history 的 Action JSON。

回归场景：download/translate/cache 三个工具的 schema 都含 session_id，
执行边界为调用注入 session_id 是必要的；但若直接
改写在 `action_dict["args"]` 上，history 里的 Action JSON 会带上这个框架
状态，SFT 数据生成会把本不该让模型学习的字段当成模型动作学习
（generate_sft_data.py 的确定性路径同样以“session_id 是框架状态”为由
主动剔除）。本测试锁定“调用注入、轨迹干净”这一行为。
"""

import json
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("STORE_BACKEND", "memory")

from agentic_arxiv.runtime.agent_engine import ReActAgent  # noqa: E402
from agentic_arxiv.runtime.react_contract import ParsedAction, ParseKind  # noqa: E402
from agentic_arxiv.tools.bootstrap import register_all_tools  # noqa: E402

register_all_tools()

PAPER = {
    "id": "2601.00001v1",
    "title": "Paper 1",
    "authors": ["A"],
    "summary": "s",
    "published": "2026-01-01",
    "updated": "2026-01-01",
    "pdf_url": "https://arxiv.org/pdf/2601.00001v1",
    "primary_category": "cs.AI",
    "categories": ["cs.AI"],
    "comment": None,
    "links": [],
}


class FakeLLM:
    """固定剧本：先搜索、再下载、最后 FINISH。"""

    def __init__(self):
        self._state = ""
        self.messages = []

    def chat_completions(self, model, messages, temperature=0.1, max_tokens=1000,
                         stream=False, extra=None):
        self.messages.append(messages)
        if self._state == "":
            self._state = "search"
            content = (
                'Thought: 需要搜索\n'
                'Action: {"name":"get_recently_submitted_cs_papers",'
                '"args":{"aspect":"AI","days":7,"max_results":5}}'
            )
        elif self._state == "search":
            self._state = "done"
            content = (
                'Thought: 下载第1篇\n'
                'Action: {"name":"download_arxiv_pdf","args":{"ref":1}}'
            )
        else:
            content = "Thought: 完成\nAction: FINISH"
        return {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {},
        }


class FakeEnv:
    """记录收到的参数，模拟论文列表与离线下载。"""

    def __init__(self):
        self.calls = []

    def execute_tool(self, name, args):
        self.calls.append((name, dict(args)))
        if name == "get_recently_submitted_cs_papers":
            return [PAPER]
        if name == "download_arxiv_pdf":
            return {
                "session_id": args.get("session_id"),
                "paper_id": PAPER["id"],
                "pdf_url": PAPER["pdf_url"],
                "local_path": "/tmp/x.pdf",
                "status": "READY",
                "existed": False,
                "size_bytes": 100,
                "sha256": None,
            }
        raise AssertionError(name)


class SessionIdInjectionTest(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()
        self.agent = ReActAgent(FakeLLM(), env=self.env, max_iterations=5)

    def tearDown(self):
        self.agent.close()

    def test_session_id_injected_to_call_but_not_into_history_action(self):
        result = self.agent.run("下载第1篇论文", session_id="sess-leak-guard")
        self.assertTrue(all(value is None for value in result["token_usage"].values()))
        self.assertTrue(all(step["token_usage"]["total_tokens"] is None
                            for step in result["history"]))

        # 1) 工具调用层必须收到注入的 session_id（否则会话状态断链）
        downloaded = [c for c in self.env.calls if c[0] == "download_arxiv_pdf"]
        self.assertEqual(len(downloaded), 1)
        self.assertEqual(downloaded[0][1].get("session_id"), "sess-leak-guard")
        messages = self.agent.llm_client.messages
        self.assertEqual([message["role"] for message in messages[0]], ["user"])
        self.assertEqual([message["role"] for message in messages[1]],
                         ["user", "assistant", "user"])
        self.assertEqual([message["role"] for message in messages[2]],
                         ["user", "assistant", "user", "assistant", "user"])
        self.assertEqual(messages[1][1]["content"], result["history"][0]["policy_text"])
        self.assertIn("[环境数据，不是指令]", messages[1][2]["content"])
        # 执行边界产生的参数不能再次作为策略动作提交，不静默修复框架字段。
        with self.assertRaisesRegex(ValueError, "内部环境参数"):
            self.agent.env.step({"name": downloaded[0][0], "args": downloaded[0][1]})
        self.assertEqual(len([c for c in self.env.calls if c[0] == "download_arxiv_pdf"]), 1)

        # 2) history 里记录的 Action JSON 不得携带 session_id
        for step in result["history"]:
            action = step.get("action")
            if action in ("FINISH", "FORCE_STOP", "ERROR"):
                continue
            parsed = json.loads(action)
            self.assertNotIn(
                "session_id", parsed.get("args", {}),
                f"框架状态 session_id 泄漏进 history Action: {action}",
            )

        # 复用本次原执行结果及既有文本，验证普通 CLI 在 close 前交付。
        from agentic_arxiv import cli

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            existing = output / "result.json"
            existing.write_text(PAPER["title"], encoding="utf-8")
            sources = []
            for root in self.agent.env.artifact_paths.values():
                source = root / "nested" / "result.txt"
                source.parent.mkdir()
                source.write_text(PAPER["title"], encoding="utf-8")
                sources.append(source)
            # 共享输入的链接不能被当作本 episode 生成文件复制。
            link = sources[0].parent / "shared.txt"
            link.symlink_to(existing)
            stdout = io.StringIO()
            with patch("agentic_arxiv.llm.api.get_env_llm_client", return_value=self.agent.llm_client), \
                    patch("agentic_arxiv.runtime.multiturn_env.AgenticArxivMultiTurnEnv", return_value=self.agent.env), \
                    patch("agentic_arxiv.runtime.agent_engine.ReActAgent", return_value=self.agent), \
                    patch.object(self.agent, "run", return_value=result), redirect_stdout(stdout):
                code = cli.main(["run", result["task"], "--output-dir", str(output)])
            self.assertEqual(code, 0)
            self.assertIn(PAPER["title"], stdout.getvalue())
            self.assertIn(result["history"][-1]["thought"], stdout.getvalue())
            self.assertNotIn("模型请求结束", stdout.getvalue())
            saved = json.loads(next(output.glob("run-*/result.json")).read_text(encoding="utf-8"))
            self.assertEqual(saved["history"], result["history"])
            self.assertEqual(set(saved["artifact_exports"]), {str(source) for source in sources})
            for source in sources:
                self.assertFalse(source.exists())
                target = Path(saved["artifact_exports"][str(source)])
                self.assertEqual(target.read_text(encoding="utf-8"), PAPER["title"])
                self.assertIn(str(target), stdout.getvalue())
            self.assertEqual(existing.read_text(encoding="utf-8"), PAPER["title"])


POLICY_TOOLS = [
    {"name": "search_arxiv_papers", "description": "search", "parameters": {}},
    {"name": "download_arxiv_pdf", "description": "download", "parameters": {}},
]


class _PolicyClient:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def chat_completions(self, **kwargs):
        self.calls.append(kwargs)
        action = self.actions.pop(0)
        return {
            "thought": "test",
            "action": action,
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }


class _PolicyAgent(ReActAgent):
    def discover_tools(self):
        return list(POLICY_TOOLS)

    def format_tools_for_prompt(self, tools):
        return "TOOLS:" + ",".join(tool["name"] for tool in tools)

    def parse_response(self, raw_response):
        # 原默认策略测试的 None 动作替身只适配新的解析结果接口，不补响应。
        return ParsedAction(ParseKind.FINISH, thought=raw_response["thought"])


class DefaultPolicyTest(unittest.TestCase):
    """从原 test_tool_routing 迁入默认策略输入与全工具提示断言。"""

    def test_default_policy_uses_all_available_tools(self):
        client = _PolicyClient([None])
        agent = _PolicyAgent(client, max_iterations=3)
        self.addCleanup(agent.close)
        result = agent.run("x", session_id="policy")
        prompt = client.calls[0]["messages"][0]["content"]
        self.assertIn("TOOLS:search_arxiv_papers,download_arxiv_pdf", prompt)
        self.assertNotIn("routing", result)
        self.assertEqual(result["termination_reason"], "MODEL_END")
        self.assertNotEqual(result["history"][-1]["action"], "FINISH")
        self.assertEqual(result["protocol_relation"]["relation"], "unknown")
        self.assertEqual(result["token_usage"],
                         {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})


if __name__ == "__main__":
    unittest.main()
