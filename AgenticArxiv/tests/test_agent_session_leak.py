# -*- coding: utf-8 -*-
"""验证 BaseAgent 的 session_id 注入不泄漏进 history 的 Action JSON。

回归场景：download/translate/cache 三个工具的 schema 都含 session_id，
`_execute_with_side_effects` 为调用强制注入 session_id 是必要的；但若直接
改写在 `action_dict["args"]` 上，history 里的 Action JSON 会带上这个框架
状态，SFT/DPO 数据生成会把本不该让模型学习的字段当成模型动作学习
（generate_sft_data.py 的确定性路径同样以“session_id 是框架状态”为由
主动剔除）。本测试锁定“调用注入、轨迹干净”这一行为。
"""

import json
import os
import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
os.environ.setdefault("STORE_BACKEND", "memory")

from agents.base_agent import BaseAgent  # noqa: E402
from agents.side_effects import LocalSideEffectManager  # noqa: E402
from tools.bootstrap import register_all_tools  # noqa: E402

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

    def chat_completions(self, model, messages, temperature=0.1, max_tokens=1000,
                         stream=False, extra=None):
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


class _TestAgent(BaseAgent):
    def __init__(self, llm, side_effect_mgr, env=None, max_iterations=5):
        super().__init__(llm, side_effect_mgr=side_effect_mgr, env=env,
                         max_iterations=max_iterations)

    def discover_tools(self):
        from tools.tool_registry import registry
        return registry.list_tools()

    def build_messages(self, task, tools_description, history_text):
        return [{"role": "user", "content": task + "\n" + history_text}], {}

    def parse_response(self, raw_response):
        import re
        content = raw_response["choices"][0]["message"]["content"]
        m = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", content, re.DOTALL)
        thought = m.group(1).strip() if m else "x"
        a = re.search(r"Action:\s*(.*?)(?=\n|$)", content, re.DOTALL)
        atext = a.group(1).strip() if a else ""
        if atext == "FINISH":
            return thought, None
        act = json.loads(atext)
        return thought, {"name": act["name"], "args": act["args"]}

    def invoke_tool(self, tool_name, args):
        raise NotImplementedError


class SessionIdInjectionTest(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv()
        self.agent = _TestAgent(
            FakeLLM(), LocalSideEffectManager(), env=self.env, max_iterations=5
        )

    def test_session_id_injected_to_call_but_not_into_history_action(self):
        result = self.agent.run("下载第1篇论文", session_id="sess-leak-guard")

        # 1) 工具调用层必须收到注入的 session_id（否则会话状态断链）
        downloaded = [c for c in self.env.calls if c[0] == "download_arxiv_pdf"]
        self.assertEqual(len(downloaded), 1)
        self.assertEqual(downloaded[0][1].get("session_id"), "sess-leak-guard")

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


if __name__ == "__main__":
    unittest.main()
