#!/usr/bin/env python3
"""stop 序列与 llm_extra 透传的测试。

不需要 LLM、网络或 GPU —— 用假的 client 捕获实际发出的请求参数。

运行：
    cd AgenticArxiv && python tests/test_llm_request_params.py
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("STORE_BACKEND", "memory")

import tools.arxiv_tool  # noqa: F401,E402
import tools.cache_status_tool  # noqa: F401,E402
import tools.pdf_download_tool  # noqa: F401,E402
import tools.pdf_translate_tool  # noqa: F401,E402

from agents.agent_engine import ReActAgent  # noqa: E402
from agents.side_effects import LocalSideEffectManager  # noqa: E402
from utils.llm_client import TransformersLLMClient  # noqa: E402


class CapturingClient:
    """记录每次请求收到的参数，并立即返回 FINISH 结束循环。"""

    def __init__(self):
        self.calls = []

    def chat_completions(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": "Thought: done\nAction: FINISH"}}],
                "usage": {}}


def run_once(**agent_kwargs):
    client = CapturingClient()
    agent = ReActAgent(client, side_effect_mgr=LocalSideEffectManager(), **agent_kwargs)
    agent.run("检索最近7天AI论文", session_id="t")
    return client.calls[0]


class TestStopSequences(unittest.TestCase):
    def test_stop_is_sent_by_default(self):
        """Observation 必须由工具产生；不设 stop 时模型会自行续写整段交互。"""
        self.assertEqual(run_once()["extra"]["stop"], ["Observation:"])

    def test_stop_can_be_disabled(self):
        class NoStop(ReActAgent):
            stop_sequences = ()
        client = CapturingClient()
        NoStop(client, side_effect_mgr=LocalSideEffectManager()).run("x", session_id="t")
        # 无任何额外参数时 BaseAgent 传 None（保持改动前的行为）
        extra = client.calls[0]["extra"]
        self.assertTrue(extra is None or "stop" not in extra, f"意外携带 stop: {extra}")


class TestLlmExtra(unittest.TestCase):
    def test_extra_is_forwarded(self):
        extra = run_once(llm_extra={"chat_template_kwargs": {"enable_thinking": False}})["extra"]
        self.assertEqual(extra["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(extra["stop"], ["Observation:"])   # 与 stop 共存

    def test_llm_extra_can_override_stop(self):
        extra = run_once(llm_extra={"stop": ["###"]})["extra"]
        self.assertEqual(extra["stop"], ["###"])

    def test_default_is_empty_dict_not_none(self):
        agent = ReActAgent(CapturingClient(), side_effect_mgr=LocalSideEffectManager())
        self.assertEqual(agent.llm_extra, {})


class TestTransformersClientResponse(unittest.TestCase):
    def test_adapts_local_generation_to_openai_response(self):
        import torch

        class Tokenizer:
            pad_token_id = 0

            def apply_chat_template(self, messages, **kwargs):
                return "prompt"

            def __call__(self, prompt, return_tensors=None):
                return {"input_ids": torch.tensor([[1, 2]])}

            def decode(self, ids, skip_special_tokens=True):
                return "Thought: x\nAction: FINISH\nObservation: invented"

        class Model:
            def parameters(self):
                return iter([torch.nn.Parameter(torch.zeros(1))])

            def generate(self, **kwargs):
                return torch.tensor([[1, 2, 3, 4, 5]])

        client = TransformersLLMClient.__new__(TransformersLLMClient)
        client.tokenizer = Tokenizer()
        client.model = Model()
        client.seed = 7
        client._calls = 0
        response = client.chat_completions(
            model="local",
            messages=[{"role": "user", "content": "x"}],
            extra={"stop": ["Observation:"]},
        )
        self.assertEqual(
            response["choices"][0]["message"]["content"],
            "Thought: x\nAction: FINISH\n",
        )
        self.assertEqual(response["usage"]["prompt_tokens"], 2)
        self.assertEqual(response["usage"]["completion_tokens"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
