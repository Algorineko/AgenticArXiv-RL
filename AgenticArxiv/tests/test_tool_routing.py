"""Offline tests for the optional inference-time tool router framework."""

import os
import sys
from pathlib import Path
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("STORE_BACKEND", "memory")

from agents.base_agent import BaseAgent  # noqa: E402
from agents.agent_engine import ReActAgent  # noqa: E402
from agents.side_effects import LocalSideEffectManager  # noqa: E402
from routing.base import RouteDecision  # noqa: E402
from routing.factory import build_tool_router_from_env  # noqa: E402
from routing.jev import JevToolRouter  # noqa: E402
from tools.bootstrap import require_all_tools  # noqa: E402


require_all_tools("tool router tests")


TOOLS = [
    {"name": "search_arxiv_papers", "description": "search", "parameters": {}},
    {"name": "download_arxiv_pdf", "description": "download", "parameters": {}},
]


class _Client:
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


class _TextClient:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def chat_completions(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }


class _Environment:
    def __init__(self):
        self.calls = []

    def execute_tool(self, name, args):
        self.calls.append((name, dict(args)))
        return {"paper_id": "2608.00001v1", "status": "READY"}


class _Agent(BaseAgent):
    def discover_tools(self):
        return list(TOOLS)

    def format_tools_for_prompt(self, tools):
        return "TOOLS:" + ",".join(tool["name"] for tool in tools)

    def build_messages(self, task, tools_info, history_text):
        return [{"role": "user", "content": tools_info}], {}

    def parse_response(self, raw_response):
        return raw_response["thought"], raw_response["action"]

    def invoke_tool(self, tool_name, args):
        raise AssertionError("the injected environment should execute tools")


class _Router:
    name = "fake-jev"

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def route(self, **kwargs):
        self.calls.append(kwargs)
        return self.decisions.pop(0)


def _agent(client, router=None):
    environment = _Environment()
    agent = _Agent(
        client,
        side_effect_mgr=LocalSideEffectManager(),
        env=environment,
        max_iterations=3,
        tool_router=router,
    )
    return agent, environment


def test_policy_mode_is_the_unchanged_default_path():
    client = _Client([None])
    agent, _ = _agent(client)
    result = agent.run("x", session_id="policy")
    prompt = client.calls[0]["messages"][0]["content"]
    assert prompt == "TOOLS:search_arxiv_papers,download_arxiv_pdf"
    assert result["routing"] == {"mode": "policy", "decisions": []}


def test_accepted_route_restricts_schema_then_executes_matching_action():
    router = _Router(
        [
            RouteDecision(
                selected_tool="download_arxiv_pdf",
                confidence=0.93,
                accepted=True,
                source="jev",
            ),
            RouteDecision(
                selected_tool="FINISH",
                confidence=0.95,
                accepted=True,
                source="jev",
            ),
        ]
    )
    client = _Client([{"name": "download_arxiv_pdf", "args": {"ref": 1}}])
    agent, environment = _agent(client, router)
    result = agent.run("download first", session_id="accepted")

    assert client.calls[0]["messages"][0]["content"] == "TOOLS:download_arxiv_pdf"
    assert environment.calls[0][0] == "download_arxiv_pdf"
    assert [step["action"] for step in result["history"]][-1] == "FINISH"
    assert all(row["used"] for row in result["routing"]["decisions"])


def test_low_confidence_route_defers_to_full_policy_prompt():
    router = _Router(
        [
            RouteDecision(
                selected_tool="download_arxiv_pdf",
                confidence=0.40,
                accepted=False,
                source="jev",
                reason="low_confidence",
            )
        ]
    )
    client = _Client([None])
    agent, _ = _agent(client, router)
    result = agent.run("x", session_id="low-confidence")

    assert client.calls[0]["messages"][0]["content"] == (
        "TOOLS:search_arxiv_papers,download_arxiv_pdf"
    )
    assert result["routing"]["decisions"][0]["used"] is False


def test_policy_router_disagreement_gets_one_full_policy_retry():
    router = _Router(
        [
            RouteDecision(
                selected_tool="download_arxiv_pdf",
                confidence=0.96,
                accepted=True,
                source="jev",
            ),
            RouteDecision(
                selected_tool="FINISH",
                confidence=0.96,
                accepted=True,
                source="jev",
            ),
        ]
    )
    client = _Client(
        [
            {"name": "search_arxiv_papers", "args": {"query": "x"}},
            {"name": "download_arxiv_pdf", "args": {"ref": 1}},
        ]
    )
    agent, environment = _agent(client, router)
    result = agent.run("download", session_id="disagreement")

    assert len(client.calls) == 2
    assert client.calls[0]["messages"][0]["content"] == "TOOLS:download_arxiv_pdf"
    assert client.calls[1]["messages"][0]["content"] == (
        "TOOLS:search_arxiv_papers,download_arxiv_pdf"
    )
    assert environment.calls[0][0] == "download_arxiv_pdf"
    route = result["routing"]["decisions"][0]
    assert route["used"] is False
    assert route["fallback_reason"] == "policy_router_disagreement"


def test_guided_router_repairs_tool_name_when_arguments_match_selected_schema():
    router = _Router(
        [
            RouteDecision(
                selected_tool="search_arxiv_papers",
                confidence=0.99,
                accepted=True,
                source="jev",
            ),
            RouteDecision(
                selected_tool="FINISH",
                confidence=0.99,
                accepted=True,
                source="jev",
            ),
        ]
    )
    client = _TextClient(
        [
            'Thought: 提取关键词参数\nAction: {"name":"get_recently_submitted_cs_papers",'
            '"args":{"query":"all:agentic reinforcement learning",'
            '"max_results":5,"days":30}}'
        ]
    )
    environment = _Environment()
    agent = ReActAgent(
        client,
        side_effect_mgr=LocalSideEffectManager(),
        env=environment,
        max_iterations=3,
        tool_router=router,
        router_argument_mode="guided",
    )

    result = agent.run("按关键词搜索 agentic reinforcement learning")

    assert environment.calls[0][0] == "search_arxiv_papers"
    assert environment.calls[0][1]["query"] == "all:agentic reinforcement learning"
    assert client.calls[0]["max_tokens"] == 400
    prompt = client.calls[0]["messages"][0]["content"]
    assert "已经做出不可更改的工具决定：search_arxiv_papers" in prompt
    route = result["routing"]["decisions"][0]
    assert route["used"] is True
    assert route["tool_name_repaired"] is True
    assert route["model_selected_tool"] == "get_recently_submitted_cs_papers"


def test_guided_router_uses_deterministic_explicit_args_without_qwen_call():
    router = _Router(
        [
            RouteDecision(
                selected_tool="search_arxiv_papers",
                confidence=0.99,
                accepted=True,
                source="jev",
            ),
            RouteDecision(
                selected_tool="FINISH",
                confidence=0.99,
                accepted=True,
                source="jev",
            ),
        ]
    )
    client = _TextClient([])
    environment = _Environment()
    agent = ReActAgent(
        client,
        side_effect_mgr=LocalSideEffectManager(),
        env=environment,
        max_iterations=3,
        tool_router=router,
        router_argument_mode="guided",
    )

    result = agent.run(
        "按关键词检索 arXiv：all:agentic reinforcement learning，"
        "最近 30 天，最多返回 5 篇论文"
    )

    assert client.calls == []
    assert environment.calls == [
        (
            "search_arxiv_papers",
            {
                "query": "all:agentic reinforcement learning",
                "days": 30,
                "max_results": 5,
            },
        )
    ]
    route = result["routing"]["decisions"][0]
    assert route["used"] is True
    assert route["argument_source"] == "deterministic"


def test_guided_router_blocks_non_positive_reference_without_tool_or_qwen_call():
    router = _Router(
        [
            RouteDecision(
                selected_tool="download_arxiv_pdf",
                confidence=0.99,
                accepted=True,
                source="jev",
            )
        ]
    )
    client = _TextClient([])
    environment = _Environment()
    agent = ReActAgent(
        client,
        side_effect_mgr=LocalSideEffectManager(),
        env=environment,
        max_iterations=3,
        tool_router=router,
        router_argument_mode="guided",
    )

    result = agent.run("下载第0篇论文")

    assert client.calls == []
    assert environment.calls == []
    assert result["history"] == [
        {
            "thought": "无法执行：论文索引 ref=0 非法，论文序号必须从 1 开始",
            "action": "FINISH",
            "observation": "任务因非法论文索引而终止，未调用任何工具",
        }
    ]
    route = result["routing"]["decisions"][0]
    assert route["used"] is False
    assert route["fallback_reason"] == "deterministic_invalid_arguments"


def test_guided_router_rejects_arguments_from_wrong_schema_then_falls_back():
    router = _Router(
        [
            RouteDecision(
                selected_tool="search_arxiv_papers",
                confidence=0.99,
                accepted=True,
                source="jev",
            ),
            RouteDecision(
                selected_tool="FINISH",
                confidence=0.99,
                accepted=True,
                source="jev",
            ),
        ]
    )
    client = _TextClient(
        [
            'Thought: 错误参数\nAction: {"name":"get_recently_submitted_cs_papers",'
            '"args":{"aspect":"AI","days":7,"max_results":5}}',
            'Thought: policy fallback\nAction: {"name":"search_arxiv_papers",'
            '"args":{"query":"all:agentic reinforcement learning","max_results":5}}',
        ]
    )
    environment = _Environment()
    agent = ReActAgent(
        client,
        side_effect_mgr=LocalSideEffectManager(),
        env=environment,
        max_iterations=3,
        tool_router=router,
        router_argument_mode="guided",
    )

    result = agent.run("按关键词搜索 agentic reinforcement learning")

    assert len(client.calls) == 2
    assert environment.calls[0][0] == "search_arxiv_papers"
    route = result["routing"]["decisions"][0]
    assert route["used"] is False
    assert route["fallback_reason"] == "routed_argument_validation_failed"
    assert route["argument_validation_error"].startswith("unknown_args:")


class _Response:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "answers": {
                "next_tool": {
                    "choice": "download_arxiv_pdf",
                    "confidence": 0.91,
                    "probabilities": {"download_arxiv_pdf": 0.91},
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }


class _Session:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return _Response()


def test_jev_adapter_accepts_typed_choice_without_exposing_key_in_result():
    session = _Session()
    router = JevToolRouter(api_key="temporary", session=session, sleep=lambda _: None)
    decision = router.route(
        task="x",
        history="",
        tools=[
            {
                "name": "get_recently_submitted_cs_papers",
                "description": "recent",
                "parameters": {},
            },
            *TOOLS,
        ],
    )
    assert decision.accepted is True
    assert decision.selected_tool == "download_arxiv_pdf"
    assert decision.confidence == 0.91
    assert "temporary" not in str(decision.to_dict())
    criteria = session.calls[0][1]["json"]["questions"]["next_tool"]["criteria"]
    assert "arbitrary keyword" in criteria["get_recently_submitted_cs_papers"]
    assert "all:" in criteria["search_arxiv_papers"]
    assert "explicit arXiv ID" in criteria["download_arxiv_pdf"]
    assert "MUST NOT be rejected" in criteria["FINISH"]


def test_jev_network_error_is_a_policy_fallback_not_an_agent_error():
    router = JevToolRouter(
        api_key="temporary",
        session=_Session(requests.exceptions.Timeout("offline")),
        max_retries=1,
        sleep=lambda _: None,
    )
    decision = router.route(task="x", history="", tools=TOOLS)
    assert decision.accepted is False
    assert decision.reason == "api_error:Timeout"


def test_factory_is_default_off_and_requires_key_when_enabled():
    with mock.patch.dict(os.environ, {"TOOL_ROUTER": "policy"}, clear=False):
        assert build_tool_router_from_env() is None
    with mock.patch.dict(
        os.environ,
        {"TOOL_ROUTER": "jev", "TYPESAFE_API_KEY": ""},
        clear=False,
    ):
        try:
            build_tool_router_from_env()
        except ValueError as exc:
            assert "TYPESAFE_API_KEY" in str(exc)
        else:
            raise AssertionError("missing Jev key should fail fast")


def test_react_agent_constructor_switches_between_policy_and_jev_from_env():
    with mock.patch.dict(os.environ, {"TOOL_ROUTER": "policy"}, clear=False):
        policy_agent = ReActAgent(
            _Client([None]),
            side_effect_mgr=LocalSideEffectManager(),
            env=_Environment(),
        )
    assert policy_agent.tool_router is None

    with mock.patch.dict(
        os.environ,
        {
            "TOOL_ROUTER": "jev",
            "TYPESAFE_API_KEY": "contract-test-placeholder",
        },
        clear=False,
    ):
        jev_agent = ReActAgent(
            _Client([None]),
            side_effect_mgr=LocalSideEffectManager(),
            env=_Environment(),
        )
    assert isinstance(jev_agent.tool_router, JevToolRouter)
