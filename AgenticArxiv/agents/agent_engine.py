"""普通文本 ReAct 循环；状态和工具执行由共同 episode 环境管理。"""

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections.abc import Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.prompt_templates import format_tool_description
from agents.react_contract import (
    ParsedAction, make_history_step, make_terminal_step,
    build_react_messages, parse_react_response, protocol_metadata,
)
from config import settings
from rl.multiturn_env import AgenticArxivMultiTurnEnv
from rl.run_manifest import read_protocol_metadata, evaluation_protocol_relation
from tools.bootstrap import require_all_tools
from tools.tool_registry import registry
from utils.logger import log


class ReActAgent:
    """只负责模型请求、共同解析、轨迹记录与轮数预算的具体运行器。"""

    agent_type = "regex"
    stop_sequences = ("Observation:",)

    def __init__(
        self, llm_client: Any, env: Optional[Any] = None,
        max_iterations: int = 6, llm_extra: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None, snapshot_path: Optional[Path] = None,
    ):
        if max_iterations < 1:
            raise ValueError("max_iterations 必须至少为 1，FINISH 也占用一轮")
        require_all_tools("ReAct")
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.llm_extra = dict(llm_extra or {})
        # 包入口与目录入口可能使用不同模块名；已具备 episode 契约的对象不再包一层。
        episode = env is not None and all(
            callable(getattr(env, method, None)) for method in ("reset", "step", "close")
        )
        self.env = env if episode else (
            AgenticArxivMultiTurnEnv(snapshot_path, mode=mode, backend=env)
        )
        self.session_id = "default"

    def discover_tools(self) -> List[Dict[str, Any]]:
        return registry.list_tools()

    def format_tools_for_prompt(self, tools: List[Dict]) -> str:
        return format_tool_description(tools)

    def build_messages(
        self, task: str, tools_description: str, history: List[Dict],
        visible_setup_context: str = "",
    ):
        return build_react_messages(
            task, tools_description, history, visible_setup_context,
        ), {}

    def training_protocol_metadata(self) -> Dict[str, Any]:
        """模型训练事实只来自明确声明或已加载本地模型的原 manifest。"""
        source = getattr(self.llm_client, "training_protocol_metadata", None)
        model_name = getattr(self.llm_client, "model_name", None)
        if source is None and getattr(self.llm_client, "tokenizer", None) is not None:
            if isinstance(model_name, (str, Path)) and model_name and Path(model_name).exists():
                source = Path(model_name)
        return read_protocol_metadata(source)

    def parse_response(self, response: Dict) -> ParsedAction:
        choices = response.get("choices") or [{}]
        content = response.get("raw_model_content")
        if content is None:
            content = (choices[0].get("message") or {}).get("content") or ""
        return self._parse_react_text(content)

    def _parse_react_text(self, response: str) -> ParsedAction:
        return parse_react_response(response)

    def _merge_llm_extra(self, extra: Optional[Dict]) -> Dict[str, Any]:
        merged = {"stop": list(self.stop_sequences)} if self.stop_sequences else {}
        merged.update(self.llm_extra)
        merged.update(extra or {})
        return merged

    @staticmethod
    def _reported_usage(response: Any) -> Dict[str, Optional[int]]:
        """缺报、空值和非法计数保持未知；真实报告的整数零是合法值。"""
        reported = response.get("usage") if isinstance(response, Mapping) else None
        reported = reported if isinstance(reported, Mapping) else {}
        return {
            key: value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            for value in [reported.get(key)]
        }

    def close(self) -> None:
        self.env.close()

    def run(
        self, task: str, agent_model: Optional[str] = None,
        session_id: str = "default", initial_history: str = "",
        task_def: Optional[Dict[str, Any]] = None,
        task_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        self.session_id = session_id
        execution_protocol = protocol_metadata(include_training_protocol=False)
        training_protocol = self.training_protocol_metadata()
        visible_state = self.env.reset(
            task_def=task_def, session_id=session_id, task_lookup=task_lookup
        )
        # 显式可见上下文仅供没有 TaskSpec 的普通调用；评测初始化以环境为准。
        if task_def is None and initial_history:
            visible_state = initial_history
        tools_description = self.format_tools_for_prompt(self.discover_tools())
        model = agent_model or settings.models.agent_model
        history: List[Dict[str, Any]] = []
        timings: List[Dict[str, int]] = []
        token_usage: Dict[str, Optional[int]] = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }
        template_hashes: List[Optional[str]] = []
        iterations = 0
        termination_reason = "MAX_TURNS"

        for _ in range(self.max_iterations):
            llm_started = None
            response_returned = False
            selected_template_hash = None
            try:
                messages, extra = self.build_messages(
                    task, tools_description, history, visible_state
                )
                extra = self._merge_llm_extra(extra)
                temperature = float(extra.pop("temperature", 0.1))
                max_tokens = int(extra.pop("max_tokens", 1000))
                template_kwargs = extra.get("chat_template_kwargs") or {}
                if isinstance(template_kwargs, Mapping):
                    selected = protocol_metadata(
                        getattr(self.llm_client, "tokenizer", None),
                        include_training_protocol=False,
                        chat_template=template_kwargs.get("chat_template"),
                    )
                    selected_template_hash = selected["chat_template_hash"]
                # 单个 episode 的任一请求模板未知或不一致，不能报默认模板为相同。
                template_hashes.append(selected_template_hash)
                execution_protocol["chat_template_hash"] = (
                    template_hashes[0] if all(item == template_hashes[0]
                                             for item in template_hashes) else None
                )
                llm_started = time.perf_counter()
                iterations += 1
                response = self.llm_client.chat_completions(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, stream=False, extra=extra or None,
                )
                response_returned = True
                usage = self._reported_usage(response)
                for key in token_usage:
                    token_usage[key] = (
                        token_usage[key] + usage[key]
                        if token_usage[key] is not None and usage[key] is not None else None
                    )
                parsed = self.parse_response(response)
            except Exception as exc:
                elapsed = int((time.perf_counter() - llm_started) * 1000) if llm_started else 0
                step = make_terminal_step("RUNTIME_ERROR", error=str(exc))
                step["llm_latency_ms"] = elapsed
                step["chat_template_hash"] = selected_template_hash
                if llm_started is not None and not response_returned:
                    token_usage = {key: None for key in token_usage}
                    step["token_usage"] = {key: None for key in token_usage}
                elif response_returned:
                    step["token_usage"] = usage
                history.append(step)
                timings.append({"llm_ms": elapsed, "tool_ms": 0})
                termination_reason = "RUNTIME_ERROR"
                break

            llm_ms = int((time.perf_counter() - llm_started) * 1000)
            tool_ms = 0
            if (response.get("choices") or [{}])[0].get("finish_reason") == "length":
                step = make_terminal_step(
                    "TOKEN_LIMIT", thought=parsed.thought,
                    raw_response=parsed.raw_response, policy_text=parsed.policy_text,
                )
                step["token_usage"] = usage
                step["llm_latency_ms"] = llm_ms
                step["chat_template_hash"] = selected_template_hash
                history.append(step)
                timings.append({"llm_ms": llm_ms, "tool_ms": 0})
                termination_reason = "TOKEN_LIMIT"
                break
            if not parsed.raw_response.strip():
                step = make_terminal_step(
                    "MODEL_END", thought=parsed.thought,
                    raw_response=parsed.raw_response, policy_text=parsed.policy_text,
                )
                step["token_usage"] = usage
                step["llm_latency_ms"] = llm_ms
                step["chat_template_hash"] = selected_template_hash
                history.append(step)
                timings.append({"llm_ms": llm_ms, "tool_ms": 0})
                termination_reason = "MODEL_END"
                break
            if parsed.kind == "CALL":
                tool_started = time.perf_counter()
                error = ""
                try:
                    result = self.env.step(parsed.action)
                except Exception as exc:
                    # 合法动作的业务错误是实际工具反馈，允许策略用剩余预算响应。
                    error = str(exc)
                    result = {"status": "FAILED", "error": error}
                tool_ms = int((time.perf_counter() - tool_started) * 1000)
                step = make_history_step(
                    parsed, result=result, error=error, token_usage=usage,
                    llm_latency_ms=llm_ms, tool_latency_ms=tool_ms,
                )
            else:
                step = make_history_step(parsed, token_usage=usage, llm_latency_ms=llm_ms)
                termination_reason = getattr(parsed.kind, "value", parsed.kind)
            step["chat_template_hash"] = selected_template_hash
            history.append(step)
            timings.append({"llm_ms": llm_ms, "tool_ms": tool_ms})
            if parsed.kind != "CALL":
                break
        else:
            history.append(make_terminal_step("MAX_TURNS", thought="达到最大迭代次数"))
            timings.append({"llm_ms": 0, "tool_ms": 0})

        total_ms = int((time.perf_counter() - started) * 1000)
        llm_ms = sum(item["llm_ms"] for item in timings)
        tool_ms = sum(item["tool_ms"] for item in timings)
        log.info(f"ReAct 结束: {termination_reason}，模型调用 {iterations} 轮")
        recorded_model = getattr(self.llm_client, "model_name", None)
        if not isinstance(recorded_model, (str, Path)) or not recorded_model:
            recorded_model = (agent_model or "") if getattr(
                self.llm_client, "tokenizer", None
            ) is not None else model
        return {
            "task": task, "msg_id": uuid.uuid4().hex, "model": str(recorded_model),
            "history": history, "setup_context": visible_state,
            "final_observation": history[-1]["observation"] if history else "",
            "termination_reason": termination_reason, "total_time_ms": total_ms,
            "iteration_count": iterations, "agent_type": self.agent_type,
            "timing": {
                "total_llm_ms": llm_ms, "total_tool_ms": tool_ms,
                "framework_overhead_ms": total_ms - llm_ms - tool_ms,
                "steps": timings,
            },
            "token_usage": token_usage,
            "chat_template_hashes": template_hashes,
            **execution_protocol,
            "protocol_metadata": execution_protocol,
            "training_protocol_metadata": training_protocol,
            "protocol_relation": evaluation_protocol_relation(training_protocol, execution_protocol),
        }
