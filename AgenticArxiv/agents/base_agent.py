# AgenticArxiv/agents/base_agent.py
import json
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List

from utils.llm_client import LLMClient
from utils.logger import log
from config import settings
from models.schemas import Paper
from tools.tool_registry import registry
from agents.side_effects import SideEffectManager, LocalSideEffectManager


# 循环终止标记。模型有时不写裸 `Action: FINISH`，而是套上和其他动作一样的
# JSON 外壳 `{"name": "FINISH", "args": {}}` —— prompt 里其余动作都是 JSON，
# 这么写很自然。若不识别，它会被当成一次工具调用：白跑一轮迭代，
# 还会在 benchmark 的 tool_call_sequence 里多出一个 "FINISH"，
# 把本来完全正确的轨迹判成 accurate=False。
TERMINAL_ACTIONS = ("FINISH", "FORCE_STOP", "ERROR")

# 两种搜索都会产生“当前会话的候选论文列表”。后续下载、翻译和缓存查询
# 都通过这份列表解析 ref，因此必须共用相同的状态写入和结果展示逻辑。
PAPER_SEARCH_ACTIONS = (
    "get_recently_submitted_cs_papers",
    "search_arxiv_papers",
)


def is_terminal_action(name: Any) -> bool:
    """判断解析出的动作名是否代表「结束」而非一次工具调用。"""
    return isinstance(name, str) and name.strip().upper() in TERMINAL_ACTIONS


class BaseAgent(ABC):
    """所有 Agent 方案的基类，封装通用的循环控制、日志、SSE、副作用逻辑"""

    agent_type: str = "regex"  # 子类覆写

    # ReAct 循环里，Observation 必须由工具真实执行产生。
    # 不设 stop 时模型会自己往下编造 Observation 与后续步骤，例如：
    #     Action: {...}
    #     Observation: 成功获取5篇论文列表，已保存至 output/...   ← 编的
    #     Thought: 任务已完成
    #     Action: FINISH
    # 解析用的正则恰好在 Observation 前截断，所以第一个动作仍能取出，
    # 但这些续写既浪费 token（实测多耗约一倍），也让"多跳任务"退化成
    # 模型自说自话，而非真的与环境交互。三种 Agent 的 prompt 都用
    # Observation: 作分隔，故在此统一设置。
    stop_sequences: Tuple[str, ...] = ("Observation:",)

    def __init__(
        self,
        llm_client: LLMClient,
        side_effect_mgr: Optional[SideEffectManager] = None,
        env: Optional[Any] = None,
        max_iterations: int = 5,
        llm_extra: Optional[Dict[str, Any]] = None,
        tool_router: Optional[Any] = None,
        router_argument_mode: str = "legacy",
    ):
        """
        Args:
            llm_client: LLM 调用客户端
            side_effect_mgr: 副作用管理器；默认 LocalSideEffectManager（无 DB/SSE）。
                Web 应用请显式传入 MySQLSideEffectManager 以保持原行为。
            env: 可选的工具执行环境（需实现 execute_tool(name, args)）。
                传入 MockArxivEnv 即可让 rollout 走快照回放而非真实 API。
            max_iterations: ReAct 最大迭代轮数
            llm_extra: 透传给 LLM 接口的额外参数，会合并进每次请求。
                例如关闭 Qwen3 系列的思维链：
                    {"chat_template_kwargs": {"enable_thinking": False}}
                思维链会让生成 token 数翻倍，而 token 用量是 benchmark 的
                核心对比指标之一，混入后三种范式之间的差异会被淹没。
            tool_router: 可选的推理时工具路由器。None 保持原始策略模型路由。
                外部路由失败、低置信度或与参数模型输出不一致时自动退回原路径。
            router_argument_mode: ``legacy`` 沿用“单工具版普通 ReAct prompt”；
                ``guided`` 使用固定工具的参数生成 prompt，并在执行前按该工具 schema
                校验参数。两种模式都只影响已接受的外部路由决定。
        """
        self.llm_client = llm_client
        self.side_effects = side_effect_mgr or LocalSideEffectManager()
        self.env = env
        self.max_iterations = max_iterations
        self.llm_extra = dict(llm_extra or {})
        self.tool_router = tool_router
        argument_mode = str(router_argument_mode).strip().lower()
        if argument_mode not in {"legacy", "guided"}:
            raise ValueError(
                "router_argument_mode must be 'legacy' or 'guided'"
            )
        self.router_argument_mode = argument_mode
        self.session_id = "default"

    # ---------- 子类必须实现 ----------

    @abstractmethod
    def discover_tools(self) -> List[Dict[str, Any]]:
        """返回可用工具列表 [{name, description, parameters}]"""

    @abstractmethod
    def build_messages(
        self, task: str, tools_info: List[Dict], history_text: str
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """构造 LLM 请求。返回 (messages, extra_payload)"""

    @abstractmethod
    def parse_response(self, raw_response: Dict) -> Tuple[str, Optional[Dict[str, Any]]]:
        """解析 LLM 响应。返回 (thought, action_dict | None 表示 FINISH)"""

    @abstractmethod
    def invoke_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """执行工具，返回原始结果"""

    # ---------- 子类可选覆写 ----------

    def format_tools_for_prompt(self, tools: List[Dict]) -> str:
        """将工具列表格式化为 prompt 中的文本描述，子类可覆写"""
        from agents.prompt_templates import format_tool_description
        return format_tool_description(tools)

    def format_history(self, steps: list) -> str:
        """将历史步骤格式化为 prompt 中的文本，子类可覆写"""
        parts = []
        for s in steps:
            parts.append(
                f"Thought: {s['thought']}\nAction: {s['action']}\nObservation: {s['observation']}"
            )
        return "\n\n".join(parts)

    def build_routed_messages(
        self,
        task: str,
        selected_tool: str,
        tool_description: str,
        history_text: str,
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Build a fixed-tool argument prompt.

        Subclasses can make the router decision explicit instead of asking the
        policy to choose a tool again.  The compatibility implementation keeps
        the historical single-tool prompt.
        """
        return self.build_messages(task, tool_description, history_text)

    # ---------- 通用执行循环 ----------

    def run(
        self,
        task: str,
        agent_model: str = None,
        session_id: str = "default",
        initial_history: str = "",
    ) -> Dict[str, Any]:
        """Run one task through the ReAct loop.

        ``initial_history`` is model-visible state that existed before the
        current user request.  Benchmark setup actions use it to expose the
        same legitimate session context that GRPO sees, without counting the
        setup as policy actions or placing it in the scored trajectory.
        """
        log.info(f"[{self.__class__.__name__}] 开始执行任务: {task}")
        run_start = time.time()
        self.session_id = session_id
        msg_id = uuid.uuid4().hex

        if agent_model is None:
            agent_model = settings.models.agent_model

        try:
            self.side_effects.create_chat_log(
                session_id, msg_id, "user", task, model=agent_model, agent_type=self.agent_type
            )
        except Exception as e:
            log.warning(f"Failed to log user message: {e}")

        tools = self.discover_tools()
        policy_tools_description = self.format_tools_for_prompt(tools)

        # Benchmark/GRPO 会显式传入同一份、由 TaskSpec.setup 派生的可见状态。
        # 此时不要再从 side-effects 追加另一种格式的论文列表，否则评测 prompt
        # 会比训练 prompt 多出标题，重新造成输入分布错位。普通 Web/API 调用没有
        # initial_history，仍沿用运行时 store 中的真实会话上下文。
        enriched_task = (
            task if initial_history.strip()
            else self._enrich_task_with_context(task, session_id)
        )

        history: List[Dict[str, str]] = []
        step_timings: List[Dict[str, int]] = []
        token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        routing_decisions: List[Dict[str, Any]] = []

        for iteration in range(self.max_iterations):
            log.info(f"第 {iteration + 1} 次迭代")

            generated_history = self.format_history(history)
            history_text = initial_history.strip()
            if generated_history:
                history_text = (
                    f"{history_text}\n\n{generated_history}"
                    if history_text else generated_history
                )
            llm_ms = 0
            tool_ms = 0
            router_ms = 0
            thought = ""
            action_dict = None
            observation = ""
            routed_tool: Optional[str] = None
            routed_tool_spec: Optional[Dict[str, Any]] = None
            deterministic_action: Optional[Dict[str, Any]] = None
            route_record: Optional[Dict[str, Any]] = None

            # The default path performs no external routing and remains byte-for-byte
            # equivalent at the prompt level.  Jev is consulted only when explicitly
            # configured on this agent.
            iteration_tools = tools
            if self.tool_router is not None:
                route_started = time.time()
                try:
                    decision = self.tool_router.route(
                        task=enriched_task,
                        history=history_text,
                        tools=tools,
                    )
                    router_ms = int((time.time() - route_started) * 1000)
                    route_record = (
                        decision.to_dict()
                        if hasattr(decision, "to_dict")
                        else dict(vars(decision))
                    )
                    route_record.update({"iteration": iteration, "used": False})
                    routing_decisions.append(route_record)
                    if decision.accepted and decision.selected_tool == "FINISH":
                        route_record["used"] = True
                        thought = f"{decision.source} router selected FINISH"
                        observation = "任务完成"
                        history.append(
                            {"thought": thought, "action": "FINISH", "observation": observation}
                        )
                        step_timings.append(
                            {"router_ms": router_ms, "llm_ms": 0, "tool_ms": 0}
                        )
                        self._log_step(
                            msg_id, iteration, thought, "FINISH", "{}",
                            observation, 0, 0, session_id,
                        )
                        break
                    if decision.accepted and decision.selected_tool:
                        selected = [
                            tool for tool in tools
                            if tool.get("name") == decision.selected_tool
                        ]
                        if selected:
                            routed_tool = decision.selected_tool
                            routed_tool_spec = selected[0]
                            iteration_tools = selected
                            route_record["argument_mode"] = self.router_argument_mode
                        else:
                            route_record["fallback_reason"] = "router_selected_unknown_tool"
                except Exception as exc:
                    router_ms = int((time.time() - route_started) * 1000)
                    route_record = {
                        "iteration": iteration,
                        "source": getattr(self.tool_router, "name", "external"),
                        "accepted": False,
                        "used": False,
                        "reason": f"router_error:{type(exc).__name__}",
                    }
                    routing_decisions.append(route_record)
                    log.warning(f"外部工具路由失败，回退策略模型: {exc}")

            tools_description = self.format_tools_for_prompt(iteration_tools)

            if routed_tool and self.router_argument_mode == "guided":
                from routing.arguments import resolve_routed_arguments

                resolution = resolve_routed_arguments(
                    tool_name=routed_tool,
                    task=enriched_task,
                    history=history_text,
                )
                assert route_record is not None
                route_record["argument_resolution"] = resolution.status
                route_record["argument_resolution_reason"] = resolution.reason
                if resolution.status == "resolved":
                    candidate = {"name": routed_tool, "args": dict(resolution.args)}
                    valid, validation_reason = self._validate_routed_action(
                        candidate,
                        routed_tool_spec or {},
                    )
                    if valid:
                        deterministic_action = candidate
                        route_record["argument_source"] = "deterministic"
                    else:
                        route_record["argument_resolution"] = "defer"
                        route_record["argument_resolution_reason"] = (
                            f"schema_validation:{validation_reason}"
                        )
                elif resolution.status == "complete":
                    # Reference exhaustion applies only to this selected tool.
                    # Earlier attempts may have failed, or the user may still
                    # need another tool. Let the full policy assess completion.
                    route_record["fallback_reason"] = "deterministic_references_exhausted"
                    routed_tool = None
                    routed_tool_spec = None
                    tools_description = policy_tools_description
                elif resolution.status == "blocked":
                    route_record["fallback_reason"] = "deterministic_invalid_arguments"
                    thought = "无法执行：论文索引 ref=0 非法，论文序号必须从 1 开始"
                    observation = "任务因非法论文索引而终止，未调用任何工具"
                    history.append(
                        {"thought": thought, "action": "FINISH", "observation": observation}
                    )
                    step_timings.append(
                        {"router_ms": router_ms, "llm_ms": 0, "tool_ms": 0}
                    )
                    self._log_step(
                        msg_id, iteration, thought, "FINISH", "{}",
                        observation, 0, 0, session_id,
                    )
                    break

            try:
                if deterministic_action is not None:
                    thought = f"使用已验证的显式参数调用 {routed_tool}"
                    action_dict = deterministic_action
                    call_ms = 0
                    usage = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                elif routed_tool and self.router_argument_mode == "guided":
                    thought, action_dict, call_ms, usage = self._request_routed_action(
                        agent_model=agent_model,
                        task=enriched_task,
                        selected_tool=routed_tool,
                        tool_description=tools_description,
                        history_text=history_text,
                    )
                else:
                    thought, action_dict, call_ms, usage = self._request_policy_action(
                        agent_model=agent_model,
                        task=enriched_task,
                        tools_description=tools_description,
                        history_text=history_text,
                    )
                llm_ms += call_ms
                for key in token_usage:
                    token_usage[key] += usage[key]

                if routed_tool and self.router_argument_mode == "guided":
                    assert route_record is not None
                    valid, validation_reason = self._validate_routed_action(
                        action_dict,
                        routed_tool_spec or {},
                    )
                    if valid:
                        model_tool = action_dict.get("name")
                        if model_tool != routed_tool:
                            route_record["model_selected_tool"] = model_tool
                            route_record["tool_name_repaired"] = True
                        action_dict = {
                            "name": routed_tool,
                            "args": dict(action_dict.get("args") or {}),
                        }
                        route_record["used"] = True
                    else:
                        route_record["fallback_reason"] = (
                            "routed_argument_validation_failed"
                        )
                        route_record["argument_validation_error"] = validation_reason
                        thought, action_dict, retry_ms, retry_usage = self._request_policy_action(
                            agent_model=agent_model,
                            task=enriched_task,
                            tools_description=policy_tools_description,
                            history_text=history_text,
                        )
                        llm_ms += retry_ms
                        for key in token_usage:
                            token_usage[key] += retry_usage[key]
                # Legacy external-router semantics: a selected tool merely narrows
                # the ordinary policy prompt.  Preserve this mode for controlled
                # ablations and backwards compatibility.
                elif routed_tool and (
                    action_dict is None or action_dict.get("name") != routed_tool
                ):
                    assert route_record is not None
                    route_record["fallback_reason"] = "policy_router_disagreement"
                    thought, action_dict, retry_ms, retry_usage = self._request_policy_action(
                        agent_model=agent_model,
                        task=enriched_task,
                        tools_description=policy_tools_description,
                        history_text=history_text,
                    )
                    llm_ms += retry_ms
                    for key in token_usage:
                        token_usage[key] += retry_usage[key]
                elif routed_tool and route_record is not None:
                    route_record["used"] = True

                log.info(f"Thought: {thought}")

                if action_dict is None:
                    log.info("任务完成")
                    observation = "任务完成"
                    history.append({"thought": thought, "action": "FINISH", "observation": observation})
                    step_timings.append(
                        {"router_ms": router_ms, "llm_ms": llm_ms, "tool_ms": 0}
                    )
                    self._log_step(msg_id, iteration, thought, "FINISH", "{}", observation, llm_ms, 0, session_id)
                    break

                # 带副作用的工具执行
                t1 = time.time()
                observation = self._execute_with_side_effects(action_dict)
                tool_ms = int((time.time() - t1) * 1000)
                log.info(f"Observation: {observation[:200]}...")

                step_timings.append(
                    {"router_ms": router_ms, "llm_ms": llm_ms, "tool_ms": tool_ms}
                )

                action_str = json.dumps(action_dict, ensure_ascii=False)
                history.append({"thought": thought, "action": action_str, "observation": observation})

                self._log_step(
                    msg_id, iteration, thought,
                    action_dict.get("name", ""), json.dumps(action_dict.get("args", {}), ensure_ascii=False),
                    observation[:4000], llm_ms, tool_ms, session_id,
                )

                if iteration == self.max_iterations - 1:
                    log.warning("达到最大迭代次数，强制结束")
                    history.append({"thought": "达到最大迭代次数", "action": "FORCE_STOP", "observation": "迭代限制"})
                    step_timings.append({"router_ms": 0, "llm_ms": 0, "tool_ms": 0})
                    self._log_step(msg_id, iteration + 1, "达到最大迭代次数", "FORCE_STOP", "", "迭代限制", 0, 0, session_id)
                    break

            except Exception as e:
                error_msg = f"LLM调用失败: {str(e)}"
                log.error(error_msg)
                history.append({"thought": "LLM调用失败", "action": "ERROR", "observation": error_msg})
                step_timings.append(
                    {"router_ms": router_ms, "llm_ms": llm_ms, "tool_ms": 0}
                )
                self._log_step(msg_id, iteration, "LLM调用失败", "ERROR", "", error_msg, llm_ms, 0, session_id)
                break

        final_observation = history[-1]["observation"] if history else "无执行结果"

        reply = ""
        for step in reversed(history):
            if step.get("action") not in ("FINISH", "FORCE_STOP", "ERROR"):
                reply = step.get("observation", "")
                break
        reply = reply or final_observation

        try:
            self.side_effects.create_chat_log(
                session_id, msg_id + "_reply", "assistant", reply,
                model=agent_model, agent_type=self.agent_type,
            )
        except Exception as e:
            log.warning(f"Failed to log assistant reply: {e}")

        total_time_ms = int((time.time() - run_start) * 1000)
        total_router_ms = sum(s.get("router_ms", 0) for s in step_timings)
        total_llm_ms = sum(s["llm_ms"] for s in step_timings)
        total_tool_ms = sum(s["tool_ms"] for s in step_timings)

        result = {
            "task": task,
            "msg_id": msg_id,
            "history": history,
            "final_observation": final_observation,
            "total_time_ms": total_time_ms,
            "iteration_count": len(history),
            "agent_type": self.agent_type,
            "routing": {
                "mode": getattr(self.tool_router, "name", "policy"),
                "decisions": routing_decisions,
            },
            "timing": {
                "total_router_ms": total_router_ms,
                "total_llm_ms": total_llm_ms,
                "total_tool_ms": total_tool_ms,
                "framework_overhead_ms": (
                    total_time_ms - total_router_ms - total_llm_ms - total_tool_ms
                ),
                "steps": step_timings,
            },
            "token_usage": token_usage,
        }
        log.info(
            f"任务执行完成，共 {len(history)} 步, 总耗时 {total_time_ms}ms "
            f"(Router {total_router_ms}ms + LLM {total_llm_ms}ms + Tool {total_tool_ms}ms)"
        )
        log.info("-" * 80)
        return result

    # ---------- LLM 请求参数 ----------

    def _request_policy_action(
        self,
        *,
        agent_model: str,
        task: str,
        tools_description: str,
        history_text: str,
    ) -> Tuple[str, Optional[Dict[str, Any]], int, Dict[str, int]]:
        """Generate and parse one policy action, preserving the original API path."""
        messages, extra = self.build_messages(task, tools_description, history_text)
        extra = self._merge_llm_extra(extra)
        eff_temperature = float(extra.pop("temperature", 0.1))
        started = time.time()
        response = self.llm_client.chat_completions(
            model=agent_model,
            messages=messages,
            temperature=eff_temperature,
            max_tokens=1000,
            stream=False,
            extra=extra or None,
        )
        elapsed_ms = int((time.time() - started) * 1000)
        raw_usage = response.get("usage") or {}
        usage = {
            "prompt_tokens": int(raw_usage.get("prompt_tokens", 0)),
            "completion_tokens": int(raw_usage.get("completion_tokens", 0)),
            "total_tokens": int(raw_usage.get("total_tokens", 0)),
        }
        thought, action = self.parse_response(response)
        return thought, action, elapsed_ms, usage

    def _request_routed_action(
        self,
        *,
        agent_model: str,
        task: str,
        selected_tool: str,
        tool_description: str,
        history_text: str,
    ) -> Tuple[str, Optional[Dict[str, Any]], int, Dict[str, int]]:
        """Generate arguments for an already selected tool and parse one action."""
        messages, extra = self.build_routed_messages(
            task, selected_tool, tool_description, history_text
        )
        extra = self._merge_llm_extra(extra)
        eff_temperature = float(extra.pop("temperature", 0.1))
        started = time.time()
        response = self.llm_client.chat_completions(
            model=agent_model,
            messages=messages,
            temperature=eff_temperature,
            max_tokens=400,
            stream=False,
            extra=extra or None,
        )
        elapsed_ms = int((time.time() - started) * 1000)
        raw_usage = response.get("usage") or {}
        usage = {
            "prompt_tokens": int(raw_usage.get("prompt_tokens", 0)),
            "completion_tokens": int(raw_usage.get("completion_tokens", 0)),
            "total_tokens": int(raw_usage.get("total_tokens", 0)),
        }
        thought, action = self.parse_response(response)
        return thought, action, elapsed_ms, usage

    @classmethod
    def _validate_routed_action(
        cls,
        action: Optional[Dict[str, Any]],
        tool: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Validate routed arguments without adding a JSON-schema dependency."""
        if not isinstance(action, dict):
            return False, "missing_action"
        args = action.get("args", {})
        if not isinstance(args, dict):
            return False, "args_not_object"
        schema = tool.get("parameters") or {}
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        unknown = sorted(set(args) - set(properties)) if properties else []
        if unknown:
            return False, f"unknown_args:{','.join(unknown)}"
        for name in required:
            if name not in args or args[name] is None or args[name] == "":
                return False, f"missing_required:{name}"
        for name, value in args.items():
            spec = properties.get(name) or {}
            if not cls._value_matches_schema(value, spec):
                return False, f"invalid_value:{name}"
        return True, "ok"

    @classmethod
    def _value_matches_schema(cls, value: Any, spec: Dict[str, Any]) -> bool:
        variants = spec.get("anyOf")
        if isinstance(variants, list):
            return any(cls._value_matches_schema(value, item) for item in variants)
        expected = spec.get("type")
        if isinstance(expected, list):
            return any(
                cls._value_matches_schema(value, {**spec, "type": item})
                for item in expected
            )
        type_checks = {
            "null": lambda item: item is None,
            "boolean": lambda item: isinstance(item, bool),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "number": lambda item: (
                isinstance(item, (int, float)) and not isinstance(item, bool)
            ),
            "string": lambda item: isinstance(item, str),
            "object": lambda item: isinstance(item, dict),
            "array": lambda item: isinstance(item, list),
        }
        if expected in type_checks and not type_checks[expected](value):
            return False
        if "enum" in spec and value not in spec["enum"]:
            return False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in spec and value < spec["minimum"]:
                return False
            if "maximum" in spec and value > spec["maximum"]:
                return False
        if isinstance(value, str) and len(value) < int(spec.get("minLength", 0)):
            return False
        return True

    def _merge_llm_extra(self, extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """合并 stop 序列与 llm_extra；build_messages 返回的 extra 优先级最高。"""
        merged: Dict[str, Any] = {}
        if self.stop_sequences:
            merged["stop"] = list(self.stop_sequences)
        merged.update(self.llm_extra)
        merged.update(extra or {})
        return merged

    # ---------- 会话上下文 ----------

    def _enrich_task_with_context(self, task: str, session_id: str) -> str:
        """向任务描述注入当前会话状态，帮助 LLM 避免重复操作"""
        try:
            papers = self.side_effects.get_last_papers(session_id)
            if papers:
                titles = [f"  {i+1}. {p.title}" for i, p in enumerate(papers[:10])]
                ctx = f"\n\n[会话上下文] 当前会话已有 {len(papers)} 篇论文:\n" + "\n".join(titles)
                ctx += "\n可直接用 ref 序号引用，无需重新搜索。"
                return task + ctx
        except Exception:
            pass
        return task

    # ---------- 工具执行（可被 env 接管） ----------

    def _dispatch_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """统一的工具执行入口：注入了 env 就走 env（快照回放），否则走子类实现。"""
        if self.env is not None:
            return self.env.execute_tool(tool_name, args)
        return self.invoke_tool(tool_name, args)

    # ---------- 通用副作用逻辑 ----------

    def _execute_with_side_effects(self, action_dict: Dict[str, Any]) -> str:
        """统一处理 session_id 覆盖、翻译异步、paper 状态写入等副作用"""
        try:
            tool_name = action_dict["name"]
            args = action_dict.get("args", {}) or {}
            # 只在调用层注入 session_id，不写回 action_dict：
            # action_dict 会被 json.dumps 进 history 的 Action JSON，
            # 若 session_id（框架状态）混入，SFT/DPO 数据生成会把
            # 本不该让模型学习的字段当成模型动作学习。
            inject_args = dict(args)

            log.info(f"执行工具: {tool_name}, 参数: {args}")

            # 强制覆盖 session_id
            try:
                tool = registry.get_tool(tool_name)
                props = (tool or {}).get("parameters", {}).get("properties", {})
                if isinstance(args, dict) and ("session_id" in props):
                    inject_args["session_id"] = self.session_id
            except Exception:
                pass

            # 验证工具存在
            available_tools = [t["name"] for t in registry.list_tools()]
            if tool_name not in available_tools:
                return f"错误: 工具 '{tool_name}' 不存在。可用工具包括: {', '.join(available_tools)}"

            # 翻译工具异步 enqueue
            if tool_name == "translate_arxiv_pdf":
                t = self.side_effects.enqueue_translate(
                    session_id=self.session_id,
                    ref=args.get("ref", None),
                    force=bool(args.get("force", False)),
                    service=args.get("service") or settings.pdf2zh_service,
                    threads=int(args.get("threads") or settings.pdf2zh_threads),
                    keep_dual=bool(args.get("keep_dual", False)),
                    paper_id=args.get("paper_id"),
                    pdf_url=args.get("pdf_url"),
                    input_pdf_path=args.get("input_pdf_path"),
                )
                return (
                    f"已创建翻译任务 task_id={t.task_id}, paper_id={t.paper_id}，状态={t.status}。"
                    f"前端可订阅 SSE: /events?session_id={self.session_id}，"
                    f"任务完成后刷新 /translate/assets 或 /pdf/assets。"
                )

            # 调用工具（注入了 env 则优先走 env）
            result = self._dispatch_tool(tool_name, inject_args)

            # paper_id 写入 last_active
            try:
                if isinstance(result, dict):
                    pid = result.get("paper_id")
                    if isinstance(pid, str) and pid.strip():
                        self.side_effects.set_last_active_paper_id(self.session_id, pid.strip())
            except Exception:
                pass

            # 两种 arXiv 搜索结果都存入同一个 session，供后续 ref 解析。
            if tool_name in PAPER_SEARCH_ACTIONS:
                # MCP 模式可能返回 dict（如 {"error": "..."}），兼容处理
                if isinstance(result, dict):
                    if "error" in result:
                        return f"工具执行失败: {result['error']}"
                    vals = list(result.values())
                    if len(vals) == 1 and isinstance(vals[0], list):
                        result = vals[0]
                if isinstance(result, list):
                    if result:
                        # replay 对未知关键词会返回带显式标记的确定性回退池，
                        # 它只用于保持环境可复现，不能伪装成真实命中，更不能
                        # 覆盖当前会话的论文列表，否则后续 ref 会指向无关论文。
                        fallback_meta = next(
                            (
                                paper.get("_mock_env")
                                for paper in result
                                if isinstance(paper, dict)
                                and isinstance(paper.get("_mock_env"), dict)
                                and paper["_mock_env"].get("offline_fallback")
                            ),
                            None,
                        )
                        if fallback_meta:
                            message = fallback_meta.get(
                                "message",
                                "关键词未命中离线快照，回退结果不代表真实匹配。",
                            )
                            return (
                                f"工具执行失败: {message}"
                                "请检查 query 与 days 是否完整匹配任务要求。"
                            )

                        papers_obj = [
                            Paper(**{key: value for key, value in paper.items() if not key.startswith("_")})
                            if isinstance(paper, dict) else paper
                            for paper in result
                        ]
                        self.side_effects.set_last_papers(self.session_id, papers_obj)
                        papers_count = len(result)
                        paper_titles = [paper.get("title", "无标题") for paper in result[:3]]
                        titles_str = "\n".join([f"  - {title}" for title in paper_titles])
                        if papers_count > 3:
                            return f"成功获取 {papers_count} 篇论文。示例论文:\n{titles_str}\n  ... 还有 {papers_count - 3} 篇论文"
                        else:
                            return f"成功获取 {papers_count} 篇论文:\n{titles_str}"
                    else:
                        return "未获取到任何论文记录，请尝试调整搜索参数（如增加天数范围）"
                else:
                    return f"工具返回结果格式异常: {type(result)}, 内容: {str(result)[:200]}"

            # 通用结果格式化
            if isinstance(result, list):
                return f"成功获取 {len(result)} 条记录"
            elif isinstance(result, str):
                return result[:1000] if len(result) > 1000 else result
            else:
                return str(result)[:1000]

        except Exception as e:
            error_msg = f"工具执行失败: {str(e)}"
            log.error(error_msg, exc_info=True)
            return error_msg

    # ---------- 日志 + SSE ----------

    def _log_step(
        self, msg_id: str, step_index: int,
        thought: str, action_name: str, action_args: str,
        observation: str, llm_ms: int, tool_ms: int,
        session_id: str,
    ):
        try:
            self.side_effects.save_agent_step(
                msg_id=msg_id, step_index=step_index,
                thought=thought, action_name=action_name,
                action_args=action_args, observation=observation,
                llm_latency_ms=llm_ms, tool_latency_ms=tool_ms,
            )
            self.side_effects.publish_sse(session_id, {
                "type": "agent_step",
                "step": {
                    "thought": thought, "action_name": action_name,
                    "observation": observation[:500], "step_index": step_index,
                    "llm_latency_ms": llm_ms, "tool_latency_ms": tool_ms,
                },
            })
        except Exception as e:
            log.warning(f"Failed to log step: {e}")
