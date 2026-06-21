# AgenticArxiv/mcp_protocol/mcp_agent.py
"""方案 B: 轻量 MCP 协议 Agent

通过 MCP JSON-RPC 协议发现和调用工具，LLM 交互方式与 ReAct 方案一致。
核心差异：工具调用走 MCP 跨进程通信，而非进程内函数调用。
"""
import asyncio
import json
import os
import re
import sys
from typing import Dict, Any, Optional, Tuple, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.base_agent import BaseAgent
from agents.prompt_templates import get_react_prompt, format_tool_description
from utils.llm_client import LLMClient
from utils.logger import log


MCP_SERVER_MODULE = "mcp_protocol.server"


class MCPAgent(BaseAgent):
    """通过 MCP 协议调用工具的 Agent"""
    agent_type = "mcp"

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client)
        self._session: Optional[ClientSession] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._mcp_tools: List[Dict[str, Any]] = []

        # 确保底层工具已注册（供 _execute_with_side_effects 使用）
        try:
            import tools.arxiv_tool  # noqa: F401
            import tools.pdf_download_tool  # noqa: F401
            import tools.pdf_translate_tool  # noqa: F401
            import tools.cache_status_tool  # noqa: F401
        except ImportError as e:
            log.warning(f"导入工具模块失败: {e}")

    def discover_tools(self) -> List[Dict[str, Any]]:
        if self._mcp_tools:
            return self._mcp_tools
        from tools.tool_registry import registry
        return registry.list_tools()

    def build_messages(
        self, task: str, tools_description: str, history_text: str
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        prompt = get_react_prompt(
            task=task,
            tools_description=tools_description,
            history=history_text,
        )
        return [{"role": "user", "content": prompt}], {}

    def parse_response(self, raw_response: Dict) -> Tuple[str, Optional[Dict[str, Any]]]:
        content = (
            raw_response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        log.debug(f"LLM响应: {content}")
        return self._parse_react_text(content)

    def invoke_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """通过 MCP tools/call JSON-RPC 调用工具"""
        if self._session is None or self._loop is None:
            from tools.tool_registry import registry
            return registry.execute_tool(tool_name, args)

        result = self._call_mcp_tool(tool_name, args)
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return result
        return result

    def run(
        self, task: str, agent_model: str = None, session_id: str = "default"
    ) -> Dict[str, Any]:
        """覆写 run 以在 MCP session 上下文中执行"""
        return self._run_with_mcp(task, agent_model, session_id)

    # ---------- MCP 相关 ----------

    def _run_with_mcp(self, task, agent_model, session_id):
        """在 MCP 会话中运行完整的 agent 循环"""

        async def _async_run():
            server_params = StdioServerParameters(
                command=sys.executable,
                args=["-m", MCP_SERVER_MODULE],
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._loop = asyncio.get_running_loop()

                    # 通过 MCP 协议获取工具列表
                    tools_result = await session.list_tools()
                    self._mcp_tools = [
                        {
                            "name": t.name,
                            "description": t.description or "",
                            "parameters": t.inputSchema if t.inputSchema else {},
                        }
                        for t in tools_result.tools
                    ]
                    log.info(f"[MCPAgent] 通过 MCP 发现 {len(self._mcp_tools)} 个工具")

                    # 关键：在线程池中运行同步的 BaseAgent.run()
                    # 保持 event loop 空闲以处理 MCP JSON-RPC 调用
                    result = await self._loop.run_in_executor(
                        None,
                        lambda: super(MCPAgent, self).run(task, agent_model, session_id),
                    )

                    self._session = None
                    self._loop = None
                    return result

        # 同步/异步桥接
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, _async_run())
                return future.result()
        else:
            return asyncio.run(_async_run())

    def _call_mcp_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        """从工作线程调度 MCP call_tool 到 event loop"""

        async def _call():
            result = await self._session.call_tool(tool_name, arguments=args)
            texts = []
            for item in result.content:
                if hasattr(item, "text"):
                    texts.append(item.text)
            return "\n".join(texts) if texts else ""

        # 从线程池调度到 event loop
        future = asyncio.run_coroutine_threadsafe(_call(), self._loop)
        return future.result(timeout=120)

    # ---------- 正则解析（复用 ReActAgent 的逻辑） ----------

    def _parse_react_text(self, response: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        from tools.tool_registry import registry

        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\nAction:|$)", response, re.DOTALL
        )
        thought = thought_match.group(1).strip() if thought_match else "未提供思考过程"

        action_match = re.search(
            r"Action:\s*(.*?)(?=\nObservation:|$)", response, re.DOTALL
        )
        action_text = action_match.group(1).strip() if action_match else ""

        if action_text.upper() == "FINISH":
            return thought, None

        try:
            json_match = re.search(r"({.*})", action_text, re.DOTALL)
            if json_match:
                action_json = json.loads(json_match.group(1))
                if isinstance(action_json, dict):
                    if "name" in action_json and "args" in action_json:
                        return thought, {"name": action_json["name"], "args": action_json["args"]}
                    else:
                        tool_names = [t["name"] for t in registry.list_tools()]
                        if tool_names:
                            return thought, {"name": tool_names[0], "args": action_json}
        except json.JSONDecodeError:
            pass

        log.error(f"无法解析Action: {action_text}")
        return thought, None
