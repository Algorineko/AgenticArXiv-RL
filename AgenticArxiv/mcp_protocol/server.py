#!/usr/bin/env python3
"""轻量 MCP Server: 将 ToolRegistry 中的工具暴露为 MCP 协议

作为独立子进程运行，通过 stdio (stdin/stdout) JSON-RPC 通信。

启动方式:
    python -m mcp_protocol.server
"""
import json
import sys
import os

# 确保项目根目录在 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 加载 .env
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 导入工具模块（触发注册）
import tools.arxiv_tool  # noqa: F401
import tools.pdf_download_tool  # noqa: F401
import tools.pdf_translate_tool  # noqa: F401
import tools.cache_status_tool  # noqa: F401

from tools.tool_registry import registry

server = Server("arxiv-tools-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """将 registry 中的工具转为 MCP Tool 对象"""
    tools = []
    for t in registry.list_tools():
        tools.append(
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["parameters"],
            )
        )
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """通过 registry 执行工具，结果序列化为 JSON 文本"""
    try:
        result = registry.execute_tool(name, arguments)
        text = json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        import traceback
        text = json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)
    return [TextContent(type="text", text=text)]


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
