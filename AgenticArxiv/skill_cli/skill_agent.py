# AgenticArxiv/skill_cli/skill_agent.py
"""方案 C: Skill/CLI 命令调用 Agent

读取 SKILL.md 文档了解可用命令 → LLM 生成 bash 命令 → subprocess 执行 → 解析 stdout JSON
"""
import json
import os
import re
import shlex
import subprocess
import sys
from typing import Dict, Any, Optional, Tuple, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.base_agent import BaseAgent
from utils.llm_client import LLMClient
from utils.logger import log
from skill_cli.skill_prompt import get_skill_prompt

# SKILL.md 路径
SKILL_DOC_PATH = os.path.join(os.path.dirname(__file__), "SKILL.md")
TOOL_CLI_PATH = os.path.join(os.path.dirname(__file__), "tool_cli.py")

# CLI 子命令 → registry 工具名映射
CLI_TO_REGISTRY = {
    "search_papers": "get_recently_submitted_cs_papers",
    "download_pdf": "download_arxiv_pdf",
    "translate_pdf": "translate_arxiv_pdf",
    "cache_status": "get_paper_cache_status",
}
REGISTRY_TO_CLI = {v: k for k, v in CLI_TO_REGISTRY.items()}


class SkillAgent(BaseAgent):
    """通过 Skill 文档 + CLI 子进程执行工具的 Agent"""
    agent_type = "skill_cli"

    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client)
        self._skill_doc = self._load_skill_doc()

        # 确保底层工具已注册（供 _execute_with_side_effects 使用）
        try:
            import tools.arxiv_tool  # noqa: F401
            import tools.pdf_download_tool  # noqa: F401
            import tools.pdf_translate_tool  # noqa: F401
            import tools.cache_status_tool  # noqa: F401
        except ImportError as e:
            log.warning(f"导入工具模块失败: {e}")

    @staticmethod
    def _load_skill_doc() -> str:
        with open(SKILL_DOC_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        # 去掉 YAML frontmatter，只保留文档正文
        match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
        if match:
            return content[match.end():]
        return content

    def discover_tools(self) -> List[Dict[str, Any]]:
        from tools.tool_registry import registry
        return registry.list_tools()

    def format_tools_for_prompt(self, tools: List[Dict]) -> str:
        # Skill 方案使用 SKILL.md 文档替代 tool description
        return self._skill_doc

    def build_messages(
        self, task: str, tools_description: str, history_text: str
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        prompt = get_skill_prompt(
            task=task,
            skill_document=tools_description,  # 这里是 SKILL.md 正文
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
        return self._parse_skill_text(content)

    def invoke_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """通过 subprocess 执行 CLI 命令（从修正后的 args 重建命令）"""
        cli_name = REGISTRY_TO_CLI.get(tool_name)
        if not cli_name:
            # 非 CLI 工具，回退到 registry 直接调用
            from tools.tool_registry import registry
            return registry.execute_tool(tool_name, args)

        # 从修正后的 args 重建命令（_execute_with_side_effects 已覆盖 session_id）
        clean_args = {k: v for k, v in args.items() if not k.startswith("_")}
        cmd_parts = [sys.executable, TOOL_CLI_PATH, cli_name]
        for k, v in clean_args.items():
            if v is not None:
                cmd_parts.append(f"--{k}={v}")

        log.info(f"[SkillAgent] 执行 CLI: {' '.join(cmd_parts)}")

        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
            )
        except subprocess.TimeoutExpired:
            return "命令执行超时 (120s)"
        except Exception as e:
            return f"命令执行异常: {e}"

        if result.returncode != 0:
            stderr = result.stderr.strip()
            return f"命令失败 (exit {result.returncode}): {stderr[:500]}"

        stdout = result.stdout.strip()
        if not stdout:
            return "命令执行成功（无输出）"

        # 尝试解析为 JSON 返回原始数据（供 _execute_with_side_effects 处理）
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return stdout[:1000]

    def format_history(self, steps: list) -> str:
        """Skill 方案用 Thought/Command/Observation 格式"""
        parts = []
        for s in steps:
            action = s.get("action", "")
            if action in ("FINISH", "FORCE_STOP", "ERROR"):
                parts.append(f"Thought: {s['thought']}\nCommand: {action}\nObservation: {s['observation']}")
            else:
                # 尝试从 action_dict 恢复原始命令
                try:
                    ad = json.loads(action)
                    raw_cmd = ad.get("args", {}).get("_raw_cmd", action)
                except (json.JSONDecodeError, AttributeError):
                    raw_cmd = action
                parts.append(
                    f"Thought: {s['thought']}\nCommand:\n```bash\n{raw_cmd}\n```\nObservation: {s['observation']}"
                )
        return "\n\n".join(parts)

    # ---------- 解析逻辑 ----------

    def _parse_skill_text(self, response: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        # 提取 Thought
        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\nCommand:|$)", response, re.DOTALL
        )
        thought = thought_match.group(1).strip() if thought_match else "未提供思考过程"

        # 提取 Command
        cmd_match = re.search(
            r"Command:\s*(.*?)(?=\nObservation:|$)", response, re.DOTALL
        )
        cmd_text = cmd_match.group(1).strip() if cmd_match else ""

        log.info(f"提取到的Command: {cmd_text[:200]}")

        if cmd_text.upper() == "FINISH":
            return thought, None

        # 提取 ```bash ... ``` 代码块
        bash_match = re.search(r"```(?:bash)?\s*\n?(.*?)\n?```", cmd_text, re.DOTALL)
        if bash_match:
            raw_cmd = bash_match.group(1).strip()
        else:
            # 退化：整行当命令
            raw_cmd = cmd_text.strip()

        if not raw_cmd or raw_cmd.upper() == "FINISH":
            return thought, None

        log.info(f"解析到的CLI命令: {raw_cmd}")

        # 从命令中解析出子命令名和参数
        tool_name, args = self._parse_cli_command(raw_cmd)
        if not tool_name:
            return thought, None

        # 映射到 registry 工具名
        registry_name = CLI_TO_REGISTRY.get(tool_name, tool_name)
        args["_raw_cmd"] = raw_cmd

        return thought, {"name": registry_name, "args": args}

    @staticmethod
    def _parse_cli_command(raw_cmd: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """从 CLI 命令字符串中解析子命令和参数"""
        try:
            parts = shlex.split(raw_cmd)
        except ValueError:
            log.error(f"命令解析失败: {raw_cmd}")
            return None, {}

        # 查找子命令 (search_papers, download_pdf, translate_pdf, cache_status)
        sub_cmd = None
        for i, p in enumerate(parts):
            if p in CLI_TO_REGISTRY:
                sub_cmd = p
                break

        if not sub_cmd:
            log.error(f"未找到有效子命令: {raw_cmd}")
            return None, {}

        # 解析 --key=value 参数
        args = {}
        for p in parts:
            m = re.match(r"--(\w+)=(.+)", p)
            if m:
                key, val = m.group(1), m.group(2)
                # 类型推断
                if val.lower() in ("true", "false"):
                    args[key] = val.lower() == "true"
                elif val.lower() == "none":
                    args[key] = None
                else:
                    try:
                        args[key] = int(val)
                    except ValueError:
                        try:
                            args[key] = float(val)
                        except ValueError:
                            args[key] = val

        return sub_cmd, args
