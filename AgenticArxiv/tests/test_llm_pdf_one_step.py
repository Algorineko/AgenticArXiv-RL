# AgenticArxiv/tests/test_llm_pdf_one_step.py
import os
import sys
import json
import re
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.arxiv_tool  # noqa: F401
import tools.pdf_download_tool  # noqa: F401
import tools.pdf_translate_tool  # noqa: F401

from tools.tool_registry import registry
from utils.llm_client import get_env_llm_client
from config import settings


def build_single_step_prompt(task: str) -> str:
    tools = registry.list_tools()

    # 为了让 LLM 更稳定地只选 PDF 工具，这里只展示两个工具
    allow = {"download_arxiv_pdf", "translate_arxiv_pdf"}
    tools = [t for t in tools if t["name"] in allow]

    tools_desc = []
    for t in tools:
        tools_desc.append(
            f"- {t['name']}: {t['description']}\n  parameters: {json.dumps(t['parameters'], ensure_ascii=False)}"
        )

    return f"""你是一个工具调用助手。你只能从下面工具中选择一个执行一次（单步），并输出严格 JSON：

可用工具：
{chr(10).join(tools_desc)}

任务：{task}

输出格式（必须严格匹配，只输出一段JSON，不要额外文字）：
{{"name":"工具名","args":{{...}}}}
"""


def parse_action_json(llm_text: str) -> dict:
    # 允许 LLM 输出里夹带一点文字，这里抓第一个 {...}
    m = re.search(r"(\{.*\})", llm_text, re.DOTALL)
    if not m:
        raise ValueError(f"LLM输出无法解析为JSON: {llm_text[:200]}")
    obj = json.loads(m.group(1))
    if not isinstance(obj, dict) or "name" not in obj or "args" not in obj:
        raise ValueError(f"LLM JSON格式不符合要求: {obj}")
    return obj


def run_one_step(task: str):
    client = get_env_llm_client()
    prompt = build_single_step_prompt(task)

    resp = client.chat_completions(
        model=settings.models.agent_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
        stream=False,
    )
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    print("=== LLM 输出 ===")
    print(content)

    action = parse_action_json(content)
    name = action["name"]
    args = action["args"]

    print("\n=== 执行工具 ===")
    print("tool =", name)
    print("args =", json.dumps(args, ensure_ascii=False))

    result = registry.execute_tool(name, args)
    print("\n=== 工具结果 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task",
        default="请翻译 paper_id=2602.11144v1 的PDF，force=false，service=bing，threads=4，keep_dual=false。注意：只允许调用一次工具。",
    )
    args = ap.parse_args()
    run_one_step(args.task)


if __name__ == "__main__":
    main()
