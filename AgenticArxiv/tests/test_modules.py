# AgenticArxiv/tests/test_modules.py
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.arxiv_tool import get_recently_submitted_cs_papers
from utils.llm_client import get_env_llm_client
from config import settings
from tools.tool_registry import registry


def test_arxiv(aspect: str, max_results: int) -> None:
    # 直接调用
    papers = get_recently_submitted_cs_papers(max_results=max_results, aspect=aspect)
    print(f"直接调用获取 {len(papers)} 篇论文")
    
    # 通过注册表调用
    print("\n" + "="*60)
    print("通过工具注册表调用:")
    result = registry.execute_tool(
        "get_recently_submitted_cs_papers",
        {"max_results": max_results, "aspect": aspect}
    )
    print(f"成功获取 {len(result)} 篇论文")


def test_llm(prompt: str) -> None:
    client = get_env_llm_client()
    resp = client.chat_completions(
        model=settings.models.agent_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000,
        stream=False,
    )
    # OpenAI-compatible 的常规字段：choices[0].message.content
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(content or resp)

def test_tools() -> None:
    """测试工具注册表"""
    print("=== 工具注册表测试 ===")
    tools = registry.list_tools()
    print(f"注册的工具数量: {len(tools)}")
    for tool in tools:
        print(f"- {tool['name']}: {tool['description']}")


def main():
    MAX_RESULTS = 20
    CS_ASPECT = "*"

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("arxiv", help="测试 ArXiv 检索")
    p1.add_argument("--aspect", default=CS_ASPECT, help='cs 子领域，如 LO/AI/LG,或 "*"')
    p1.add_argument("--max", type=int, default=MAX_RESULTS, help="最大结果数")

    p2 = sub.add_parser("llm", help="测试 LLM chat/completions")
    p2.add_argument(
        "--prompt", default="请介绍计算机科学领域新颖的研究课题", help="LLM 提示词"
    )

    p3 = sub.add_parser("tools", help="测试工具注册表")
    p3.add_argument("--list", action="store_true", help="列出所有工具")

    args = parser.parse_args()

    if args.cmd == "arxiv":
        test_arxiv(aspect=args.aspect, max_results=args.max)
    elif args.cmd == "llm":
        test_llm(prompt=args.prompt)
    elif args.cmd == "tools": 
        test_tools()

if __name__ == "__main__":
    main()
