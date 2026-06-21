# AgenticArxiv/tests/test_tool_registry.py
import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 注册工具
import tools.arxiv_tool  # noqa: F401
import tools.pdf_download_tool  # noqa: F401
import tools.pdf_translate_tool  # noqa: F401

from tools.tool_registry import registry
from tools.arxiv_tool import cs_categories


def test_list_tools():
    """测试列出所有工具"""
    print("=== 可用工具列表 ===")
    tools = registry.list_tools()
    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool['name']}")
        print(f"   描述: {tool['description']}")
        print(
            f"   参数: {json.dumps(tool['parameters'], ensure_ascii=False, indent=4)}"
        )
        print()


def test_execute_tool(name: str, args: dict):
    """测试执行特定工具"""
    print(f"=== 测试工具: {name} ===")
    print(f"参数: {json.dumps(args, ensure_ascii=False)}")

    try:
        result = registry.execute_tool(name, args)
        if name == "get_recently_submitted_cs_papers":
            print(f"成功获取 {len(result)} 篇论文")
            # 显示前几篇论文的标题
            for i, paper in enumerate(result[:3], 1):
                print(f"  {i}. {paper['title'][:50]}...")
            if len(result) > 3:
                print("  ...")
        elif name == "format_papers_console":
            print("格式化结果:")
            print(result[:200] + "..." if len(result) > 200 else result)
    except Exception as e:
        print(f"工具执行失败: {str(e)}")


def test_aspect_validation():
    """测试参数验证"""
    print("=== 测试参数验证 ===")

    # 测试无效的aspect
    try:
        result = registry.execute_tool(
            "get_recently_submitted_cs_papers",
            {"aspect": "INVALID_CATEGORY", "max_results": 2},
        )
        print(type(result))
    except Exception as e:
        print(f"参数验证生效: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="测试工具注册表")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list 命令
    subparsers.add_parser("list", help="列出所有注册的工具")

    # test 命令
    test_parser = subparsers.add_parser("test", help="测试特定工具")
    test_parser.add_argument(
        "--tool",
        required=True,
        choices=["arxiv", "format"],
        help="要测试的工具: arxiv 或 format",
    )
    test_parser.add_argument(
        "--max-results", type=int, default=3, help="最大结果数（仅限arxiv工具）"
    )
    test_parser.add_argument(
        "--aspect",
        default="AI",
        help=f"CS子领域，可选值: {', '.join(list(cs_categories.keys())[:5])}...",
    )
    test_parser.add_argument(
        "--days", type=int, default=7, help="查询天数（仅限arxiv工具）"
    )

    # validate 命令
    subparsers.add_parser("validate", help="测试参数验证")

    args = parser.parse_args()

    if args.command == "list":
        test_list_tools()

    elif args.command == "test":
        if args.tool == "arxiv":
            test_execute_tool(
                "get_recently_submitted_cs_papers",
                {
                    "max_results": args.max_results,
                    "aspect": args.aspect,
                    "days": args.days,
                },
            )
        elif args.tool == "format":
            # 先获取一些论文数据用于测试格式化工具
            papers = registry.execute_tool(
                "get_recently_submitted_cs_papers", {"max_results": 2, "aspect": "AI"}
            )
            test_execute_tool(
                "format_papers_console", {"papers": papers, "top_authors": 2}
            )

    elif args.command == "validate":
        test_aspect_validation()


if __name__ == "__main__":
    main()
