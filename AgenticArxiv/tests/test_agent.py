# AgenticArxiv/test_agent.py
import sys
import os

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 确保工具被注册
import tools.arxiv_tool

from tools.tool_registry import registry
from agents.prompt_templates import format_tool_description

def test_prompt_generation():
    """测试提示词生成"""
    print("=== 测试工具描述生成 ===")
    
    # 获取工具列表
    tools = registry.list_tools()
    print(f"发现 {len(tools)} 个工具")
    
    # 生成工具描述
    tools_description = format_tool_description(tools)
    print("\n生成的工具描述:")
    print("-" * 50)
    print(tools_description)
    print("-" * 50)
    
    # 测试任务
    task = "获取最近7天内人工智能(AI)领域的最新论文,最多获取10篇"
    print(f"\n任务: {task}")
    
    # 模拟LLM应该做什么
    print("\n预期的LLM推理:")
    print("Thought: 我需要获取最近7天AI领域的论文。")
    print("         可用的工具是 'get_recently_submitted_cs_papers'，它可以按子领域和天数筛选论文。")
    print("         我需要设置 aspect='AI', days=7, max_results=10")
    print("Action: {\"name\": \"get_recently_submitted_cs_papers\", \"args\": {\"aspect\": \"AI\", \"days\": 7, \"max_results\": 10}}")
    
    # 实际执行测试
    print("\n=== 实际执行测试 ===")
    try:
        result = registry.execute_tool(
            "get_recently_submitted_cs_papers",
            {"aspect": "AI", "days": 7, "max_results": 10}
        )
        print(f"成功获取 {len(result)} 篇论文")
        if result:
            print("\n前3篇论文:")
            for i, paper in enumerate(result[:3], 1):
                print(f"{i}. {paper['title']}")
    except Exception as e:
        print(f"执行失败: {e}")

if __name__ == "__main__":
    test_prompt_generation()