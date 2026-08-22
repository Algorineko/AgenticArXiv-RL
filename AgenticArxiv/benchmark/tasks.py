# AgenticArxiv/benchmark/tasks.py
"""标准化测试任务集，用于对比三种 Agent 模式的性能和准确性。"""

from typing import List, Dict, Any, Optional


BENCHMARK_TASKS: List[Dict[str, Any]] = [
    # === 类型 1: 简单搜索（单工具） ===
    {
        "id": "search_01",
        "task": "检索最近7天内人工智能(cs.AI)方向的论文，最多5篇",
        "expected_tools": ["get_recently_submitted_cs_papers"],
        "expected_tool_args": [
            {"aspect": "AI", "days": 7, "max_results": 5}
        ],
        "expected_termination": "FINISH",
        "category": "search",
    },
    {
        "id": "search_02",
        "task": "获取最近3天机器学习(cs.LG)方向的最新论文，最多10篇",
        "expected_tools": ["get_recently_submitted_cs_papers"],
        "expected_tool_args": [
            {"aspect": "LG", "days": 3, "max_results": 10}
        ],
        "expected_termination": "FINISH",
        "category": "search",
    },
    {
        "id": "search_03",
        "task": "搜索最近7天自然语言处理(cs.CL)方向的论文，最多5篇",
        "expected_tools": ["get_recently_submitted_cs_papers"],
        "expected_tool_args": [
            {"aspect": "CL", "days": 7, "max_results": 5}
        ],
        "expected_termination": "FINISH",
        "category": "search",
    },
    {
        "id": "search_04",
        "task": "检索最近7天计算机科学全部方向的论文，最多10篇",
        "expected_tools": ["get_recently_submitted_cs_papers"],
        "expected_tool_args": [
            {"aspect": "*", "days": 7, "max_results": 10}
        ],
        "expected_termination": "FINISH",
        "category": "search",
    },

    # === 类型 2: 下载 PDF（依赖搜索） ===
    {
        "id": "download_01",
        "task": "下载第1篇论文的PDF",
        "expected_tools": ["download_arxiv_pdf"],
        "expected_termination": "FINISH",
        "category": "download",
        "depends_on": "search_01",
    },

    # === 类型 3: 翻译 PDF（异步任务，依赖下载） ===
    {
        "id": "translate_01",
        "task": "翻译第1篇论文",
        "expected_tools": ["translate_arxiv_pdf"],
        "expected_termination": "FINISH",
        "category": "translate",
        "depends_on": "download_01",
    },

    # === 类型 4: 缓存查询（依赖搜索） ===
    {
        "id": "cache_01",
        "task": "查看第1篇论文的缓存状态",
        "expected_tools": ["get_paper_cache_status"],
        "expected_termination": "FINISH",
        "category": "cache",
        "depends_on": "search_01",
    },

    # === 类型 5: 多步骤复合任务 ===
    {
        "id": "composite_01",
        "task": "搜索最近7天计算机视觉(cs.CV)的论文(最多3篇)，然后下载第1篇",
        "expected_tools": ["get_recently_submitted_cs_papers", "download_arxiv_pdf"],
        "expected_tool_args": [
            {"aspect": "CV", "days": 7, "max_results": 3},
            None,
        ],
        "expected_termination": "FINISH",
        "category": "composite",
    },

    # === 类型 6: 开源代码仓库检索与下载 ===
    {
        "id": "github_search_01",
        "task": "在 GitHub 搜索 Python 实现的 RAG 项目，按 stars 降序返回前5个",
        "expected_tools": ["search_github_repositories"],
        "expected_tool_args": [{"query": "RAG", "language": "Python", "sort": "stars", "order": "desc", "max_results": 5}],
        "expected_termination": "FINISH", "category": "code_search",
    },
    {
        "id": "github_search_02",
        "task": "去 GitHub 找最近更新的 TypeScript MCP server 项目，最多3个",
        "expected_tools": ["search_github_repositories"],
        "expected_tool_args": [{"query": "MCP server", "language": "TypeScript", "sort": "updated", "order": "desc", "max_results": 3}],
        "expected_termination": "FINISH", "category": "code_search",
    },
    {
        "id": "gitee_search_01",
        "task": "在 Gitee 搜索 Java 微服务项目，按 stars 降序返回前5个",
        "expected_tools": ["search_gitee_repositories"],
        "expected_tool_args": [{"query": "微服务", "language": "Java", "sort": "stars", "order": "desc", "max_results": 5}],
        "expected_termination": "FINISH", "category": "code_search",
    },
    {
        "id": "gitee_search_02",
        "task": "去 Gitee 找 Python 大模型应用项目，按最近更新排序，最多3个",
        "expected_tools": ["search_gitee_repositories"],
        "expected_tool_args": [{"query": "大模型应用", "language": "Python", "sort": "updated", "order": "desc", "max_results": 3}],
        "expected_termination": "FINISH", "category": "code_search",
    },
    {
        "id": "github_download_direct_01",
        "task": "下载 GitHub 仓库 langchain-ai/langchain 的 main 分支源码",
        "expected_tools": ["download_github_repository"],
        "expected_tool_args": [{"repository": "langchain-ai/langchain", "ref": "main"}],
        "expected_termination": "FINISH", "category": "code_download",
    },
    {
        "id": "gitee_download_direct_01",
        "task": "下载 Gitee 仓库 dromara/hutool 的 v5-master 分支源码",
        "expected_tools": ["download_gitee_repository"],
        "expected_tool_args": [{"repository": "dromara/hutool", "ref": "v5-master"}],
        "expected_termination": "FINISH", "category": "code_download",
    },
    {
        "id": "github_search_download_01",
        "task": "在 GitHub 搜索 Python Agent 框架，按 stars 排序取前3个，然后下载第1个仓库",
        "expected_tools": ["search_github_repositories", "download_github_repository"],
        "expected_tool_args": [
            {"query": "AI agent framework", "language": "Python", "sort": "stars", "order": "desc", "max_results": 3},
            {"repository": 1},
        ],
        "expected_termination": "FINISH", "category": "code_composite",
    },
    {
        "id": "gitee_search_download_01",
        "task": "在 Gitee 搜索 Java 工作流引擎，返回前3个，然后下载第2个仓库",
        "expected_tools": ["search_gitee_repositories", "download_gitee_repository"],
        "expected_tool_args": [
            {"query": "工作流引擎", "language": "Java", "sort": "stars", "order": "desc", "max_results": 3},
            {"repository": 2},
        ],
        "expected_termination": "FINISH", "category": "code_composite",
    },
    {
        "id": "cross_platform_search_01",
        "task": "分别在 GitHub 和 Gitee 搜索向量数据库项目，各返回3个，不要下载",
        "expected_tools": ["search_github_repositories", "search_gitee_repositories"],
        "expected_tool_args": [
            {"query": "vector database", "max_results": 3},
            {"query": "向量数据库", "max_results": 3},
        ],
        "expected_termination": "FINISH", "category": "code_composite",
    },
    {
        "id": "github_tag_download_01",
        "task": "从 GitHub 下载 psf/requests 仓库的 v2.32.3 标签源码，不需要先搜索",
        "expected_tools": ["download_github_repository"],
        "expected_tool_args": [{"repository": "psf/requests", "ref": "v2.32.3"}],
        "expected_termination": "FINISH", "category": "code_download",
    },
]


def get_tasks_by_category(category: str) -> List[Dict[str, Any]]:
    return [t for t in BENCHMARK_TASKS if t["category"] == category]


def get_task_by_id(task_id: str) -> Optional[Dict[str, Any]]:
    for t in BENCHMARK_TASKS:
        if t["id"] == task_id:
            return t
    return None


def get_all_tasks() -> List[Dict[str, Any]]:
    return list(BENCHMARK_TASKS)


def get_dependency_chain(task_id: str) -> List[str]:
    """返回任务的依赖链（从最早的依赖到当前任务）"""
    chain = []
    current = task_id
    while current:
        task = get_task_by_id(current)
        if task is None:
            break
        chain.append(current)
        current = task.get("depends_on")
    chain.reverse()
    return chain
