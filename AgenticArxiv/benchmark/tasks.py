# AgenticArxiv/benchmark/tasks.py
"""标准化测试任务集，用于对比三种 Agent 模式的性能和准确性。"""

from typing import List, Dict, Any, Optional


BENCHMARK_TASKS: List[Dict[str, Any]] = [
    # === 类型 1: 简单搜索（单工具） ===
    {
        "id": "search_01",
        "task": "检索最近7天内人工智能(cs.AI)方向的论文，最多5篇",
        "expected_tools": ["get_recently_submitted_cs_papers"],
        "expected_termination": "FINISH",
        "category": "search",
    },
    {
        "id": "search_02",
        "task": "获取最近3天机器学习(cs.LG)方向的最新论文，最多10篇",
        "expected_tools": ["get_recently_submitted_cs_papers"],
        "expected_termination": "FINISH",
        "category": "search",
    },
    {
        "id": "search_03",
        "task": "搜索最近7天自然语言处理(cs.CL)方向的论文，最多5篇",
        "expected_tools": ["get_recently_submitted_cs_papers"],
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
        "expected_termination": "FINISH",
        "category": "composite",
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
