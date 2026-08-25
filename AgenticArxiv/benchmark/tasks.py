# AgenticArxiv/benchmark/tasks.py
"""标准化测试任务集，用于对比三种 Agent 模式的性能和准确性。

这里是 8 条冒烟任务；完整的 58 条基准集在 `benchmark/tasks_expanded.py`
（`--task-set expanded`）。两边共用 `task_spec.TaskSpec`：`expected_tools`
与 `expected_tool_args` 由同一份 `steps` 派生，不再手写两份平行列表。

改成派生不是为了少敲字。原来 `download_01` / `translate_01` / `cache_01`
三条只写了 `expected_tools`，`expected_tool_args` 缺省为 None，于是
`argument_match_score` 返回 None、`RewardCalculator` 把 argument 档整个
踢出加权分母——任务照跑、分照打，参数从此不再被检查。代价是可以量出来的：
`benchmark/run_baselines.py` 里那个纯随机挑一个工具的退化策略，只要蒙对
工具名就能在这三条上拿到满分 1.000，因为没有任何参数标准答案能拆穿它。
"""

from typing import List, Dict, Any, Optional

from benchmark.task_spec import Step, TaskSpec, build


BENCHMARK_SPECS: List[TaskSpec] = [
    # === 类型 1: 简单搜索（单工具） ===
    # 四条的措辞刻意不统一（检索 / 获取 / 搜索），用来看模型对表述的鲁棒性，
    # 因此不套 family() 模板——模板会把描述抹平成一个句式。
    TaskSpec(
        id="search_01",
        task="检索最近7天内人工智能(cs.AI)方向的论文，最多5篇",
        steps=(Step("get_recently_submitted_cs_papers",
                    {"aspect": "AI", "days": 7, "max_results": 5}),),
        category="search",
    ),
    TaskSpec(
        id="search_02",
        task="获取最近3天机器学习(cs.LG)方向的最新论文，最多10篇",
        steps=(Step("get_recently_submitted_cs_papers",
                    {"aspect": "LG", "days": 3, "max_results": 10}),),
        category="search",
    ),
    TaskSpec(
        id="search_03",
        task="搜索最近7天自然语言处理(cs.CL)方向的论文，最多5篇",
        steps=(Step("get_recently_submitted_cs_papers",
                    {"aspect": "CL", "days": 7, "max_results": 5}),),
        category="search",
    ),
    TaskSpec(
        id="search_04",
        task="检索最近7天计算机科学全部方向的论文，最多10篇",
        steps=(Step("get_recently_submitted_cs_papers",
                    {"aspect": "*", "days": 7, "max_results": 10}),),
        category="search",
    ),

    # === 类型 2: 下载 PDF（依赖搜索） ===
    TaskSpec(
        id="download_01",
        task="下载第1篇论文的PDF",
        steps=(Step("download_arxiv_pdf", {"ref": 1}),),
        category="download",
        depends_on="search_01",
    ),

    # === 类型 3: 翻译 PDF（异步任务，依赖下载） ===
    TaskSpec(
        id="translate_01",
        task="翻译第1篇论文",
        steps=(Step("translate_arxiv_pdf", {"ref": 1}),),
        category="translate",
        depends_on="download_01",
    ),

    # === 类型 4: 缓存查询（依赖搜索） ===
    TaskSpec(
        id="cache_01",
        task="查看第1篇论文的缓存状态",
        steps=(Step("get_paper_cache_status", {"ref": 1}),),
        category="cache",
        depends_on="search_01",
    ),

    # === 类型 5: 多步骤复合任务 ===
    TaskSpec(
        id="composite_01",
        task="搜索最近7天计算机视觉(cs.CV)的论文(最多3篇)，然后下载第1篇",
        steps=(
            Step("get_recently_submitted_cs_papers",
                 {"aspect": "CV", "days": 7, "max_results": 3}),
            # 原来这一步写的是 None（不校验参数）。任务明说了「第1篇」，
            # 有标准答案就该校验，否则第二步的参数是免检的。
            Step("download_arxiv_pdf", {"ref": 1}),
        ),
        category="composite",
    ),
]


BENCHMARK_TASKS: List[Dict[str, Any]] = build(BENCHMARK_SPECS)


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
