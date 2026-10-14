"""在独立 replay episode 中显式准备任务的语义参考答案。

策略执行与本服务分别消费同一纯初始化计划。参考标准步骤仅在此服务中执行，
它们不构造模型上下文，也不由评分或历史只读重评分隐式调用。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from agentic_arxiv.config import workspace_path
from agentic_arxiv.tasks.initialization import compile_initialization


PAPER_TOOLS = {
    "download_arxiv_pdf",
    "translate_arxiv_pdf",
    "get_paper_cache_status",
    "get_paper_content",
    "summarize_paper",
    "extract_paper_figures",
    "analyze_figure",
    "get_translated_content",
}


def _call(environment: Any, name: str, args: Dict[str, Any]) -> Any:
    # 标准步骤可能含框架元数据，维持原参考执行的清理边界。
    clean = {k: v for k, v in dict(args or {}).items() if k != "session_id"}
    return environment.step({"name": name, "args": clean})


def attach_expected_paper_ids(
    tasks: Iterable[Dict[str, Any]], snapshot_path: Path | str, *,
    task_lookup: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """返回含逐步 expected_paper_ids 的任务副本，不修改策略 episode。

    搜索步骤目标为 None，论文工具目标来自冻结快照的实际解析结果。
    无法执行的标准轨迹保持立即失败，不能静默弱化评测或奖励。
    """
    from agentic_arxiv.runtime.multiturn_env import AgenticArxivMultiTurnEnv

    path = workspace_path(snapshot_path)
    if not path.exists():
        raise FileNotFoundError(f"语义标准答案需要离线快照: {path}")

    materialized = list(tasks)
    initialization_tasks = dict(task_lookup or {})
    initialization_tasks.update({task.get("id"): task for task in materialized})
    enriched: List[Dict[str, Any]] = []
    for original in materialized:
        task = deepcopy(original)
        tools = list(task.get("expected_tools") or [])
        args_list = task.get("expected_tool_args")
        if args_list is None:
            enriched.append(task)
            continue
        if len(args_list) != len(tools):
            raise ValueError(
                f"任务 {task.get('id')} 的 expected_tools/expected_tool_args 长度不一致"
            )

        environment = AgenticArxivMultiTurnEnv(path, mode="replay")
        try:
            plan = compile_initialization(task, task_lookup=initialization_tasks)
            environment.reset(initialization_plan=plan)

            expected_papers: List[Optional[str]] = []
            for index, (name, args) in enumerate(zip(tools, args_list)):
                result = _call(environment, name, args or {})
                paper_id = result.get("paper_id") if isinstance(result, dict) else None
                if name in PAPER_TOOLS and not paper_id:
                    raise ValueError(
                        f"任务 {task.get('id')} 第 {index + 1} 步 {name} 未解析到 paper_id"
                    )
                expected_papers.append(str(paper_id) if paper_id else None)
        except Exception as exc:
            raise ValueError(
                f"任务 {task.get('id')} 的离线语义标准答案无法执行: {exc}"
            ) from exc
        finally:
            environment.close()

        task["expected_paper_ids"] = expected_papers
        enriched.append(task)
    return enriched
