"""Derive semantic paper targets by replaying each task's gold steps.

The benchmark accepts several equivalent ``ref`` forms (one-based index,
arXiv id, title fragment, or ``null`` for the last active paper).  Comparing
the raw argument therefore produces both false positives and false negatives.
In offline mode the snapshot is the frozen world state, so we can execute the
declarative setup and gold steps once and record the paper id each paper tool
actually resolves to.  Benchmark and RL then share the same semantic targets.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


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
    # setup dictionaries used by Benchmark may carry framework metadata.
    clean = {k: v for k, v in dict(args or {}).items() if k != "session_id"}
    return environment.step({"name": name, "args": clean})


def attach_expected_paper_ids(
    tasks: Iterable[Dict[str, Any]], snapshot_path: Path | str, *,
    task_lookup: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return copied tasks enriched with per-step semantic paper ids.

    ``expected_paper_ids`` is parallel to ``expected_tools``.  Search steps
    contain ``None``; paper operations contain the arXiv id resolved by the
    frozen offline environment.  A broken gold trajectory fails fast instead
    of silently weakening the evaluator or the RL reward.
    """
    from rl.multiturn_env import AgenticArxivMultiTurnEnv

    path = Path(snapshot_path)
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
            # Gold actions remain on the evaluator side.  Initialization uses
            # the same setup/dependency boundary as policy episodes.
            environment.reset(task_def=task, task_lookup=initialization_tasks)

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
