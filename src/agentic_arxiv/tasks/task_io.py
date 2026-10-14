"""训练 / 评测入口共用的任务池加载。

任务池二选一（默认 tasks/tasks.py 的冒烟任务，--task-set expanded
换完整基准集），GRPO 还额外支持按切分过滤。各入口的
冒烟集提示文案仍留在各自文件里，避免一份文案伺候三种语境。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_arxiv.tasks.splits import load_split


def load_tasks(task_set: str, split: Optional[str] = None, *,
               workspace: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    """按名字选任务池，可选按切分过滤。

    task_set="expanded" 取 tasks/tasks_expanded.py 的完整基准集，
    其余取值一律回落到 tasks/tasks.py 的冒烟任务（默认值不能变：
    换了会让此前所有训练曲线不可比）。

    split 走 tasks.splits.load_split：只保留切分里的任务，并按切分
    文件里的 id 排序输出，保证同一份切分每次拿到同样的顺序；切分里出现
    当前任务池没有的 id 时直接报错——切分按完整任务集划定，缺任务说明
    任务池选错了，此时训练集会静默变小，训出来的东西与切分不对应。
    """
    # 延迟导入：轻量入口 import 时不必加载任务集。
    if task_set == "expanded":
        from agentic_arxiv.tasks.tasks_expanded import get_expanded_tasks
        pool = get_expanded_tasks()
    else:
        from agentic_arxiv.tasks.tasks import get_all_tasks
        pool = get_all_tasks()

    if not split:
        return pool

    wanted = set(load_split(split, workspace=workspace))
    by_id = {t["id"]: t for t in pool}
    chosen = [by_id[tid] for tid in sorted(wanted) if tid in by_id]
    missing = wanted - set(by_id)
    if missing:
        raise SystemExit(
            f"❌ 切分 '{split}' 里有 {len(missing)} 条任务不在当前任务集中，"
            f"例如 {sorted(missing)[:3]}\n"
            "   切分按 --task_set expanded 的完整任务集划定"
        )
    print(f"📑 使用切分 {split}（{len(chosen)} 条）")
    return chosen
