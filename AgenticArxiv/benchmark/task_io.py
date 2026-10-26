# AgenticArxiv/benchmark/task_io.py
"""训练 / 评测入口共用的任务池加载。

train_grpo / train_opd / badcase_replay 此前各有一份 ``_load_tasks``：
任务池二选一（默认 benchmark/tasks.py 的冒烟任务，--task_set expanded
换完整基准集），GRPO 还额外支持按切分过滤。这里收敛成一份；各入口的
冒烟集提示文案仍留在各自文件里，避免一份文案伺候三种语境。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from benchmark.splits import load_split

#: BFCL 任务的提交协议说明（拼进任务文本；submit_calls 契约见 rl/bfcl_env.py）。
#: calls 是「JSON 数组字符串」，元素形如 'name(k=v, ...)' —— 元素内的调用
#: 参数统一用单引号，避免与外层 JSON 双引号转义打架。
_BFCL_PROTOCOL = (
    "[提交协议] 本环境只有一个可调用工具 submit_calls：把你的完整函数调用链"
    "一次性提交，不要逐个函数分多轮调用。每个调用写成文本形式 name(k=v, ...)，"
    "参数用字面量（字符串参数用单引号），全部调用按顺序放进一个 JSON 数组，"
    "再整体作为 calls 字符串参数提交：\n"
    'Action: {"name": "submit_calls", "args": {"calls": '
    '"[\\"cd(folder=\'docs\')\\", \\"send_mail(to=\'a@b.com\', subject=\'hi\')\\"]"}}\n'
    "即 calls 的值是一个 JSON 数组字符串：数组元素用转义的双引号包住，调用内的"
    "字符串参数用单引号。提交后用 Action: FINISH 结束。"
)


def _bfcl_signature(tool: Dict[str, Any]) -> str:
    """一个 BFCL 函数 -> 'name(req[, opt: type])' 紧凑签名。

    本域 parquet 的 schema 形状与全局注册表不同：properties 里大量键取值为
    null（数据集级参数超集，与当前工具无关），format_tool_description 对
    None spec 会直接 AttributeError。有意义的参数 = required 列表 +
    properties 里带 dict spec 的可选键（gold 链会用到的可选参数，如
    tail(lines=N)、ls(a=True)，都在这一档）。
    """
    params = tool.get("parameters")
    params = params if isinstance(params, dict) else {}
    props = params.get("properties")
    props = props if isinstance(props, dict) else {}
    required = params.get("required")
    required = [str(name) for name in list(required)] if required is not None else []

    def render(names: List[str]) -> str:
        parts = []
        for name in names:
            spec = props.get(name)
            spec = spec if isinstance(spec, dict) else {}
            kind = spec.get("type")
            parts.append(f"{name}: {kind}" if kind else name)
        return ", ".join(parts)

    signature = render(required)
    optional = [
        str(key) for key, spec in props.items()
        if isinstance(spec, dict) and str(key) not in required
    ]
    optional_text = render(optional)
    if optional_text:
        signature = f"{signature}[, {optional_text}]" if signature else f"[{optional_text}]"
    return signature


def _bfcl_task_text(task: Dict[str, Any]) -> str:
    """BFCL 任务 -> ReAct 任务文本: 用户问题 + 逐任务函数 schema + 提交协议。

    BFCL 的可用函数逐任务不同（schema 由任务自带），全局注册表的 arXiv
    工具对它没有意义。沿用代码域先例（域工具说明写进任务文本，prompt 模板
    里的全局工具列表当作噪音），把 schema 与元工具 submit_calls 一起写进
    任务文本，build_prompt_dataset 无需感知域差异。
    """
    functions = "\n".join(
        f"- {tool.get('name', '')}({_bfcl_signature(tool)}): {tool.get('description', '')}"
        for tool in task["tools"]
    )
    return (
        f"{task['prompt']}\n\n"
        "本任务可用的函数及其参数（必选参数在前，方括号内为可选参数）：\n"
        f"{functions}\n\n"
        f"{_BFCL_PROTOCOL}"
    )


def load_tasks(task_set: str, split: Optional[str] = None) -> List[Dict[str, Any]]:
    """按名字选任务池，可选按切分过滤。

    task_set="expanded" 取 benchmark/tasks_expanded.py 的完整基准集，
    task_set="bfcl" 取 benchmark/tasks_bfcl.py 的 BFCL-V3 multi_turn_base
    200 任务（任务文本由 _bfcl_task_text 现场拼装），其余取值一律回落到
    benchmark/tasks.py 的冒烟任务（默认值不能变：换了会让此前所有训练
    曲线不可比）。

    split 走 benchmark.splits.load_split：只保留切分里的任务，并按切分
    文件里的 id 排序输出，保证同一份切分每次拿到同样的顺序；切分里出现
    当前任务池没有的 id 时直接报错——切分按完整任务集划定，缺任务说明
    任务池选错了，此时训练集会静默变小，训出来的东西与切分不对应。
    """
    # 延迟导入：badcase_replay 是轻量 CI 入口，import 时不必加载任务集。
    if task_set == "code":
        from AgenticArxiv.benchmark.tasks_code import get_code_tasks
        pool = get_code_tasks()
    elif task_set == "bfcl":
        from AgenticArxiv.benchmark.tasks_bfcl import get_bfcl_tasks
        pool = [{**task, "task": _bfcl_task_text(task)} for task in get_bfcl_tasks()]
    elif task_set == "expanded":
        from benchmark.tasks_expanded import get_expanded_tasks
        pool = get_expanded_tasks()
    else:
        from benchmark.tasks import get_all_tasks
        pool = get_all_tasks()

    if not split:
        return pool

    wanted = set(load_split(split))
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
