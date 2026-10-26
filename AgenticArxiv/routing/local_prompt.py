"""本地 next-tool 判别头的纯文本层（无 torch 依赖，可被 CI 裸 import）。

训练（``rl/train_jev_rlcd.py``）、评测（``scripts/eval_jev_local_router.py``）
与线上 backend 三层共用这里的渲染，保证 prompt 逐字节同源。

设计约定：

- prompt 不入库。数据行只存 ``state_text``（会话状态），判据文案在本层
  拼装——补改判据不需要重写数据集，manifest 钉住
  ``ROUTER_PROMPT_VERSION`` + 判据文本 SHA 即可复现；
- 判据复用 :data:`routing.jev.ROUTING_CRITERIA`（外部 Jev 与本地头共用，
  口径一致）；
- 候选后缀统一为 ``Next tool: {name}``，11 类（registry 10 工具 + FINISH）
  顺序固定，softmax 分布的类序即此顺序。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from routing.jev import FINISH_CRITERION, ROUTING_CRITERIA

ROUTER_PROMPT_VERSION = "jev_router_prompt_v1"

FINISH_TOOL = "FINISH"

# 打分后缀的前缀与填空格式。trailing 空格会被 BPE 吃掉，留给自己断言。
CANDIDATE_SUFFIX_TEMPLATE = "Next tool: {name}"
# 共享前缀（verbalizer scoring 时从 z 聚合中剔除的 token 段）
CANDIDATE_SUFFIX_PREFIX = CANDIDATE_SUFFIX_TEMPLATE.format(name="")


def candidate_names(tools: Sequence[Mapping[str, Any]] | None = None) -> List[str]:
    """候选类序：registry 给定顺序的 10 工具 + FINISH（固定第 11 类）。"""
    names = [str(tool.get("name") or "") for tool in (tools or [])]
    return [name for name in names if name] + [FINISH_TOOL]


def router_criteria(tools: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """与 :meth:`routing.jev.JevToolRouter.route` 完全同构的判据表。

    区分性判据（ROUTING_CRITERIA）缺失的工具退回 description 兜底——
    与外部 Jev 同一规则，两个路径的可见信息永远一致。
    """
    criteria: Dict[str, str] = {}
    for tool in tools:
        name = str(tool.get("name") or "")
        if not name:
            continue
        description = str(tool.get("description") or name)
        distinction = ROUTING_CRITERIA.get(name)
        criteria[name] = (
            f"{distinction} Project tool description: {description}"
            if distinction
            else description
        )
    criteria[FINISH_TOOL] = FINISH_CRITERION
    return criteria


def candidate_suffix_texts(names: Sequence[str]) -> List[str]:
    """每个候选的打分后缀文本（与 ``candidate_names`` 同序）。"""
    return [CANDIDATE_SUFFIX_TEMPLATE.format(name=name) for name in names]


def bounded_state(state_text: str, max_chars: int = 12000) -> str:
    """会话状态截断（尾部保留），与 ``JevToolRouter._bounded_history`` 同口径。"""
    clean = (state_text or "").strip() or "(no prior tool observations)"
    if len(clean) <= max_chars:
        return clean
    return "[earlier context truncated]\n" + clean[-max_chars:]


def build_router_prompt(
    *,
    task: str,
    state_text: str,
    tools: Sequence[Mapping[str, Any]],
    max_state_chars: int = 12000,
) -> str:
    """拼装判别头的 user prompt（训练/评测/线上唯一来源）。"""
    criteria = router_criteria(tools)
    lines = [
        "You are the next-tool router of an arXiv research agent. "
        "Pick the single tool that should be called next, or FINISH when "
        "no tool call is needed.",
        "",
        "[Task]",
        task.strip(),
        "",
        "[Session state]",
        bounded_state(state_text, max_state_chars),
        "",
        "[Candidate tools]",
    ]
    for name in candidate_names(tools):
        lines.append(f"- {name}: {criteria[name]}")
    lines += [
        "",
        "Answer with exactly one line in the format:",
        f"{CANDIDATE_SUFFIX_TEMPLATE.format(name='<name from the candidate list>')}",
    ]
    return "\n".join(lines)
