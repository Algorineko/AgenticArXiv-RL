"""工具注册的统一入口。

## 为什么需要这个模块

三个 Agent（regex / mcp / skill_cli）各自复制了同一段注册代码：

    try:
        import tools.arxiv_tool
        import tools.pdf_download_tool
        import tools.pdf_translate_tool
        import tools.cache_status_tool
    except ImportError as e:
        log.warning(f"导入工具模块失败: {e}")

四个 import 共用一个 try，所以**缺任何一个依赖，四个工具就全都注册不上**。
`tools.arxiv_tool` 依赖第三方包 `arxiv`，它排在第一个 —— 环境里没有 `arxiv`
时，连纯本地、零依赖的 `cache_status_tool` 也一起消失。

后果不是报错而是**换了个人格**：registry 为空 → prompt 里的工具列表是空的
→ 模型开始编工具名。实测在一台没装 `arxiv` 的机器上跑 benchmark，模型调用了
`search_arxiv`、`check_download_history` 这些根本不存在的工具，96 条运行全部
正常产出报告、给出成功率、写进 CSV，没有任何一处提示工具没装上。

这里把注册收敛成一处，并且：
  - 逐个模块独立导入，一个失败不牵连其余；
  - 返回失败明细而不是吞掉；
  - 给需要「结果必须可信」的入口（benchmark / RL）一个硬校验函数。
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Tuple

from tools.tool_registry import registry

#: 模块 -> 它提供的工具名。用于校验「导入成功」是否真的等于「工具可用」。
TOOL_MODULES: Dict[str, Tuple[str, ...]] = {
    "tools.arxiv_tool": (
        "get_recently_submitted_cs_papers",
        "search_arxiv_papers",
    ),
    "tools.pdf_download_tool": ("download_arxiv_pdf",),
    "tools.pdf_translate_tool": ("translate_arxiv_pdf",),
    "tools.cache_status_tool": ("get_paper_cache_status",),
}


def register_all_tools() -> Dict[str, str]:
    """逐个导入工具模块，返回 {模块名: 错误信息}（全部成功则为空字典）。

    每个模块独立 try —— 一个第三方依赖缺失不应该让其余工具也消失。
    """
    failures: Dict[str, str] = {}
    for module in TOOL_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:                      # noqa: BLE001
            failures[module] = f"{type(exc).__name__}: {exc}"
    return failures


def missing_tools() -> List[str]:
    """当前 registry 里还缺哪些工具。"""
    registered = {t["name"] for t in registry.list_tools()}
    return sorted(
        name
        for names in TOOL_MODULES.values()
        for name in names
        if name not in registered
    )


def registered_tool_count() -> int:
    """已注册工具数。让调用方不必各自 import registry。"""
    return len(registry.list_tools())


def require_all_tools(context: str = "本次运行") -> None:
    """工具没注册齐就直接失败。

    给 benchmark / RL 这类「跑完才发现结果无意义」代价极高的入口用。
    工具列表不全时模型不会报错，它会编工具名，然后一切照常产出 ——
    那种数字比没有数字更危险。
    """
    failures = register_all_tools()
    missing = missing_tools()
    if not missing:
        return

    expected_count = sum(len(names) for names in TOOL_MODULES.values())
    lines = [f"❌ {context}需要全部 {expected_count} 个工具，当前缺少: {missing}"]
    for module, error in failures.items():
        lines.append(f"   {module}: {error}")
    lines.append("   工具列表不全时模型不会报错，而是会编造工具名 ——")
    lines.append("   benchmark 照样跑完并给出成功率，那些数字是无意义的。")
    if any("arxiv" in e for e in failures.values()):
        lines.append("   缺 arxiv 包时: pip install arxiv")
    raise SystemExit("\n".join(lines))
