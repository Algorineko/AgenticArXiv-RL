"""工具产物路径的上下文绑定；不临时改写全局 settings。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Mapping, Optional

_paths: ContextVar[Optional[Mapping[str, Path]]] = ContextVar("episode_artifact_paths", default=None)


def safe_artifact_name(value: str) -> str:
    from utils.pdf_downloader import safe_filename

    # safe_filename 保留点；目录名不得把 "." 或 ".." 解释为路径导航。
    return safe_filename(str(value)).strip(".") or "paper"


def artifact_path(setting_name: str, default: Optional[str] = None) -> str:
    """episode 工具使用自有根；独立快照构建继续使用显式配置的路径。"""
    bound = _paths.get()
    if bound is not None and setting_name in bound:
        return str(bound[setting_name])
    if default is not None:
        return str(default)
    from config import settings

    return str(getattr(settings, setting_name))


@contextmanager
def bind_artifact_paths(paths: Mapping[str, Path]):
    token = _paths.set(dict(paths))
    try:
        yield
    finally:
        _paths.reset(token)


def search_output_path(requested: Optional[str], default: str) -> str:
    """保留显式输出位置，但 episode 不覆盖外部既有资产也不清理它们。"""
    bound = _paths.get()
    if bound is None:
        return requested or default
    if requested is None:
        return str(bound["search_path"] / "recent_cs_papers.txt")
    output = Path(requested).expanduser().resolve()
    owned = any(output.is_relative_to(Path(root).resolve()) for root in bound.values())
    if output.exists() and not owned:
        raise ValueError("output_path 已存在且不属于当前 episode；不能覆盖共享输入或用户资产")
    return str(output)
