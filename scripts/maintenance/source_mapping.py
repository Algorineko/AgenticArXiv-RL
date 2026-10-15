"""维护审计的独立来源映射；旧身份与当前实现哈希分别记录。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DEFAULT_SOURCE_MAP = Path("scripts/maintenance/source_map.json")


def workspace_path(workspace: Path, value: str | Path) -> Path:
    value = Path(value)
    return (value if value.is_absolute() else workspace / value).resolve()


def load_source_map(workspace: Path, path: str | Path = DEFAULT_SOURCE_MAP) -> dict:
    path = workspace_path(workspace, path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("sources"), dict):
        raise ValueError(f"来源映射格式不受支持：{path}")
    return document


def mapped_paths(workspace: Path, source: str, mapping: dict) -> list[Path]:
    if source not in mapping["sources"]:
        raise ValueError(f"缺少旧源码的显式迁移映射：{source}")
    record = mapping["sources"][source]
    current = record.get("current_paths")
    if not isinstance(current, list) or not record.get("reason"):
        raise ValueError(f"迁移映射必须声明 current_paths 和 reason：{source}")
    paths = []
    for name in current:
        path = workspace_path(workspace, name)
        if not path.is_relative_to(workspace):
            raise ValueError(f"源码映射越出显式工作区：{name}")
        if not path.is_file():
            raise FileNotFoundError(f"映射的当前源码不存在：{path}")
        paths.append(path)
    return paths


def source_index(workspace: Path, sources: tuple[str, ...], mapping: dict) -> dict:
    """退出记录也要显式；不能把新源码哈希冒充旧源码指纹。"""
    index = {}
    for source in sources:
        paths = mapped_paths(workspace, source, mapping)
        record = mapping["sources"][source]
        original = record.get("original_sha256")
        if original is not None and (not isinstance(original, str) or len(original) != 64):
            raise ValueError(f"旧源码指纹格式不正确：{source}")
        old_path = workspace_path(workspace, source)
        index[source] = {
            "source": source if old_path.is_file() else None,
            "status": "mapped" if paths else "retired",
            "original_source": source,
            "sha256": original,
            "sha256_scope": "architecture_migration 起点的旧源码身份，不是当前实现哈希",
            "reason": record["reason"],
            "current_sources": [{
                "path": path.relative_to(workspace).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            } for path in paths],
        }
    return index
