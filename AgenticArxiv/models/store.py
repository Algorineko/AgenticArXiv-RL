# AgenticArxiv/models/store.py
"""Store 后端分发器。

保持 `from models.store import store` 这一历史调用方式不变，
但让底层实现可切换，从而使离线 RL 路径不再强依赖 MySQL：

    STORE_BACKEND=auto    （默认）sqlalchemy 可用则用 MySQL，否则回落内存
    STORE_BACKEND=memory  强制内存（RL 训练 / 测试 / 无数据库演示）
    STORE_BACKEND=mysql   强制 MySQL（Web 应用；缺依赖时直接报错而非静默降级）

`store` 是一个代理对象：运行时调用 use_memory_store() 切换后端后，
即使其他模块早已 `from models.store import store`，也会自动指向新后端。
"""
from __future__ import annotations

import os
from typing import Any, Optional

_BACKEND_ENV = "STORE_BACKEND"
_backend: Optional[Any] = None


def _make_store() -> Any:
    backend = os.getenv(_BACKEND_ENV, "auto").strip().lower()

    if backend == "memory":
        from models.store_memory import MemoryStore
        return MemoryStore()

    if backend == "mysql":
        from models.store_mysql import Store
        return Store()

    # auto：优先 MySQL，缺依赖时静默回落内存（离线训练场景）
    try:
        from models.store_mysql import Store
        return Store()
    except Exception:  # ImportError（无 sqlalchemy/pymysql）或配置异常
        from models.store_memory import MemoryStore
        return MemoryStore()


def get_store() -> Any:
    """返回当前后端实例（首次调用时惰性创建）。"""
    global _backend
    if _backend is None:
        _backend = _make_store()
    return _backend


class _StoreProxy:
    """透明代理，使后端切换对所有既有 import 立即生效。"""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_store(), name)

    def __repr__(self) -> str:
        return f"<StoreProxy backend={type(get_store()).__name__}>"


store = _StoreProxy()


def use_memory_store(reset: bool = True) -> Any:
    """把全局 store 切换为内存实现，并返回它。

    RL rollout 在进程启动早期调用一次即可保证全程无外部副作用。
    """
    global _backend
    from models.store_memory import MemoryStore

    if not isinstance(_backend, MemoryStore):
        _backend = MemoryStore()
    elif reset:
        _backend.reset()
    return _backend


def is_memory_backend() -> bool:
    from models.store_memory import MemoryStore
    return isinstance(get_store(), MemoryStore)
