"""本地会话与资产 Store 门面。

工具通过这一门面访问当前上下文绑定的状态，不依赖训练器或服务层。
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

_backend: Optional[Any] = None
_bound_store: ContextVar[Optional[Any]] = ContextVar("episode_store", default=None)


def _make_store() -> Any:
    from agentic_arxiv.state.store_memory import MemoryStore
    return MemoryStore(make_dirs=False)


def get_store() -> Any:
    """返回当前上下文的 store；直接工具调用使用惰性本地默认实例。"""
    bound = _bound_store.get()
    if bound is not None:
        return bound
    global _backend
    if _backend is None:
        _backend = _make_store()
    return _backend


class _StoreProxy:
    """透明代理，工具只能看到当前执行上下文的会话与资产。"""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_store(), name)

    def __repr__(self) -> str:
        return f"<StoreProxy backend={type(get_store()).__name__}>"


store = _StoreProxy()


def use_memory_store(reset: bool = True) -> Any:
    """为直接工具调用绑定当前上下文的内存状态，不切换进程全局后端。

    正式 episode 使用 ``bind_store``，在调用结束后恢复先前绑定。
    """
    from agentic_arxiv.state.store_memory import MemoryStore

    current = _bound_store.get()
    if not isinstance(current, MemoryStore):
        current = MemoryStore(make_dirs=False)
        _bound_store.set(current)
    elif reset:
        current.reset()
    return current


@contextmanager
def bind_store(instance: Any):
    """在一次执行边界绑定 episode 独有 store，支持嵌套和并行上下文。"""
    token = _bound_store.set(instance)
    try:
        yield instance
    finally:
        _bound_store.reset(token)


def is_memory_backend() -> bool:
    from agentic_arxiv.state.store_memory import MemoryStore
    return isinstance(get_store(), MemoryStore)
