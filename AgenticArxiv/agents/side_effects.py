"""副作用管理器（解耦 DB / SSE / translate / store 等副作用）。

BaseAgent 的执行循环本身是纯粹的 ReAct 逻辑，但历史上混入了四类副作用：
    1. chat log / agent step 写 MySQL
    2. SSE 事件推送
    3. 翻译任务异步入队（起线程 + 调 pdf2zh）
    4. 会话记忆读写（last_papers / last_active_paper_id）

本模块把它们抽成可注入接口，提供三种实现：

    NoOpSideEffectManager    完全无操作（连会话记忆都不保留）
    LocalSideEffectManager   ★ RL 默认：会话记忆走内存 store，无 DB/SSE/线程
    MySQLSideEffectManager   Web 应用原行为（懒加载，只有用到才 import fastapi/sqlalchemy）

注意第 4 类不能简单 no-op：download / translate / cache 三类任务都要靠会话记忆
解析 ref（"第1篇"、null 指代），若会话记忆失效，这些任务在 RL 环境里必然失败，
奖励信号会被污染成"全是工具执行失败"。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TranslateHandle:
    """enqueue_translate 的返回值（只保留 BaseAgent 用到的字段）。"""

    task_id: str
    paper_id: Optional[str]
    status: str = "PENDING"


class SideEffectManager(ABC):
    """副作用管理器抽象接口。"""

    # ---- 日志 / 推送 ----

    @abstractmethod
    def create_chat_log(
        self,
        session_id: str,
        msg_id: str,
        role: str,
        content: str,
        model: Optional[str] = None,
        agent_type: Optional[str] = None,
    ) -> None:
        """记录对话日志。"""

    @abstractmethod
    def save_agent_step(
        self,
        msg_id: str,
        step_index: int,
        thought: str,
        action_name: str,
        action_args: str,
        observation: str,
        llm_latency_ms: int,
        tool_latency_ms: int,
    ) -> None:
        """保存 Agent 执行步骤。"""

    @abstractmethod
    def publish_sse(self, session_id: str, event_data: Dict[str, Any]) -> None:
        """发布 SSE 事件。"""

    # ---- 异步任务 ----

    @abstractmethod
    def enqueue_translate(self, session_id: str, **kwargs) -> TranslateHandle:
        """入队翻译任务，返回任务句柄。"""

    # ---- 会话记忆（读 + 写）----

    @abstractmethod
    def set_last_papers(self, session_id: str, papers: list) -> None:
        """写入会话论文列表。"""

    @abstractmethod
    def get_last_papers(self, session_id: str) -> List[Any]:
        """读取会话论文列表（用于 prompt 上下文注入）。"""

    @abstractmethod
    def set_last_active_paper_id(self, session_id: str, paper_id: str) -> None:
        """写入最近操作的论文 ID。"""


class NoOpSideEffectManager(SideEffectManager):
    """完全无操作实现。

    连会话记忆都不保留 —— 只适合"单步、无状态"的任务（如纯搜索）。
    多步/指代类任务请用 LocalSideEffectManager。
    """

    def create_chat_log(self, *args, **kwargs) -> None:
        pass

    def save_agent_step(self, *args, **kwargs) -> None:
        pass

    def publish_sse(self, *args, **kwargs) -> None:
        pass

    def enqueue_translate(self, session_id: str, **kwargs) -> TranslateHandle:
        return TranslateHandle(task_id="noop", paper_id=kwargs.get("paper_id"), status="PENDING")

    def set_last_papers(self, *args, **kwargs) -> None:
        pass

    def get_last_papers(self, session_id: str) -> List[Any]:
        return []

    def set_last_active_paper_id(self, *args, **kwargs) -> None:
        pass


class LocalSideEffectManager(SideEffectManager):
    """离线 RL 默认实现：会话记忆走内存 store，其余副作用全部关闭。

    - chat log / agent step：不落库（轨迹由 rl/trajectory.py 负责持久化）
    - SSE：丢弃
    - 翻译：不起线程、不调 pdf2zh，只返回一个确定性的 mock 句柄
      （观测文本形状与线上一致，因此指标/奖励计算不受影响）
    - 会话记忆：委托给 models.store 的当前后端（RL 下为 MemoryStore）
    """

    def __init__(self, store: Any = None, fake_translate: bool = True):
        if store is None:
            from models.store import store as global_store
            store = global_store
        self.store = store
        self.fake_translate = fake_translate
        self._translate_seq = 0
        self.published_events: List[Dict[str, Any]] = []  # 便于测试断言

    def create_chat_log(self, *args, **kwargs) -> None:
        pass

    def save_agent_step(self, *args, **kwargs) -> None:
        pass

    def publish_sse(self, session_id: str, event_data: Dict[str, Any]) -> None:
        # 保留在内存里，方便调试/断言，但不做任何 IO
        self.published_events.append({"session_id": session_id, **event_data})

    def enqueue_translate(self, session_id: str, **kwargs) -> TranslateHandle:
        if not self.fake_translate:
            from services.runtime import translate_runner
            return translate_runner.enqueue(session_id=session_id, **kwargs)

        # 确定性 mock：task_id 单调递增，便于快照回放时结果稳定
        self._translate_seq += 1
        paper_id = kwargs.get("paper_id")
        if not paper_id:
            ref = kwargs.get("ref")
            paper = self.store.resolve_paper(session_id, ref)
            paper_id = paper.id if paper else self.store.get_last_active_paper_id(session_id)
        if paper_id:
            self.store.set_last_active_paper_id(session_id, paper_id)
        return TranslateHandle(
            task_id=f"mocktask{self._translate_seq:04d}",
            paper_id=paper_id,
            status="PENDING",
        )

    def set_last_papers(self, session_id: str, papers: list) -> None:
        self.store.set_last_papers(session_id, papers)

    def get_last_papers(self, session_id: str) -> List[Any]:
        return self.store.get_last_papers(session_id)

    def set_last_active_paper_id(self, session_id: str, paper_id: str) -> None:
        self.store.set_last_active_paper_id(session_id, paper_id)


class MySQLSideEffectManager(SideEffectManager):
    """Web 应用原行为：MySQL 落库 + SSE 推送 + 异步翻译线程。

    所有重依赖（sqlalchemy / fastapi / 线程池）都在方法内部懒加载，
    因此只要不实例化本类，离线 RL 路径就不会碰到它们。
    """

    def __init__(self):
        from models.store import store
        from services.log_service import log_service
        from services.runtime import event_bus, translate_runner

        self.store = store
        self.log_service = log_service
        self.event_bus = event_bus
        self.translate_runner = translate_runner

    def create_chat_log(
        self,
        session_id: str,
        msg_id: str,
        role: str,
        content: str,
        model: Optional[str] = None,
        agent_type: Optional[str] = None,
    ) -> None:
        self.log_service.create_chat_log(
            session_id, msg_id, role, content, model=model, agent_type=agent_type
        )

    def save_agent_step(
        self,
        msg_id: str,
        step_index: int,
        thought: str,
        action_name: str,
        action_args: str,
        observation: str,
        llm_latency_ms: int,
        tool_latency_ms: int,
    ) -> None:
        self.log_service.save_agent_step(
            msg_id=msg_id,
            step_index=step_index,
            thought=thought,
            action_name=action_name,
            action_args=action_args,
            observation=observation,
            llm_latency_ms=llm_latency_ms,
            tool_latency_ms=tool_latency_ms,
        )

    def publish_sse(self, session_id: str, event_data: Dict[str, Any]) -> None:
        self.event_bus.publish(session_id, event_data)

    def enqueue_translate(self, session_id: str, **kwargs) -> TranslateHandle:
        t = self.translate_runner.enqueue(session_id=session_id, **kwargs)
        return TranslateHandle(task_id=t.task_id, paper_id=t.paper_id, status=t.status)

    def set_last_papers(self, session_id: str, papers: list) -> None:
        self.store.set_last_papers(session_id, papers)

    def get_last_papers(self, session_id: str) -> List[Any]:
        return self.store.get_last_papers(session_id)

    def set_last_active_paper_id(self, session_id: str, paper_id: str) -> None:
        self.store.set_last_active_paper_id(session_id, paper_id)


def default_side_effect_manager() -> SideEffectManager:
    """按运行环境挑默认实现：能连 MySQL 就用 MySQL，否则本地内存。"""
    try:
        return MySQLSideEffectManager()
    except Exception:
        return LocalSideEffectManager()
