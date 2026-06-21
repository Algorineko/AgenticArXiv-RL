"""副作用管理器（解耦 DB/SSE/translate/store 等副作用）

RL 训练时用 NoOpSideEffectManager（无操作），
生产环境可用 MySQLSideEffectManager（原逻辑）。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class SideEffectManager(ABC):
    """副作用管理器抽象接口"""

    @abstractmethod
    def create_chat_log(
        self, session_id: str, msg_id: str, role: str, content: str, model: str
    ) -> None:
        """记录对话日志（DB）"""
        pass

    @abstractmethod
    def save_agent_step(
        self, session_id: str, msg_id: str, step_index: int, step_data: Dict[str, Any]
    ) -> None:
        """保存 Agent 执行步骤（DB）"""
        pass

    @abstractmethod
    def publish_sse(self, session_id: str, event_data: Dict[str, Any]) -> None:
        """发布 SSE 事件（实时推送到前端）"""
        pass

    @abstractmethod
    def enqueue_translate(
        self, paper_id: int, session_id: str, **kwargs
    ) -> None:
        """入队翻译任务（异步）"""
        pass

    @abstractmethod
    def set_last_papers(self, session_id: str, papers: list) -> None:
        """设置会话最近论文列表（store）"""
        pass

    @abstractmethod
    def set_last_active_paper_id(self, session_id: str, paper_id: int) -> None:
        """设置会话最后活跃论文 ID（store）"""
        pass


class NoOpSideEffectManager(SideEffectManager):
    """无操作实现（用于离线 RL 训练）"""

    def create_chat_log(self, *args, **kwargs):
        pass

    def save_agent_step(self, *args, **kwargs):
        pass

    def publish_sse(self, *args, **kwargs):
        pass

    def enqueue_translate(self, *args, **kwargs):
        pass

    def set_last_papers(self, *args, **kwargs):
        pass

    def set_last_active_paper_id(self, *args, **kwargs):
        pass
