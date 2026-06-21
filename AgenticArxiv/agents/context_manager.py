# AgenticArxiv/agents/context_manager.py
from typing import List, Dict, Any
from dataclasses import dataclass
import json

@dataclass
class ReactStep:
    """ReAct的每一步"""
    thought: str
    action: str
    observation: str
    
    def format(self) -> str:
        """格式化为文本"""
        return f"""Thought: {self.thought}
Action: {self.action}
Observation: {self.observation}"""

class ContextManager:
    """管理Agent的上下文历史"""
    
    def __init__(self, max_steps: int = 10):
        self.history: List[ReactStep] = []
        self.max_steps = max_steps
        
    def add_step(self, thought: str, action: str, observation: str):
        """添加一个ReAct步骤"""
        step = ReactStep(thought, action, observation)
        self.history.append(step)
        
        # 如果超过最大步数，移除最早的一步
        if len(self.history) > self.max_steps:
            self.history = self.history[-self.max_steps:]
    
    def get_history_text(self) -> str:
        """获取历史记录的文本表示"""
        return "\n\n".join([step.format() for step in self.history])
    
    def get_full_history(self) -> List[Dict[str, str]]:
        """获取完整的历史记录"""
        return [
            {
                "thought": step.thought,
                "action": step.action,
                "observation": step.observation
            }
            for step in self.history
        ]
    
    def clear(self):
        """清空历史记录"""
        self.history.clear()