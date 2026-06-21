# AgenticArxiv/tools/tool_registry.py
from typing import Dict, Any, Callable, List, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ToolRegistry:
    """
    工具注册表，管理所有可用的工具
    """

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        parameter_schema: Dict[str, Any],
        func: Callable,
    ) -> None:
        """
        注册一个工具

        Args:
            name: 工具名称（唯一标识）
            description: 工具描述, 说明工具的功能
            parameter_schema: 参数模式, 遵循JSON Schema规范
            func: 工具函数
        """
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameter_schema,
            "func": func,
        }

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """获取指定名称的工具"""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有可用工具"""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
            for tool in self._tools.values()
        ]

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        执行指定工具

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            工具执行结果

        Raises:
            ValueError: 如果工具不存在或参数无效
        """
        if name not in self._tools:
            raise ValueError(f"工具 '{name}' 未注册")

        tool = self._tools[name]
        try:
            # 将JSON参数传递给工具函数
            return tool["func"](**arguments)
        except TypeError as e:
            raise ValueError(f"工具参数错误: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"工具执行失败: {str(e)}")


# 创建全局工具注册表实例
registry = ToolRegistry()
