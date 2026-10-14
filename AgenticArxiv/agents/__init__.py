# AgenticArxiv/agents/__init__.py
from .prompt_templates import get_react_prompt, format_tool_description

__all__ = [
    "ReActAgent",
    "get_react_prompt",
    "format_tool_description",
]


def __getattr__(name):
    if name == "ReActAgent":
        from .agent_engine import ReActAgent
        return ReActAgent
    raise AttributeError(name)
