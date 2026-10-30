"""Reuse the original tool registry without registering legacy business tools."""
from copy import deepcopy

from AgenticArxiv.tools.tool_registry import ToolRegistry


class RegisteredToolEnvironment:
    """Adapt the pilot environment contract through an isolated ToolRegistry.

    The wrapped environment retains validation, execution and final evaluation.
    This adapter adds no scoring or parameter coercion. Invalid names/argument
    containers go directly to that validator, preserving its error semantics.
    """

    def __init__(self, environment):
        self.environment = environment
        self.registry = ToolRegistry()
        self._tools = deepcopy(environment.tools())
        for tool in self._tools:
            spec = tool["function"]
            name = spec["name"]
            if self.registry.get_tool(name) is not None:
                raise ValueError(f"Duplicate tool: {name}")
            self.registry.register_tool(
                name, spec["description"], deepcopy(spec["parameters"]),
                self._bind(name),
            )

    def _bind(self, name):
        def dispatch(**arguments):
            return self.environment.execute(name, arguments)
        return dispatch

    def tools(self):
        return deepcopy(self._tools)

    def execute(self, name, arguments):
        if not isinstance(name, str) or self.registry.get_tool(name) is None or not isinstance(arguments, dict):
            return self.environment.execute(name, arguments)
        return self.registry.execute_tool(name, arguments)

    def check(self):
        return self.environment.check()

    def close(self):
        return self.environment.close()
