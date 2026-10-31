"""Explicit same-API batching on top of the shared primitive budget."""
from copy import deepcopy
import re

from .batch_environment import BoundedBatchEnvironment


class SameAPIBatchEnvironment(BoundedBatchEnvironment):
    """Accept concrete argument objects for independent calls to one app API."""

    def __init__(self, environment, primitive_budget=128, max_batch=16):
        super().__init__(environment, primitive_budget, max_batch)
        if "call_api" not in self._names or "batch_api_calls" in self._names:
            raise ValueError("An unambiguous call_api primitive is required")

    def tools(self):
        return super().tools() + [{"type": "function", "function": {
            "name": "batch_api_calls",
            "description": "Call one documented app API with multiple independent, concrete argument objects. "
                           "Each item consumes one primitive attempt; results are ordered. "
                           "Stops on a proxy error, without rolling back earlier operations.",
            "parameters": {"type": "object", "properties": {
                "app_name": {"type": "string"}, "api_name": {"type": "string"},
                "arguments_list": {"type": "array", "minItems": 1, "maxItems": self.max_batch,
                                   "items": {"type": "object"}}},
                "required": ["app_name", "api_name", "arguments_list"], "additionalProperties": False}}}]

    def execute(self, name, arguments):
        if name != "batch_api_calls":
            return super().execute(name, arguments)
        if not isinstance(arguments, dict) or set(arguments) != {"app_name", "api_name", "arguments_list"}:
            return {"error": "batch_api_calls requires app_name, api_name, arguments_list"}
        app, api, items = arguments["app_name"], arguments["api_name"], arguments["arguments_list"]
        if not all(isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]*", value)
                   for value in (app, api)):
            return {"error": "invalid API identifier"}
        if not isinstance(items, list) or not 1 <= len(items) <= self.max_batch or not all(
                isinstance(item, dict) for item in items):
            return {"error": "invalid arguments_list"}
        if self.primitive_attempts + len(items) > self.primitive_budget:
            return {"error": "primitive_budget_exhausted"}
        results = []
        for item in items:
            result = self._primitive("call_api", {"app_name": app, "api_name": api,
                                                   "arguments": deepcopy(item)})
            results.append(result)
            if isinstance(result, dict) and "error" in result:
                return {"error": "batch stopped after primitive error", "results": results}
        return {"results": results}
