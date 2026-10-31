"""Bounded, explicit tool batching without model-authored Python execution."""
from copy import deepcopy


class BoundedBatchEnvironment:
    def __init__(self, environment, primitive_budget=128, max_batch=16):
        if primitive_budget < 1 or not 1 <= max_batch <= 16:
            raise ValueError("Invalid primitive budget or batch size")
        self.environment = environment
        self.primitive_budget = primitive_budget
        self.max_batch = max_batch
        self.primitive_attempts = 0
        self._tools = deepcopy(environment.tools())
        self._names = {tool["function"]["name"] for tool in self._tools}
        if "batch_tools" in self._names:
            raise ValueError("batch_tools name already registered")

    def tools(self):
        return deepcopy(self._tools) + [{"type": "function", "function": {
            "name": "batch_tools",
            "description": "Execute up to 16 independent existing tool calls in order. "
                           "Each child has name and arguments. No references to earlier child results. "
                           "Every child consumes the same primitive budget as a direct call. "
                           "Execution stops on a tool error; earlier side effects are retained.",
            "parameters": {"type": "object", "properties": {"calls": {
                "type": "array", "minItems": 1, "maxItems": self.max_batch,
                "items": {"type": "object", "properties": {
                    "name": {"type": "string", "enum": sorted(self._names)},
                    "arguments": {"type": "object"}},
                    "required": ["name", "arguments"], "additionalProperties": False}}},
                "required": ["calls"], "additionalProperties": False}}}]

    def execute(self, name, arguments):
        if name != "batch_tools":
            return self._primitive(name, arguments)
        if not isinstance(arguments, dict) or set(arguments) != {"calls"}:
            return {"error": "batch_tools requires exactly calls"}
        calls = arguments["calls"]
        if not isinstance(calls, list) or not 1 <= len(calls) <= self.max_batch:
            return {"error": "invalid batch size"}
        # Reject structural errors and insufficient budget before any dispatch.
        for call in calls:
            if (not isinstance(call, dict) or set(call) != {"name", "arguments"}
                    or not isinstance(call["name"], str) or call["name"] not in self._names
                    or not isinstance(call["arguments"], dict)):
                return {"error": "invalid batch child"}
        if self.primitive_attempts + len(calls) > self.primitive_budget:
            return {"error": "primitive_budget_exhausted"}
        results = []
        for call in calls:
            result = self._primitive(call["name"], call["arguments"])
            results.append({"name": call["name"], "result": result})
            if isinstance(result, dict) and "error" in result:
                return {"error": "batch stopped after primitive error", "results": results}
        return {"results": results}

    def _primitive(self, name, arguments):
        if self.primitive_attempts >= self.primitive_budget:
            return {"error": "primitive_budget_exhausted"}
        # Count attempted dispatches conservatively, including invalid calls.
        self.primitive_attempts += 1
        return self.environment.execute(name, arguments)

    def check(self):
        return self.environment.check()

    def close(self):
        return self.environment.close()
