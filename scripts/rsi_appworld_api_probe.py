"""Two train-only AppWorld API probes; protected traces stay outside Git."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from rsi_api_pilot import API, episode, utc
from rsi_cpu_tasks import function


class AppWorldEnvironment:
    def __init__(self, task_id, experiment):
        from appworld import AppWorld
        self.context = AppWorld(task_id=task_id, experiment_name=experiment,
                                load_ground_truth=True, ground_truth_mode="minimal", random_seed=45,
                                timeout_seconds=10, max_interactions=48, max_api_calls_per_interaction=1)
        self.world = self.context.__enter__()

    def tools(self):
        string = {"type": "string"}
        return [function("list_apps", "Discover available applications and their descriptions.", {}),
                function("list_apis", "List APIs of one application.", {"app_name": string}, ("app_name",)),
                function("api_doc", "Read the exact parameters of an API before calling it.",
                         {"app_name": string, "api_name": string}, ("app_name", "api_name")),
                function("call_api", "Call a documented app API. Arguments is a JSON object, not code. "
                         "Use supervisor.complete_task only after satisfying the request.",
                         {"app_name": string, "api_name": string, "arguments": {"type": "object"}},
                         ("app_name", "api_name", "arguments"))]

    def execute(self, name, args):
        schema = next((x["function"]["parameters"] for x in self.tools() if x["function"]["name"] == name), None)
        if schema is None or not isinstance(args, dict):
            return {"error": "unknown tool or invalid arguments"}
        if set(args) - set(schema["properties"]) or set(schema["required"]) - set(args):
            return {"error": "missing or extra arguments"}
        app = args.get("app_name", "api_docs")
        method = args.get("api_name", "show_app_descriptions")
        if not all(isinstance(x, str) and re.fullmatch(r"[a-z][a-z0-9_]*", x) for x in (app, method)):
            return {"error": "invalid API identifier"}
        if name == "list_apps":
            app, method, kwargs = "api_docs", "show_app_descriptions", {}
        elif name == "list_apis":
            kwargs = {"app_name": app}
            app, method = "api_docs", "show_api_descriptions"
        elif name == "api_doc":
            kwargs = {"app_name": app, "api_name": method}
            app, method = "api_docs", "show_api_doc"
        else:
            kwargs = args["arguments"]
            if not isinstance(kwargs, dict):
                return {"error": "arguments must be a JSON object"}
        # Only identifiers validated above and repr of JSON values enter this fixed
        # expression. The model cannot submit Python source, imports or attributes.
        code = f"print(apis.{app}.{method}(**{kwargs!r}))"
        try:
            observation = self.world.execute(code)
            if "Traceback (most recent call last)" in observation:
                return {"error": observation}
            return {"observation": observation}
        except Exception as exc:
            return {"error": str(exc)}

    def check(self):
        stats = self.world.evaluate().to_dict(stats_only=True)
        return {"success": stats["success"], "checks": {"official_task_success": stats["success"]}}

    def close(self):
        self.context.__exit__(None, None, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--private-output", type=Path, required=True)
    p.add_argument("--public-output", type=Path, required=True)
    args = p.parse_args()
    from appworld import load_task_ids
    from appworld.common.path_store import path_store
    path_store.update_root(str(args.root))
    selected = load_task_ids("train", num_tasks_per_scenario=1)[:2]
    assert len(selected) == 2
    args.private_output.mkdir(parents=True, exist_ok=False)
    if args.public_output.exists():
        raise ValueError("public output already exists")
    limits = {"calls": 30, "input_tokens": 200000, "output_tokens": 30000}
    api = API("https://api.qnaigc.com/v1", "doubao-seed-2.0-mini", args.private_output,
              limits, min_interval=15, max_completion_tokens=1024)
    protocol = {"stage": "external train-only diagnostic, no RSI effect", "created_utc": utc(),
                "task_id_sha256": [hashlib.sha256(x.encode()).hexdigest() for x in selected],
                "max_steps": 12, "max_parallel_tools": 4, "limits": limits,
                "model": api.model, "min_request_interval_s": 15, "max_completion_tokens": 1024,
                "sources": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ("rsi_api_pilot.py", "rsi_cpu_tasks.py", "rsi_appworld_api_probe.py")}}
    (args.private_output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    for i, task_id in enumerate(selected):
        env = AppWorldEnvironment(task_id, f"rsi_external_api_probe_v1_{i}")
        task = {"id": hashlib.sha256(task_id.encode()).hexdigest(), "domain": "appworld",
                "prompt": env.world.task.instruction + "\nSupervisor identity: " + str(env.world.task.supervisor) +
                "\nUse discovery tools to find the APIs and their parameters. Perform the actual requested "
                "operations, then call supervisor.complete_task with the appropriate answer if required."}
        trace = episode(api, task, repeat=0, max_steps=12, environment=env)
        records.append({"task_id_sha256": task["id"], "success": trace["verdict"]["success"],
                        "stop": trace["stop"], "model_steps": len(trace["steps"]),
                        "tool_calls": trace["tool_calls"], "tool_errors": trace["tool_errors"]})
        result = {"protocol": protocol, "episodes": records, "usage": api.totals(),
                  "complete": len(records) == len(selected),
                  "protected_trace_location": str(args.private_output),
                  "limitation": "Two deterministic train scenarios; not a benchmark score or method comparison."}
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        args.public_output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(records[-1]), flush=True)
        if trace["stop"] in ("api_error", "global_budget"):
            break


if __name__ == "__main__":
    main()
