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
    def __init__(self, task_id, experiment, max_interactions=48):
        from appworld import AppWorld
        self.context = AppWorld(task_id=task_id, experiment_name=experiment,
                                load_ground_truth=True, ground_truth_mode="minimal", random_seed=45,
                                timeout_seconds=10, max_interactions=max_interactions, max_api_calls_per_interaction=1)
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
    p.add_argument("--harness-file", type=Path)
    p.add_argument("--experiment-prefix", default="rsi_external_api_probe_v1")
    p.add_argument("--registry", action="store_true", help="Use the original framework's isolated ToolRegistry")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--max-calls", type=int, default=30)
    p.add_argument("--max-input-tokens", type=int, default=200000)
    p.add_argument("--max-output-tokens", type=int, default=30000)
    p.add_argument("--completion-tokens", type=int, default=1024)
    p.add_argument("--batch", action="store_true")
    p.add_argument("--primitive-budget", type=int, default=128)
    p.add_argument("--train-pool-file", type=Path)
    p.add_argument("--train-pool-manifest", type=Path)
    p.add_argument("--train-count", type=int, default=4)
    args = p.parse_args()
    if not (1 <= args.max_steps <= 24 and 1 <= args.max_calls <= 100
            and 1 <= args.max_input_tokens <= 1500000 and 1 <= args.max_output_tokens <= 100000
            and 1 <= args.completion_tokens <= 2048 and 1 <= args.primitive_budget <= 128
            and 1 <= args.train_count <= 4):
        p.error("Development caps: 24 steps, 100 attempts, 1.5M input, 100k output, 2048 output/response, 128 primitive calls")
    if args.batch and not args.registry:
        p.error("Batching requires --registry for tracked framework integration")
    if bool(args.train_pool_file) != bool(args.train_pool_manifest):
        p.error("Both training pool file and public manifest are required")
    if args.registry:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from AgenticArxiv.rsi.tool_environment import RegisteredToolEnvironment
    if args.batch:
        from AgenticArxiv.rsi.batch_environment import BoundedBatchEnvironment
    from appworld import load_task_ids
    from appworld.common.path_store import path_store
    path_store.update_root(str(args.root))
    selected = load_task_ids("train", num_tasks_per_scenario=1)[:2]
    assert len(selected) == 2
    pool_metadata = None
    if args.train_pool_file:
        private = json.loads(args.train_pool_file.read_text())
        public = json.loads(args.train_pool_manifest.read_text())
        selected = []
        train_ids = set(load_task_ids("train"))
        for index, entry in enumerate(private["meta_train"][:args.train_count]):
            task_id = entry["tasks"]["inner_train"]
            published = public["pools"]["meta_train"][index]
            assert hashlib.sha256(task_id.encode()).hexdigest() == published["task_sha256_by_role"]["inner_train"]
            assert hashlib.sha256(entry["generator"].encode()).hexdigest() == published["generator_sha256"]
            assert task_id in train_ids
            selected.append(task_id)
        assert len(selected) == args.train_count and len(set(selected)) == len(selected)
        pool_metadata = {"public_manifest_sha256": hashlib.sha256(args.train_pool_manifest.read_bytes()).hexdigest(),
                         "selection": "Frozen meta_train prefix, inner_train only", "count": len(selected)}
    args.private_output.mkdir(parents=True, exist_ok=False)
    if args.public_output.exists():
        raise ValueError("public output already exists")
    limits = {"calls": args.max_calls, "input_tokens": args.max_input_tokens, "output_tokens": args.max_output_tokens}
    api = API("https://api.qnaigc.com/v1", "doubao-seed-2.0-mini", args.private_output,
              limits, min_interval=15, max_completion_tokens=args.completion_tokens)
    harness = args.harness_file.read_text() if args.harness_file else ""
    protocol = {"stage": "external train-only diagnostic, no RSI effect", "created_utc": utc(),
                "task_id_sha256": [hashlib.sha256(x.encode()).hexdigest() for x in selected],
                "max_steps": args.max_steps, "max_parallel_tools": 4, "limits": limits,
                "isolated_framework_registry": args.registry,
                "bounded_batch_interface": args.batch, "primitive_budget": args.primitive_budget if args.batch else None,
                "max_environment_interactions": args.primitive_budget if args.batch else 48,
                "training_pool": pool_metadata,
                "model": api.model, "min_request_interval_s": 15, "max_completion_tokens": args.completion_tokens,
                "harness_sha256": hashlib.sha256(harness.encode()).hexdigest(),
                "experiment_prefix": args.experiment_prefix,
                "sources": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ("rsi_api_pilot.py", "rsi_cpu_tasks.py", "rsi_appworld_api_probe.py")}}
    (args.private_output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    frozen = args.private_output / "frozen_sources"
    frozen.mkdir()
    for name in protocol["sources"]:
        (frozen / name).write_bytes((Path(__file__).parent / name).read_bytes())
    if args.registry:
        repo = Path(__file__).resolve().parents[1]
        protocol["framework_sources"] = {}
        framework_paths = ["AgenticArxiv/tools/tool_registry.py", "AgenticArxiv/rsi/tool_environment.py"]
        if args.batch:
            framework_paths.append("AgenticArxiv/rsi/batch_environment.py")
        for name in framework_paths:
            content = (repo / name).read_bytes()
            protocol["framework_sources"][name] = hashlib.sha256(content).hexdigest()
            target = args.private_output / "frozen_framework_sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        (args.private_output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records = []
    for i, task_id in enumerate(selected):
        env = AppWorldEnvironment(task_id, f"{args.experiment_prefix}_{i}",
                                  max_interactions=protocol["max_environment_interactions"])
        task = {"id": hashlib.sha256(task_id.encode()).hexdigest(), "domain": "appworld",
                "prompt": env.world.task.instruction + "\nSupervisor identity: " + str(env.world.task.supervisor) +
                "\nUse discovery tools to find the APIs and their parameters. Perform the actual requested "
                "operations, then call supervisor.complete_task with the appropriate answer if required."}
        primitive_env = None
        if args.batch:
            primitive_env = BoundedBatchEnvironment(env, primitive_budget=args.primitive_budget)
            env = primitive_env
        if args.registry:
            env = RegisteredToolEnvironment(env)
        trace = episode(api, task, repeat=0, max_steps=args.max_steps, environment=env, harness_instruction=harness)
        records.append({"task_id_sha256": task["id"], "success": trace["verdict"]["success"],
                        "stop": trace["stop"], "model_steps": len(trace["steps"]),
                        "tool_calls": trace["tool_calls"], "tool_errors": trace["tool_errors"],
                        "primitive_attempts": primitive_env.primitive_attempts if primitive_env else trace["tool_calls"]})
        result = {"protocol": protocol, "episodes": records, "usage": api.totals(),
                  "complete": len(records) == len(selected),
                  "protected_trace_location": str(args.private_output),
                  "limitation": "Preselected train-only diagnostic; not a benchmark score or RSI method comparison."}
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        args.public_output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(records[-1]), flush=True)
        if trace["stop"] in ("api_error", "global_budget"):
            break


if __name__ == "__main__":
    main()
