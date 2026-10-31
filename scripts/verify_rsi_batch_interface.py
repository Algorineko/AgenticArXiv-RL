"""Native AppWorld/registry batch parity on an already exposed train task."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from AgenticArxiv.rsi.batch_environment import BoundedBatchEnvironment
from AgenticArxiv.rsi.tool_environment import RegisteredToolEnvironment
from rsi_appworld_api_probe import AppWorldEnvironment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from appworld import load_task_ids
    from appworld.common.path_store import path_store
    path_store.update_root(args.root)
    task = load_task_ids("train", num_tasks_per_scenario=1)[0]
    calls = [{"name": "list_apps", "arguments": {}},
             {"name": "list_apis", "arguments": {"app_name": "supervisor"}},
             {"name": "api_doc", "arguments": {"app_name": "supervisor", "api_name": "complete_task"}}]
    serial = AppWorldEnvironment(task, "rsi_batch_preflight_serial")
    try:
        expected = [{"name": c["name"], "result": serial.execute(c["name"], c["arguments"])} for c in calls]
        serial_verdict = serial.check()
    finally:
        serial.close()
    raw = AppWorldEnvironment(task, "rsi_batch_preflight_batched")
    budget = BoundedBatchEnvironment(raw, primitive_budget=3)
    env = RegisteredToolEnvironment(budget)
    try:
        result = env.execute("batch_tools", {"calls": calls})
        verdict = env.check()
        exhausted = env.execute("list_apps", {})
        checks = {"observations_identical": result == {"results": expected},
                  "verdict_identical": verdict == serial_verdict,
                  "no_op_rejected": not verdict["success"],
                  "exact_primitive_accounting": budget.primitive_attempts == 3,
                  "budget_exhaustion_rejected": exhausted == {"error": "primitive_budget_exhausted"},
                  "no_extra_world_interactions": raw.world.num_interactions == 3}
    finally:
        env.close()
    paths = ("AgenticArxiv/rsi/batch_environment.py", "AgenticArxiv/rsi/tool_environment.py",
             "AgenticArxiv/tools/tool_registry.py", "scripts/rsi_appworld_api_probe.py",
             "scripts/verify_rsi_batch_interface.py")
    report = {"checks": checks, "passed": all(checks.values()), "llm_api_calls": 0,
              "primitive_calls_per_world": 3, "task_id_sha256": hashlib.sha256(task.encode()).hexdigest(),
              "sources": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in paths},
              "scope": "Exposed train documentation parity only; state-changing batch invariants tested separately on SQLite."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
