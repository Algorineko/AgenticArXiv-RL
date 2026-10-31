"""Check same-API batching against actual AppWorld primitive calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from AgenticArxiv.rsi.api_batch_environment import SameAPIBatchEnvironment
from AgenticArxiv.rsi.tool_environment import RegisteredToolEnvironment
from rsi_appworld_api_probe import AppWorldEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from appworld import load_task_ids
    from appworld.common.path_store import path_store
    path_store.update_root(args.root)
    task = load_task_ids("train", num_tasks_per_scenario=1)[0]
    items = [{"app_name": "supervisor"}, {"app_name": "api_docs"}]
    raw = AppWorldEnvironment(task, "rsi_same_api_serial")
    try:
        expected = [raw.execute("call_api", {"app_name": "api_docs", "api_name": "show_api_descriptions",
                                             "arguments": item}) for item in items]
        verdict = raw.check()
    finally:
        raw.close()
    raw = AppWorldEnvironment(task, "rsi_same_api_batch")
    batch = SameAPIBatchEnvironment(raw, primitive_budget=2)
    env = RegisteredToolEnvironment(batch)
    try:
        result = env.execute("batch_api_calls", {"app_name": "api_docs", "api_name": "show_api_descriptions",
                                                  "arguments_list": items})
        exhausted = env.execute("batch_api_calls", {"app_name": "api_docs", "api_name": "show_api_descriptions",
                                                     "arguments_list": items[:1]})
        checks = {"observations_match": result == {"results": expected},
                  "verdict_matches": env.check() == verdict,
                  "no_op_rejected": not verdict["success"],
                  "primitive_budget_exact": batch.primitive_attempts == 2,
                  "exhaustion_rejected": exhausted == {"error": "primitive_budget_exhausted"},
                  "world_interactions_exact": raw.world.num_interactions == 2}
    finally:
        env.close()
    paths = ["AgenticArxiv/rsi/api_batch_environment.py", "AgenticArxiv/rsi/batch_environment.py",
             "AgenticArxiv/rsi/tool_environment.py", "scripts/verify_rsi_same_api_batch.py"]
    result = {"passed": all(checks.values()), "checks": checks, "llm_api_calls": 0,
              "task_id_sha256": hashlib.sha256(task.encode()).hexdigest(),
              "sources": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
              "scope": "Exposed train documentation only; no method effectiveness claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
