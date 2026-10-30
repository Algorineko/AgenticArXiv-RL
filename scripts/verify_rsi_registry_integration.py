"""Replay exposed development traces through original AgenticArxiv registry.

No API calls; protected observations are compared in memory and never exported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from AgenticArxiv.rsi.tool_environment import RegisteredToolEnvironment
from AgenticArxiv.tools.tool_registry import registry as legacy_registry
from AgenticArxiv.rsi.trajectory_adapter import to_trajectory, from_trajectory
from AgenticArxiv.rl.trajectory import save_trajectory, load_trajectories
from rsi_appworld_api_probe import AppWorldEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--private-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists")
    from appworld import load_task_ids
    from appworld.common.path_store import path_store
    path_store.update_root(args.root)
    # Explicitly restrict this integration audit to the two exposed scenarios.
    allowed = {hashlib.sha256(t.encode()).hexdigest(): t
               for t in load_task_ids("train", num_tasks_per_scenario=1)[:2]}
    initial_registry = legacy_registry.list_tools()
    counts = {"episodes": 0, "observations": 0, "observation_mismatches": 0,
              "verdict_mismatches": 0, "schema_mismatches": 0,
              "invalid_call_mismatches": 0, "trajectory_roundtrip_mismatches": 0,
              "trajectory_roundtrips": 0, "api_calls": 0}
    for run_index, run in enumerate(args.private_run):
        protocol = json.loads((run / "protocol.json").read_text())
        for name, expected in protocol["sources"].items():
            assert hashlib.sha256((run / "frozen_sources" / name).read_bytes()).hexdigest() == expected
        for index, line in enumerate((run / "traces.jsonl").read_text().splitlines()):
            trace = json.loads(line)
            assert trace["task_id"] in allowed
            # Protected content never enters the public report or repository.
            with tempfile.TemporaryDirectory(prefix="rsi_trajectory_", dir=run.parent) as tmp:
                target = Path(tmp) / "roundtrip.jsonl"
                save_trajectory(to_trajectory(trace, model=protocol["model"]), target)
                restored = load_trajectories(target)
                counts["trajectory_roundtrip_mismatches"] += int(
                    len(restored) != 1 or from_trajectory(restored[0]) != trace)
                counts["trajectory_roundtrips"] += 1
            raw = AppWorldEnvironment(allowed[trace["task_id"]], f"rsi_registry_replay_{run_index}_{index}")
            env = RegisteredToolEnvironment(raw)
            try:
                counts["schema_mismatches"] += int(env.tools() != raw.tools())
                invalid = [("unknown", {}), ("call_api", []),
                           ("call_api", {"app_name": "bad.name", "api_name": "x", "arguments": {}}),
                           ("api_doc", {"app_name": "supervisor"}),
                           ("list_apps", {"unexpected": 1})]
                for name, arguments in invalid:
                    counts["invalid_call_mismatches"] += int(env.execute(name, arguments) != raw.execute(name, arguments))
                for step in trace["steps"]:
                    operations = step["assistant"].get("tool_calls", [])
                    observed = step["observations"]
                    if not observed:
                        continue
                    assert len(operations) == len(observed)
                    for operation, observation in zip(operations, observed):
                        function = operation.get("function") or {}
                        try:
                            actual = env.execute(function.get("name"), json.loads(function.get("arguments", "{}")))
                        except (ValueError, TypeError):
                            actual = {"error": "malformed tool arguments"}
                        counts["observations"] += 1
                        counts["observation_mismatches"] += int(actual != observation["result"])
                counts["verdict_mismatches"] += int(env.check() != trace["verdict"])
                counts["episodes"] += 1
            finally:
                env.close()
    counts["legacy_global_registry_unchanged"] = legacy_registry.list_tools() == initial_registry
    counts["sources"] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        "AgenticArxiv/tools/tool_registry.py", "AgenticArxiv/rsi/tool_environment.py",
        "AgenticArxiv/rl/trajectory.py", "AgenticArxiv/rsi/trajectory_adapter.py",
        "scripts/rsi_appworld_api_probe.py", "scripts/verify_rsi_registry_integration.py")}
    counts["passed"] = counts["legacy_global_registry_unchanged"] and all(
        counts[k] == 0 for k in counts if k.endswith("mismatches"))
    counts["scope"] = "Exposed train traces only; registry integration audit, no RSI effectiveness claim."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(counts, output, indent=2)
        output.write("\n")
    print(json.dumps(counts))
    if not counts["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
