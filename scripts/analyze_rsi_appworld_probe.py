"""Replay private AppWorld train traces; export only aggregate audit metadata."""
import argparse
import atexit
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--private-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from appworld import load_task_ids
    from appworld.common.path_store import path_store
    path_store.update_root(str(args.root))
    ids = {hashlib.sha256(t.encode()).hexdigest(): t
           for t in load_task_ids("train")}
    traces = [json.loads(x) for x in (args.private_run / "traces.jsonl").read_text().splitlines()]
    protocol = json.loads((args.private_run / "protocol.json").read_text())
    source_checks = {}
    for name, expected in protocol["sources"].items():
        archived = args.private_run / "frozen_sources" / name
        src = archived if archived.exists() else Path(__file__).parent / name
        source_checks[name] = hashlib.sha256(src.read_bytes()).hexdigest() == expected
    framework_checks = {}
    for name, expected in protocol.get("framework_sources", {}).items():
        archived = args.private_run / "frozen_framework_sources" / name
        framework_checks[name] = hashlib.sha256(archived.read_bytes()).hexdigest() == expected
        source_checks[name] = framework_checks[name]
    for name, expected in protocol.get("backend_sources", {}).items():
        archived = args.private_run / "frozen_backend_sources" / name
        source_checks[name] = hashlib.sha256(archived.read_bytes()).hexdigest() == expected
    assert all(source_checks.values()), source_checks
    # Load the actual archived implementation, not a later worktree version.
    frozen = args.private_run / "frozen_sources"
    proxy_source = frozen / "rsi_appworld_api_probe.py"
    if not proxy_source.exists():
        proxy_source = Path(__file__).parent / "rsi_appworld_api_probe.py"
    if frozen.exists():
        sys.path.insert(0, str(frozen))
    module_spec = importlib.util.spec_from_file_location("rsi_frozen_appworld_proxy", proxy_source)
    proxy_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(proxy_module)
    AppWorldEnvironment = proxy_module.AppWorldEnvironment
    framework_temp = None
    if framework_checks:
        framework_temp = tempfile.mkdtemp(prefix="rsi-replay-sources-")
        atexit.register(shutil.rmtree, framework_temp, ignore_errors=True)
        framework_root = Path(framework_temp)
        for package in ("AgenticArxiv", "AgenticArxiv/rsi", "AgenticArxiv/tools"):
            (framework_root / package).mkdir(parents=True, exist_ok=True)
            (framework_root / package / "__init__.py").touch()
        for name in framework_checks:
            target = framework_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.private_run / "frozen_framework_sources" / name, target)
        sys.path.insert(0, framework_temp)
    if protocol.get("isolated_framework_registry"):
        from AgenticArxiv.rsi.tool_environment import RegisteredToolEnvironment
    if protocol.get("bounded_batch_interface"):
        from AgenticArxiv.rsi.batch_environment import BoundedBatchEnvironment
    if protocol.get("same_api_batch_interface"):
        from AgenticArxiv.rsi.api_batch_environment import SameAPIBatchEnvironment
    counts = {"observations_replayed": 0, "observation_mismatches": 0,
              "goal_verdict_mismatches": 0, "episodes": len(traces),
              "primitive_attempts_by_episode": []}
    for index, trace in enumerate(traces):
        assert trace["task_id"] in protocol["task_id_sha256"]
        # Older frozen proxy constructors used the fixed default of 48.
        interaction_cap = protocol.get("max_environment_interactions", 48)
        if "max_interactions" in __import__("inspect").signature(AppWorldEnvironment).parameters:
            env = AppWorldEnvironment(ids[trace["task_id"]], f"rsi_external_api_replay_v1_{index}",
                                      max_interactions=interaction_cap)
        else:
            assert interaction_cap == 48
            env = AppWorldEnvironment(ids[trace["task_id"]], f"rsi_external_api_replay_v1_{index}")
        primitive_env = None
        if protocol.get("bounded_batch_interface"):
            wrapper = SameAPIBatchEnvironment if protocol.get("same_api_batch_interface") else BoundedBatchEnvironment
            primitive_env = wrapper(env, primitive_budget=protocol["primitive_budget"])
            env = primitive_env
        if protocol.get("isolated_framework_registry"):
            env = RegisteredToolEnvironment(env)
        try:
            for step in trace["steps"]:
                operations = step["assistant"].get("tool_calls", [])
                observations = step["observations"]
                if not observations:
                    continue
                assert len(operations) == len(observations)
                for operation, observed in zip(operations, observations):
                    function = operation.get("function") or {}
                    try:
                        arguments = json.loads(function.get("arguments", "{}"))
                        actual = env.execute(function.get("name"), arguments)
                    except (ValueError, TypeError):
                        actual = {"error": "malformed tool arguments"}
                    counts["observations_replayed"] += 1
                    counts["observation_mismatches"] += int(actual != observed["result"])
            counts["goal_verdict_mismatches"] += int(env.check() != trace["verdict"])
            if primitive_env:
                assert primitive_env.primitive_attempts <= protocol["primitive_budget"]
                counts["primitive_attempts_by_episode"].append({"task_id_sha256": trace["task_id"],
                                                               "attempts": primitive_env.primitive_attempts})
        finally:
            env.close()
    counts["all_observations_match"] = counts["observation_mismatches"] == 0
    counts["all_goal_verdicts_match"] = counts["goal_verdict_mismatches"] == 0
    counts["source_digests_match"] = source_checks
    counts["replay_proxy_source_sha256"] = hashlib.sha256(proxy_source.read_bytes()).hexdigest()
    counts["replay_framework_source_sha256"] = {
        name: hashlib.sha256((Path(framework_temp) / name).read_bytes()).hexdigest()
        for name in framework_checks}
    counts["source_execution"] = "archived snapshots when available"
    counts["note"] = "Only train scenarios replayed. Mismatches, if present, are reported without normalization."
    with args.output.open("x") as f:
        json.dump(counts, f, indent=2)
        f.write("\n")
    print(json.dumps(counts))
    if not counts["all_observations_match"] or not counts["all_goal_verdicts_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
