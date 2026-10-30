"""Replay private AppWorld train traces; export only aggregate audit metadata."""
import argparse
import hashlib
import json
from pathlib import Path

from rsi_appworld_api_probe import AppWorldEnvironment


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
           for t in load_task_ids("train", num_tasks_per_scenario=1)}
    traces = [json.loads(x) for x in (args.private_run / "traces.jsonl").read_text().splitlines()]
    protocol = json.loads((args.private_run / "protocol.json").read_text())
    source_checks = {}
    for name, expected in protocol["sources"].items():
        archived = args.private_run / "frozen_sources" / name
        src = archived if archived.exists() else Path(__file__).parent / name
        source_checks[name] = hashlib.sha256(src.read_bytes()).hexdigest() == expected
    assert all(source_checks.values()), source_checks
    counts = {"observations_replayed": 0, "observation_mismatches": 0,
              "goal_verdict_mismatches": 0, "episodes": len(traces)}
    for index, trace in enumerate(traces):
        assert trace["task_id"] in protocol["task_id_sha256"]
        env = AppWorldEnvironment(ids[trace["task_id"]], f"rsi_external_api_replay_v1_{index}")
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
        finally:
            env.close()
    counts["all_observations_match"] = counts["observation_mismatches"] == 0
    counts["all_goal_verdicts_match"] = counts["goal_verdict_mismatches"] == 0
    counts["source_digests_match"] = source_checks
    counts["replay_proxy_source_sha256"] = hashlib.sha256((Path(__file__).parent / "rsi_appworld_api_probe.py").read_bytes()).hexdigest()
    counts["note"] = "Only train scenarios replayed. Mismatches, if present, are reported without normalization."
    with args.output.open("x") as f:
        json.dump(counts, f, indent=2)
        f.write("\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
