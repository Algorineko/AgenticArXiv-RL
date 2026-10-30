"""Readiness check on one AppWorld train task; no LLM calls or test-split reads.

Only aggregate metrics and encrypted-source digests are written to the research
archive. Protected instructions, API documentation and checker details stay in
the external AppWorld runtime directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output file already exists")
    from appworld import AppWorld, load_task_ids
    from appworld.common.path_store import path_store
    import appworld

    path_store.update_root(str(args.root))
    task_ids = load_task_ids("train", num_tasks_per_scenario=1)
    if not task_ids:
        raise RuntimeError("no train tasks available")
    bundle_dir = Path(appworld.__file__).parent / ".source"
    result = {"stage": "external CPU engine preflight, not a model evaluation",
              "python": platform.python_version(), "appworld_version": importlib.metadata.version("appworld"),
              "root": str(args.root), "dataset_read": "train only",
              "train_scenarios_available": len(task_ids),
              "selected_task_id_sha256": hashlib.sha256(task_ids[0].encode()).hexdigest(),
              "encrypted_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in bundle_dir.glob("*.bundle")},
              "dependencies": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())}
    # Ground truth exists in the host solely for its official evaluator; no model is
    # invoked, and instructions/solutions/tests are never exported into this record.
    with AppWorld(task_id=task_ids[0], experiment_name="rsi_cpu_engine_preflight_v1",
                  load_ground_truth=True, ground_truth_mode="minimal", random_seed=45,
                  timeout_seconds=10, max_interactions=5) as world:
        observation = world.execute("print(apis.api_docs.show_app_descriptions())")
        if "Traceback" in observation or not observation.strip():
            raise RuntimeError("API documentation discovery failed")
        result["documentation_discovery_operational"] = True
        result["documentation_observation_characters"] = len(observation)
        initial = world.evaluate().to_dict(stats_only=True)
        result["no_op_verdict"] = {k: initial[k] for k in ("success", "num_tests")}
        assert not initial["success"], "selected train task has no headroom at initial state"
        result["gpu_used"] = False
        result["model_calls"] = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "dependencies"}))


if __name__ == "__main__":
    main()
