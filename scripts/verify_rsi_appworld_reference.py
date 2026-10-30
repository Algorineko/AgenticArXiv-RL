"""Run official reference solutions for the two exposed train probes only.

Reference code and its execution logs stay inside the external AppWorld cache;
only aggregate success and opaque task hashes are exported. No LLM is called.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-native-calls", type=int, default=1000)
    args = p.parse_args()
    if not 1 <= args.max_native_calls <= 1000:
        p.error("Reference native-call cap must be between 1 and 1000")
    from appworld import AppWorld, load_task_ids
    from appworld.common.path_store import path_store
    path_store.update_root(args.root)
    results = []
    for i, task in enumerate(load_task_ids("train", num_tasks_per_scenario=1)[:2]):
        with AppWorld(task_id=task, experiment_name=f"rsi_reference_cap{args.max_native_calls}_{i}",
                      load_ground_truth=True, ground_truth_mode="full", random_seed=45,
                      timeout_seconds=30, max_interactions=1,
                      max_api_calls_per_interaction=args.max_native_calls) as world:
            # This is trusted official benchmark reference code, not model output.
            code = world.task.ground_truth.compiled_solution_code + "\nsolution(apis, requester)"
            observation = world.execute(code)
            native_calls = len(world.requester.request_tracker.requests)
            result = world.evaluate().to_dict(stats_only=True)
            results.append({"task_id_sha256": hashlib.sha256(task.encode()).hexdigest(),
                            "reference_success": result["success"], "official_check_count": result["num_tests"],
                            "native_api_calls": native_calls,
                            "execution_traceback": "Traceback" in observation,
                            "reference_code_sha256": hashlib.sha256(code.encode()).hexdigest()})
    with open(args.output, "x") as f:
        json.dump({"dataset": "train only", "model_calls": 0,
                   "max_native_calls_in_single_reference_execution": args.max_native_calls,
                   "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "limitation": "Trusted Python reference, not a model or proxy-policy capability bound. Failure does not prove no shorter policy exists.",
                   "results": results}, f, indent=2)
        f.write("\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
