"""Run official reference solutions for the two exposed train probes only.

Reference code and its execution logs stay inside the external AppWorld cache;
only aggregate success and opaque task hashes are exported. No LLM is called.
"""
import argparse
import hashlib
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from appworld import AppWorld, load_task_ids
    from appworld.common.path_store import path_store
    path_store.update_root(args.root)
    results = []
    for i, task in enumerate(load_task_ids("train", num_tasks_per_scenario=1)[:2]):
        with AppWorld(task_id=task, experiment_name=f"rsi_reference_v1_{i}",
                      load_ground_truth=True, ground_truth_mode="full", random_seed=45,
                      timeout_seconds=30) as world:
            # This is trusted official benchmark reference code, not model output.
            code = world.task.ground_truth.compiled_solution_code + "\nsolution(apis, requester)"
            world.execute(code)
            result = world.evaluate().to_dict(stats_only=True)
            results.append({"task_id_sha256": hashlib.sha256(task.encode()).hexdigest(),
                            "reference_success": result["success"], "official_check_count": result["num_tests"]})
    with open(args.output, "x") as f:
        json.dump({"dataset": "train only", "model_calls": 0, "results": results}, f, indent=2)
        f.write("\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
