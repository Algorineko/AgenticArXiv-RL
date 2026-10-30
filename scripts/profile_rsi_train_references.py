"""Profile a preselected prefix of meta-train references, never final/dev tasks.

References are trusted host diagnostics, not accessible experience for an agent.
Their API-call counts describe these solutions, not optimal-policy lower bounds.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--private-pool", type=Path, required=True)
    p.add_argument("--public-pool", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=4)
    args = p.parse_args()
    if not 1 <= args.count <= 4 or args.output.exists():
        p.error("Use one to four preselected meta-train groups and a new output")
    from appworld import AppWorld, load_task_ids
    from appworld.common.path_store import path_store
    path_store.update_root(args.root)
    private = json.loads(args.private_pool.read_text())
    public = json.loads(args.public_pool.read_text())
    selected = private["meta_train"][:args.count]
    registered_train = set(load_task_ids("train"))
    results = []
    for index, entry in enumerate(selected):
        task_id = entry["tasks"]["inner_train"]
        published = public["pools"]["meta_train"][index]
        task_hash = hashlib.sha256(task_id.encode()).hexdigest()
        assert task_hash == published["task_sha256_by_role"]["inner_train"]
        assert task_id in registered_train
        with AppWorld(task_id=task_id, experiment_name=f"rsi_train_reference_profile_v1_{index}",
                      load_ground_truth=True, ground_truth_mode="full", random_seed=45,
                      timeout_seconds=30, max_interactions=1,
                      max_api_calls_per_interaction=256) as world:
            noop = world.evaluate().to_dict(stats_only=True)["success"]
            code = world.task.ground_truth.compiled_solution_code + "\nsolution(apis, requester)"
            observation = world.execute(code)
            native_calls = len(world.requester.request_tracker.requests)
            verdict = world.evaluate().to_dict(stats_only=True)
            results.append({"task_id_sha256": task_hash,
                            "generator_sha256": published["generator_sha256"],
                            "no_op_success": noop, "reference_success": verdict["success"],
                            "native_api_calls": native_calls,
                            "execution_traceback": "Traceback" in observation,
                            "reference_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                            "official_check_count": verdict["num_tests"]})
    result = {"selection": "First four (or shorter prefix) frozen meta_train groups; inner_train only, no score-based selection",
              "model_calls": 0, "max_native_calls": 256,
              "meta_dev_tasks_read": 0, "locked_final_tasks_read": 0,
              "public_pool_sha256": hashlib.sha256(args.public_pool.read_bytes()).hexdigest(),
              "profiler_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reference_experience_policy": "Source/observations stay private and are never passed to improvers or solvers",
              "limitation": "Trusted program counts, not an agent score or policy-length lower bound",
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
