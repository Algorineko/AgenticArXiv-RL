"""ID-only pool provenance audit; never load tasks, solutions or checkers."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--private-pool", type=Path, required=True)
    p.add_argument("--public-pool", type=Path, required=True)
    p.add_argument("--previous-private-pool", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from appworld import load_task_ids
    from appworld.task import task_id_to_generator_id, task_id_to_number
    from appworld.common.path_store import path_store
    path_store.update_root(args.root)
    registry = load_task_ids("train")
    excluded = {task_id_to_generator_id(t) for t in load_task_ids("train", num_tasks_per_scenario=1)[:2]}
    private = json.loads(args.private_pool.read_text())
    public = json.loads(args.public_pool.read_text())
    expected_counts = {"meta_train": 12, "meta_dev": 8, "meta_final_locked": 8}
    assert set(private) == set(public["pools"]) == set(expected_counts)
    assert public["train_registry_sha256"] == sha(json.dumps(sorted(registry)))
    builder = Path(__file__).with_name("build_rsi_appworld_pool.py")
    assert public["builder_source_sha256"] == hashlib.sha256(builder.read_bytes()).hexdigest()
    seen_groups, seen_tasks = set(), set()
    for pool, number in expected_counts.items():
        assert len(private[pool]) == len(public["pools"][pool]) == number
        for entry, published in zip(private[pool], public["pools"][pool]):
            generator = entry["generator"]
            assert generator not in seen_groups | excluded
            seen_groups.add(generator)
            assert published["generator_sha256"] == sha(generator)
            roles = entry["tasks"]
            assert set(roles) == {"inner_train", "inner_selection", "inner_audit"}
            assert published["task_sha256_by_role"] == {k: sha(v) for k, v in roles.items()}
            expected = sorted((t for t in registry if task_id_to_generator_id(t) == generator),
                              key=task_id_to_number)
            assert [roles[r] for r in ("inner_train", "inner_selection", "inner_audit")] == expected
            for task in roles.values():
                assert task in registry and task not in seen_tasks
                seen_tasks.add(task)
    assert seen_groups == {task_id_to_generator_id(t) for t in registry} - excluded
    same_membership = None
    if args.previous_private_pool:
        same_membership = private == json.loads(args.previous_private_pool.read_text())
        assert same_membership
    result = {"passed": True, "generators": len(seen_groups), "instances": len(seen_tasks),
              "excluded_exposed_generators": len(excluded),
              "same_membership_as_previous": same_membership,
              "public_manifest_sha256": hashlib.sha256(args.public_pool.read_bytes()).hexdigest(),
              "verifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "task_instructions_loaded": 0, "solutions_loaded": 0, "checker_labels_loaded": 0,
              "scope": "ID and provenance checks only; semantic independence, difficulty and contamination not established."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
