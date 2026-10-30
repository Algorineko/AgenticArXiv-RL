"""Allocate unused AppWorld train generators from IDs only, before optimization.

This freezes an eligibility inventory, not a claim that task difficulty or
semantic lineage has been validated. Only hashes leave the private runtime.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--public-output", type=Path, required=True)
    p.add_argument("--private-output", type=Path, required=True)
    args = p.parse_args()
    from appworld import load_task_ids
    from appworld.task import task_id_to_generator_id, task_id_to_number
    from appworld.common.path_store import path_store
    path_store.update_root(args.root)
    groups = collections.defaultdict(list)
    for task in load_task_ids("train"):
        groups[task_id_to_generator_id(task)].append(task)
    exposed = {task_id_to_generator_id(t) for t in load_task_ids("train", num_tasks_per_scenario=1)[:2]}
    eligible = sorted((g for g in groups if g not in exposed), key=lambda g: sha("rsi-pool-20261030:" + g))
    assert len(eligible) == 28
    public, private = {}, {}
    for name, subset in (("meta_train", eligible[:12]), ("meta_dev", eligible[12:20]), ("meta_final_locked", eligible[20:])):
        public[name], private[name] = [], []
        for generator in subset:
            tasks = sorted(groups[generator], key=task_id_to_number)
            assert len(tasks) == 3
            roles = dict(zip(("inner_train", "inner_selection", "inner_audit"), tasks))
            public[name].append({"generator_sha256": sha(generator), "task_sha256_by_role": {k: sha(v) for k, v in roles.items()}})
            private[name].append({"generator": generator, "tasks": roles})
    metadata = {"source": "AppWorld 0.1.3.post1 official train ID registry only",
                "builder_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "train_registry_sha256": sha(json.dumps(sorted(load_task_ids("train")))),
                "selection_seed": "rsi-pool-20261030", "source_train_tasks": 90,
                "excluded_exposed_generators": sorted(sha(g) for g in exposed),
                "counts": {k: {"generators": len(v), "instances": 3 * len(v)} for k, v in public.items()},
                "status": "Eligibility split only; healthy baseline and semantic-lineage audit still required before P1",
                "boundary": "Generator IDs disjoint; shared applications, APIs and semantic primitives remain. Not an OOD claim.",
                "pretraining_contamination_unknown": True,
                "not_an_official_AppWorld_test_score": True,
                "final_policy": "Do not execute/read task instructions, solutions or checker labels in meta_final_locked before a separate frozen final protocol.",
                "pools": public}
    for path, data in ((args.public_output, metadata), (args.private_output, private)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
    print(json.dumps(metadata["counts"]))


if __name__ == "__main__":
    main()
