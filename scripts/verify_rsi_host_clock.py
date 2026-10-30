"""Check host accounting clocks under AppWorld's simulated-time context."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

from rsi_api_pilot import monotonic, utc


def sample():
    native_start = monotonic()
    patched_start = time.monotonic()
    time.sleep(0.08)
    return {"accounting_elapsed_s": monotonic() - native_start,
            "environment_monotonic_elapsed_s": time.monotonic() - patched_start,
            "accounting_matches_system_utc": abs(
                datetime.datetime.fromisoformat(utc()).timestamp() -
                time.clock_gettime(time.CLOCK_REALTIME)) < 1,
            "environment_utc_matches_system": abs(
                datetime.datetime.now(datetime.timezone.utc).timestamp() -
                time.clock_gettime(time.CLOCK_REALTIME)) < 1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from appworld import AppWorld, load_task_ids
    from appworld.common.path_store import path_store
    path_store.update_root(args.root)
    outside = sample()
    with AppWorld(task_id=load_task_ids("train", num_tasks_per_scenario=1)[0],
                  experiment_name="rsi_host_clock_verification", load_ground_truth=False):
        inside = sample()
    result = {"outside": outside, "inside": inside,
              "api_calls": 0, "read_locked_final_tasks": 0,
              "api_source_sha256": hashlib.sha256(Path(__file__).with_name("rsi_api_pilot.py").read_bytes()).hexdigest(),
              "verifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result["passed"] = all(s["accounting_elapsed_s"] >= .06 and s["accounting_matches_system_utc"]
                           for s in (inside, outside)) and inside["environment_monotonic_elapsed_s"] == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
