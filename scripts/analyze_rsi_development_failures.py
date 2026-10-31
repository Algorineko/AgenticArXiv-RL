"""Export structural failure counts, never private AppWorld content or API names."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


PROXY_NAMES = {"list_apps", "list_apis", "api_doc", "call_api"}
KNOWN_ERRORS = {"invalid batch child", "invalid batch size", "primitive_budget_exhausted",
                "batch stopped after primitive error", "invalid arguments_list", "invalid API identifier"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--private-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.private_run / "protocol.json").read_text())
    traces = [json.loads(line) for line in (args.private_run / "traces.jsonl").read_text().splitlines()]
    if len(traces) != len(protocol["task_id_sha256"]):
        raise ValueError("Full planned trace set required")
    rows = []
    for trace in traces:
        if trace["task_id"] not in protocol["task_id_sha256"]:
            raise ValueError("Trace outside the frozen development selection")
        tool_counts, errors, malformed = collections.Counter(), collections.Counter(), collections.Counter()
        batch_children = 0
        for step in trace["steps"]:
            operations, observations = step["assistant"].get("tool_calls", []), step["observations"]
            if observations and len(operations) != len(observations):
                raise ValueError("Misaligned dispatched calls and observations")
            for operation, observation in zip(operations, observations):
                function = operation.get("function") or {}
                name = function.get("name")
                tool_counts[name if name in PROXY_NAMES | {"batch_tools", "batch_api_calls"} else "other"] += 1
                result = observation["result"]
                if "error" in result:
                    error = result["error"]
                    errors[error if error in KNOWN_ERRORS else "other_proxy_error"] += 1
                if name not in ("batch_tools", "batch_api_calls"):
                    continue
                batch_children += len(result.get("results", []))
                try:
                    arguments = json.loads(function.get("arguments", "{}"))
                except (TypeError, ValueError):
                    malformed["malformed_json"] += 1
                    continue
                if name == "batch_tools" and isinstance(arguments, dict):
                    children = arguments.get("calls")
                    if isinstance(children, list):
                        if any(isinstance(child, dict) and child.get("name") not in PROXY_NAMES for child in children):
                            malformed["contains_non_proxy_child_name"] += 1
                        if len(children) > 16:
                            malformed["more_than_16_children"] += 1
        rows.append({"task_id_sha256": trace["task_id"], "goal_success": trace["verdict"]["success"],
                     "stop": trace["stop"], "model_steps": len(trace["steps"]),
                     "proxy_tool_counts": dict(tool_counts), "proxy_errors": dict(errors),
                     "batch_structural_findings": dict(malformed), "batch_children_executed": batch_children})
    result = {"dataset": "preselected train/inner_train only", "episodes": rows,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model_calls": 0, "locked_final_tasks_read": 0,
              "limitation": "Proxy errors and output structure only; domain-level API failures are not fully classified."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"episodes": len(rows), "successes": sum(row["goal_success"] for row in rows),
                      "batch_name_confusion_episodes": sum(bool(row["batch_structural_findings"].get("contains_non_proxy_child_name"))
                                                           for row in rows)}))


if __name__ == "__main__":
    main()
