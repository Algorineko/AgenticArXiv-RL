"""Post hoc classification of native multi-API tool-error observations."""
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/rescue_bfcl_native_multapi_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(text):
    lower = text.lower()
    if lower.startswith("error during execution:"):
        return "executor_or_dispatch_exception"
    if any(token in lower for token in (
        "unlocked", "brake pedal", "press the brake", "fuel above",
        "cannot fill gas above", "tank capacity",
    )):
        return "state_precondition_or_capacity_rejection"
    if any(token in lower for token in (
        "no such file", "no such directory", "path not allowed",
        "unsupported path", "invalid character", "cannot write",
        "cannot move", "cannot touch",
    )):
        return "filesystem_path_or_object_rejection"
    if any(token in lower for token in (
        "not found", "no available route", "distance not found",
    )):
        return "missing_entity_or_domain_data"
    return "other_simulator_returned_error"


def main():
    protocol = ART / "protocol.json"
    reference_trace = ART / "model_runs/reference/traces.jsonl"
    references = {
        row["case_id"]: row
        for row in map(json.loads, reference_trace.read_text().splitlines())
    }
    arms = {}
    for arm in ("base_1p5", "base_7b"):
        trace = ART / "model_runs" / arm / "traces.jsonl"
        categories = Counter()
        parse_errors = 0
        attempts = 0
        turns_with_tool_error = 0
        error_turns_with_later_successful_call = 0
        later_successful_calls = 0
        error_turns_with_later_gold_aligned_action = 0
        later_gold_aligned_calls = 0
        successful_turns_after_error = 0
        examples = {}
        for line in trace.read_text().splitlines():
            record = json.loads(line)
            for turn in record["turns"]:
                saw_error = False
                recovered = False
                gold_aligned_after_error = False
                gold_calls = [
                    action
                    for step in references[record["case_id"]]["turns"][turn["turn"]]["steps"]
                    for action in step.get("actions", [])
                ]
                gold_cursor = 0
                for step in turn["steps"]:
                    if "parse_error" in step:
                        parse_errors += 1
                        continue
                    actions = step.get("actions", [])
                    observations = step.get("observations", [])
                    if len(actions) != len(observations):
                        raise ValueError(f"Action/observation mismatch in {arm}/{record['case_id']}")
                    for action, observation in zip(actions, observations):
                        attempts += 1
                        if not observation.get("tool_error"):
                            if saw_error:
                                recovered = True
                                later_successful_calls += 1
                            matched_gold = None
                            for index in range(gold_cursor, len(gold_calls)):
                                if action == gold_calls[index]:
                                    matched_gold = index
                                    break
                            if matched_gold is not None:
                                gold_cursor = matched_gold + 1
                                if saw_error:
                                    gold_aligned_after_error = True
                                    later_gold_aligned_calls += 1
                            continue
                        saw_error = True
                        label = classify(observation.get("text", ""))
                        categories[label] += 1
                        examples.setdefault(label, {
                            "tool": action["name"],
                            "observation": observation.get("text", "")[:240],
                        })
                if saw_error:
                    turns_with_tool_error += 1
                    if recovered:
                        error_turns_with_later_successful_call += 1
                    if gold_aligned_after_error:
                        error_turns_with_later_gold_aligned_action += 1
                    if turn.get("success"):
                        successful_turns_after_error += 1
        summary = {
            "function_attempts": attempts,
            "parse_error_steps": parse_errors,
            "tool_error_calls": sum(categories.values()),
            "simulator_returned_error_calls": sum(v for k, v in categories.items() if k != "executor_or_dispatch_exception"),
            "executor_or_dispatch_exceptions": categories["executor_or_dispatch_exception"],
            "turns_with_tool_error": turns_with_tool_error,
            "error_turns_with_later_successful_call": error_turns_with_later_successful_call,
            "later_successful_calls_after_error": later_successful_calls,
            "error_turns_with_later_gold_aligned_action": error_turns_with_later_gold_aligned_action,
            "later_gold_aligned_calls_after_error": later_gold_aligned_calls,
            "turns_successful_despite_prior_tool_error": successful_turns_after_error,
            "categories": dict(sorted(categories.items())),
            "examples": examples,
            "trace_sha256": digest(trace),
        }
        arms[arm] = summary
    total_errors = sum(x["tool_error_calls"] for x in arms.values())
    total_sim_errors = sum(x["simulator_returned_error_calls"] for x in arms.values())
    result = {
        "kind": "posthoc_native_multapi_error_taxonomy_v1",
        "protocol_sha256": digest(protocol),
        "arms": arms,
        "combined": {
            "tool_error_calls": total_errors,
            "simulator_returned_error_calls": total_sim_errors,
            "simulator_return_share": round(total_sim_errors / total_errors, 6),
        },
        "reference_trace_sha256": digest(reference_trace),
        "method": "Classify each runtime-marked tool_error from its recorded observation. Exceptions caught by the dispatcher are separated from simulator method responses. Simulator response families use explicit, inspectable keyword rules. Recovery counts record any later successful call and, separately, calls matching a subsequent exact gold name/argument pair in order. This is descriptive and post hoc, not a validated latent-error model.",
    }
    out = ART / "error_taxonomy.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["combined"], sort_keys=True))


if __name__ == "__main__":
    main()
