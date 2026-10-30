"""Replay every recorded tool operation and verify P0 accounting offline."""
import argparse
import hashlib
import json
from pathlib import Path

from rsi_cpu_tasks import Environment


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(root):
    protocol = json.loads((root / "protocol.json").read_text())
    tasks = {t["id"]: t for t in json.loads((root / "catalog.json").read_text())}
    traces = read_lines(root / "traces.jsonl")
    calls = read_lines(root / "api_calls.jsonl")
    summary = json.loads((root / "summary.json").read_text())
    assert len({t["episode_id"] for t in traces}) == len(traces)
    assert [c["call"] for c in calls] == list(range(1, len(calls) + 1))
    replayed, domain = 0, {}
    for trace in traces:
        env = Environment(tasks[trace["task_id"]])
        attempts = errors = 0
        try:
            assert len(trace["steps"]) <= protocol["max_steps"]
            for step in trace["steps"]:
                operations = step["assistant"].get("tool_calls", [])
                observations = step["observations"]
                if not observations:
                    continue  # truncated generation or refused parallel batch was not dispatched
                assert len(operations) == len(observations) <= protocol["max_parallel_tools"]
                for operation, recorded in zip(operations, observations):
                    function = operation.get("function") or {}
                    try:
                        arguments = json.loads(function.get("arguments", "{}"))
                        result = env.execute(function.get("name"), arguments)
                    except (TypeError, ValueError):
                        result = {"error": "malformed tool arguments"}
                    assert result == recorded["result"], (trace["episode_id"], recorded, result)
                    assert function.get("name") == recorded["name"]
                    attempts += 1
                    errors += int("error" in result)
                    replayed += 1
            assert env.check() == trace["verdict"], trace["episode_id"]
            assert attempts == trace["tool_calls"] and errors == trace["tool_errors"]
            d = domain.setdefault(trace["domain"], {"episodes": 0, "successes": 0})
            d["episodes"] += 1
            d["successes"] += int(trace["verdict"]["success"])
        finally:
            env.close()
    assert len(traces) == summary["episodes"]
    assert sum(t["verdict"]["success"] for t in traces) == summary["successes"]
    assert domain == summary["by_domain"]
    assert len(calls) == summary["usage"]["calls"] <= protocol["limits"]["calls"]
    accounted_input = accounted_output = unknown = 0
    for call in calls:
        usage = (call.get("response") or {}).get("usage") or {}
        if call["usage_accounting"] == "provider_reported":
            accounted_input += usage["prompt_tokens"]
            accounted_output += usage["completion_tokens"]
        else:
            unknown += 1
            accounted_input += call["reserved_input_upper_estimate"]
            accounted_output += call["request"]["max_tokens"]
    assert accounted_input == summary["usage"]["accounted_input_tokens"]
    assert accounted_output == summary["usage"]["accounted_output_tokens"]
    assert unknown == summary["usage"]["calls_without_provider_usage"]
    source_checks = {}
    for filename, expected in protocol["source_sha256"].items():
        archived = root / "frozen_sources" / filename
        actual = hashlib.sha256((archived if archived.exists() else Path(__file__).parent / filename).read_bytes()).hexdigest()
        source_checks[filename] = actual == expected
    assert all(source_checks.values()), source_checks
    groups = {}
    for trace in traces:
        groups.setdefault(trace["task_id"], []).append(trace)
    pairs = [g for g in groups.values() if len(g) == 2]
    return {"tool_observations_replayed": replayed, "all_observations_match": True,
            "all_goal_checks_match": True, "accounting_matches": True, "source_digests_match": source_checks,
            "complete": summary["complete"], "episodes": len(traces), "successes": summary["successes"],
            "by_domain": domain, "repeat_pairs": len(pairs),
            "success_disagreements_between_repeats": sum(g[0]["verdict"]["success"] != g[1]["verdict"]["success"] for g in pairs),
            "limitation": "Eight development instances with related templates; no independence or RSI effect claim."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.run)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, ensure_ascii=False))
