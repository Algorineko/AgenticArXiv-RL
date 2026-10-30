"""Audit all four frozen development arms before emitting descriptive results."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AgenticArxiv"))
from benchmark.rescue_catalog import sha256, verify_catalog
from rl.self_evolve import strict_success

ARMS = ("base_baseline", "grpo_baseline", "base_completion_check", "grpo_completion_check")


def analyze(catalog, runs):
    _, tasks, splits = verify_catalog(catalog)
    families = {t["id"]: t["rescue_family"] for t in tasks}
    expected = set(splits["split"]["search_dev"])
    catalog_hash = sha256(catalog / "manifest.json")
    manifests, scores, output = {}, {}, {}
    for arm in ARMS:
        directory = runs / arm
        manifest = json.loads((directory / "run_manifest.json").read_text())
        summary = json.loads((directory / "benchmark/summary.json").read_text())
        rows = summary["details"]
        traces = [json.loads(x) for x in (directory / "traces.jsonl").read_text().splitlines()]
        if (manifest["status"] != "completed" or manifest["policy"] != "model"
                or manifest["split"] != "search_dev" or manifest["final_test_model_evaluated"]
                or manifest["catalog_manifest_sha256"] != catalog_hash
                or summary["error_count"] or summary.get("errors")):
            raise ValueError(f"{arm}: incomplete, wrong-split, changed-catalog, or execution-error run")
        if (len(rows) != len(expected) or len(traces) != len(expected)
                or {x["task_id"] for x in rows} != expected
                or {x["task_id"] for x in traces} != expected
                or set(manifest["completed_task_ids"]) != expected
                or any(x["trial"] != 0 or x["agent_type"] != "regex" for x in rows)):
            raise ValueError(f"{arm}: invalid task/trial coverage")
        if sha256(directory / "traces.jsonl") != manifest["traces_sha256"]:
            raise ValueError(f"{arm}: modified traces")
        if manifest["variant"] != arm.split("_", 1)[1]:
            raise ValueError(f"{arm}: wrong prompt variant")
        if any(manifest[k] != v for k, v in {
                "seed": 45, "repeat": 1, "decode": "greedy", "max_new_tokens": 384, "max_iterations": 6}.items()):
            raise ValueError(f"{arm}: changed generation budget")
        values = {x["task_id"]: strict_success(x) for x in rows}
        if sum(values.values()) != manifest["strict_successes"]:
            raise ValueError(f"{arm}: score does not regenerate")
        counts = defaultdict(lambda: {"successes": 0, "n": 0})
        for row in rows:
            family = counts[families[row["task_id"]]]
            family["n"] += 1
            family["successes"] += int(values[row["task_id"]])
        usage = {k: sum(t["raw_result"]["token_usage"][k] for t in traces)
                 for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        if usage["total_tokens"] != sum(x["tokens"] for x in rows):
            raise ValueError(f"{arm}: token accounting mismatch")
        output[arm] = {"model_id": manifest["model_id"], "strict_successes": sum(values.values()),
            "n": len(rows), "false_finishes": sum(bool(x["false_finish"]) for x in rows),
            "force_stops": sum(x["termination"] == "FORCE_STOP" for x in rows),
            "single_action_finishes": sum(
                len(t["raw_result"]["history"]) == 1
                and str(t["raw_result"]["history"][0].get("action", "")).strip().upper() == "FINISH"
                for t in traces),
            "parse_failures": sum(x["parse_fail"] for x in rows),
            "tool_failures": sum(x["tool_fail"] for x in rows),
            "by_family": dict(counts), "token_usage": usage,
            "rollout_seconds": round(sum(x["total_ms"] for x in rows) / 1000, 3),
            "wall_seconds_including_setup": manifest["wall_seconds"],
            "manifest_sha256": sha256(directory / "run_manifest.json"),
            "summary_sha256": sha256(directory / "benchmark/summary.json"),
            "traces_sha256": manifest["traces_sha256"]}
        manifests[arm], scores[arm] = manifest, values
    for arm in ARMS[1:]:
        if manifests[arm]["source_sha256"] != manifests[ARMS[0]]["source_sha256"]:
            raise ValueError("Model arms ran different evaluation/scoring code")
    for model in ("base", "grpo"):
        a, b = manifests[model + "_baseline"], manifests[model + "_completion_check"]
        if a["weights_sha256"] != b["weights_sha256"] or a["model_id"] != b["model_id"]:
            raise ValueError("Prompt comparison uses different checkpoint weights")
    comparisons = {}
    for model in ("base", "grpo"):
        a, b = scores[model + "_baseline"], scores[model + "_completion_check"]
        comparisons[model + "_prompt_change"] = {
            "fixed": sorted(t for t in expected if not a[t] and b[t]),
            "regressed": sorted(t for t in expected if a[t] and not b[t]),
            "both_correct": sum(a[t] and b[t] for t in expected),
            "both_wrong": sum(not a[t] and not b[t] for t in expected)}
    return {"kind": "descriptive_development_screen_only", "catalog_manifest_sha256": catalog_hash,
            "arms": output, "paired_prompt_changes": comparisons,
            "independent_task_families": 6, "parameter_instances_per_family": 4,
            "final_test_model_evaluated": False, "promotion_val_model_evaluated": False,
            "limitations": ["No confidence/significance claim from correlated parameter variants",
                            "Base versus SFT+GRPO pipeline does not isolate an RL effect",
                            "Strict scoring requires the prescribed action chain, not an independent goal-state verifier",
                            "Paper-disjoint task targets do not restrict tool access to the other catalog categories",
                            "No autonomous evolution, reasoning, PDF-quality, or online-service claim"]}


def markdown(data):
    lines = ["# Frozen development screen v1", "",
        "24 development instances in six related task families; one deterministic trial per arm. "
        "Promotion and final model evaluations remain closed.", "",
        "| Arm | Strict | False finish | Total tokens | Rollout seconds |",
        "|---|---:|---:|---:|---:|"]
    for arm, row in data["arms"].items():
        lines.append(f"| {arm} | {row['strict_successes']}/{row['n']} | {row['false_finishes']} | "
                     f"{row['token_usage']['total_tokens']} | {row['rollout_seconds']} |")
    lines += ["", "## Family counts", "", "| Family | Base | GRPO | Base + check | GRPO + check |",
              "|---|---:|---:|---:|---:|"]
    for family in data["arms"][ARMS[0]]["by_family"]:
        cells = [str(data["arms"][a]["by_family"][family]["successes"]) + "/4" for a in ARMS]
        lines.append("| " + family + " | " + " | ".join(cells) + " |")
    lines += ["", "## Paired prompt changes", ""]
    for name, row in data["paired_prompt_changes"].items():
        lines += [f"- {name}: {len(row['fixed'])} fixed, {len(row['regressed'])} regressed; "
                  f"{row['both_correct']} correct in both, {row['both_wrong']} wrong in both."]
    lines += ["", "## Interpretation limits", ""]
    lines.extend("- " + x for x in data["limitations"])
    lines += ["", "Raw task counts are descriptive. Family and parameter dependence prohibit treating "
              "the 24 instances as 24 independent research replications. Rollout time excludes initial "
              "weight hashing/loading; shared-server timing is not a controlled throughput comparison.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = analyze(args.catalog, args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "analysis.json").open("x") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    with (args.output / "analysis.md").open("x") as f:
        f.write(markdown(data))
    print(json.dumps({a: {k: d[k] for k in ("strict_successes", "n", "false_finishes")}
                      for a, d in data["arms"].items()}))


if __name__ == "__main__":
    main()
