"""Offline upstream-GEPA plumbing test; scripted solver/reflection, no LLMs.

Toy scores in this script are test fixtures, never method effectiveness data.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from AgenticArxiv.rsi.gepa_adapter import NativeTraceGEPAAdapter
from rsi_cpu_tasks import Environment, catalog

UPSTREAM_COMMIT = "80a62c879ecdeb8741faac72a844dceb1c18691f"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gepa-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    subprocess.run(["git", "-C", str(args.gepa_source), "diff", "--exit-code",
                    UPSTREAM_COMMIT, "--", "src/gepa"], check=True, capture_output=True)
    sys.path.insert(0, str(args.gepa_source / "src"))
    import gepa
    from gepa import ScoreThresholdStopper
    specimens = catalog()[:2]
    fixtures = {t["id"]: t for t in specimens}
    tasks = [{"id": t["id"], "prompt": t["prompt"], "domain": t["domain"]} for t in specimens]

    def scripted_episode(task, harness):
        env = Environment(fixtures[task["id"]])
        steps = []
        try:
            if "reserve-exact" in harness:
                initial = env.execute("inspect_order", {})
                for sku, required in fixtures[task["id"]]["lines"].items():
                    missing = required - initial["reserved"].get(sku, 0)
                    if missing:
                        result = env.execute("reserve_item", {"sku": sku, "quantity": missing})
                        steps.append({"operation": "reserve_item", "result": result})
                steps.append({"operation": "dispatch_order", "result": env.execute("dispatch_order", {})})
            return {"task_id": task["id"], "steps": steps, "stop": "scripted_fixture",
                    "verdict": env.check()}
        finally:
            env.close()

    adapter = NativeTraceGEPAAdapter(scripted_episode, fixtures)
    reflections = []

    def scripted_reflection(prompt):
        reflections.append(prompt)
        return "```\nreserve-exact\n```"

    allowlist_rejected = False
    try:
        adapter.evaluate([{"id": "not-allowed", "prompt": "", "domain": "inventory"}],
                         {"solver_harness": "noop"})
    except ValueError:
        allowlist_rejected = True
    assert adapter.evaluated_episodes == 0
    with tempfile.TemporaryDirectory(prefix="gepa_offline_", dir=args.gepa_source.parent) as run_dir:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = gepa.optimize(seed_candidate={"solver_harness": "noop"}, trainset=tasks, valset=tasks,
                                   adapter=adapter, reflection_lm=scripted_reflection,
                                   reflection_minibatch_size=1, max_metric_calls=12,
                                   stop_callbacks=ScoreThresholdStopper(1.0),
                                   run_dir=run_dir, seed=0, display_progress_bar=False)
    checks = {"allowlist_rejected_before_execution": allowlist_rejected,
              "official_engine_proposed_and_evaluated": len(result.candidates) >= 2 and bool(reflections),
              "independent_sqlite_fixture_passed": result.best_candidate == {"solver_harness": "reserve-exact"}
                                                       and result.best_score == 1.0,
              "metric_count_matches_adapter": result.total_metric_calls == adapter.evaluated_episodes}
    source_paths = ("AgenticArxiv/rsi/gepa_adapter.py", "scripts/verify_rsi_gepa_adapter.py", "scripts/rsi_cpu_tasks.py")
    report = {"passed": all(checks.values()), "checks": checks,
              "upstream_url": "https://github.com/gepa-ai/gepa", "upstream_commit": UPSTREAM_COMMIT,
              "upstream_core_unchanged": True, "llm_api_calls": 0,
              "scripted_reflection_calls": len(reflections), "fixture_episodes": adapter.evaluated_episodes,
              "sources": {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in source_paths},
              "scope": "Synthetic plumbing fixture only; scripted policies, same toy train/val. No GEPA or RSI effectiveness result."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
