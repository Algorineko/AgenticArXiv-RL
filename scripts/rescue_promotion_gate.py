"""Development-only promotion gate; final-test scores are never input.

The frozen split file must contain train/search_dev/promotion_val/final_test.
This checks the declared IDs, not semantic/parent-family independence. New
research must audit parent families separately, and limit promotion-set access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "AgenticArxiv"))
from rl.self_evolve import gate, pass_k  # noqa: E402

SPLITS = ("train", "search_dev", "promotion_val", "final_test")


def evaluate(previous, candidate, split_payload, *, k=1, min_gain=0.0):
    if min_gain < 0:
        raise ValueError("min_gain must be non-negative")
    splits = split_payload.get("split", {})
    seen = set()
    for name in SPLITS:
        ids = splits.get(name)
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
            raise ValueError(f"Missing/non-empty string-list split: {name}")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f"Duplicate or overlapping task IDs: {name}")
        seen.update(ids)
    expected = set(splits["promotion_val"])
    for label, summary in (("previous", previous), ("candidate", candidate)):
        rows = summary.get("details", [])
        actual = {row.get("task_id") for row in rows}
        if actual != expected:
            raise ValueError(f"{label}: must cover exactly promotion_val; test/train/dev IDs forbidden")
        if summary.get("error_count", 0) or summary.get("errors"):
            raise ValueError(f"{label}: execution errors invalidate promotion")
        agents = {row.get("agent_type") for row in rows}
        if agents != {"regex"}:
            raise ValueError(f"{label}: expected one regex policy, got {agents}")
        trials = [(row.get("task_id"), row.get("trial")) for row in rows]
        if len(set(trials)) != len(trials):
            raise ValueError(f"{label}: duplicate task/trial rows")
    result = gate({"promotion_val": previous}, {"promotion_val": candidate},
                  cases_before=0, cases_after=0, k=k, splits=("promotion_val",))
    before, after = pass_k(previous, k), pass_k(candidate, k)
    gain = after - before if before is not None and after is not None else None
    accepted = result.passed and gain is not None and gain > min_gain
    return {"kind": "development_promotion_only", "accepted": accepted,
            "previous_score": before, "candidate_score": after, "gain": gain,
            "strict_gain_threshold": min_gain, "k": k, "regression_gate": result.to_dict(),
            "final_test_used": False,
            "limitations": "No statistical significance, family-disjointness, cost, or regression-suite claim."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--split-sha256", required=True, help="Hash committed before the search")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--min-gain", type=float, default=0.0)
    args = p.parse_args()
    raw = args.split.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.split_sha256:
        p.error("Split manifest differs from its pre-search hash")
    result = evaluate(json.loads(args.previous.read_text()), json.loads(args.candidate.read_text()),
                      json.loads(raw), k=args.k, min_gain=args.min_gain)
    result["split_sha256"] = args.split_sha256
    result["input_sha256"] = {label: hashlib.sha256(path.read_bytes()).hexdigest()
                              for label, path in (("previous", args.previous), ("candidate", args.candidate))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
