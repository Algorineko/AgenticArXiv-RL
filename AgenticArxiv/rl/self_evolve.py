"""Bounded self-evolution data loop (README Roadmap P3).

One round:

    diagnose  held-out eval summary.json  ->  per-template-family weakness report
    mine      failing traces              ->  per-round badcase file (library only grows)
    select    weak families               ->  train-split parents to re-derive
    synthesize (scripts/generate_parametric_sft_data.py --only-parents ...)
    freeze    every input/output          ->  round manifest with SHA256 lineage
    retrain   opaque command (never imported here)
    gate      new four-split eval vs. previous round -> accept / reject

Everything except ``retrain`` is a pure function over JSON files: no model, no
network, no torch. The loop deliberately reuses the existing pieces (benchmark
report schema, ``benchmark.badcases.capture``, the parametric derivations and
the SFT manifest conventions) instead of adding new mechanisms, and it never
changes rewards, tools or task templates. Promotion of a round is decided by
``gate`` only; nothing under ``data/`` is touched by this module.

Usage (from the repository root)::

    python -m AgenticArxiv.rl.self_evolve diagnose --eval dev=eval_results/x/dev/summary.json ...
    python -m AgenticArxiv.rl.self_evolve run --round 1 --eval ... --split data/splits/v3_81.json \
        --snapshot data/mock_arxiv_snapshot.json --parametric-manifest data/sft/sft_v2_parametric_seed.jsonl.manifest.json
    python -m AgenticArxiv.rl.self_evolve gate --round-dir artifacts/self_evolve/round_1 \
        --prev dev=... --new dev=... --cases-before 14 --cases-after 17
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from benchmark.splits import template_key  # noqa: E402

GATE_SPLITS: Tuple[str, ...] = ("dev", "iid_test", "ood_test")
PASS_K = 3

FAILURE_MODES: Tuple[str, ...] = (
    "not_completed",
    "wrong_tools",
    "wrong_args",
    "wrong_ref",
    "false_finish",
    "parse_fail",
    "tool_fail",
    "terminal_semantics",
)

ROUND_MANIFEST_KIND = "self_evolve_round"
BADCASE_FILE_NAME = "eval_cases.open.jsonl"  # per-round, human-merged into eval/eval_cases.jsonl
WEAKNESS_FILE_NAME = "weakness_report.json"
SELECTION_FILE_NAME = "selection.json"
MANIFEST_FILE_NAME = "manifest.json"
GATE_FILE_NAME = "gate_result.json"


# --------------------------------------------------------------------------- #
# hashing / io
# --------------------------------------------------------------------------- #

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def parse_split_assignments(items: Sequence[str]) -> Dict[str, Path]:
    """``["dev=path/a.json", "iid_test=path/b.json"]`` -> ``{"dev": Path(...), ...}``."""
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected SPLIT=PATH, got {item!r}")
        split, raw = item.split("=", 1)
        split = split.strip()
        if not split or not raw.strip():
            raise ValueError(f"expected SPLIT=PATH, got {item!r}")
        if split in out:
            raise ValueError(f"split {split!r} given twice")
        out[split] = Path(raw.strip())
    return out


# --------------------------------------------------------------------------- #
# per-trial verdicts (mirrors benchmark.metrics.is_strict_success on the
# ``details`` rows that benchmark/report.py writes into summary.json)
# --------------------------------------------------------------------------- #

def _score_ok(row: Mapping[str, Any], score_key: str, applicable_key: str) -> bool:
    score = row.get(score_key)
    if score is None:
        return not row.get(applicable_key, True)
    return float(score) == 1.0


def strict_success(row: Mapping[str, Any]) -> bool:
    """Same predicate as ``is_strict_success`` but over a summary ``details`` row."""
    return bool(
        row.get("completed")
        and row.get("tool_accurate")
        and _score_ok(row, "arg_score", "arg_applicable")
        and _score_ok(row, "ref_score", "ref_applicable")
        and int(row.get("parse_fail") or 0) == 0
        and int(row.get("tool_fail") or 0) == 0
        and row.get("terminal_semantics_accurate", True)
    )


def failure_modes(row: Mapping[str, Any]) -> List[str]:
    """Which strict-success clauses a trial violated (empty list == strict success)."""
    modes: List[str] = []
    if not row.get("completed"):
        modes.append("not_completed")
    if not row.get("tool_accurate"):
        modes.append("wrong_tools")
    if not _score_ok(row, "arg_score", "arg_applicable"):
        modes.append("wrong_args")
    if not _score_ok(row, "ref_score", "ref_applicable"):
        modes.append("wrong_ref")
    if row.get("false_finish"):
        modes.append("false_finish")
    if int(row.get("parse_fail") or 0) > 0:
        modes.append("parse_fail")
    if int(row.get("tool_fail") or 0) > 0:
        modes.append("tool_fail")
    if not row.get("terminal_semantics_accurate", True):
        modes.append("terminal_semantics")
    return modes


def pass_at_k(successes: int, trials: int, k: int) -> Optional[float]:
    """Probability that k trials drawn without replacement from ``trials`` all succeed."""
    if trials < k:
        return None
    if successes < k:
        return 0.0
    return math.comb(successes, k) / math.comb(trials, k)


def trials_by_task(summary: Mapping[str, Any]) -> Dict[str, List[Mapping[str, Any]]]:
    out: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in summary.get("details") or []:
        out[str(row["task_id"])].append(row)
    return dict(out)


def pass_k_by_task(summary: Mapping[str, Any], k: int = PASS_K) -> Dict[str, Optional[float]]:
    return {
        task_id: pass_at_k(sum(strict_success(r) for r in rows), len(rows), k)
        for task_id, rows in trials_by_task(summary).items()
    }


def pass_k(summary: Mapping[str, Any], k: int = PASS_K) -> Optional[float]:
    """Task-averaged pass^k, skipping tasks with fewer than k trials (report.py convention)."""
    values = [v for v in pass_k_by_task(summary, k).values() if v is not None]
    return sum(values) / len(values) if values else None


# --------------------------------------------------------------------------- #
# diagnose
# --------------------------------------------------------------------------- #

@dataclass
class TaskWeakness:
    task_id: str
    family: str
    trials: int
    strict_successes: int
    pass_k: Optional[float]
    failure_modes: Dict[str, int] = field(default_factory=dict)
    in_train: bool = False

    @property
    def strict_rate(self) -> float:
        return self.strict_successes / self.trials if self.trials else 0.0


@dataclass
class FamilyWeakness:
    family: str
    task_ids: List[str]
    trials: int
    strict_successes: int
    pass_k: Optional[float]
    failure_modes: Dict[str, int]
    train_task_ids: List[str]

    @property
    def strict_rate(self) -> float:
        return self.strict_successes / self.trials if self.trials else 0.0

    @property
    def actionable(self) -> bool:
        return bool(self.train_task_ids)

    @property
    def dominant_failure(self) -> Optional[str]:
        if not self.failure_modes:
            return None
        return sorted(self.failure_modes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["strict_rate"] = self.strict_rate
        out["actionable"] = self.actionable
        out["dominant_failure"] = self.dominant_failure
        return out


@dataclass
class WeaknessReport:
    families: List[FamilyWeakness]
    tasks: Dict[str, TaskWeakness]
    sources: List[Dict[str, Any]]
    k: int = PASS_K

    def weak_families(self, *, max_strict_rate: float) -> List[FamilyWeakness]:
        return [f for f in self.families if f.actionable and f.strict_rate <= max_strict_rate]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "self_evolve_weakness_report",
            "k": self.k,
            "sources": self.sources,
            "families": [f.to_dict() for f in self.families],
            "tasks": {
                tid: {**asdict(t), "strict_rate": t.strict_rate}
                for tid, t in sorted(self.tasks.items())
            },
        }


def family_of(task: Mapping[str, Any]) -> str:
    return template_key(task)[0]


def diagnose(
    summaries: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    train_ids: Iterable[str],
    *,
    k: int = PASS_K,
    sources: Optional[Sequence[Mapping[str, Any]]] = None,
) -> WeaknessReport:
    """Aggregate held-out trials into per-family weaknesses.

    A family is *actionable* only if the train split contains at least one task
    of that family: those are the only parents the loop may re-derive data from,
    so weaknesses in ood-only families are reported but never acted on.
    """
    train = set(map(str, train_ids))
    task_by_id = {str(t["id"]): t for t in tasks}
    family_by_id = {tid: family_of(t) for tid, t in task_by_id.items()}

    per_task: Dict[str, TaskWeakness] = {}
    for summary in summaries:
        for task_id, rows in trials_by_task(summary).items():
            if task_id not in task_by_id:
                continue
            tw = per_task.get(task_id) or TaskWeakness(
                task_id=task_id, family=family_by_id[task_id], trials=0,
                strict_successes=0, pass_k=None, in_train=task_id in train,
            )
            for row in rows:
                tw.trials += 1
                modes = failure_modes(row)
                if not modes:
                    tw.strict_successes += 1
                for m in modes:
                    tw.failure_modes[m] = tw.failure_modes.get(m, 0) + 1
            per_task[task_id] = tw
    for tw in per_task.values():
        tw.pass_k = pass_at_k(tw.strict_successes, tw.trials, k)

    grouped: Dict[str, List[TaskWeakness]] = defaultdict(list)
    for tw in per_task.values():
        grouped[tw.family].append(tw)

    families: List[FamilyWeakness] = []
    for fam, items in grouped.items():
        modes: Dict[str, int] = defaultdict(int)
        for tw in items:
            for m, n in tw.failure_modes.items():
                modes[m] += n
        pk_values = [tw.pass_k for tw in items if tw.pass_k is not None]
        families.append(FamilyWeakness(
            family=fam,
            task_ids=sorted(tw.task_id for tw in items),
            trials=sum(tw.trials for tw in items),
            strict_successes=sum(tw.strict_successes for tw in items),
            pass_k=(sum(pk_values) / len(pk_values)) if pk_values else None,
            failure_modes=dict(sorted(modes.items())),
            train_task_ids=sorted(tid for tid in task_by_id if family_by_id[tid] == fam and tid in train),
        ))
    families.sort(key=lambda f: (f.strict_rate, -f.trials, f.family))
    return WeaknessReport(families=families, tasks=per_task, sources=list(sources or []), k=k)


# --------------------------------------------------------------------------- #
# select
# --------------------------------------------------------------------------- #

@dataclass
class Selection:
    parents: List[str]
    by_family: Dict[str, List[str]]
    skipped_families: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "self_evolve_selection", **asdict(self)}


def derived_parent_map(parametric_manifest: Mapping[str, Any]) -> Dict[str, str]:
    """``derived_task_id -> parent_task_id`` from a parametric seed manifest."""
    tasks = parametric_manifest.get("tasks") or []
    out = {str(t["derived_task_id"]): str(t["parent_task_id"]) for t in tasks}
    if not out:
        raise ValueError("parametric manifest has no 'tasks' lineage; cannot select parents")
    return out


def select_parents(
    report: WeaknessReport,
    derived_parents: Mapping[str, str],
    train_ids: Iterable[str],
    *,
    max_strict_rate: float = 0.5,
    max_families: int = 3,
    parents_per_family: int = 2,
) -> Selection:
    """Pick train-split parents, worst family first, that already have parametric derivations.

    Deterministic: families by weakness order, parents by their own held-out
    strict rate (ascending; unseen parents last) and then by id.
    """
    train = set(map(str, train_ids))
    derivable = {p for p in derived_parents.values() if p in train}
    parent_family = {p: report.tasks[p].family for p in derivable if p in report.tasks}

    by_family: Dict[str, List[str]] = {}
    skipped: Dict[str, str] = {}
    for fam in report.weak_families(max_strict_rate=max_strict_rate):
        if len(by_family) >= max_families:
            skipped[fam.family] = "max_families reached"
            continue
        candidates = [p for p in derivable if p in fam.train_task_ids]
        if not candidates:
            skipped[fam.family] = "no train parent with parametric derivations"
            continue

        def order(pid: str) -> Tuple[float, str]:
            tw = report.tasks.get(pid)
            return (tw.strict_rate if tw else 2.0, pid)

        chosen = sorted(candidates, key=order)[:parents_per_family]
        by_family[fam.family] = chosen
    del parent_family
    parents = sorted({p for ps in by_family.values() for p in ps})
    return Selection(parents=parents, by_family=by_family, skipped_families=skipped)


# --------------------------------------------------------------------------- #
# mine
# --------------------------------------------------------------------------- #

def mine_badcases(
    traces_path: Path,
    tasks: Sequence[Mapping[str, Any]],
    out_path: Path,
    *,
    source: str,
    training_step: int = 100,
) -> int:
    """Capture silent failures from a traces.jsonl into a per-round case file.

    Uses ``benchmark.badcases.capture`` unchanged, so the selection rule and the
    ``reproduces_when`` conditions are exactly those of ``eval/badcase_replay.py``.
    The file is per round on purpose: merging into ``eval/eval_cases.jsonl`` and
    flipping ``open -> fixed`` stay human steps (see eval/readme.md).
    """
    from benchmark.badcases import capture, dump_cases  # local import: pulls in rl.reward

    samples = []
    for line in Path(traces_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            samples.append((row["task_id"], row.get("history") or []))
    cases = capture(samples, tasks, source=source, training_step=training_step)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_cases(cases, out_path)
    return len(cases)


def count_cases(path: Path) -> int:
    if not Path(path).exists():
        return 0
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


# --------------------------------------------------------------------------- #
# mix: base train mix + targeted rows -> one audited training file
# --------------------------------------------------------------------------- #

BASE_SOURCE = "base_train_mix"
TARGETED_SOURCE = "self_evolve_targeted"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_round_mix(
    base_rows: Sequence[Mapping[str, Any]],
    targeted_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Append the round's targeted rows to the base mix, keeping the base contract.

    Every row must carry a ``sample_sha256`` matching its ``messages`` (the SFT
    audit relies on it); a targeted row whose fingerprint already exists in the
    base is dropped rather than duplicated, and the count of dropped rows is
    returned. Rows are tagged with ``mixture_source`` and shuffled with a fixed
    seed so the file is reproducible.
    """
    import random

    if not base_rows or not targeted_rows:
        raise ValueError("both the base mix and the targeted rows must be non-empty")
    mixed: List[Dict[str, Any]] = []
    seen: set = set()
    dropped = 0
    for source, rows in ((BASE_SOURCE, base_rows), (TARGETED_SOURCE, targeted_rows)):
        for index, original in enumerate(rows):
            row = dict(original)
            fingerprint = canonical_hash(row.get("messages"))
            if row.get("sample_sha256") != fingerprint:
                raise ValueError(f"{source} row {index}: sample_sha256 does not match messages")
            if fingerprint in seen:
                if source == BASE_SOURCE:
                    raise ValueError(f"base mix contains duplicate messages: {fingerprint}")
                dropped += 1
                continue
            seen.add(fingerprint)
            row["mixture_source"] = source
            mixed.append(row)
    random.Random(seed).shuffle(mixed)
    return mixed, dropped


def write_round_mix(
    base_path: Path,
    targeted_path: Path,
    output_path: Path,
    *,
    round_id: int,
    seed: int = 42,
) -> Dict[str, Any]:
    """Write ``output_path`` plus the ``qlora_sft_train_mix`` manifest train_sft audits."""
    base_path, targeted_path, output_path = Path(base_path), Path(targeted_path), Path(output_path)
    if output_path.resolve() in {base_path.resolve(), targeted_path.resolve()}:
        raise ValueError("output must not overwrite an input")
    base_rows, targeted_rows = read_jsonl(base_path), read_jsonl(targeted_path)
    mixed, dropped = build_round_mix(base_rows, targeted_rows, seed=seed)
    write_jsonl(output_path, mixed)

    def source_entry(path: Path, rows: int) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"path": str(path).replace("\\", "/"), "sha256": sha256_file(path), "rows": rows}
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        if manifest_path.exists():
            entry["manifest_sha256"] = sha256_file(manifest_path)
        return entry

    counts: Dict[str, int] = {}
    for row in mixed:
        counts[row["mixture_source"]] = counts.get(row["mixture_source"], 0) + 1
    manifest = {
        "version": 1,
        "kind": "qlora_sft_train_mix",
        "self_evolve_round": round_id,
        "shuffle_seed": seed,
        "output": str(output_path).replace("\\", "/"),
        "output_sha256": sha256_file(output_path),
        "output_rows": len(mixed),
        "unique_sample_fingerprints": len({row["sample_sha256"] for row in mixed}),
        "semantic_task_instances": len({row.get("source_task_id") for row in mixed}),
        "source_counts": dict(sorted(counts.items())),
        "targeted_rows_dropped_as_duplicates": dropped,
        "sources": [source_entry(base_path, len(base_rows)), source_entry(targeted_path, len(targeted_rows))],
        "policy": (
            "The base mix is kept whole; the round's targeted rows are appended, deduplicated "
            "against the base by sample_sha256, and the file order is deterministically shuffled."
        ),
    }
    dump_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return manifest


# --------------------------------------------------------------------------- #
# freeze
# --------------------------------------------------------------------------- #

def git_revision(repo_root: Path) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def freeze_round(
    round_dir: Path,
    files: Mapping[str, Path],
    *,
    round_id: int,
    meta: Optional[Mapping[str, Any]] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write ``manifest.json`` pinning every input/output of the round by SHA256.

    ``files`` maps a role name (``eval:dev``, ``split``, ``snapshot``,
    ``targeted_seed`` ...) to a path; every path must exist. The manifest is
    what a later ``gate`` and any reader of the round can audit against.
    """
    entries: Dict[str, Dict[str, Any]] = {}
    for role in sorted(files):
        path = Path(files[role])
        if not path.exists():
            raise FileNotFoundError(f"{role}: {path}")
        entries[role] = {
            "path": str(path).replace("\\", "/"),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "version": 1,
        "kind": ROUND_MANIFEST_KIND,
        "round": round_id,
        "git_revision": git_revision(repo_root or Path(PACKAGE_ROOT).parent),
        "files": entries,
        "meta": dict(meta or {}),
    }
    manifest["lineage_sha256"] = canonical_hash({"round": round_id, "files": entries})
    dump_json(Path(round_dir) / MANIFEST_FILE_NAME, manifest)
    return manifest


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #

@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GateResult:
    passed: bool
    checks: List[GateCheck]
    dev_tolerance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "self_evolve_gate_result",
            "passed": self.passed,
            "dev_tolerance": self.dev_tolerance,
            "checks": [asdict(c) for c in self.checks],
        }

    def summary(self) -> str:
        lines = [f"GATE {'PASSED' if self.passed else 'REJECTED'}"]
        for c in self.checks:
            mark = "ok " if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name} {json.dumps(c.detail, ensure_ascii=False)}")
        return "\n".join(lines)


def gate(
    prev_summaries: Mapping[str, Mapping[str, Any]],
    new_summaries: Mapping[str, Mapping[str, Any]],
    *,
    cases_before: int,
    cases_after: int,
    split_path: Optional[Path] = None,
    frozen_split_sha256: Optional[str] = None,
    dev_tolerance: float = 0.0,
    k: int = PASS_K,
    splits: Sequence[str] = GATE_SPLITS,
) -> GateResult:
    """Accept a round only if nothing the loop promised to protect got worse.

    1. strict pass^k on every held-out split must not regress. ``iid_test`` and
       ``ood_test`` are always strict; only ``dev`` (n=8, noisy) may be given an
       opt-in ``dev_tolerance`` (e.g. 0.125 = one task), which is recorded in the
       result so a relaxed gate is never silent;
    2. the badcase library may only grow;
    3. the held-out split file must be byte-identical to the one frozen in the
       round manifest (no data leakage by moving tasks between splits).
    """
    if dev_tolerance < 0:
        raise ValueError("dev_tolerance must be >= 0")
    checks: List[GateCheck] = []
    for split in splits:
        tolerance = dev_tolerance if split == "dev" else 0.0
        prev = prev_summaries.get(split)
        new = new_summaries.get(split)
        if prev is None or new is None:
            checks.append(GateCheck(f"pass^{k}:{split}", False,
                                    {"error": "missing summary for split"}))
            continue
        p_prev, p_new = pass_k(prev, k), pass_k(new, k)
        if p_prev is None or p_new is None:
            checks.append(GateCheck(f"pass^{k}:{split}", False,
                                    {"error": f"fewer than {k} trials per task"}))
            continue
        checks.append(GateCheck(
            f"pass^{k}:{split}",
            p_new + 1e-12 >= p_prev - tolerance,
            {"previous": round(p_prev, 4), "new": round(p_new, 4), "tolerance": tolerance},
        ))
    checks.append(GateCheck(
        "badcases_non_decreasing", cases_after >= cases_before,
        {"before": cases_before, "after": cases_after},
    ))
    if frozen_split_sha256 is not None:
        actual = sha256_file(split_path) if split_path and Path(split_path).exists() else None
        checks.append(GateCheck(
            "heldout_split_frozen", actual == frozen_split_sha256,
            {"frozen": frozen_split_sha256, "actual": actual},
        ))
    return GateResult(passed=all(c.passed for c in checks), checks=checks, dev_tolerance=dev_tolerance)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _load_tasks(task_set: str) -> List[Dict[str, Any]]:
    if task_set == "expanded":
        from benchmark.tasks_expanded import get_expanded_tasks
        return get_expanded_tasks()
    from benchmark.tasks import get_all_tasks
    return get_all_tasks()


def _load_summaries(assignments: Mapping[str, Path]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summaries, sources = {}, []
    for split, path in assignments.items():
        summaries[split] = load_json(path)
        sources.append({"split": split, "path": str(path).replace("\\", "/"), "sha256": sha256_file(path)})
    return summaries, sources


def _train_ids(split_payload: Mapping[str, Any]) -> List[str]:
    return [str(t) for t in (split_payload.get("split") or {}).get("train", [])]


def cmd_diagnose(args: argparse.Namespace) -> int:
    assignments = parse_split_assignments(args.eval)
    summaries, sources = _load_summaries(assignments)
    split_payload = load_json(Path(args.split))
    report = diagnose(list(summaries.values()), _load_tasks(args.task_set),
                      _train_ids(split_payload), k=args.k, sources=sources)
    if args.output:
        dump_json(Path(args.output), report.to_dict())
    for fam in report.families:
        flag = "actionable" if fam.actionable else "ood-only "
        print(f"{fam.family:<20} strict={fam.strict_rate:5.2f} "
              f"pass^{args.k}={'n/a' if fam.pass_k is None else f'{fam.pass_k:.2f}'} "
              f"trials={fam.trials:<3} {flag} dominant={fam.dominant_failure}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    prev, _ = _load_summaries(parse_split_assignments(args.prev))
    new, _ = _load_summaries(parse_split_assignments(args.new))
    frozen = None
    if args.round_dir:
        manifest = load_json(Path(args.round_dir) / MANIFEST_FILE_NAME)
        frozen = (manifest.get("files") or {}).get("split", {}).get("sha256")
    result = gate(prev, new, cases_before=args.cases_before, cases_after=args.cases_after,
                  split_path=Path(args.split) if args.split else None,
                  frozen_split_sha256=frozen, dev_tolerance=args.dev_tolerance, k=args.k)
    if args.round_dir:
        dump_json(Path(args.round_dir) / GATE_FILE_NAME, result.to_dict())
    print(result.summary())
    return 0 if result.passed else 1


def cmd_run(args: argparse.Namespace) -> int:
    repo_root = Path(PACKAGE_ROOT).parent
    round_dir = Path(args.artifacts_dir) / f"round_{args.round}"
    round_dir.mkdir(parents=True, exist_ok=True)

    assignments = parse_split_assignments(args.eval)
    summaries, sources = _load_summaries(assignments)
    split_path, snapshot_path = Path(args.split), Path(args.snapshot)
    split_payload = load_json(split_path)
    train_ids = _train_ids(split_payload)
    tasks = _load_tasks(args.task_set)

    # 1. diagnose
    report = diagnose(list(summaries.values()), tasks, train_ids, k=args.k, sources=sources)
    dump_json(round_dir / WEAKNESS_FILE_NAME, report.to_dict())

    # 2. mine (optional: needs the traces of the same eval)
    files: Dict[str, Path] = {f"eval:{s}": p for s, p in assignments.items()}
    files["split"] = split_path
    files["snapshot"] = snapshot_path
    if args.traces:
        n = mine_badcases(Path(args.traces), tasks, round_dir / BADCASE_FILE_NAME,
                          source=f"self_evolve:round_{args.round}")
        files["badcases"] = round_dir / BADCASE_FILE_NAME
        print(f"mined {n} badcases -> {round_dir / BADCASE_FILE_NAME}")

    # 3. select
    selection = select_parents(
        report, derived_parent_map(load_json(Path(args.parametric_manifest))), train_ids,
        max_strict_rate=args.max_strict_rate, max_families=args.max_families,
        parents_per_family=args.parents_per_family,
    )
    dump_json(round_dir / SELECTION_FILE_NAME, selection.to_dict())
    print(f"selected parents: {selection.parents or '(none)'}")

    # 4. synthesize (subprocess: the derivation script owns env execution + manifest)
    if selection.parents and not args.no_synthesize:
        seed_out = round_dir / "targeted_parametric_seed.jsonl"
        cmd = [sys.executable, str(repo_root / "scripts" / "generate_parametric_sft_data.py"),
               "--split-file", str(split_path), "--snapshot", str(snapshot_path),
               "--output", str(seed_out), "--only-parents", ",".join(selection.parents)]
        if args.include_t5:
            cmd.append("--include-t5")
        print("synthesize:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(repo_root), check=True)
        files["targeted_seed"] = seed_out
        files["targeted_seed_manifest"] = seed_out.with_suffix(seed_out.suffix + ".manifest.json")

    # 5. freeze
    manifest = freeze_round(
        round_dir, files, round_id=args.round, repo_root=repo_root,
        meta={"k": args.k, "max_strict_rate": args.max_strict_rate,
              "selection": selection.to_dict(), "task_set": args.task_set},
    )
    print(f"froze round {args.round}: lineage {manifest['lineage_sha256'][:12]}… -> {round_dir / MANIFEST_FILE_NAME}")

    # 6. retrain (opaque)
    if args.train_cmd:
        print("retrain:", args.train_cmd)
        subprocess.run(args.train_cmd, shell=True, cwd=str(repo_root), check=True)
    print("next: evaluate the new policy on the held-out splits, then run "
          f"`self_evolve gate --round-dir {round_dir} --prev ... --new ... --cases-before N --cases-after M`")
    return 0


def cmd_mix(args: argparse.Namespace) -> int:
    manifest = write_round_mix(Path(args.base), Path(args.targeted), Path(args.output),
                               round_id=args.round, seed=args.seed)
    print(f"mixed {manifest['output_rows']} rows ({manifest['source_counts']}, "
          f"{manifest['targeted_rows_dropped_as_duplicates']} targeted duplicates dropped) -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="self_evolve",
        description="Bounded self-evolution loop: diagnose -> mine -> select -> synthesize -> mix -> freeze -> gate",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("mix", help="append the round's targeted rows to the base train mix, with an audited manifest")
    p.add_argument("--base", required=True, help="base train mix jsonl (kind qlora_sft_train_mix)")
    p.add_argument("--targeted", required=True,
                   help="the round's targeted rows, e.g. the linguistic augmentation of targeted_parametric_seed.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_mix)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--task-set", choices=["default", "expanded"], default="expanded")
        p.add_argument("--k", type=int, default=PASS_K, help="k of strict pass^k (default 3)")

    p = sub.add_parser("diagnose", help="per-family weakness report from held-out summary.json files")
    p.add_argument("--eval", action="append", required=True, metavar="SPLIT=summary.json")
    p.add_argument("--split", required=True, help="frozen split file, e.g. data/splits/v3_81.json")
    p.add_argument("--output", default=None)
    add_common(p)
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("gate", help="accept/reject a round against the previous one")
    p.add_argument("--prev", action="append", required=True, metavar="SPLIT=summary.json")
    p.add_argument("--new", action="append", required=True, metavar="SPLIT=summary.json")
    p.add_argument("--cases-before", type=int, required=True)
    p.add_argument("--cases-after", type=int, required=True)
    p.add_argument("--round-dir", default=None, help="round dir with manifest.json (checks frozen split)")
    p.add_argument("--split", default=None, help="current split file to compare with the frozen one")
    p.add_argument("--dev-tolerance", type=float, default=0.0,
                   help="opt-in pass^k slack for the noisy dev split only (e.g. 0.125 = one task); "
                        "iid_test / ood_test are always strict; recorded in gate_result.json")
    add_common(p)
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("run", help="steps 1-5 (+ optional retrain) for one round")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--eval", action="append", required=True, metavar="SPLIT=summary.json")
    p.add_argument("--traces", default=None, help="traces.jsonl of the same eval (enables badcase mining)")
    p.add_argument("--split", required=True)
    p.add_argument("--snapshot", default="data/mock_arxiv_snapshot.json")
    p.add_argument("--parametric-manifest", default="data/sft/sft_v2_parametric_seed.jsonl.manifest.json",
                   help="parametric seed manifest providing derived->parent lineage")
    p.add_argument("--artifacts-dir", default="artifacts/self_evolve")
    p.add_argument("--max-strict-rate", type=float, default=0.5)
    p.add_argument("--max-families", type=int, default=3)
    p.add_argument("--parents-per-family", type=int, default=2)
    p.add_argument("--include-t5", action="store_true")
    p.add_argument("--no-synthesize", action="store_true", help="stop after selection + freeze")
    p.add_argument("--train-cmd", default=None, help="opaque retrain command run from the repo root")
    add_common(p)
    p.set_defaults(func=cmd_run)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
