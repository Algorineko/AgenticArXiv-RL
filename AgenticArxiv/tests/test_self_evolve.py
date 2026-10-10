"""Bounded self-evolution loop: pure-JSON components, no model, no torch."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rl import self_evolve as se  # noqa: E402
from benchmark.tasks_expanded import get_expanded_tasks  # noqa: E402
from generate_parametric_sft_data import (  # noqa: E402
    build_parametric_tasks,
    filter_derived_by_parents,
)

SPLIT_PATH = REPO_ROOT / "data" / "splits" / "v3_81.json"
PARAMETRIC_MANIFEST = REPO_ROOT / "data" / "sft" / "sft_v2_parametric_seed.jsonl.manifest.json"


def trial(task_id, *, ok=True, **overrides):
    """A ``details`` row as benchmark/report.py writes it; ``ok`` builds a strict success."""
    row = {
        "task_id": task_id, "agent_type": "regex", "trial": 0,
        "completed": True, "termination": "FINISH", "tool_accurate": True,
        "arg_score": 1.0, "arg_applicable": True, "false_finish": False,
        "ref_score": 1.0, "ref_applicable": True, "parse_fail": 0, "tool_fail": 0,
        "terminal_semantics_accurate": True,
    }
    if not ok:
        row.update({"tool_accurate": False})
    row.update(overrides)
    return row


def summary(*rows):
    return {"model": "fake", "sample_count": len(rows), "details": list(rows)}


class TrialVerdictTest(unittest.TestCase):
    def test_strict_success_mirrors_metrics_predicate(self):
        self.assertTrue(se.strict_success(trial("t")))
        for bad in (
            {"completed": False}, {"tool_accurate": False}, {"arg_score": 0.5},
            {"ref_score": 0.0}, {"parse_fail": 1}, {"tool_fail": 2},
            {"terminal_semantics_accurate": False},
        ):
            with self.subTest(bad=bad):
                self.assertFalse(se.strict_success(trial("t", **bad)))

    def test_non_applicable_scores_do_not_count_as_failures(self):
        row = trial("t", arg_score=None, arg_applicable=False, ref_score=None, ref_applicable=False)
        self.assertTrue(se.strict_success(row))
        self.assertEqual(se.failure_modes(row), [])

    def test_failure_modes_name_every_violated_clause(self):
        row = trial("t", completed=False, false_finish=True, arg_score=0.0)
        self.assertEqual(se.failure_modes(row), ["not_completed", "wrong_args", "false_finish"])

    def test_pass_at_k(self):
        self.assertIsNone(se.pass_at_k(2, 2, 3))          # fewer trials than k -> skipped
        self.assertEqual(se.pass_at_k(3, 3, 3), 1.0)
        self.assertEqual(se.pass_at_k(2, 3, 3), 0.0)
        self.assertAlmostEqual(se.pass_at_k(3, 4, 3), 0.25)  # C(3,3)/C(4,3)
        self.assertAlmostEqual(se.pass_at_k(2, 3, 2), 1 / 3)  # C(2,2)/C(3,2)

    def test_pass_k_is_task_averaged_and_skips_short_tasks(self):
        s = summary(
            trial("a"), trial("a"), trial("a"),
            trial("b", ok=False), trial("b"), trial("b"),
            trial("c"), trial("c"),                       # only two trials: skipped
        )
        self.assertAlmostEqual(se.pass_k(s, 3), (1.0 + 0.0) / 2)
        self.assertIsNone(se.pass_k(summary(trial("x")), 3))


class DiagnoseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = get_expanded_tasks()
        cls.split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        cls.train = cls.split["split"]["train"]
        by_family = {}
        for t in cls.tasks:
            by_family.setdefault(se.family_of(t), []).append(t["id"])
        cls.by_family = by_family

    def test_families_sorted_worst_first_with_dominant_failure(self):
        search = self.by_family["search"]
        infeasible = self.by_family["infeasible"]
        s = summary(
            *[trial(search[0]) for _ in range(3)],
            *[trial(infeasible[0], ok=False, false_finish=True) for _ in range(3)],
        )
        report = se.diagnose([s], self.tasks, self.train)
        self.assertEqual([f.family for f in report.families], ["infeasible", "search"])
        worst = report.families[0]
        self.assertEqual(worst.strict_rate, 0.0)
        self.assertEqual(worst.pass_k, 0.0)
        # equal counts: ties resolve by name so the report is deterministic
        self.assertEqual(worst.failure_modes, {"false_finish": 3, "wrong_tools": 3})
        self.assertEqual(worst.dominant_failure, "false_finish")
        self.assertEqual(report.families[1].pass_k, 1.0)

    def test_ood_only_families_are_reported_but_not_actionable(self):
        ood_ids = self.split["split"]["ood_test"]
        ood_family = se.family_of(next(t for t in self.tasks if t["id"] == ood_ids[0]))
        report = se.diagnose([summary(trial(ood_ids[0], ok=False))], self.tasks, self.train)
        fam = next(f for f in report.families if f.family == ood_family)
        # composite(3)/composite(4) share the "composite" label with train tasks of shorter
        # chains, so actionability is decided by whether *any* train task carries the label.
        self.assertEqual(fam.actionable, bool(fam.train_task_ids))
        self.assertEqual(fam.train_task_ids, sorted(
            t["id"] for t in self.tasks if se.family_of(t) == ood_family and t["id"] in self.train))

    def test_unknown_task_ids_are_ignored_and_sources_recorded(self):
        report = se.diagnose([summary(trial("not-a-task"))], self.tasks, self.train,
                             sources=[{"split": "dev", "sha256": "x"}])
        self.assertEqual(report.families, [])
        self.assertEqual(report.sources, [{"split": "dev", "sha256": "x"}])
        self.assertEqual(report.to_dict()["kind"], "self_evolve_weakness_report")


class SelectParentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = get_expanded_tasks()
        cls.split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        cls.train = cls.split["split"]["train"]
        cls.derived = se.derived_parent_map(json.loads(PARAMETRIC_MANIFEST.read_text(encoding="utf-8")))

    def test_lineage_comes_from_manifest_and_parents_are_train_only(self):
        parents = set(self.derived.values())
        self.assertTrue(parents)
        self.assertTrue(parents <= set(self.train), parents - set(self.train))
        with self.assertRaises(ValueError):
            se.derived_parent_map({"tasks": []})

    def test_selection_is_deterministic_worst_first_and_bounded(self):
        parents_by_family = {}
        for p in set(self.derived.values()):
            fam = se.family_of(next(t for t in self.tasks if t["id"] == p))
            parents_by_family.setdefault(fam, []).append(p)
        fam_a, fam_b = sorted(f for f, ps in parents_by_family.items() if len(ps) >= 3)[:2]
        pa, pb = sorted(parents_by_family[fam_a]), sorted(parents_by_family[fam_b])
        s = summary(
            *[trial(pa[0], ok=False) for _ in range(3)],      # worst parent in fam_a
            *[trial(pa[1], ok=False) for _ in range(2)], trial(pa[1]),
            *[trial(pa[2]) for _ in range(3)],                # healthy parent, must not be chosen first
            *[trial(pb[0], ok=False) for _ in range(2)], trial(pb[0]), trial(pb[0]),
        )
        report = se.diagnose([s], self.tasks, self.train)
        sel = se.select_parents(report, self.derived, self.train,
                                max_strict_rate=0.6, max_families=1, parents_per_family=2)
        self.assertEqual(sel.by_family, {fam_a: [pa[0], pa[1]]})
        self.assertEqual(sel.parents, sorted([pa[0], pa[1]]))
        self.assertEqual(sel.skipped_families, {fam_b: "max_families reached"})
        again = se.select_parents(report, self.derived, self.train,
                                  max_strict_rate=0.6, max_families=1, parents_per_family=2)
        self.assertEqual(sel, again)

    def test_healthy_families_and_underivable_parents_are_skipped(self):
        parent = next(iter(set(self.derived.values())))
        report = se.diagnose([summary(*[trial(parent) for _ in range(3)])], self.tasks, self.train)
        self.assertEqual(se.select_parents(report, self.derived, self.train).parents, [])
        # a weak family whose parents have no derivations is reported as skipped, never selected
        weak = summary(*[trial(parent, ok=False) for _ in range(3)])
        report = se.diagnose([weak], self.tasks, self.train)
        sel = se.select_parents(report, {"d": "some-other-train-task"}, self.train)
        self.assertEqual(sel.parents, [])
        self.assertIn("no train parent", next(iter(sel.skipped_families.values())))


class OnlyParentsFilterTest(unittest.TestCase):
    def test_filter_keeps_only_requested_parents(self):
        derived = build_parametric_tasks()
        parents = sorted({d.parent_task_id for d in derived})[:2]
        kept = filter_derived_by_parents(derived, parents)
        self.assertTrue(kept)
        self.assertEqual({d.parent_task_id for d in kept}, set(parents))
        self.assertEqual(len(kept), sum(d.parent_task_id in parents for d in derived))

    def test_unknown_or_empty_parents_fail_loudly(self):
        derived = build_parametric_tasks()
        with self.assertRaises(ValueError):
            filter_derived_by_parents(derived, ["no_such_parent"])
        with self.assertRaises(ValueError):
            filter_derived_by_parents(derived, [" "])


class MineBadcasesTest(unittest.TestCase):
    def test_capture_writes_per_round_file_using_library_rules(self):
        tasks = get_expanded_tasks()
        infeasible = next(t for t in tasks if se.family_of(t) == "infeasible")
        search = next(t for t in tasks if se.family_of(t) == "search")
        with tempfile.TemporaryDirectory() as tmp:
            traces = Path(tmp) / "traces.jsonl"
            rows = [
                # silent failure: calls a tool on an infeasible task, then FINISH
                {"task_id": infeasible["id"], "history": [
                    {"action": json.dumps({"name": "get_recently_submitted_cs_papers",
                                           "args": {"aspect": "AI", "days": 7, "max_results": 5}}),
                     "observation": "x"},
                    {"action": "FINISH", "observation": ""},
                ]},
                # loud failure (no FINISH): not a badcase by library rules
                {"task_id": search["id"], "history": [{"action": "garbage", "observation": "parse error"}]},
            ]
            traces.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            out = Path(tmp) / "round" / se.BADCASE_FILE_NAME
            n = se.mine_badcases(traces, tasks, out, source="self_evolve:test")
            self.assertEqual(n, 1)
            case = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(case["task_id"], infeasible["id"])
            self.assertEqual(case["status"], "open")
            self.assertEqual(case["source"], "self_evolve:test")
            self.assertEqual(se.count_cases(out), 1)
            self.assertEqual(se.count_cases(Path(tmp) / "missing.jsonl"), 0)


class FreezeAndGateTest(unittest.TestCase):
    def test_freeze_round_pins_every_file_and_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.json"
            a.write_text("{}", encoding="utf-8")
            m1 = se.freeze_round(Path(tmp) / "r1", {"split": a}, round_id=1, meta={"k": 3})
            m2 = se.freeze_round(Path(tmp) / "r1b", {"split": a}, round_id=1)
            self.assertEqual(m1["kind"], se.ROUND_MANIFEST_KIND)
            self.assertEqual(m1["files"]["split"]["sha256"], se.sha256_file(a))
            self.assertEqual(m1["lineage_sha256"], m2["lineage_sha256"])
            self.assertEqual(json.loads((Path(tmp) / "r1" / se.MANIFEST_FILE_NAME).read_text("utf-8"))["meta"], {"k": 3})
            with self.assertRaises(FileNotFoundError):
                se.freeze_round(Path(tmp) / "r2", {"split": Path(tmp) / "missing"}, round_id=2)

    def _eval(self, dev_ok, iid_ok, ood_ok):
        def split(ok):
            return summary(*[trial("t1", ok=ok) for _ in range(3)], *[trial("t2") for _ in range(3)])
        return {"dev": split(dev_ok), "iid_test": split(iid_ok), "ood_test": split(ood_ok)}

    def test_gate_rejects_any_regression_and_accepts_equal_or_better(self):
        prev = self._eval(True, True, True)
        worse = self._eval(True, False, True)
        res = se.gate(prev, worse, cases_before=14, cases_after=14)
        self.assertFalse(res.passed)
        failed = [c.name for c in res.checks if not c.passed]
        self.assertEqual(failed, ["pass^3:iid_test"])
        self.assertTrue(se.gate(prev, prev, cases_before=14, cases_after=15).passed)
        better = se.gate(self._eval(False, True, True), prev, cases_before=0, cases_after=0)
        self.assertTrue(better.passed)

    def test_dev_tolerance_is_opt_in_dev_only_and_recorded(self):
        prev = self._eval(True, True, True)
        dev_worse = self._eval(False, True, True)  # dev pass^3 drops 1.0 -> 0.5
        self.assertFalse(se.gate(prev, dev_worse, cases_before=1, cases_after=1).passed)
        relaxed = se.gate(prev, dev_worse, cases_before=1, cases_after=1, dev_tolerance=0.5)
        self.assertTrue(relaxed.passed)
        self.assertEqual(relaxed.to_dict()["dev_tolerance"], 0.5)
        self.assertEqual(next(c.detail["tolerance"] for c in relaxed.checks if c.name == "pass^3:dev"), 0.5)
        # the same slack never applies to the held-out test splits
        iid_worse = self._eval(True, False, True)
        strict = se.gate(prev, iid_worse, cases_before=1, cases_after=1, dev_tolerance=0.5)
        self.assertEqual([c.name for c in strict.checks if not c.passed], ["pass^3:iid_test"])
        self.assertEqual(next(c.detail["tolerance"] for c in strict.checks if c.name == "pass^3:iid_test"), 0.0)
        with self.assertRaises(ValueError):
            se.gate(prev, prev, cases_before=0, cases_after=0, dev_tolerance=-0.1)

    def test_badcase_library_must_not_shrink(self):
        prev = self._eval(True, True, True)
        shrink = se.gate(prev, prev, cases_before=14, cases_after=13)
        self.assertEqual([c.name for c in shrink.checks if not c.passed], ["badcases_non_decreasing"])

    def test_gate_checks_frozen_split_and_missing_inputs(self):
        prev = self._eval(True, True, True)
        with tempfile.TemporaryDirectory() as tmp:
            split = Path(tmp) / "split.json"
            split.write_text('{"split": {}}', encoding="utf-8")
            frozen = se.sha256_file(split)
            ok = se.gate(prev, prev, cases_before=0, cases_after=0,
                         split_path=split, frozen_split_sha256=frozen)
            self.assertTrue(ok.passed)
            split.write_text('{"split": {"train": ["leak"]}}', encoding="utf-8")
            moved = se.gate(prev, prev, cases_before=0, cases_after=0,
                            split_path=split, frozen_split_sha256=frozen)
            self.assertEqual([c.name for c in moved.checks if not c.passed], ["heldout_split_frozen"])
        missing = se.gate(prev, {"dev": prev["dev"]}, cases_before=0, cases_after=0)
        self.assertFalse(missing.passed)
        short = se.gate(prev, {**prev, "dev": summary(trial("t1"))}, cases_before=0, cases_after=0)
        self.assertIn("fewer than 3", next(c.detail["error"] for c in short.checks if not c.passed))

    def test_summary_text_and_dict(self):
        res = se.gate(self._eval(True, True, True), self._eval(True, True, True),
                      cases_before=0, cases_after=0)
        self.assertTrue(res.summary().startswith("GATE PASSED"))
        self.assertEqual(res.to_dict()["kind"], "self_evolve_gate_result")


class CliTest(unittest.TestCase):
    def test_split_assignment_parsing(self):
        self.assertEqual(se.parse_split_assignments(["dev=a.json", "iid_test= b.json"]),
                         {"dev": Path("a.json"), "iid_test": Path("b.json")})
        for bad in (["dev"], ["=x"], ["dev=a", "dev=b"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                se.parse_split_assignments(bad)

    def test_run_without_synthesis_writes_report_selection_and_manifest(self):
        tasks = get_expanded_tasks()
        split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        derived = se.derived_parent_map(json.loads(PARAMETRIC_MANIFEST.read_text(encoding="utf-8")))
        parent = sorted(set(derived.values()))[0]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dev = tmp / "dev.json"
            dev.write_text(json.dumps(summary(*[trial(parent, ok=False) for _ in range(3)])), encoding="utf-8")
            snapshot = tmp / "snapshot.json"
            snapshot.write_text("{}", encoding="utf-8")
            rc = se.main([
                "run", "--round", "7", "--eval", f"dev={dev}", "--split", str(SPLIT_PATH),
                "--snapshot", str(snapshot), "--parametric-manifest", str(PARAMETRIC_MANIFEST),
                "--artifacts-dir", str(tmp / "art"), "--no-synthesize", "--parents-per-family", "1",
            ])
            self.assertEqual(rc, 0)
            round_dir = tmp / "art" / "round_7"
            report = json.loads((round_dir / se.WEAKNESS_FILE_NAME).read_text("utf-8"))
            selection = json.loads((round_dir / se.SELECTION_FILE_NAME).read_text("utf-8"))
            manifest = json.loads((round_dir / se.MANIFEST_FILE_NAME).read_text("utf-8"))
            self.assertEqual(report["families"][0]["strict_rate"], 0.0)
            self.assertEqual(selection["parents"], [parent])
            self.assertEqual(set(manifest["files"]), {"eval:dev", "split", "snapshot"})
            self.assertEqual(manifest["files"]["split"]["sha256"], se.sha256_file(SPLIT_PATH))
            self.assertEqual(manifest["meta"]["selection"]["parents"], [parent])
            self.assertNotIn("targeted_seed", manifest["files"])
        del tasks, split

    def test_module_help_runs_from_repo_root(self):
        proc = subprocess.run([sys.executable, "-m", "AgenticArxiv.rl.self_evolve", "--help"],
                              cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
                              env={**os.environ})
        self.assertEqual(proc.returncode, 0, proc.stderr[-1500:])
        self.assertIn("diagnose", proc.stdout)


class RoundMixTest(unittest.TestCase):
    @staticmethod
    def _row(label, task="t1"):
        messages = [{"role": "user", "content": label}, {"role": "assistant", "content": "Thought: x\nAction: FINISH"}]
        return {"source_task_id": task, "messages": messages, "sample_sha256": se.canonical_hash(messages)}

    def test_mix_tags_sources_drops_targeted_duplicates_and_is_deterministic(self):
        base = [self._row("b1"), self._row("b2", "t2")]
        targeted = [self._row("x1", "t3"), self._row("b1")]  # second one duplicates a base row
        mixed, dropped = se.build_round_mix(base, targeted, seed=7)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(mixed), 3)
        self.assertEqual({r["mixture_source"] for r in mixed}, {se.BASE_SOURCE, se.TARGETED_SOURCE})
        self.assertEqual(mixed, se.build_round_mix(base, targeted, seed=7)[0])
        self.assertNotEqual([r["sample_sha256"] for r in mixed], [r["sample_sha256"] for r in base + targeted[:1]])

    def test_mix_rejects_bad_fingerprints_empty_inputs_and_base_duplicates(self):
        good, bad = self._row("ok"), self._row("tampered")
        bad["sample_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "sample_sha256"):
            se.build_round_mix([good], [bad], seed=1)
        with self.assertRaises(ValueError):
            se.build_round_mix([], [good], seed=1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            se.build_round_mix([good, dict(good)], [self._row("t")], seed=1)

    def test_write_round_mix_manifest_satisfies_the_sft_audit_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base, targeted, out = tmp / "base.jsonl", tmp / "targeted.jsonl", tmp / "round" / "mix.jsonl"
            se.write_jsonl(base, [self._row("b1"), self._row("b2", "t2")])
            se.write_jsonl(targeted, [self._row("x1", "t3")])
            manifest = se.write_round_mix(base, targeted, out, round_id=3, seed=5)
            rows = se.read_jsonl(out)
            # the same checks rl/train_sft.py::_verify_data_manifest applies before training
            self.assertEqual(manifest["kind"], "qlora_sft_train_mix")
            self.assertEqual(manifest["output_sha256"], se.sha256_file(out))
            self.assertEqual(manifest["output_rows"], len(rows))
            self.assertEqual(manifest["unique_sample_fingerprints"], len(rows))
            self.assertEqual(manifest["semantic_task_instances"], 3)
            self.assertEqual(manifest["self_evolve_round"], 3)
            self.assertEqual(manifest["source_counts"], {se.BASE_SOURCE: 2, se.TARGETED_SOURCE: 1})
            self.assertEqual([s["rows"] for s in manifest["sources"]], [2, 1])
            self.assertTrue((out.with_suffix(out.suffix + ".manifest.json")).exists())
            with self.assertRaisesRegex(ValueError, "overwrite"):
                se.write_round_mix(base, targeted, base, round_id=3)


if __name__ == "__main__":
    unittest.main()
