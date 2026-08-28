"""坏例回放。

回放只重跑打分器，不跑 Agent —— 用例里存的是死轨迹。所以这些测试
不需要 LLM、网络或工具，也因此能进 CI。
"""

import json
import tempfile
import unittest
from pathlib import Path

from benchmark.badcases import (
    STATUS_FIXED,
    STATUS_OPEN,
    BadCase,
    capture,
    dump_cases,
    is_badcase,
    load_cases,
    matches,
    replay,
    verdict_of,
)
from benchmark.tasks_expanded import EXPANDED_TASKS

CASES_PATH = Path(__file__).resolve().parents[2] / "eval" / "eval_cases.jsonl"

SEARCH = '{"name": "get_recently_submitted_cs_papers", "args": {"aspect": "CL", "days": 7, "max_results": 5}}'
BY_ID = {t["id"]: t for t in EXPANDED_TASKS}
TASK = BY_ID["search_CL_7d_5"]


def step(action):
    return {"thought": "", "action": action, "observation": "obs"}


def case(**kw):
    base = dict(case_id="c", task_id=TASK["id"], history=[step("FINISH")],
                reproduces_when={"false_finish": True})
    base.update(kw)
    return BadCase(**base)


class BadCaseValidationTest(unittest.TestCase):
    def test_rejects_an_unknown_status(self):
        with self.assertRaises(ValueError):
            case(status="maybe")

    def test_rejects_an_empty_condition(self):
        """没有判定条件的用例，回放时无从判断是否复现。"""
        with self.assertRaises(ValueError):
            case(reproduces_when={})

    def test_rejects_a_condition_on_a_field_that_does_not_exist(self):
        """写错字段名会让条件永远不成立，静默把用例变成摆设。"""
        with self.assertRaises(ValueError) as ctx:
            case(reproduces_when={"flase_finish": True})
        self.assertIn("false_finish", str(ctx.exception))


class MatchesTest(unittest.TestCase):
    def test_scalars_compare_by_equality(self):
        self.assertTrue(matches({"false_finish": True}, {"false_finish": True}))
        self.assertFalse(matches({"false_finish": False}, {"false_finish": True}))

    def test_threshold_form_for_reward_hacking(self):
        """「这种行为不许拿到 0.6 分」是阈值断言，写死等于某个数会在调权后失效。"""
        self.assertTrue(matches({"reward": 0.7}, {"reward": {"ge": 0.6}}))
        self.assertFalse(matches({"reward": 0.5}, {"reward": {"ge": 0.6}}))

    def test_all_keys_must_hold(self):
        verdict = {"task_completed": True, "tool_call_accurate": False}
        self.assertTrue(matches(verdict, {"task_completed": True, "tool_call_accurate": False}))
        self.assertFalse(matches(verdict, {"task_completed": True, "tool_call_accurate": True}))

    def test_a_missing_field_never_matches_a_threshold(self):
        self.assertFalse(matches({}, {"reward": {"ge": 0.0}}))

    def test_unknown_operator_is_an_error(self):
        with self.assertRaises(ValueError):
            matches({"reward": 1.0}, {"reward": {"approximately": 1.0}})


class ReplayOutcomeTest(unittest.TestCase):
    """四个象限：状态 x 是否复现。"""

    def _replay_one(self, c):
        return replay([c], EXPANDED_TASKS)[0]

    def test_open_case_that_still_reproduces(self):
        out = self._replay_one(case(status=STATUS_OPEN, reproduces_when={"false_finish": True}))
        self.assertTrue(out.reproduces)
        self.assertEqual(out.outcome, "still_open")
        self.assertFalse(out.is_regression)

    def test_open_case_that_no_longer_reproduces_is_newly_fixed(self):
        out = self._replay_one(case(status=STATUS_OPEN, reproduces_when={"false_finish": False}))
        self.assertEqual(out.outcome, "newly_fixed")

    def test_fixed_case_that_reproduces_again_is_a_regression(self):
        out = self._replay_one(case(status=STATUS_FIXED, reproduces_when={"false_finish": True}))
        self.assertEqual(out.outcome, "regressed")
        self.assertTrue(out.is_regression)

    def test_fixed_case_that_stays_fixed(self):
        out = self._replay_one(case(status=STATUS_FIXED, reproduces_when={"false_finish": False}))
        self.assertEqual(out.outcome, "stays_fixed")

    def test_reward_drift_is_reported_against_the_captured_value(self):
        c = case(status=STATUS_FIXED, reproduces_when={"false_finish": False},
                 captured={"reward": -1.0})
        out = self._replay_one(c)
        self.assertAlmostEqual(out.reward_drift, out.verdict["reward"] + 1.0, places=6)

    def test_no_drift_reported_when_nothing_was_captured(self):
        self.assertIsNone(self._replay_one(case()).reward_drift)

    def test_a_case_pointing_at_a_deleted_task_fails_loudly(self):
        """静默跳过等于悄悄少测；任务改名时绑在它上面的用例必须一起处理。"""
        with self.assertRaises(KeyError):
            replay([case(task_id="task_that_was_deleted")], EXPANDED_TASKS)


class SelectionTest(unittest.TestCase):
    def test_only_runs_that_claim_to_be_done_are_candidates(self):
        """异常终止一眼可见，不需要用例来提醒。"""
        self.assertFalse(is_badcase({"termination_type": "ERROR", "false_finish": True}))

    def test_a_false_finish_is_a_badcase(self):
        self.assertTrue(is_badcase({"termination_type": "FINISH", "false_finish": True,
                                    "tool_call_accurate": True, "ref_score": 1.0}))

    def test_resolving_to_the_wrong_paper_is_a_badcase(self):
        self.assertTrue(is_badcase({"termination_type": "FINISH", "false_finish": False,
                                    "tool_call_accurate": True, "ref_score": 0.0}))

    def test_a_clean_run_is_not_a_badcase(self):
        self.assertFalse(is_badcase({"termination_type": "FINISH", "false_finish": False,
                                     "tool_call_accurate": True, "ref_score": 1.0}))


class CaptureTest(unittest.TestCase):
    def test_picks_the_failures_and_leaves_the_rest(self):
        samples = [
            (TASK["id"], [step("FINISH")]),                 # 什么都没做就收工
            (TASK["id"], [step(SEARCH), step("FINISH")]),   # 正解
        ]
        found = capture(samples, EXPANDED_TASKS, source="unit-test")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].reproduces_when, {"false_finish": True})
        self.assertEqual(found[0].status, STATUS_OPEN)
        self.assertEqual(found[0].source, "unit-test")

    def test_records_the_verdict_at_capture_time(self):
        found = capture([(TASK["id"], [step("FINISH")])], EXPANDED_TASKS)
        self.assertIn("reward", found[0].captured)

    def test_repeated_failures_on_one_task_get_distinct_ids(self):
        samples = [(TASK["id"], [step("FINISH")])] * 3
        ids = [c.case_id for c in capture(samples, EXPANDED_TASKS)]
        self.assertEqual(len(set(ids)), 3)

    def test_unknown_tasks_and_empty_histories_are_skipped(self):
        self.assertEqual(capture([("nope", [step("FINISH")]), (TASK["id"], [])],
                                 EXPANDED_TASKS), [])


class CaseFileTest(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.jsonl"
            original = [case(case_id="a", note="hi", captured={"reward": 0.5}),
                        case(case_id="b", status=STATUS_FIXED)]
            dump_cases(original, path)
            self.assertEqual([c.to_dict() for c in load_cases(path)],
                             [c.to_dict() for c in original])

    def test_duplicate_case_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.jsonl"
            dump_cases([case(case_id="dup"), case(case_id="dup")], path)
            with self.assertRaises(ValueError):
                load_cases(path)

    def test_blank_lines_and_comments_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.jsonl"
            path.write_text("// 说明\n\n"
                            + json.dumps(case(case_id="x").to_dict(), ensure_ascii=False) + "\n")
            self.assertEqual(len(load_cases(path)), 1)

    def test_a_malformed_line_names_its_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.jsonl"
            path.write_text(json.dumps(case(case_id="ok").to_dict(), ensure_ascii=False)
                            + "\nnot json\n")
            with self.assertRaises(ValueError) as ctx:
                load_cases(path)
            self.assertIn(":2", str(ctx.exception))

    def test_a_case_missing_required_fields_names_its_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.jsonl"
            path.write_text('{"case_id": "half-written"}\n')
            with self.assertRaises(ValueError) as ctx:
                load_cases(path)
            self.assertIn(":1", str(ctx.exception))


class ShippedCasesTest(unittest.TestCase):
    """仓库里那份用例库本身。这条测试就是回归闸门。"""

    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases(CASES_PATH)

    def test_the_case_library_is_not_empty(self):
        self.assertTrue(self.cases)

    def test_every_case_points_at_a_task_that_still_exists(self):
        known = {t["id"] for t in EXPANDED_TASKS}
        for c in self.cases:
            self.assertIn(c.task_id, known, c.case_id)

    def test_every_case_explains_itself(self):
        """用例没有说明就没人敢改它，只会被整条注释掉。"""
        for c in self.cases:
            self.assertTrue(c.note.strip(), c.case_id)

    def test_no_fixed_case_reproduces(self):
        """修好过的毛病不许再出现——这是整个模块存在的理由。"""
        regressions = [o.case.case_id for o in replay(self.cases, EXPANDED_TASKS)
                       if o.is_regression]
        self.assertEqual(regressions, [])

    def test_open_cases_still_reproduce(self):
        """open 却已经不复现，说明它早该改成 fixed，留着会让人误以为毛病还在。"""
        stale = [o.case.case_id for o in replay(self.cases, EXPANDED_TASKS)
                 if o.outcome == "newly_fixed"]
        self.assertEqual(stale, [], "这些用例已修复，把 status 改成 fixed")


if __name__ == "__main__":
    unittest.main()
