"""任务声明层的测试。

两条主线：
  1. `expected_tools` / `expected_tool_args` 由同一份 steps 派生，不可能漂移；
  2. 重构不能动到原有 43 条 —— data/splits/v1.json 里的成功率是 ~1600 次
     运行测出来的，任务文本或标准答案一变，那些数字就全部作废。
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.task_spec import Step, TaskSpec, build, family  # noqa: E402
from benchmark.tasks_expanded import EXPANDED_TASKS  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures_expanded_v1.json"


class DerivationTest(unittest.TestCase):
    def test_tools_and_args_come_from_the_same_steps(self):
        spec = TaskSpec(
            id="t", task="x",
            steps=(Step("get_recently_submitted_cs_papers", {"aspect": "AI"}),
                   Step("download_arxiv_pdf", {"ref": 1})),
        )
        task = spec.to_task()
        self.assertEqual(task["expected_tools"],
                         ["get_recently_submitted_cs_papers", "download_arxiv_pdf"])
        self.assertEqual(task["expected_tool_args"], [{"aspect": "AI"}, {"ref": 1}])

    def test_no_task_can_have_mismatched_ground_truth(self):
        """benchmark/tasks.py 里 download_01 / translate_01 / cache_01 就是
        expected_tools 有一项、expected_tool_args 为 None 的状态——
        argument_match_score 返回 None，参数这一档被整个踢出加权分母，
        任务照跑、分照打，只是参数从此不再被检查。这里保证扩充集不会重演。
        """
        for t in EXPANDED_TASKS:
            with self.subTest(task=t["id"]):
                self.assertIsNotNone(t["expected_tool_args"], t["id"])
                self.assertEqual(len(t["expected_tools"]), len(t["expected_tool_args"]))

    def test_empty_steps_means_no_tool_call_not_missing_ground_truth(self):
        """两者必须区分得开：`[]` = 正确行为是不调工具，`None` = 忘了写。"""
        task = TaskSpec(id="t", task="做不到的事").to_task()
        self.assertEqual(task["expected_tools"], [])
        self.assertEqual(task["expected_tool_args"], [])

    def test_args_are_copied_not_shared(self):
        shared = {"ref": 1}
        spec = TaskSpec(id="t", task="x", steps=(Step("download_arxiv_pdf", shared),))
        task = spec.to_task()
        task["expected_tool_args"][0]["ref"] = 99
        self.assertEqual(shared["ref"], 1)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            build([TaskSpec(id="dup", task="a"), TaskSpec(id="dup", task="b")])


class FamilyTest(unittest.TestCase):
    def test_text_and_ground_truth_share_one_param_set(self):
        """描述里写 7 天、标准答案却写 30 天——这种不一致要在结构上不可能。"""
        specs = family(
            task_id=lambda p: f"s_{p['days']}",
            text=lambda p: f"检索最近{p['days']}天的论文",
            steps=lambda p: [Step("get_recently_submitted_cs_papers", {"days": p["days"]})],
            params=[{"days": 7}, {"days": 30}],
            category="search",
        )
        for spec in specs:
            task = spec.to_task()
            days = task["expected_tool_args"][0]["days"]
            self.assertIn(f"最近{days}天", task["task"])


class BackwardCompatibilityTest(unittest.TestCase):
    """原 43 条必须逐字段不变，否则 data/splits/v1.json 的成功率全部作废。"""

    def setUp(self):
        self.old = {t["id"]: t for t in json.loads(FIXTURE.read_text(encoding="utf-8"))}
        self.new = {t["id"]: t for t in EXPANDED_TASKS}

    def test_no_original_task_disappeared(self):
        self.assertEqual(set(self.old) - set(self.new), set())

    def test_every_original_task_is_byte_identical(self):
        for task_id, old in self.old.items():
            with self.subTest(task=task_id):
                self.assertEqual(json.dumps(old, sort_keys=True, ensure_ascii=False),
                                 json.dumps(self.new[task_id], sort_keys=True, ensure_ascii=False))

    def test_split_file_ids_all_still_exist(self):
        split_path = Path(__file__).resolve().parents[2] / "data" / "splits" / "v1.json"
        if not split_path.exists():
            self.skipTest("没有持久化切分")
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        known = set(self.new)
        for group, ids in payload["split"].items():
            with self.subTest(group=group):
                self.assertEqual(set(ids) - known, set(), f"切分 {group} 引用了不存在的任务")


class ExpandedSetShapeTest(unittest.TestCase):
    def test_reaches_the_fifty_task_target(self):
        """README TODO P1 要求扩充到 50+。"""
        self.assertGreaterEqual(len(EXPANDED_TASKS), 50)

    def test_ids_unique(self):
        ids = [t["id"] for t in EXPANDED_TASKS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_has_infeasible_tasks(self):
        """整套任务此前只奖励「做对了什么」，从不惩罚「做了不该做的事」。"""
        infeasible = [t for t in EXPANDED_TASKS if t["category"] == "infeasible"]
        self.assertGreaterEqual(len(infeasible), 3)
        for t in infeasible:
            with self.subTest(task=t["id"]):
                self.assertEqual(t["expected_tools"], [], "不可行任务的正确行为是不调工具")

    def test_long_chains_declare_enough_iterations(self):
        """Agent 默认 5 轮 = 4 次工具调用 + FINISH。

        超出预算的链会被判成 FORCE_STOP —— 那是「预算不够」而不是「不会规划」，
        混在一起会让长链任务的失败率完全没法解读。
        """
        for t in EXPANDED_TASKS:
            needed = len(t["expected_tools"]) + 1        # +1 给 FINISH
            budget = t.get("max_iterations", 5)
            with self.subTest(task=t["id"]):
                self.assertGreaterEqual(
                    budget, needed,
                    f"{t['id']} 需要 {needed} 轮，预算只有 {budget}",
                )

    def test_every_ref_using_task_has_papers_in_session(self):
        """用序号 / null 指代但既不自己搜、又没有 setup 的任务是无解的。"""
        for t in EXPANDED_TASKS:
            if t["category"] == "infeasible":
                continue      # 无解正是这一类的题意
            uses_ref = any("ref" in (a or {}) for a in t["expected_tool_args"])
            self_searches = (t["expected_tools"] or [None])[0] == "get_recently_submitted_cs_papers"
            with self.subTest(task=t["id"]):
                if uses_ref and not self_searches:
                    self.assertIn("setup", t, f"{t['id']} 用了 ref 却没有会话状态")

    def test_setup_provides_enough_papers_for_every_ordinal_ref(self):
        for t in EXPANDED_TASKS:
            if "setup" not in t:
                continue
            seeded = max((a["args"].get("max_results", 0) for a in t["setup"]
                          if a["name"] == "get_recently_submitted_cs_papers"), default=0)
            ordinals = [a["ref"] for a in t["expected_tool_args"]
                        if isinstance(a.get("ref"), int)]
            if ordinals:
                with self.subTest(task=t["id"]):
                    self.assertLessEqual(max(ordinals), seeded,
                                         f"{t['id']} 引用第 {max(ordinals)} 篇，setup 只铺了 {seeded} 篇")


class InfeasibleScoringTest(unittest.TestCase):
    """不可行任务必须真的能把「幻觉调用」和「正确拒绝」拉开分差。

    只加任务不够 —— 加之前 `_check_tool_sequence` 在 expected 为空时无条件
    返回 True，硬调一个不存在的 ref 也算 accurate，outcome 还给满分。
    """

    def _score(self, task_id, history):
        from rl.reward import RewardCalculator
        task = next(t for t in EXPANDED_TASKS if t["id"] == task_id)
        result = {"history": history, "timing": {}, "token_usage": {},
                  "iteration_count": len(history)}
        return RewardCalculator().compute_reward_breakdown(task, result, training_step=100)

    REFUSE = [{"thought": "会话里只有 5 篇，做不到", "action": "FINISH", "observation": "任务完成"}]
    HALLUCINATE = [
        {"thought": "下载第20篇", "action": '{"name": "download_arxiv_pdf", "args": {"ref": 20}}',
         "observation": "工具执行失败: 未找到论文"},
        {"thought": "", "action": "FINISH", "observation": "任务完成"},
    ]

    def test_refusing_scores_full_marks(self):
        breakdown, metrics = self._score("infeasible_index_out_of_range", self.REFUSE)
        self.assertTrue(metrics.tool_call_accurate)
        self.assertEqual(breakdown.tool, 1.0)

    def test_hallucinated_call_is_penalised(self):
        breakdown, metrics = self._score("infeasible_index_out_of_range", self.HALLUCINATE)
        self.assertFalse(metrics.tool_call_accurate)
        self.assertEqual(breakdown.tool, -1.0)

    def test_gap_is_wide_enough_to_learn_from(self):
        """分差太小，GRPO 组内相对优势就区分不出这两种行为。"""
        good, _ = self._score("infeasible_index_out_of_range", self.REFUSE)
        bad, _ = self._score("infeasible_index_out_of_range", self.HALLUCINATE)
        self.assertGreater(good.total - bad.total, 0.5)

    def test_penalty_matches_calling_the_wrong_tool_on_a_normal_task(self):
        """「本该什么都不做却动了手」不该比「做错了」罚得更轻。"""
        from rl.reward import RewardCalculator
        calc = RewardCalculator()
        self.assertEqual(calc._tool_score(["download_arxiv_pdf"], []), -1.0)
        self.assertEqual(calc._tool_score(["download_arxiv_pdf"],
                                          ["get_recently_submitted_cs_papers"]), -1.0)


class RunnerHonoursIterationBudgetTest(unittest.TestCase):
    """任务声明的 max_iterations 必须真的传到 Agent。

    只在任务字典里写上预算是不够的 —— runner 以前无条件用 Agent 的默认 5 轮，
    5 步链因此必然 FORCE_STOP，看起来像「不会规划」，实际是预算不够。
    """

    def _runner(self):
        from benchmark.runner import BenchmarkRunner
        runner = BenchmarkRunner.__new__(BenchmarkRunner)   # 不走 __init__，免得建 LLM 客户端
        runner.llm_extra = None
        runner._llm_client = object()      # ReActAgent 只是存下来，不会调用
        runner._side_effects = lambda: None
        runner._tool_env = lambda: None
        return runner

    def test_budget_reaches_the_agent(self):
        agent = self._runner()._create_agent("regex", 7)
        self.assertEqual(agent.max_iterations, 7)

    def test_default_is_left_untouched_when_task_declares_nothing(self):
        agent = self._runner()._create_agent("regex", None)
        self.assertEqual(agent.max_iterations, 5)

    def test_task_dict_budget_is_what_gets_passed(self):
        """端到端：从任务字典取到的值，就是 Agent 拿到的值。"""
        task = next(t for t in EXPANDED_TASKS
                    if t["id"] == "chain_cv5_cache_dl_tr_cache")
        agent = self._runner()._create_agent("regex", task.get("max_iterations"))
        self.assertEqual(agent.max_iterations, task["max_iterations"])
        self.assertGreater(agent.max_iterations, len(task["expected_tools"]))


if __name__ == "__main__":
    unittest.main()
