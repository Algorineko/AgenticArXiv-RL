"""奖励区分度的回归护栏。

README TODO P1 的落点是「……奖励才有区分度」。这个文件把「有区分度」
变成可执行的断言：借 `benchmark/run_baselines.py` 的确定性退化策略，
逐类目卡住「参考解法」与「退化策略」之间的分差。

这些数字曾经不成立。修之前实测（training_step=100，full weights）：

    类目        参考   always_search  最小分差
    search     1.000      0.833        0.167   <- 无视任务永远搜 cs.AI，
                                                  在「检索 cs.CL」上拿 0.933
    infeasible 1.000      0.165       (负分才对) <- 正确答案是一次都别调，
                                                  乱调一个反而拿正分

根因有四处，都在参数档：

1. `argument_match_score` 把一半分给键覆盖率，而把键填齐是免费的；
2. 期望值为 None（「该键应缺省」）时键覆盖率把方向判反了，
   正确的省略与错误的传值同为 0.5，null 对照任务毫无区分；
3. `expected_tool_args == []`（一次都不该调）落到了「没有标准答案」
   那条分支，返回 1.0；
4. 参数分不看是哪个工具调的，`get_paper_cache_status(ref=1)` 能顶替
   `download_arxiv_pdf(ref=1)` 拿满分。

外加 `_outcome_score` 只看工具名，参数全错也给满分，权重 3。
"""

import sys
import unittest
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.baselines import (  # noqa: E402
    ALL_POLICIES,
    category_gap_failures,
    evaluate_baselines,
    resolve_policies,
)
from benchmark.tasks import get_all_tasks  # noqa: E402
from benchmark.tasks_expanded import get_expanded_tasks  # noqa: E402

# 与 run_baselines.py 的 --min-reference-gap 默认值保持一致
MIN_GAP = 0.3


def _score(tasks):
    return evaluate_baselines(
        tasks, resolve_policies(list(ALL_POLICIES)),
        seed=42, random_samples=20, training_step=100,
    )


def _by_category(results):
    """{category: {policy: mean_reward}}"""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        grouped[r.category][r.policy].append(r.reward)
    return {
        cat: {policy: mean(vals) for policy, vals in rows.items()}
        for cat, rows in grouped.items()
    }


class PerCategoryRewardGapTest(unittest.TestCase):
    """总体均值差看不见单类目的洞：search 曾经只有 0.167，总体却 PASS。"""

    @classmethod
    def setUpClass(cls):
        cls.expanded_rows = _score(get_expanded_tasks())
        cls.default_rows = _score(get_all_tasks())
        cls.expanded = _by_category(cls.expanded_rows)

    def test_expanded_set_separates_every_category(self):
        # infeasible 上 always_finish **就是**参考解法（正确行为是一次工具
        # 都不调），category_gap_failures 已按 exact_reference 把这类行剔除。
        self.assertEqual(category_gap_failures(self.expanded_rows, min_gap=MIN_GAP), [])

    def test_default_set_separates_every_category(self):
        self.assertEqual(category_gap_failures(self.default_rows, min_gap=MIN_GAP), [])

    def test_calling_a_tool_on_an_infeasible_task_is_punished(self):
        """正确答案是「什么都别做」，动了手必须是负分而不只是低分。"""
        rewards = self.expanded["infeasible"]
        self.assertEqual(rewards["reference"], 1.0)
        self.assertEqual(rewards["always_finish"], 1.0)
        for policy in ("always_search", "random_tool"):
            with self.subTest(policy=policy):
                self.assertLess(rewards[policy], 0.0)

    def test_ignoring_the_task_costs_more_than_a_third_of_the_score(self):
        """always_search 无视任务、永远发同一个 cs.AI 查询。

        修前它在 search 类目上平均 0.833；这里只钉一个上界，不钉具体值，
        免得以后调权重时变成需要同步维护的魔数。
        """
        self.assertLess(self.expanded["search"]["always_search"], 0.7)


class DefaultTaskSetDerivationTest(unittest.TestCase):
    """benchmark/tasks.py 迁到 TaskSpec 之后不该再有免检步骤。"""

    def test_every_task_declares_argument_oracles(self):
        for task in get_all_tasks():
            with self.subTest(task_id=task["id"]):
                args = task["expected_tool_args"]
                self.assertIsNotNone(
                    args, "expected_tool_args 为 None 会让 argument 档整个退出加权"
                )
                self.assertEqual(len(args), len(task["expected_tools"]))
                self.assertNotIn(None, args, "不该再有免检步骤")

    def test_dependency_chain_survives_the_migration(self):
        from benchmark.tasks import get_dependency_chain

        self.assertEqual(
            get_dependency_chain("translate_01"),
            ["search_01", "download_01", "translate_01"],
        )


if __name__ == "__main__":
    unittest.main()
