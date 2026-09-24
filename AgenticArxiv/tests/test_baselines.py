"""Regression tests for the deterministic benchmark-score baselines."""

from dataclasses import replace
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.baselines import (  # noqa: E402
    ALL_POLICIES,
    category_gap_failures,
    evaluate_baselines,
    highest_scoring_tasks,
    reference_gap_failures,
    render_markdown,
    resolve_policies,
    summarize_baselines,
)
from benchmark.tasks_expanded import get_expanded_tasks  # noqa: E402


class BaselineDiagnosticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = get_expanded_tasks()

    def _results(self, names, random_samples=20):
        return evaluate_baselines(
            self.tasks, resolve_policies(names), seed=42,
            random_samples=random_samples, training_step=100,
        )

    def test_reference_trajectory_is_the_score_ceiling(self):
        results = self._results(["reference"])
        self.assertEqual(len(results), len(self.tasks))
        self.assertTrue(all(result.reward == 1.0 for result in results))
        self.assertTrue(all(result.exact_tool_path for result in results))

    def test_reference_figure_analysis_has_a_valid_synthetic_answer(self):
        """参考策略可使用合成观察值，但图表答案必须满足结果契约。"""
        task = next(
            item for item in self.tasks
            if item["category"] == "figure_analysis"
        )
        reference = resolve_policies(["reference"])[0]
        result = reference.build_result(task, seed=42)
        figure_steps = [
            step for step in result["history"]
            if '"name": "analyze_figure"' in step["action"]
        ]
        self.assertEqual(len(figure_steps), 1)
        observation = json.loads(figure_steps[0]["observation"])
        self.assertTrue(observation["paper_id"].strip())
        self.assertTrue(observation["answer"].strip())

    def test_always_finish_is_not_mistaken_for_the_reference(self):
        results = self._results(["reference", "always_finish"])
        summaries = {item.policy: item for item in summarize_baselines(results)}

        # FINISH is a termination signal, not proof that task work happened.
        self.assertEqual(summaries["always_finish"].finish_rate, 1.0)
        self.assertLess(summaries["always_finish"].exact_tool_path_rate, 1.0)
        self.assertLess(
            summaries["always_finish"].mean_reward,
            summaries["reference"].mean_reward,
        )

    def test_degenerate_policies_keep_the_required_reference_gap(self):
        results = self._results(list(ALL_POLICIES))
        summaries = summarize_baselines(results)
        self.assertEqual(reference_gap_failures(summaries, min_gap=0.3), [])

    def test_health_guard_rejects_a_near_reference_policy(self):
        summaries = summarize_baselines(self._results(list(ALL_POLICIES)))
        summaries = [
            replace(summary, mean_reward=0.95)
            if summary.policy == "always_search"
            else summary
            for summary in summaries
        ]
        failures = reference_gap_failures(summaries, min_gap=0.3)
        self.assertTrue(any("always_search" in failure for failure in failures))

    def test_health_guard_requires_the_reference_policy(self):
        summaries = summarize_baselines(self._results(["always_finish"]))
        self.assertIn("reference policy is required", reference_gap_failures(summaries)[0])

    def test_random_tool_is_reproducible_for_a_fixed_seed(self):
        first = self._results(["random_tool"], random_samples=5)
        second = self._results(["random_tool"], random_samples=5)
        self.assertEqual(first, second)

    def test_random_tool_reports_multiple_samples_and_variance(self):
        results = self._results(["random_tool"], random_samples=5)
        summary = summarize_baselines(results)[0]
        self.assertEqual(summary.task_count, len(self.tasks))
        self.assertEqual(summary.sample_count, len(self.tasks) * 5)
        self.assertGreater(summary.reward_std, 0.0)

    def test_missing_argument_oracles_are_excluded_from_the_mean(self):
        from benchmark.tasks import get_all_tasks

        results = evaluate_baselines(
            get_all_tasks(), resolve_policies(["always_finish"]), training_step=100
        )
        self.assertEqual(summarize_baselines(results)[0].mean_arg_score, 0.0)

    def test_per_category_check_catches_what_the_aggregate_mean_dilutes(self):
        """总体均值把单类目的洞摊平了。

        实测过的例子：`always_search` 在 search 类目上距参考仅 0.167，
        而总体均值差 0.832 —— 总体闸 PASS，逐类目闸 FAIL。
        """
        results = self._results(list(ALL_POLICIES))
        self.assertEqual(category_gap_failures(results, min_gap=0.3), [])

        leaky = [
            replace(row, reward=0.95)
            if (row.category == "search" and row.policy == "always_search")
            else row
            for row in results
        ]
        summaries = summarize_baselines(leaky)
        self.assertEqual(reference_gap_failures(summaries, min_gap=0.3), [])
        failures = category_gap_failures(leaky, min_gap=0.3)
        self.assertTrue(any("search/always_search" in f for f in failures), failures)

    def test_empty_tool_path_does_not_make_false_finish_a_reference(self):
        """blocked 任务还要解释原因，空工具路径不是完整参考答案。"""
        results = self._results(["reference", "always_finish"])
        infeasible = [r for r in results if r.category == "infeasible"]
        self.assertTrue(infeasible)
        self.assertTrue(all(
            not r.exact_reference for r in infeasible if r.policy == "always_finish"
        ))
        self.assertFalse(any(
            "infeasible/always_finish" in f
            for f in category_gap_failures(results, min_gap=0.3)
        ))

    def test_parameter_boundary_infeasible_case_rejects_tool_calls(self):
        """zero_index：参数非法时正确行为是不调工具，weak policy 必然掉分。"""
        task = next(t for t in self.tasks if t["id"] == "infeasible_zero_index")
        results = evaluate_baselines(
            [task], resolve_policies(["always_search", "random_tool"]),
            seed=42, training_step=100,
        )
        self.assertTrue(all(r.reward < 1.0 for r in results))

    def test_wrong_args_policy_triggers_arg_score_gate(self):
        """wrong_args：工具名对、参数出错，arg_score 必须低于 reference。"""
        summaries = {s.policy: s for s in summarize_baselines(
            self._results(["reference", "wrong_args"])
        )}
        self.assertLess(
            summaries["wrong_args"].mean_arg_score,
            summaries["reference"].mean_arg_score,
        )

    def test_markdown_labels_finish_as_a_separate_signal(self):
        results = self._results(["reference", "always_finish"])
        summaries = summarize_baselines(results)
        top_tasks = highest_scoring_tasks(results, limit_per_policy=3)
        markdown = render_markdown(
            summaries, task_set="expanded", seed=42, training_step=100,
            top_tasks=top_tasks, min_reference_gap=0.3,
        )
        self.assertIn("Finish rate", markdown)
        self.assertIn("Exact tool path", markdown)
        self.assertIn("Highest-scoring non-reference trajectories", markdown)
        self.assertIn("Health check: **PASS**", markdown)
        always_finish_tasks = {
            task.task_id for task in top_tasks if task.policy == "always_finish"
        }
        self.assertIn("infeasible_index_out_of_range", always_finish_tasks)


if __name__ == "__main__":
    unittest.main()
