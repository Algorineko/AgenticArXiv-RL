"""Regression tests for the deterministic benchmark-score baselines."""

from dataclasses import replace
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.baselines import (  # noqa: E402
    ALL_POLICIES,
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
        self.assertNotIn("infeasible_index_out_of_range", always_finish_tasks)


if __name__ == "__main__":
    unittest.main()
