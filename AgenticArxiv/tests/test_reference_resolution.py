"""指代解析准确率：工具解析到的论文，是不是任务指的那篇。

`ref` 支持序号 / arXiv ID / 标题子串 / null，是本 agent 真正的难点。
按字符串比对参数对它同时会假阳性和假阴性，所以改比工具返回值里的 paper_id：
指代形式随便用，但必须落到正确的那篇论文。
"""

import json
import unittest

from benchmark.metrics import (
    extract_metrics,
    reference_resolution_score,
    resolved_paper_id,
)

P1, P2 = "2608.14539v1", "2608.14548v1"


def _step(observation, name="download_arxiv_pdf", args=None):
    return {"thought": "t",
            "action": json.dumps({"name": name, "args": args or {}}),
            "observation": observation}


def _dict_obs(pid):
    # 真实形态：base_agent 对 dict 结果做 str()，得到 Python repr
    return str({"session_id": "s", "paper_id": pid, "status": "READY"})


class ResolvedPaperIdTest(unittest.TestCase):
    """三种真实写法都要认，取自 traces/ 里的实际观测。"""

    def test_python_repr_single_quotes(self):
        self.assertEqual(resolved_paper_id(_dict_obs(P1)), P1)

    def test_json_double_quotes(self):
        self.assertEqual(resolved_paper_id(json.dumps({"paper_id": P1})), P1)

    def test_key_value_text_form(self):
        self.assertEqual(
            resolved_paper_id(f"已创建翻译任务 task_id=t1, paper_id={P1}，状态=PENDING"), P1)

    def test_explicit_none_is_not_a_paper(self):
        self.assertIsNone(resolved_paper_id("已创建翻译任务 task_id=t1, paper_id=None，状态=PENDING"))

    def test_error_observation_yields_nothing(self):
        self.assertIsNone(resolved_paper_id("工具执行失败: 未找到论文"))

    def test_search_observation_yields_nothing(self):
        self.assertIsNone(resolved_paper_id("成功获取 5 篇论文:\n  - Some Title"))

    def test_handles_non_string_observations(self):
        self.assertIsNone(resolved_paper_id(None))
        self.assertEqual(resolved_paper_id({"paper_id": P1}), P1)


class ReferenceResolutionScoreTest(unittest.TestCase):
    def test_returns_none_when_task_declares_no_expected_paper(self):
        self.assertIsNone(reference_resolution_score([_step(_dict_obs(P1))], None))

    def test_correct_paper_scores_one(self):
        self.assertEqual(reference_resolution_score([_step(_dict_obs(P1))], P1), 1.0)

    def test_wrong_paper_scores_zero(self):
        self.assertEqual(reference_resolution_score([_step(_dict_obs(P2))], P1), 0.0)

    def test_touching_no_paper_at_all_scores_zero(self):
        # agent 压根没碰论文，不能算它没错
        self.assertEqual(reference_resolution_score([_step("成功获取 5 篇论文")], P1), 0.0)

    def test_averages_over_resolving_steps(self):
        history = [_step(_dict_obs(P1)), _step(_dict_obs(P2))]
        self.assertEqual(reference_resolution_score(history, P1), 0.5)

    def test_search_steps_are_not_counted(self):
        # 检索不解析单篇论文，不该稀释分数
        history = [_step("成功获取 5 篇论文", name="get_recently_submitted_cs_papers"),
                   _step(_dict_obs(P1))]
        self.assertEqual(reference_resolution_score(history, P1), 1.0)

    def test_version_suffix_is_ignored(self):
        # v1 与 v2 是同一篇论文的不同版本
        self.assertEqual(reference_resolution_score([_step(_dict_obs("2608.14539v2"))], "2608.14539v1"), 1.0)

    def test_different_paper_is_not_excused_by_version_stripping(self):
        self.assertEqual(reference_resolution_score([_step(_dict_obs("2608.99999v1"))], "2608.14539v1"), 0.0)


class WhyNotArgumentMatchingTest(unittest.TestCase):
    """这两条正是引入本指标的理由 —— 参数比对在此处会给出错误结论。"""

    def test_catches_a_wrong_paper_that_argument_matching_would_pass(self):
        # 参数字符串与期望完全一致，但标题子串匹配到了另一篇
        history = [_step(_dict_obs(P2), args={"ref": "Learning State"})]
        self.assertEqual(reference_resolution_score(history, P1), 0.0)

    def test_credits_a_different_ref_form_that_argument_matching_would_fail(self):
        # 用序号而非标题子串，参数分接近 0，但解析到同一篇
        history = [_step(_dict_obs(P1), args={"ref": 3})]
        self.assertEqual(reference_resolution_score(history, P1), 1.0)


class ExtractMetricsIntegrationTest(unittest.TestCase):
    def _metrics(self, history, task):
        return extract_metrics(task, {"history": history}, "regex", 0)

    def test_task_without_expected_paper_is_not_penalised(self):
        m = self._metrics([_step(_dict_obs(P1))], {"id": "t", "expected_tools": []})
        self.assertEqual(m.ref_score, 1.0)

    def test_task_with_expected_paper_is_scored(self):
        task = {"id": "t", "expected_tools": ["download_arxiv_pdf"], "expected_paper": P1}
        self.assertEqual(self._metrics([_step(_dict_obs(P1))], task).ref_score, 1.0)
        self.assertEqual(self._metrics([_step(_dict_obs(P2))], task).ref_score, 0.0)

    def test_right_tool_wrong_paper_is_only_visible_in_ref_score(self):
        # 工具序列完全正确，唯有 ref_score 暴露下错了论文
        task = {"id": "t", "expected_tools": ["download_arxiv_pdf"], "expected_paper": P1}
        m = self._metrics([_step(_dict_obs(P2)), {"thought": "d", "action": "FINISH", "observation": "任务完成"}], task)
        self.assertTrue(m.tool_call_accurate)
        self.assertFalse(m.false_finish)
        self.assertEqual(m.ref_score, 0.0)


if __name__ == "__main__":
    unittest.main()
