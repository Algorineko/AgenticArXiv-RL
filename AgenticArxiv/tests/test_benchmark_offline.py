"""离线回放、setup 铺状态、参数准确率的单元测试（不需要 LLM / 网络 / 数据库）。"""

import json
import unittest

from benchmark.metrics import argument_match_score
from benchmark.runner import BenchmarkRunner


def _step(name, args):
    return {"thought": "t", "action": json.dumps({"name": name, "args": args}), "observation": ""}


class _FakeEnv:
    """记录被调用的工具，并返回可预测的结果。"""

    def __init__(self, results=None):
        self.calls = []
        self._results = results or {}

    def execute_tool(self, name, args):
        self.calls.append((name, dict(args)))
        return self._results.get(name)


class _FakeSideEffects:
    def __init__(self):
        self.papers = {}
        self.active = {}
        self.translated = []

    def set_last_papers(self, session_id, papers):
        self.papers[session_id] = papers

    def set_last_active_paper_id(self, session_id, paper_id):
        self.active[session_id] = paper_id

    def enqueue_translate(self, **kwargs):
        self.translated.append(kwargs)


class ArgumentMatchScoreTest(unittest.TestCase):
    """参数级打分：只认取值，认工具身份，区分「不该调」与「不校验」。"""

    def test_returns_none_when_task_declares_no_expected_args(self):
        self.assertIsNone(argument_match_score([_step("t", {"a": 1})], None))

    def test_exact_match_scores_one(self):
        history = [_step("download_arxiv_pdf", {"ref": 2})]
        self.assertEqual(argument_match_score(history, [{"ref": 2}]), 1.0)

    def test_right_key_wrong_value_scores_zero(self):
        # 曾经是 0.5：键覆盖率占一半分，而把键填齐是免费的。
        history = [_step("download_arxiv_pdf", {"ref": 9})]
        self.assertEqual(argument_match_score(history, [{"ref": 2}]), 0.0)

    def test_missing_step_scores_zero(self):
        self.assertEqual(argument_match_score([], [{"ref": 2}]), 0.0)

    def test_none_entries_skip_that_step(self):
        history = [_step("a", {"x": 1}), _step("b", {"ref": 2})]
        self.assertEqual(argument_match_score(history, [None, {"ref": 2}]), 1.0)

    def test_extra_session_id_does_not_penalise(self):
        # session_id 由框架注入，不该算模型的错
        history = [_step("download_arxiv_pdf", {"ref": 2, "session_id": "s"})]
        self.assertEqual(argument_match_score(history, [{"ref": 2}]), 1.0)

    def test_empty_expected_args_means_call_nothing(self):
        """[] 是「正确行为是一次都不调」，不是「没写标准答案」。

        后者是 None。两者曾经都走到末尾的 `else 1.0`，于是 infeasible
        任务上乱调一个工具反而白拿满参数分（+1.0 × 权重 2），把
        `_tool_score` 给的 -1.0 抵掉大半。
        """
        self.assertEqual(argument_match_score([], []), 1.0)
        self.assertEqual(argument_match_score([_step("search", {"aspect": "AI"})], []), 0.0)

    def test_expected_none_value_means_the_key_should_be_omitted(self):
        """`{"ref": None}` = 用当前活跃论文；省略 ref 才是正确写法。

        旧口径下「省略」被判键覆盖率 0、「错传 ref=1」反倒键覆盖率满分，
        两者都落在 0.5——而 ref_form / state 里那些 null 对照任务存在的
        唯一目的就是区分这两种行为。
        """
        expected = [{"ref": None}]
        self.assertEqual(argument_match_score([_step("translate_arxiv_pdf", {})], expected), 1.0)
        self.assertEqual(
            argument_match_score([_step("translate_arxiv_pdf", {"ref": None})], expected), 1.0
        )
        self.assertEqual(
            argument_match_score([_step("translate_arxiv_pdf", {"ref": 1})], expected), 0.0
        )

    def test_arguments_only_count_when_the_tool_itself_is_right(self):
        """参数分不能与工具名脱钩，否则等于替调错的工具背书。"""
        history = [_step("get_paper_cache_status", {"ref": 1})]
        self.assertEqual(argument_match_score(history, [{"ref": 1}]), 1.0)  # 不传 expected_tools
        self.assertEqual(
            argument_match_score(history, [{"ref": 1}], ["download_arxiv_pdf"]), 0.0
        )
        self.assertEqual(
            argument_match_score(history, [{"ref": 1}], ["get_paper_cache_status"]), 1.0
        )


class ApplySetupTest(unittest.TestCase):
    """setup 直接调工具铺状态，不再走一遍完整 Agent。"""

    def _runner(self, env, side_fx):
        r = BenchmarkRunner(agent_types=["regex"], repeat=1, offline=True)
        r._env, r._side_fx = env, side_fx
        return r

    def test_no_setup_is_a_no_op(self):
        env, fx = _FakeEnv(), _FakeSideEffects()
        self._runner(env, fx)._apply_setup({"id": "t"}, "s1")
        self.assertEqual(env.calls, [])

    def test_search_setup_seeds_the_paper_list(self):
        papers = [{"id": "2608.1v1", "title": "A"}, {"id": "2608.2v1", "title": "B"}]
        env = _FakeEnv({"get_recently_submitted_cs_papers": papers})
        fx = _FakeSideEffects()
        task = {"id": "t", "setup": [
            {"name": "get_recently_submitted_cs_papers", "args": {"aspect": "AI", "days": 7}}]}
        self._runner(env, fx)._apply_setup(task, "s1")

        self.assertEqual(env.calls[0][0], "get_recently_submitted_cs_papers")
        self.assertEqual(env.calls[0][1]["session_id"], "s1")   # session_id 被注入
        self.assertEqual([p.id for p in fx.papers["s1"]], ["2608.1v1", "2608.2v1"])

    def test_download_setup_marks_the_active_paper(self):
        env = _FakeEnv({"download_arxiv_pdf": {"paper_id": "2608.1v1", "status": "READY"}})
        fx = _FakeSideEffects()
        task = {"id": "t", "setup": [{"name": "download_arxiv_pdf", "args": {"ref": 1}}]}
        self._runner(env, fx)._apply_setup(task, "s1")
        self.assertEqual(fx.active["s1"], "2608.1v1")

    def test_translate_setup_is_enqueued_not_executed(self):
        env, fx = _FakeEnv(), _FakeSideEffects()
        task = {"id": "t", "setup": [{"name": "translate_arxiv_pdf", "args": {"ref": 1}}]}
        self._runner(env, fx)._apply_setup(task, "s1")
        self.assertEqual(env.calls, [])                          # 不同步执行
        self.assertEqual(fx.translated[0]["session_id"], "s1")


class OfflineWiringTest(unittest.TestCase):
    def test_online_runner_injects_no_env(self):
        self.assertIsNone(BenchmarkRunner(offline=False)._tool_env())

    def test_missing_snapshot_fails_loudly(self):
        r = BenchmarkRunner(offline=True, snapshot="/nonexistent/snap.json")
        with self.assertRaises(SystemExit):
            r._tool_env()

    def test_offline_forces_local_side_effects(self):
        from agents.side_effects import LocalSideEffectManager
        self.assertIsInstance(BenchmarkRunner(offline=True)._side_effects(),
                              LocalSideEffectManager)


if __name__ == "__main__":
    unittest.main()
