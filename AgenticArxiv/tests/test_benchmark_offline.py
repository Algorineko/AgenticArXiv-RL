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
    """语义须与提取前的 rl/reward.py::_argument_score 一致。"""

    def test_returns_none_when_task_declares_no_expected_args(self):
        self.assertIsNone(argument_match_score([_step("t", {"a": 1})], None))

    def test_exact_match_scores_one(self):
        history = [_step("download_arxiv_pdf", {"ref": 2})]
        self.assertEqual(argument_match_score(history, [{"ref": 2}]), 1.0)

    def test_right_key_wrong_value_scores_half(self):
        history = [_step("download_arxiv_pdf", {"ref": 9})]
        self.assertEqual(argument_match_score(history, [{"ref": 2}]), 0.5)

    def test_missing_step_scores_zero(self):
        self.assertEqual(argument_match_score([], [{"ref": 2}]), 0.0)

    def test_none_entries_skip_that_step(self):
        history = [_step("a", {"x": 1}), _step("b", {"ref": 2})]
        self.assertEqual(argument_match_score(history, [None, {"ref": 2}]), 1.0)

    def test_extra_session_id_does_not_penalise(self):
        # session_id 由框架注入，不该算模型的错
        history = [_step("download_arxiv_pdf", {"ref": 2, "session_id": "s"})]
        self.assertEqual(argument_match_score(history, [{"ref": 2}]), 1.0)


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
