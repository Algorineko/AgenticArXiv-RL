"""离线回放、setup 铺状态、参数准确率的单元测试（不需要 LLM / 网络 / 数据库）。"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from benchmark.metrics import _match_arg_value, argument_match_score
from benchmark.runner import BenchmarkRunner


def _step(name, args):
    return {
        "thought": "t",
        "action": json.dumps({"name": name, "args": args}),
        "observation": "",
    }


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
        self.assertEqual(
            argument_match_score([_step("search", {"aspect": "AI"})], []), 0.0
        )

    def test_expected_none_value_means_the_key_should_be_omitted(self):
        """`{"ref": None}` = 用当前活跃论文；省略 ref 才是正确写法。

        旧口径下「省略」被判键覆盖率 0、「错传 ref=1」反倒键覆盖率满分，
        两者都落在 0.5——而 ref_form / state 里那些 null 对照任务存在的
        唯一目的就是区分这两种行为。
        """
        expected = [{"ref": None}]
        self.assertEqual(
            argument_match_score([_step("translate_arxiv_pdf", {})], expected), 1.0
        )
        self.assertEqual(
            argument_match_score(
                [_step("translate_arxiv_pdf", {"ref": None})], expected
            ),
            1.0,
        )
        self.assertEqual(
            argument_match_score([_step("translate_arxiv_pdf", {"ref": 1})], expected),
            0.0,
        )

    def test_arguments_only_count_when_the_tool_itself_is_right(self):
        """参数分不能与工具名脱钩，否则等于替调错的工具背书。"""
        history = [_step("get_paper_cache_status", {"ref": 1})]
        self.assertEqual(
            argument_match_score(history, [{"ref": 1}]), 1.0
        )  # 不传 expected_tools
        self.assertEqual(
            argument_match_score(history, [{"ref": 1}], ["download_arxiv_pdf"]), 0.0
        )
        self.assertEqual(
            argument_match_score(history, [{"ref": 1}], ["get_paper_cache_status"]), 1.0
        )


class ArgValueEquivalenceTest(unittest.TestCase):
    """取值等价判定：允许写法差异，不允许语义差异。

    `_match_arg_value` 让 "cs.AI"/"AI"、"1"/1、"第1篇"/1 这类同义写法不被
    误判为错，这是对的——模型换个写法不该扣分。但它的宽松度必须止步于写法：
    一旦放行了语义不同的取值，参数档就重新变成「照签名填齐就有分」，
    也就是 issue #17 第 2 条描述的那个洞。
    """

    def test_aspect_accepts_the_cs_prefix_and_case_variants(self):
        for predicted in ("cs.AI", "AI", "cs.ai", " CS.AI "):
            self.assertTrue(_match_arg_value(predicted, "cs.AI", "aspect"), predicted)

    def test_aspect_still_rejects_a_different_category(self):
        self.assertFalse(_match_arg_value("cs.AI", "cs.CL", "aspect"))

    def test_ref_accepts_string_and_ordinal_spellings(self):
        for predicted in (1, "1", "1.0", "第1篇", "第 1 篇"):
            self.assertTrue(_match_arg_value(predicted, 1, "ref"), predicted)

    def test_ref_does_not_drop_the_minus_sign(self):
        r"""回归：`\d+` 从 "-1" 里只抠得到 "1"，于是 ref=-1 被判成 ref=1。

        ref 是 1-based 序号（tools/pdf_download_tool.py），-1 是越界值，
        也正是 wrong_args 基线用来制造错误参数的取值——这个洞让它在
        composite 类目上白拿 0.226 分，离参考轨迹只剩 0.335 的差距。
        """
        self.assertFalse(_match_arg_value(-1, 1, "ref"))
        self.assertFalse(_match_arg_value("-1", 1, "ref"))
        self.assertFalse(_match_arg_value(-3, 3, "ref"))
        self.assertFalse(_match_arg_value("ref=-1", 1, "ref"))

    def test_numeric_and_string_forms_agree_outside_the_named_keys(self):
        self.assertTrue(_match_arg_value("7", 7, "days"))
        self.assertTrue(_match_arg_value(5.0, 5, "max_results"))
        self.assertFalse(_match_arg_value(7, 30, "days"))

    def test_expected_none_means_the_key_should_be_omitted(self):
        self.assertTrue(_match_arg_value(None, None, "ref"))
        self.assertFalse(_match_arg_value(1, None, "ref"))
        self.assertFalse(_match_arg_value(None, 1, "ref"))

    def test_wrong_args_baseline_earns_nothing_on_a_ref_oracle(self):
        """端到端确认：wrong_args 的 ref=-1 在参数档上应当颗粒无收。"""
        history = [_step("download_arxiv_pdf", {"ref": -1})]
        self.assertEqual(
            argument_match_score(history, [{"ref": 1}], ["download_arxiv_pdf"]), 0.0
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
        task = {
            "id": "t",
            "setup": [
                {
                    "name": "get_recently_submitted_cs_papers",
                    "args": {"aspect": "AI", "days": 7},
                }
            ],
        }
        self._runner(env, fx)._apply_setup(task, "s1")

        self.assertEqual(env.calls[0][0], "get_recently_submitted_cs_papers")
        self.assertEqual(env.calls[0][1]["session_id"], "s1")  # session_id 被注入
        self.assertEqual([p.id for p in fx.papers["s1"]], ["2608.1v1", "2608.2v1"])

    def test_download_setup_marks_the_active_paper(self):
        env = _FakeEnv(
            {"download_arxiv_pdf": {"paper_id": "2608.1v1", "status": "READY"}}
        )
        fx = _FakeSideEffects()
        task = {
            "id": "t",
            "setup": [{"name": "download_arxiv_pdf", "args": {"ref": 1}}],
        }
        self._runner(env, fx)._apply_setup(task, "s1")
        self.assertEqual(fx.active["s1"], "2608.1v1")

    def test_translate_setup_is_enqueued_not_executed(self):
        env, fx = _FakeEnv(), _FakeSideEffects()
        task = {
            "id": "t",
            "setup": [{"name": "translate_arxiv_pdf", "args": {"ref": 1}}],
        }
        self._runner(env, fx)._apply_setup(task, "s1")
        self.assertEqual(env.calls, [])  # 不同步执行
        self.assertEqual(fx.translated[0]["session_id"], "s1")


class OfflineWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 每个测试类必须能独立运行，不能依赖其他测试先导入并注册工具。
        from tools.bootstrap import register_all_tools

        register_all_tools()

    def test_online_runner_injects_no_env(self):
        self.assertIsNone(BenchmarkRunner(offline=False)._tool_env())

    def test_missing_snapshot_fails_loudly(self):
        r = BenchmarkRunner(offline=True, snapshot="/nonexistent/snap.json")
        with self.assertRaises(SystemExit):
            r._tool_env()

    def test_offline_forces_local_side_effects(self):
        from agents.side_effects import LocalSideEffectManager

        self.assertIsInstance(
            BenchmarkRunner(offline=True)._side_effects(), LocalSideEffectManager
        )

    def test_empty_session_translate_fails_instead_of_creating_none_paper_task(self):
        from agents.side_effects import LocalSideEffectManager

        fx = LocalSideEffectManager()
        with self.assertRaisesRegex(ValueError, "未找到指代对象"):
            fx.enqueue_translate(session_id="empty-translate-test", ref=None)

    def test_offline_download_state_and_timestamps_ignore_existing_disk_files(self):
        import config
        from models.schemas import Paper
        from models.store import store, use_memory_store
        from rl.env import MockArxivEnv

        paper = Paper(
            id="2601.00001v1",
            title="Deterministic Offline Paper",
            pdf_url="https://arxiv.org/pdf/2601.00001v1.pdf",
        )
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch(
                "config.settings",
                replace(config.settings, pdf_raw_path=tmpdir),
            ),
        ):
            # 预先放入一个文件，模拟上一轮 benchmark 的磁盘残留。
            existing = Path(tmpdir) / "2601.00001v1.pdf"
            existing.write_bytes(b"an old file with a different size")

            use_memory_store(reset=True)
            store.set_last_papers("trial", [paper])
            env = MockArxivEnv(mode="replay")
            first = env.execute_tool(
                "download_arxiv_pdf", {"session_id": "trial", "ref": 1}
            )
            second = env.execute_tool(
                "download_arxiv_pdf", {"session_id": "trial", "ref": 1}
            )
            first_asset = store.get_pdf_asset(paper.id)

            self.assertFalse(first["existed"])
            self.assertTrue(second["existed"])
            self.assertEqual(first["size_bytes"], 39)
            self.assertEqual(str(first_asset.downloaded_at), "2000-01-01 00:00:00")
            self.assertEqual(str(first_asset.updated_at), "2000-01-01 00:00:00")

            use_memory_store(reset=True)
            store.set_last_papers("trial", [paper])
            another_env = MockArxivEnv(mode="replay")
            repeated_run = another_env.execute_tool(
                "download_arxiv_pdf", {"session_id": "trial", "ref": 1}
            )
            self.assertFalse(repeated_run["existed"])

    def test_offline_trial_reset_clears_env_and_translate_runtime_state(self):
        from agents.side_effects import LocalSideEffectManager
        from models.store import store, use_memory_store

        class FakeEnv:
            def __init__(self):
                self.reset_count = 0

            def reset_runtime_state(self):
                self.reset_count += 1

        use_memory_store(reset=True)
        store.set_last_active_paper_id("old", "2601.00001v1")
        runner = BenchmarkRunner(offline=True)
        runner._env = FakeEnv()
        runner._side_fx = LocalSideEffectManager()
        runner._side_fx._translate_seq = 7
        runner._side_fx.published_events.append({"old": True})

        runner._reset_offline_trial_state()

        self.assertIsNone(store.get_last_active_paper_id("old"))
        self.assertEqual(runner._env.reset_count, 1)
        self.assertEqual(runner._side_fx._translate_seq, 0)
        self.assertEqual(runner._side_fx.published_events, [])


class ReplayArgumentValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.bootstrap import register_all_tools

        register_all_tools()

    def test_replay_rejects_arguments_the_real_tool_does_not_accept(self):
        from rl.env import MockArxivEnv

        with self.assertRaisesRegex(ValueError, "unexpected keyword argument 'query'"):
            MockArxivEnv._validate_tool_arguments(
                "get_recently_submitted_cs_papers",
                {"aspect": "AR", "query": "Image Restoration", "days": 7},
            )

    def test_framework_session_id_is_allowed_for_setup_search(self):
        from rl.env import MockArxivEnv

        MockArxivEnv._validate_tool_arguments(
            "get_recently_submitted_cs_papers",
            {"aspect": "CV", "days": 7, "max_results": 5, "session_id": "setup"},
        )

    def test_replay_rejects_empty_keyword_query_like_the_real_tool(self):
        from rl.env import MockArxivEnv

        with self.assertRaisesRegex(ValueError, "query 不能为空"):
            MockArxivEnv._validate_tool_arguments(
                "search_arxiv_papers",
                {"query": "   ", "max_results": 3, "days": 30},
            )


class SearchSessionSyncTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.bootstrap import register_all_tools

        register_all_tools()

    def test_both_search_tools_seed_the_same_session_memory_and_show_titles(self):
        from agents.agent_engine import ReActAgent
        from agents.side_effects import LocalSideEffectManager

        paper = {"id": "2601.00001v1", "title": "Keyword Result"}
        for tool_name in ("get_recently_submitted_cs_papers", "search_arxiv_papers"):
            with self.subTest(tool=tool_name):
                session_id = f"search-sync-{tool_name}"
                fx = LocalSideEffectManager()
                env = _FakeEnv({tool_name: [paper]})
                agent = ReActAgent(llm_client=None, side_effect_mgr=fx, env=env)
                agent.session_id = session_id

                observation = agent._execute_with_side_effects(
                    {"name": tool_name, "args": {}}
                )

                self.assertIn("Keyword Result", observation)
                self.assertEqual(
                    [p.id for p in fx.get_last_papers(session_id)],
                    ["2601.00001v1"],
                )

    def test_keyword_fallback_is_reported_as_failure_and_does_not_seed_session(self):
        from agents.agent_engine import ReActAgent
        from agents.side_effects import LocalSideEffectManager

        paper = {
            "id": "2601.00001v1",
            "title": "Unrelated Fallback",
            "_mock_env": {
                "offline_fallback": True,
                "message": "关键词未命中离线快照；返回固定回退子集，不代表查询匹配结果。",
            },
        }
        session_id = "keyword-fallback-not-success"
        fx = LocalSideEffectManager()
        env = _FakeEnv({"search_arxiv_papers": [paper]})
        agent = ReActAgent(llm_client=None, side_effect_mgr=fx, env=env)
        agent.session_id = session_id

        observation = agent._execute_with_side_effects(
            {
                "name": "search_arxiv_papers",
                "args": {"query": "all:unknown", "days": 30, "max_results": 3},
            }
        )

        self.assertIn("工具执行失败", observation)
        self.assertIn("不代表查询匹配结果", observation)
        self.assertEqual(fx.get_last_papers(session_id), [])


class LocalLlmBackendTest(unittest.TestCase):
    def test_benchmark_resets_a_stable_stream_for_each_task_trial(self):
        class FakeLocalClient:
            def __init__(self):
                self.streams = []

            def start_generation_stream(self, key):
                self.streams.append(key)

        runner = BenchmarkRunner(model="/models/qwen", llm_backend="transformers")
        fake = FakeLocalClient()
        runner._llm_client = fake

        runner._start_generation_stream("task-a", 2)

        self.assertEqual(fake.streams, ["task-a:2"])

    def test_api_remains_the_default_backend(self):
        sentinel = object()
        with (
            mock.patch(
                "benchmark.runner.get_env_llm_client", return_value=sentinel
            ) as api,
            mock.patch("benchmark.runner.TransformersLLMClient") as local,
        ):
            runner = BenchmarkRunner(model="api-model")
            self.assertIs(runner.llm_client, sentinel)
            api.assert_called_once_with()
            local.assert_not_called()

    def test_transformers_backend_loads_the_local_model_without_api_credentials(self):
        sentinel = object()
        with (
            mock.patch("benchmark.runner.get_env_llm_client") as api,
            mock.patch(
                "benchmark.runner.TransformersLLMClient", return_value=sentinel
            ) as local,
        ):
            runner = BenchmarkRunner(
                model="/models/qwen",
                llm_backend="transformers",
                local_device="cuda",
                local_dtype="bfloat16",
                generation_seed=7,
            )
            self.assertIs(runner.llm_client, sentinel)
            local.assert_called_once_with(
                model="/models/qwen",
                device="cuda",
                dtype="bfloat16",
                seed=7,
            )
            api.assert_not_called()

    def test_unknown_backend_fails_before_any_model_is_loaded(self):
        with self.assertRaises(ValueError):
            BenchmarkRunner(llm_backend="unknown")


if __name__ == "__main__":
    unittest.main()
