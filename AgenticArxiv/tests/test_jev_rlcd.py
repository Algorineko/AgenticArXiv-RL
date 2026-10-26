"""RLCD 判别头纯函数单测（无 GPU、无模型加载，CI 裸跑）。

覆盖六组：
1. 对比损失方向性（∂L/∂z_gold < 0、∂L/∂z_neg > 0）；
2. 候选打分两条路径（cache / batch）数值一致 + z 聚合；
3. 后缀 batch 组装与 tokenize round-trip；
4. prompt 形状与 state 截断（不依赖 gold 字段）；
5. cache_layers 三代结构探测 + expand_cache；
6. 数据集合并/防泄漏纯函数。

``scripts/build_jev_rlcd_dataset.py`` 通过 importlib 以文件路径加载
（scripts/ 不是包）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
for path in (str(PACKAGE_ROOT), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from rl import jev_scoring  # noqa: E402
from routing.local_prompt import (  # noqa: E402
    CANDIDATE_SUFFIX_TEMPLATE,
    bounded_state,
    build_router_prompt,
    candidate_names,
    candidate_suffix_texts,
)


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_jev_rlcd_dataset",
        REPO_ROOT / "scripts" / "build_jev_rlcd_dataset.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContrastiveLossDirection(unittest.TestCase):
    """RLCD 对比项必须「gold 涨、neg 压」，方向由梯度符号验证。"""

    def test_gradient_direction(self):
        z = torch.tensor([0.3, 0.9, -0.2], requires_grad=True)
        loss = jev_scoring.router_contrastive_loss(
            z, beta=1.0, gold_idx=0, negative_idxs=[1, 2], margin=0.0
        )
        loss.backward()
        self.assertLess(float(z.grad[0]), 0.0, "gold 方向梯度应为负（推高 gold）")
        self.assertGreater(float(z.grad[1]), 0.0, "负样本方向梯度应为正（压低负样本）")
        self.assertGreater(float(z.grad[2]), 0.0)

    def test_loss_decreases_when_gap_grows(self):
        close = jev_scoring.router_contrastive_loss(
            torch.tensor([0.5, 0.4, 0.0]), 1.0, 0, [1], 0.0
        )
        apart = jev_scoring.router_contrastive_loss(
            torch.tensor([2.0, -1.0, 0.0]), 1.0, 0, [1], 0.0
        )
        self.assertLess(float(apart), float(close))

    def test_no_negatives_returns_zero(self):
        z = torch.zeros(11)
        loss = jev_scoring.router_contrastive_loss(z, 1.0, 0, [], 0.0)
        self.assertEqual(float(loss), 0.0)

    def test_ce_prefers_gold(self):
        z_good = torch.tensor([2.0, 0.1, -1.0])
        z_bad = torch.tensor([-1.0, 2.0, 0.1])
        self.assertLess(
            float(jev_scoring.router_ce_loss(z_good, 0.5, 0)),
            float(jev_scoring.router_ce_loss(z_bad, 0.5, 0)),
        )


class _FakeOut:
    def __init__(self, hidden, cache=None):
        self.last_hidden_state = hidden
        self.past_key_values = cache


class _FakeCache:
    """只实现 to_legacy_cache 的最小 cache（被 expand_cache 消费）。"""

    def __init__(self):
        self._layers = []

    def append(self, key, value):
        self._layers.append((key, value))

    def to_legacy_cache(self):
        return list(self._layers)


class _FakeTransformer(torch.nn.Module):
    """h[i] = emb[t_i] + 0.01·Σ_{j≤i} emb[t_j] 的确定性注意力替身。

    cache 路径从展开后的 cache 取 prompt 累计和，与 batch 路径对同一
    绝对位置必须算出相同 hidden——这正是两条打分路径一致性的检验。
    """

    def __init__(self, hidden: int = 4, vocab: int = 7):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        with torch.no_grad():
            grid = torch.arange(vocab * hidden, dtype=torch.float32)
            self.emb.weight.copy_(grid.reshape(vocab, hidden) / 50.0)

    def forward(self, input_ids=None, attention_mask=None,
                past_key_values=None, use_cache=False, **kwargs):
        hidden = self.emb(input_ids)
        # prefix 保持与 hidden 同维（(B,1,H)），防止 3-D+4-D 广播升维
        prefix = torch.zeros(hidden.shape[0], 1, hidden.shape[-1])
        if past_key_values is not None:
            layers = jev_scoring.cache_layers(past_key_values)
            prefix = layers[0][1][:, :, -1, :]
        cumulative = torch.cumsum(hidden, dim=1) + prefix
        out = hidden + 0.01 * cumulative
        cache = None
        if use_cache:
            cache = _FakeCache()
            cache.append(cumulative[:, None, :, :], cumulative[:, None, :, :])
        return _FakeOut(out, cache)


class _FakeLMModel(torch.nn.Module):
    def __init__(self, hidden: int = 4, vocab: int = 7):
        super().__init__()
        self.transformer = _FakeTransformer(hidden, vocab)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(
                torch.arange(vocab * hidden, dtype=torch.float32)
                .reshape(vocab, hidden) / 30.0
            )

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def get_output_embeddings(self):
        return self.head


class CandidateScoringConsistency(unittest.TestCase):
    """cache 路径与 batch 路径必须对同一决策点给出相同 z。"""

    def setUp(self):
        names = ["alpha_tool", "beta_tool", "FINISH"]
        self.suffix_texts = candidate_suffix_texts(names)
        self.suffix_ids = jev_scoring.tokenize_suffixes(
            _StubTokenizer(self.suffix_texts), self.suffix_texts
        )
        # fake 词表必须覆盖桩 tokenizer 的全部 id 与 prompt id
        max_id = max(
            max(ids) for ids in self.suffix_ids
        )
        self.model = _FakeLMModel(vocab=max_id + 2)

    def test_cache_and_batch_paths_agree(self):
        prompt_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        suffix_ids, suffix_lens = jev_scoring.build_suffix_batch(
            self.suffix_ids, torch.device("cpu")
        )
        z_cache = jev_scoring.score_candidates(
            self.model, prompt_ids, suffix_ids, suffix_lens,
            agg="mean", scoring="cache",
        )
        z_batch = jev_scoring.score_candidates(
            self.model, prompt_ids, suffix_ids, suffix_lens,
            agg="mean", scoring="batch",
        )
        self.assertEqual(z_cache.shape, (3,))
        torch.testing.assert_close(z_cache, z_batch, rtol=1e-5, atol=1e-5)

    def test_sum_aggregation_scales_with_length(self):
        prompt_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        suffix_ids, suffix_lens = jev_scoring.build_suffix_batch(
            self.suffix_ids, torch.device("cpu")
        )
        z_mean = jev_scoring.score_candidates(
            self.model, prompt_ids, suffix_ids, suffix_lens,
            agg="mean", scoring="batch",
        )
        z_sum = jev_scoring.score_candidates(
            self.model, prompt_ids, suffix_ids, suffix_lens,
            agg="sum", scoring="batch",
        )
        for i, length in enumerate(suffix_lens):
            self.assertAlmostEqual(
                float(z_sum[i]), float(z_mean[i]) * length, places=5
            )

    def test_name_span_scoring_paths_agree(self):
        prefix = "Next tool: "
        starts = [len(prefix)] * len(self.suffix_texts)
        prompt_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        suffix_ids, suffix_lens = jev_scoring.build_suffix_batch(
            self.suffix_ids, torch.device("cpu")
        )
        z_cache = jev_scoring.score_candidates(
            self.model, prompt_ids, suffix_ids, suffix_lens,
            agg="mean", scoring="cache", name_starts=starts,
        )
        z_batch = jev_scoring.score_candidates(
            self.model, prompt_ids, suffix_ids, suffix_lens,
            agg="mean", scoring="batch", name_starts=starts,
        )
        torch.testing.assert_close(z_cache, z_batch, rtol=1e-5, atol=1e-5)
        # span 聚合的每行 token 数 = len - start，sum 版按比例可查
        z_sum = jev_scoring.score_candidates(
            self.model, prompt_ids, suffix_ids, suffix_lens,
            agg="sum", scoring="batch", name_starts=starts,
        )
        for i, length in enumerate(suffix_lens):
            self.assertAlmostEqual(
                float(z_sum[i]), float(z_batch[i]) * (length - starts[i]), places=5
            )

    def test_name_token_starts_char_tokenizer(self):
        tokenizer = _StubTokenizer(self.suffix_texts)
        starts = jev_scoring.name_token_starts(
            tokenizer, self.suffix_texts, "Next tool: "
        )
        # 字符级 tokenizer 有精确边界：start = len("Next tool: ")
        self.assertEqual(set(starts), {len("Next tool: ")})

    def test_decision_distribution_sums_to_one(self):
        z = torch.tensor([0.2, 0.8, -0.1])
        selected, confidence, probabilities = jev_scoring.decision_from_z(
            z, ["a", "b", "FINISH"], tau=0.5
        )
        self.assertEqual(selected, "b")
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)
        self.assertAlmostEqual(confidence, probabilities["b"], places=6)


class _StubTokenizer:
    """词表 = 文本本身的最小 tokenizer（round-trip 恒真）。"""

    def __init__(self, texts):
        self.vocab = {}
        for text in texts:
            for ch in text:
                self.vocab.setdefault(ch, len(self.vocab) + 1)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.vocab[ch] for ch in text]}

    def decode(self, ids):
        inverse = {i: ch for ch, i in self.vocab.items()}
        return "".join(inverse[int(i)] for i in ids)


class SuffixBatchAndRoundTrip(unittest.TestCase):
    def test_round_trip_rejects_boundary_merge(self):
        tokenizer = _StubTokenizer(["Next tool: alpha_tool"])

        class Merging(_StubTokenizer):
            def __call__(self, text, add_special_tokens=False):
                # 模拟 BPE 把尾字符并进前一 token
                ids = super().__call__(text)["input_ids"]
                return {"input_ids": ids[:-1]}

            def decode(self, ids):
                inverse = {i: ch for ch, i in self.vocab.items()}
                return "".join(inverse[int(i)] for i in ids)

        with self.assertRaises(ValueError):
            jev_scoring.tokenize_suffixes(
                Merging("Next tool: alpha_tool"), ["Next tool: alpha_tool"]
            )

    def test_build_suffix_batch_right_pads(self):
        ids = [[1, 2, 3], [4, 5], [6]]
        batch, lens = jev_scoring.build_suffix_batch(ids, torch.device("cpu"))
        self.assertEqual(batch.shape, (3, 3))
        self.assertEqual(lens, [3, 2, 1])
        self.assertEqual(batch[1].tolist(), [4, 5, 0])
        self.assertEqual(batch[2].tolist(), [6, 0, 0])


class PromptLayer(unittest.TestCase):
    def setUp(self):
        self.tools = [
            {"name": "search_arxiv_papers", "description": "关键词检索"},
            {"name": "download_arxiv_pdf", "description": "下载 PDF"},
        ]

    def test_prompt_contains_candidates_and_not_gold_fields(self):
        prompt = build_router_prompt(
            task="下载第 1 篇论文",
            state_text="[任务开始前的会话状态]\n- 空会话",
            tools=self.tools,
        )
        for needle in (
            "下载第 1 篇论文",
            "search_arxiv_papers",
            "download_arxiv_pdf",
            "FINISH",
            CANDIDATE_SUFFIX_TEMPLATE.format(name="<name from the candidate list>"),
        ):
            self.assertIn(needle, prompt)
        for banned in ("expected_tools", "expected_tool_args", "gold_tool"):
            self.assertNotIn(banned, prompt)

    def test_candidate_names_appends_finish_last(self):
        names = candidate_names(self.tools)
        self.assertEqual(names, ["search_arxiv_papers", "download_arxiv_pdf", "FINISH"])

    def test_bounded_state_tail_kept(self):
        state = "x" * 200 + "TAIL"
        bounded = bounded_state(state, max_chars=100)
        self.assertTrue(bounded.startswith("[earlier context truncated]"))
        self.assertTrue(bounded.endswith("TAIL"))
        self.assertEqual(len(bounded), len("[earlier context truncated]\n") + 100)

    def test_visible_context_empty_setup_has_no_history(self):
        builder = _load_builder()
        task = {
            "id": "t1",
            "task": "任务",
            "setup": [],
            "expected_tools": ["download_arxiv_pdf"],
            "category": "download",
        }
        context = builder.visible_context(task, papers=[])
        self.assertNotIn("Action:", context)
        self.assertNotIn("Observation:", context)
        self.assertNotIn("download_arxiv_pdf", context)


class CacheLayerProbing(unittest.TestCase):
    def test_legacy_cache(self):
        cache = types.SimpleNamespace(
            to_legacy_cache=lambda: [("k1", "v1"), ("k2", "v2")]
        )
        self.assertEqual(
            jev_scoring.cache_layers(cache), [("k1", "v1"), ("k2", "v2")]
        )

    def test_key_value_lists(self):
        cache = types.SimpleNamespace(
            key_cache=["k1"], value_cache=["v1"]
        )
        self.assertEqual(jev_scoring.cache_layers(cache), [("k1", "v1")])

    def test_layers_attribute(self):
        cache = types.SimpleNamespace(
            layers=[types.SimpleNamespace(keys="k", values="v")]
        )
        self.assertEqual(jev_scoring.cache_layers(cache), [("k", "v")])

    def test_unknown_structure_raises_loudly(self):
        with self.assertRaises(RuntimeError):
            jev_scoring.cache_layers(types.SimpleNamespace())

    def test_expand_cache_replicates_batch(self):
        from transformers.cache_utils import DynamicCache

        source = DynamicCache()
        k = torch.ones(1, 2, 3, 4)
        v = torch.full((1, 2, 3, 4), 2.0)
        source.update(k, v, 0)
        source.update(k * 3, v * 3, 1)
        expanded = jev_scoring.expand_cache(source, batch=5)
        layers = jev_scoring.cache_layers(expanded)
        self.assertEqual(len(layers), 2)
        self.assertEqual(layers[0][0].shape, (5, 2, 3, 4))
        self.assertTrue(torch.all(layers[0][1] == 2.0))
        self.assertTrue(torch.all(layers[1][0] == 3.0))


class BuilderPureFunctions(unittest.TestCase):
    def setUp(self):
        self.builder = _load_builder()

    def test_merge_attaches_negatives_to_positive_row(self):
        pos = {
            "decision_id": "d1", "source": "policy_success",
            "task_id": "t1", "state_text": "S", "gold_tool": "FINISH",
            "gold_args": {}, "negative_tools": [], "prefix_clean": True,
        }
        neg = {
            "decision_id": "d2", "source": "policy_failure_neg",
            "task_id": "t1", "state_text": "S", "gold_tool": "FINISH",
            "gold_args": {}, "negative_tools": ["download_arxiv_pdf"],
            "negative_provenance": [], "prefix_clean": True,
            "error_pattern": "infeasible_called_tool",
        }
        merged = self.builder.merge_rows([pos], [neg])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["gold_tool"], "FINISH")
        self.assertEqual(
            merged[0]["negative_tools"], ["download_arxiv_pdf"]
        )
        self.assertEqual(
            merged[0]["error_pattern"], "infeasible_called_tool"
        )

    def test_merge_caps_negatives_and_excludes_gold(self):
        rows = []
        for tool in ("a", "b", "c", "d"):
            rows.append(
                {
                    "decision_id": f"n{tool}", "source": "policy_failure_neg",
                    "task_id": "t1", "state_text": "S", "gold_tool": "FINISH",
                    "gold_args": {}, "negative_tools": [tool],
                    "negative_provenance": [], "prefix_clean": True,
                    "error_pattern": "other",
                }
            )
        pos = {
            "decision_id": "d1", "source": "policy_success",
            "task_id": "t1", "state_text": "S", "gold_tool": "FINISH",
            "gold_args": {}, "negative_tools": [], "prefix_clean": True,
        }
        merged = self.builder.merge_rows([pos], rows)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["negative_tools"]), 2)
        self.assertNotIn("FINISH", merged[0]["negative_tools"])

    def test_negatives_only_state_is_kept_with_gold_label(self):
        neg = {
            "decision_id": "d2", "source": "policy_failure_neg",
            "task_id": "t9", "state_text": "S9", "gold_tool": "search_arxiv_papers",
            "gold_args": {}, "negative_tools": ["download_arxiv_pdf"],
            "negative_provenance": [], "prefix_clean": True,
            "error_pattern": "title_ref_wrong_search",
        }
        merged = self.builder.merge_rows([], [neg])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["gold_tool"], "search_arxiv_papers")
        self.assertEqual(merged[0]["negative_tools"], ["download_arxiv_pdf"])

    def test_error_patterns(self):
        classify = self.builder.classify_error_pattern
        infeasible = {"expected_tools": [], "task": "下载不存在的第 20 篇"}
        self.assertEqual(
            classify(infeasible, "FINISH", "download_arxiv_pdf"),
            "infeasible_called_tool",
        )
        ref_task = {
            "expected_tools": ["download_arxiv_pdf"],
            "task": "下载标题为《Example》的论文",
        }
        self.assertEqual(
            classify(ref_task, "download_arxiv_pdf", "search_arxiv_papers"),
            "title_ref_wrong_search",
        )
        recent = {
            "expected_tools": ["get_recently_submitted_cs_papers"],
            "task": "检索最近 7 天 cs.AI 论文",
        }
        self.assertEqual(
            classify(recent, "get_recently_submitted_cs_papers",
                     "search_arxiv_papers"),
            "recent_vs_keyword",
        )

    def test_leak_assertion_rejects_holdout_task_in_train(self):
        task_split = {"t1": "train", "t2": "iid_test", "t3": "dev",
                      "t4": "ood_test"}
        row = {
            "task_id": "t2", "split": "train",
            "state_text": "s", "gold_tool": "FINISH",
        }
        with self.assertRaises(SystemExit):
            self.builder.leak_assertions([row], task_split)

    def test_decision_id_is_stable(self):
        a = self.builder.decision_id("state", "task", 0, "gold_replay")
        b = self.builder.decision_id("state", "task", 0, "gold_replay")
        c = self.builder.decision_id("state", "task", 1, "gold_replay")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_gold_rows_end_with_finish(self):
        """gold 链的决策点数 = len(expected_tools) + 1（链尾 FINISH）。"""
        # 该行为由 DatasetBuilder.gold_rows 的循环结构保证，这里用
        # 离线快照做一次真实小回放（snapshot 是仓库冻结资产，CPU 可跑）。
        builder = self.builder.DatasetBuilder(
            snapshot_path=REPO_ROOT / "data" / "mock_arxiv_snapshot.json",
            split_file=REPO_ROOT / "data" / "splits" / "v3_81.json",
            max_state_chars=12000,
            replay_check=True,
        )
        task = builder.tasks_by_id["search_AI_1d_3"]
        rows = builder.gold_rows(task)
        expected = task["expected_tools"]
        self.assertEqual(len(rows), len(expected) + 1)
        self.assertEqual([r["gold_tool"] for r in rows], expected + ["FINISH"])
        self.assertTrue(
            all(json.dumps(r["state_text"], ensure_ascii=False) for r in rows)
        )
        self.assertNotIn("expected_tools", rows[0]["state_text"])


if __name__ == "__main__":
    unittest.main()
