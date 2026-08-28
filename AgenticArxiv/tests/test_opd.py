#!/usr/bin/env python3
"""OPD 训练脚本的纯逻辑单测（不加载真模型、不需要 GPU）。

覆盖四类「静默失败」防线：
1. GKD 导入门控（TRL 把 GKDTrainer 搬过家，导不上来要在启动时报错）；
2. beta 的 KL 方向（beta=1 必须是 reverse-KL D_KL(π_student‖π_teacher)，
   这是 OPD 的方向；将来 TRL 若翻转语义，这里会响）；
3. 数据集适配（chat template 渲染必须与 GRPO rollout 一致）；
4. 配置过滤 / 路径解析 / 长度体检等纯函数。

运行：
    cd AgenticArxiv && python tests/test_opd.py
    （需要已安装 trl / transformers / torch 的环境，数值用例缺 trl 时自动跳过）
"""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
os.environ.setdefault("STORE_BACKEND", "memory")

import torch  # noqa: E402

import rl.train_opd as opd  # noqa: E402
from benchmark.tasks import get_all_tasks  # noqa: E402


def _trl_available() -> bool:
    try:
        opd._import_gkd()
        return True
    except SystemExit:
        return False


class _FakeTokenizer:
    """apply_chat_template 记录调用参数；文本按 4 字符 1 token 近似。"""

    def __init__(self, fail_template: bool = False):
        self.fail_template = fail_template
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if self.fail_template:
            raise ValueError("no chat template")
        self.calls.append({"messages": messages, "add_generation_prompt": add_generation_prompt})
        return f"<tpl>{messages[0]['content']}"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1] * (len(text) // 4 + 1)}


class _ModulePatch:
    """临时替换 sys.modules 里的导入键，模拟 TRL 各版本的 GKD 位置。"""

    def __init__(self, **modules):
        self.modules = modules
        self._saved = {}

    def __enter__(self):
        for name, mod in self.modules.items():
            self._saved[name] = sys.modules.get(name, "__absent__")
            sys.modules[name] = mod
        return self

    def __exit__(self, *exc):
        for name, old in self._saved.items():
            if old == "__absent__":
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        return False


class ImportGateTest(unittest.TestCase):
    """GKD 导入门控：优先 experimental 路径，回退顶层，全缺时响亮报错。"""

    def test_prefers_experimental_path(self):
        fake = SimpleNamespace(GKDConfig="cfg", GKDTrainer="trainer")
        with _ModulePatch(**{"trl.experimental.gkd": fake}):
            cfg, trainer = opd._import_gkd()
        self.assertEqual((cfg, trainer), ("cfg", "trainer"))

    def test_falls_back_to_top_level(self):
        fake_trl = SimpleNamespace(GKDConfig="cfg", GKDTrainer="trainer")
        with _ModulePatch(**{
            "trl.experimental.gkd": None,  # None 在 sys.modules 里 ⇒ ImportError
            "trl": fake_trl,
        }):
            cfg, trainer = opd._import_gkd()
        self.assertEqual((cfg, trainer), ("cfg", "trainer"))

    def test_missing_everywhere_raises_system_exit(self):
        with _ModulePatch(**{"trl.experimental.gkd": None, "trl": None}):
            with self.assertRaises(SystemExit):
                opd._import_gkd()


class FilterCfgKwargsTest(unittest.TestCase):
    def test_drops_unknown_fields_and_keeps_order_irrelevant(self):
        cfg, dropped = opd._filter_cfg_kwargs({"a": 1, "b": 2, "c": 3}, {"a", "b"})
        self.assertEqual(cfg, {"a": 1, "b": 2})
        self.assertEqual(dropped, ["c"])

    def test_no_valid_fields_drops_everything(self):
        cfg, dropped = opd._filter_cfg_kwargs({"a": 1}, set())
        self.assertEqual(cfg, {})
        self.assertEqual(dropped, ["a"])


class ResolveModelPathTest(unittest.TestCase):
    def test_existing_local_path_resolved(self):
        resolved = opd._resolve_model_path("AgenticArxiv")
        self.assertTrue(resolved.endswith("AgenticArxiv"))

    def test_hf_repo_name_passthrough(self):
        self.assertEqual(opd._resolve_model_path("Qwen/Qwen2.5-7B-Instruct"), "Qwen/Qwen2.5-7B-Instruct")

    def test_missing_local_path_fails_loudly(self):
        with self.assertRaises(SystemExit):
            opd._resolve_model_path("outputs/definitely-not-here")
        with self.assertRaises(SystemExit):
            opd._resolve_model_path("./outputs/definitely-not-here")


class LoadTasksTest(unittest.TestCase):
    def test_default_is_the_smoke_set(self):
        self.assertEqual([t["id"] for t in opd._load_tasks("default")], [t["id"] for t in get_all_tasks()])

    def test_expanded_is_the_full_pool(self):
        self.assertGreater(len(opd._load_tasks("expanded")), len(get_all_tasks()))


class BuildOpdDatasetTest(unittest.TestCase):
    def test_rows_render_with_generation_prompt(self):
        tokenizer = _FakeTokenizer()
        tasks = get_all_tasks()[:2]
        dataset, max_prompt_tokens = opd.build_opd_dataset(tasks, tokenizer)

        self.assertEqual(len(dataset), len(tasks))
        self.assertTrue(all(c["add_generation_prompt"] for c in tokenizer.calls))
        row = dataset[0]
        self.assertTrue(row["prompt"].startswith("<tpl>"))
        self.assertTrue(row["input_ids"])
        self.assertTrue(row["task_id"])
        self.assertGreater(max_prompt_tokens, 0)
        self.assertEqual(row["input_ids"], [1] * (len(row["prompt"]) // 4 + 1))

    def test_template_failure_fails_loudly(self):
        with self.assertRaises(SystemExit):
            opd.build_opd_dataset(get_all_tasks()[:1], _FakeTokenizer(fail_template=True))


class TeacherLoadKwargsTest(unittest.TestCase):
    def test_bf16_on_cuda_precision(self):
        with patch.object(opd, "_precision_flags", return_value={"bf16": True}):
            kwargs = opd._teacher_load_kwargs()
        self.assertEqual(list(kwargs.values()), [torch.bfloat16])
        self.assertIn(next(iter(kwargs)), ("dtype", "torch_dtype"))

    def test_fp32_elsewhere(self):
        with patch.object(opd, "_precision_flags", return_value={}):
            self.assertEqual(opd._teacher_load_kwargs(), {})


class GoldActionTokensTest(unittest.TestCase):
    def test_longest_tool_name_wins(self):
        tokenizer = SimpleNamespace(__call__=lambda self, text, **kw: {"input_ids": list(range(len(text)))})
        tasks = [{"expected_tools": ["short", "a_much_longer_tool_name"]}]
        need = opd._gold_action_tokens(tokenizer, tasks)
        longest = "a_much_longer_tool_name"
        expected = len(f'Thought: xxx\nAction: {{"name": "{longest}", "args": {{}}}}')
        self.assertEqual(need, expected)

    def test_tasks_without_tools_yield_zero(self):
        tokenizer = SimpleNamespace(__call__=lambda self, text, **kw: {"input_ids": list(range(len(text)))})
        self.assertEqual(opd._gold_action_tokens(tokenizer, [{"expected_tools": []}]), 0)


@unittest.skipUnless(_trl_available(), "需要可导入的 TRL GKD 实现做数值对照")
class ReverseKLDirectionTest(unittest.TestCase):
    """把「beta=1 是 reverse-KL」钉成数值事实。

    OPD 要的是 D_KL(π_student‖π_teacher)（mode-seeking，学生分布被拉向教师
    在学生自身高概率动作上的行为）。若将来 TRL 翻转 beta 语义，这两个用例
    会替运行环境把方向重新验一遍。
    """

    @classmethod
    def setUpClass(cls):
        _, cls.gkd_trainer = opd._import_gkd()
        cls.loss = cls.gkd_trainer.generalized_jsd_loss

    def _hand_kl(self, p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
        log_p = torch.log_softmax(p_logits, dim=-1)
        log_q = torch.log_softmax(q_logits, dim=-1)
        return float((log_p.exp() * (log_p - log_q)).sum())

    def test_beta_1_is_reverse_kl_student_first(self):
        student = torch.tensor([[[2.0, 0.0, -1.0, 0.5]]])
        teacher = torch.tensor([[[0.0, 1.0, 3.0, -0.5]]])
        loss = float(self.loss(student, teacher, beta=1.0))
        self.assertAlmostEqual(loss, self._hand_kl(student, teacher), places=5)

    def test_beta_0_is_the_opposite_direction(self):
        student = torch.tensor([[[2.0, 0.0, -1.0, 0.5]]])
        teacher = torch.tensor([[[0.0, 1.0, 3.0, -0.5]]])
        loss_beta0 = float(self.loss(student, teacher, beta=0.0))
        self.assertAlmostEqual(loss_beta0, self._hand_kl(teacher, student), places=5)
        # 方向不同 ⇒ 两个损失不相等；相等说明语义被翻转，OPD 方向失效
        self.assertNotAlmostEqual(
            loss_beta0, self._hand_kl(student, teacher), places=5,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
