"""训练脚本的两个「把静默失败变成响亮失败」的守卫。

两个失败模式都不会让训练崩：跑完、保存 checkpoint、日志看着正常，
只是模型什么都没学到。
"""

import unittest
import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from benchmark.splits import load_split
from benchmark.tasks import get_all_tasks
from benchmark.tasks_expanded import get_expanded_tasks
from rl.precision import (
    assert_quantization_is_single_process,
    is_distributed_launch,
    pin_single_gpu,
    precision_flags,
)
from rl.train_grpo import (
    DAPO_PRESET,
    DEFAULT_BETA,
    RewardVarianceGuard,
    _build_reward_calculator,
    _load_tasks,
    resolve_dapo_options,
)
from rl.train_sft import (
    QLORA_TARGET_MODULES,
    _assert_lora_only_trainable,
    _check_lengths,
    _filter_dataclass_kwargs,
    _messages_of,
    _sha256_file,
    _to_prompt_completion,
    _verify_data_manifest,
)


class ConfigCompatibilityTest(unittest.TestCase):
    def test_filters_fields_missing_from_installed_dataclass_version(self):
        @dataclass
        class FakeConfig:
            output_dir: str
            max_length: int = 1024

        filtered, dropped = _filter_dataclass_kwargs(
            FakeConfig,
            {"output_dir": "out", "max_length": 4096, "use_cache": False},
        )

        self.assertEqual(filtered, {"output_dir": "out", "max_length": 4096})
        self.assertEqual(dropped, ["use_cache"])


class RewardCurriculumConfigTest(unittest.TestCase):
    def test_zero_steps_enables_full_semantic_weights_immediately(self):
        weights = _build_reward_calculator(0).schedule(training_step=0)
        self.assertEqual(
            (weights.format, weights.tool, weights.argument, weights.process, weights.outcome),
            (1.0, 3.0, 2.0, 1.0, 3.0),
        )

    def test_default_probe_schedule_keeps_early_scaled_weights(self):
        weights = _build_reward_calculator(30).schedule(training_step=0)
        self.assertEqual(
            (weights.format, weights.tool, weights.argument, weights.process, weights.outcome),
            (1.0, 1.0, 2.0 / 3.0, 1.0, 1.0),
        )

    def test_negative_curriculum_steps_fail_fast(self):
        with self.assertRaisesRegex(SystemExit, "不能为负数"):
            _build_reward_calculator(-1)


class GrpoV4SplitTest(unittest.TestCase):
    def test_formal_rl_split_is_train_only_and_matches_frozen_selection(self):
        from pathlib import Path
        import json

        repo = Path(__file__).resolve().parents[2]
        v2 = json.loads((repo / "data/splits/v2_62.json").read_text(encoding="utf-8"))
        v4_path = repo / "data/splits/v4_grpo_train.json"
        v4 = json.loads(v4_path.read_text(encoding="utf-8"))
        selected = set(load_split(f"{v4_path}:rl_train"))

        self.assertEqual(len(selected), 7)
        self.assertTrue(selected <= set(v2["split"]["train"]))
        heldout = set(v2["split"]["dev"] + v2["split"]["iid_test"] + v2["split"]["ood_test"])
        self.assertFalse(selected & heldout)
        self.assertEqual(
            v4["split"]["excluded_low_variance"],
            ["chain_lg10_dl_tr_last"],
        )
        for task_id in selected:
            self.assertGreaterEqual(
                v4["audit"][task_id]["informative_group_fraction"], 0.375
            )


class _FakeTokenizer:
    """按字符数近似 token 数，够用来测长度守卫的分支。"""

    def apply_chat_template(self, messages, tokenize=False):
        return "".join(m.get("content", "") for m in messages)

    def __call__(self, text):
        return {"input_ids": list(range(len(text)))}


def _row(n_prompt, n_completion):
    return {
        "prompt": [{"role": "user", "content": "p" * n_prompt}],
        "completion": [{"role": "assistant", "content": "c" * n_completion}],
    }


class MessagesOfTest(unittest.TestCase):
    def test_prompt_completion_format(self):
        self.assertEqual(len(_messages_of(_row(3, 2))), 2)

    def test_messages_format(self):
        row = {"messages": [{"role": "user", "content": "x"}]}
        self.assertEqual(_messages_of(row), row["messages"])


class PromptCompletionConversionTest(unittest.TestCase):
    def test_splits_last_assistant_message_from_prompt(self):
        row = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": '{"tool": "search"}'},
            ]
        }

        converted = _to_prompt_completion(row)

        self.assertEqual([m["role"] for m in converted["prompt"]], ["system", "user"])
        self.assertEqual(converted["completion"], [row["messages"][-1]])

    def test_rejects_sample_without_final_assistant_message(self):
        row = {"messages": [{"role": "user", "content": "task"}]}
        with self.assertRaisesRegex(ValueError, "assistant"):
            _to_prompt_completion(row)


class SftRowLoadingTest(unittest.TestCase):
    """审计列不能把训练数据加载搞崩。

    参数化 SFT 数据每行都带 `generation_parameters` 之类的溯源结构，它们的
    形状逐行不同（检索行没有 section/style/max_words，解读行有）。`datasets`
    为整个文件推断一套 Arrow schema，一行多一个键就会让整个数据集 cast 失败
    —— 曾经就是这样：2436 条数据一条都加载不进来。
    """

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "rows.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, rows):
        import json as _json

        self.path.write_text(
            "\n".join(_json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )

    def test_keeps_only_the_training_columns(self):
        from rl.train_sft import _load_sft_rows

        self._write([{
            "messages": [{"role": "user", "content": "t"}],
            "generation_parameters": {"aspect": "AI"},
            "sample_sha256": "abc",
        }])

        rows = _load_sft_rows(self.path)

        self.assertEqual(rows, [{"messages": [{"role": "user", "content": "t"}]}])

    def test_rows_with_different_provenance_still_load(self):
        from rl.train_sft import _load_sft_rows

        self._write([
            {"messages": [{"role": "user", "content": "a"}],
             "generation_parameters": {"aspect": "AI", "ref": 1}},
            {"messages": [{"role": "user", "content": "b"}],
             "generation_parameters": {"aspect": "CV", "ref": 2, "style": "tldr",
                                       "max_words": 60, "section": None}},
        ])

        rows = _load_sft_rows(self.path)

        self.assertEqual(len(rows), 2)
        self.assertTrue(all(set(r) == {"messages"} for r in rows))

    def test_prompt_completion_rows_pass_through(self):
        from rl.train_sft import _load_sft_rows

        self._write([{
            "prompt": [{"role": "user", "content": "t"}],
            "completion": [{"role": "assistant", "content": "a"}],
            "task_id": "x",
        }])

        rows = _load_sft_rows(self.path)

        self.assertEqual(set(rows[0]), {"prompt", "completion"})

    def test_blank_lines_are_ignored_and_empty_input_fails(self):
        from rl.train_sft import _load_sft_rows

        self.path.write_text("\n\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            _load_sft_rows(self.path)


class CheckLengthsTest(unittest.TestCase):
    def test_passes_when_everything_fits(self):
        _check_lengths(_FakeTokenizer(), [_row(10, 5)] * 4, max_length=100)

    def test_aborts_when_most_samples_overflow(self):
        # 这是实测到的真实情形：207/207 超过 max_length=1024
        with self.assertRaises(SystemExit) as ctx:
            _check_lengths(_FakeTokenizer(), [_row(900, 80)] * 10, max_length=100)
        self.assertIn("max_length", str(ctx.exception))

    def test_suggests_a_length_that_actually_fits(self):
        with self.assertRaises(SystemExit) as ctx:
            _check_lengths(_FakeTokenizer(), [_row(200, 50)] * 10, max_length=100)
        self.assertIn("314", str(ctx.exception))     # max 250 + 64

    def test_tolerates_a_single_outlier(self):
        rows = [_row(10, 5)] * 200 + [_row(900, 80)]
        _check_lengths(_FakeTokenizer(), rows, max_length=100)   # <=1% 只告警

    def test_empty_dataset_aborts(self):
        with self.assertRaises(SystemExit):
            _check_lengths(_FakeTokenizer(), [], max_length=100)


class SftDataManifestTest(unittest.TestCase):
    def test_accepts_matching_frozen_mix(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "train.jsonl"
            data.write_text('{"messages": []}\n', encoding="utf-8")
            manifest = data.with_suffix(data.suffix + ".manifest.json")
            manifest.write_text(json.dumps({
                "kind": "qlora_sft_train_mix",
                "output_sha256": _sha256_file(data),
                "output_rows": 1,
                "unique_sample_fingerprints": 1,
                "semantic_task_instances": 1,
            }), encoding="utf-8")
            self.assertEqual(_verify_data_manifest(data)["output_rows"], 1)

    def test_rejects_data_changed_after_manifest(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "train.jsonl"
            data.write_text('{"messages": []}\n', encoding="utf-8")
            manifest = data.with_suffix(data.suffix + ".manifest.json")
            manifest.write_text(json.dumps({
                "kind": "qlora_sft_train_mix",
                "output_sha256": _sha256_file(data),
                "output_rows": 1,
                "unique_sample_fingerprints": 1,
            }), encoding="utf-8")
            data.write_text('{"messages": [1]}\n', encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "SHA256"):
                _verify_data_manifest(data)


class QLoRAGuardTest(unittest.TestCase):
    def test_qwen_attention_and_mlp_projections_are_targeted(self):
        self.assertEqual(
            set(QLORA_TARGET_MODULES),
            {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
        )

    def test_rejects_model_that_is_not_actually_4bit(self):
        import torch

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_weight = torch.nn.Parameter(torch.ones(1))
                self.is_loaded_in_4bit = False

        with self.assertRaisesRegex(SystemExit, "4-bit"):
            _assert_lora_only_trainable(FakeModel())

    def test_rejects_non_lora_trainable_parameter(self):
        import torch

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base_weight = torch.nn.Parameter(torch.ones(1))
                self.is_loaded_in_4bit = True

        with self.assertRaisesRegex(SystemExit, "非LoRA"):
            _assert_lora_only_trainable(FakeModel())


class BooleanFlagContractTest(unittest.TestCase):
    def test_verify_boolean_optional_action_accepts_both_forms(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--verify", action=argparse.BooleanOptionalAction, default=False
        )
        self.assertTrue(parser.parse_args(["--verify"]).verify)
        self.assertFalse(parser.parse_args(["--no-verify"]).verify)
        self.assertFalse(parser.parse_args([]).verify)


class RewardVarianceGuardTest(unittest.TestCase):
    def _fire(self, guard, n, step=1, **logs):
        """step 现在会影响行为：开局零方差是故障，练久了是收敛。"""
        control = SimpleNamespace(should_training_stop=False)
        for _ in range(n):
            guard.on_log(Mock(), SimpleNamespace(global_step=step), control,
                         logs=dict(logs))
        return control

    def test_healthy_variance_never_stops(self):
        guard = RewardVarianceGuard(patience=3)
        control = self._fire(guard, 10, frac_reward_zero_std=0.0, reward_std=0.4)
        self.assertFalse(control.should_training_stop)
        self.assertFalse(guard.tripped)

    def test_stops_after_patience_consecutive_dead_steps(self):
        guard = RewardVarianceGuard(patience=3)
        control = self._fire(guard, 3, frac_reward_zero_std=1.0, reward=-0.8393)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(guard.tripped)

    def test_does_not_stop_before_patience(self):
        guard = RewardVarianceGuard(patience=3)
        control = self._fire(guard, 2, frac_reward_zero_std=1.0)
        self.assertFalse(control.should_training_stop)

    def test_streak_resets_on_a_healthy_step(self):
        guard = RewardVarianceGuard(patience=3)
        self._fire(guard, 2, frac_reward_zero_std=1.0)
        self._fire(guard, 1, frac_reward_zero_std=0.0)
        control = self._fire(guard, 2, frac_reward_zero_std=1.0)
        self.assertFalse(control.should_training_stop)

    def test_falls_back_to_reward_std_when_frac_absent(self):
        guard = RewardVarianceGuard(patience=2)
        control = self._fire(guard, 2, reward_std=0.0)
        self.assertTrue(control.should_training_stop)

    def test_ignores_logs_without_reward_fields(self):
        guard = RewardVarianceGuard(patience=1)
        control = self._fire(guard, 5, loss=0.1, epoch=1.0)
        self.assertFalse(control.should_training_stop)

    def test_zero_variance_after_the_grace_period_is_convergence_not_failure(self):
        """练到没梯度 ≠ 开局就没梯度。

        --split rl_train 只有十来条中间带任务，学会之后每组都是满分，
        方差自然归零。把它判成故障会白扔一个已经练好的 checkpoint。
        """
        guard = RewardVarianceGuard(patience=3, grace_steps=20)
        control = self._fire(guard, 3, step=50, frac_reward_zero_std=1.0, reward=1.0)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(guard.converged)
        self.assertFalse(guard.tripped)      # 不以失败退出，checkpoint 保留
        self.assertEqual(guard.stop_step, 50)

    def test_zero_variance_inside_the_grace_period_is_still_a_failure(self):
        """开局就零方差多半是基座还吐不出可解析动作，那是真故障。"""
        guard = RewardVarianceGuard(patience=3, grace_steps=20)
        control = self._fire(guard, 3, step=5, frac_reward_zero_std=1.0, reward=-0.84)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(guard.tripped)
        self.assertFalse(guard.converged)

    def test_the_boundary_step_still_counts_as_grace(self):
        guard = RewardVarianceGuard(patience=1, grace_steps=20)
        self._fire(guard, 1, step=20, frac_reward_zero_std=1.0)
        self.assertTrue(guard.tripped)

    def test_an_unusable_global_step_falls_back_to_failure(self):
        """拿不到步数时按开局处理：宁可多报一次故障，也不要因它崩掉。"""
        guard = RewardVarianceGuard(patience=1, grace_steps=20)
        control = SimpleNamespace(should_training_stop=False)
        guard.on_log(Mock(), Mock(), control, logs={"frac_reward_zero_std": 1.0})
        self.assertTrue(guard.tripped)


class PrecisionFlagsTest(unittest.TestCase):
    """fp16 的 GradScaler 不接受 bf16 梯度；在 bf16 权重的模型上会硬崩：
    NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda ... for 'BFloat16'
    """

    def _flags(self, cuda, bf16):
        import torch
        from unittest.mock import patch
        with patch.object(torch.cuda, "is_available", return_value=cuda), \
             patch.object(torch.cuda, "is_bf16_supported", return_value=bf16):
            return precision_flags()

    def test_no_mixed_precision_off_cuda(self):
        self.assertEqual(self._flags(cuda=False, bf16=False), {"use_cpu": True})

    def test_prefers_bf16_when_supported(self):
        self.assertEqual(self._flags(cuda=True, bf16=True), {"bf16": True})

    def test_never_sets_fp16_alongside_bf16(self):
        # 同时置位会让 GradScaler 介入 bf16 梯度，正是崩溃的来源
        self.assertNotIn("fp16", self._flags(cuda=True, bf16=True))

    def test_falls_back_to_fp16_without_bf16(self):
        self.assertEqual(self._flags(cuda=True, bf16=False), {"fp16": True})

    def test_all_three_trainers_agree(self):
        from rl.train_dpo import _precision_flags as dpo
        from rl.train_grpo import _precision_flags as grpo
        from rl.train_sft import _precision_flags as sft
        self.assertEqual(sft(), dpo())
        self.assertEqual(dpo(), grpo())


class DistributedLaunchGuardTest(unittest.TestCase):
    """多卡启动下不能再钉单卡，QLoRA 也不能静默按单卡跑。"""

    def _env(self, **values):
        import os
        from unittest.mock import patch

        return patch.dict(os.environ, values, clear=False)

    def test_no_launch_marker_means_single_process(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LOCAL_RANK", None)
            self.assertFalse(is_distributed_launch())

    def test_local_rank_marks_a_distributed_launch(self):
        with self._env(LOCAL_RANK="1"):
            self.assertTrue(is_distributed_launch())

    def test_pin_single_gpu_is_skipped_under_launch(self):
        config = SimpleNamespace()
        with self._env(LOCAL_RANK="0"):
            pin_single_gpu(config)
        self.assertFalse(hasattr(config, "_n_gpu"))

    def test_pin_single_gpu_applies_without_launch(self):
        import os
        from unittest.mock import patch

        config = SimpleNamespace()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LOCAL_RANK", None)
            os.environ.pop("RANK", None)
            pin_single_gpu(config)
        self.assertEqual(config._n_gpu, 1)

    def test_qlora_is_rejected_under_launch(self):
        with self._env(LOCAL_RANK="0"):
            with self.assertRaises(SystemExit):
                assert_quantization_is_single_process(True)

    def test_full_precision_is_allowed_under_launch(self):
        with self._env(LOCAL_RANK="0"):
            assert_quantization_is_single_process(False)


class DapoPresetTest(unittest.TestCase):
    """DAPO 预设只填空，不覆盖显式给出的参数。"""

    def _resolve(self, dapo, **overrides):
        kwargs = {
            "loss_type": None,
            "epsilon_high": None,
            "mask_truncated_completions": None,
            "beta": DEFAULT_BETA,
        }
        kwargs.update(overrides)
        return resolve_dapo_options(dapo=dapo, **kwargs)

    def test_without_the_flag_nothing_changes(self):
        self.assertEqual(self._resolve(False), (None, None, None, DEFAULT_BETA))

    def test_preset_fills_every_default(self):
        self.assertEqual(
            self._resolve(True),
            (
                DAPO_PRESET["loss_type"],
                DAPO_PRESET["epsilon_high"],
                DAPO_PRESET["mask_truncated_completions"],
                DAPO_PRESET["beta"],
            ),
        )

    def test_explicit_values_beat_the_preset(self):
        self.assertEqual(
            self._resolve(
                True,
                loss_type="dr_grpo",
                epsilon_high=0.2,
                mask_truncated_completions=False,
                beta=0.1,
            ),
            ("dr_grpo", 0.2, False, 0.1),
        )

    def test_a_changed_beta_survives_the_preset(self):
        """beta 没有「未设置」哨兵值，所以只认「没动过默认值」这一种情况。"""
        self.assertEqual(self._resolve(True, beta=0.2)[3], 0.2)
        self.assertEqual(self._resolve(True, beta=DEFAULT_BETA)[3], 0.0)

    def test_preset_values_are_trl_field_names(self):
        import dataclasses

        from trl import GRPOConfig

        fields = {f.name for f in dataclasses.fields(GRPOConfig)}
        for name in DAPO_PRESET:
            with self.subTest(field=name):
                self.assertIn(name, fields)


class StageVerificationDefaultTest(unittest.TestCase):
    """五个训练阶段的 `--verify` 默认值必须一致，而且必须默认开启。

    README 把阶段验证写成每个阶段都有的质量闸门（「每个阶段产出模型须过最低
    质量阈值……`--no-verify` 可跳过」）。SFT 与 DPO 曾经默认关闭、GRPO/PPO/OPD
    默认开启，于是最基础的两步实际上没有闸门：SFT 训坏了要一路跑到 GRPO 才看得
    出来，而中间那轮 DPO 数据采样已经白烧掉一两个小时。
    """

    STAGES = ("train_sft", "train_dpo", "train_grpo", "train_ppo", "train_opd")

    def test_sources_declare_verify_defaulting_to_true(self):
        import re
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        for stage in self.STAGES:
            source = (repo / "AgenticArxiv" / "rl" / f"{stage}.py").read_text(encoding="utf-8")
            with self.subTest(stage=stage):
                match = re.search(
                    r'"--verify",\s*action=[^,]+,\s*default=(True|False)', source
                )
                self.assertIsNotNone(match, f"{stage} 里找不到 --verify 的默认值")
                self.assertEqual(
                    match.group(1),
                    "True",
                    f"{stage} 的 --verify 默认关闭，阶段闸门对这一步形同不存在",
                )

    def test_every_stage_accepts_no_verify(self):
        """「可跳过」要真的能跳过：必须是 BooleanOptionalAction。"""
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        for stage in self.STAGES:
            source = (repo / "AgenticArxiv" / "rl" / f"{stage}.py").read_text(encoding="utf-8")
            with self.subTest(stage=stage):
                self.assertIn("argparse.BooleanOptionalAction", source)


class ToolRegistrationCoverageTest(unittest.TestCase):
    """所有训练入口都必须注册齐工具。

    每个入口曾经各自手写一段 `import tools.xxx  # 触发注册`，四行里只列了当时
    存在的四个工具。后来加了 `search_arxiv_papers`、`get_paper_content`、
    `summarize_paper`、`extract_paper_figures`，这些入口没有跟着更新 —— 后果不是
    报错，而是 prompt 里的工具列表少了几个：策略根本不知道那些工具存在，对应的
    任务永远做不成，日志上却什么都看不出来。现在统一走 tools/bootstrap.py。
    """

    ENTRY_POINTS = (
        "rl.train_grpo",
        "rl.train_opd",
        "rl.stage_verifier",
        "rl.build_snapshot",
    )

    def test_every_entry_point_registers_the_full_tool_set(self):
        from tools.bootstrap import missing_tools, register_all_tools

        # 先把注册表清干净不可行（注册是模块级副作用），所以改为逐个 import
        # 入口模块后再整体核对：任何入口只要漏注册，这里就会看见缺项。
        for module in self.ENTRY_POINTS:
            with self.subTest(module=module):
                __import__(module)

        register_all_tools()
        self.assertEqual(missing_tools(), [])

    def test_entry_points_do_not_hand_list_tool_modules(self):
        """手写模块清单本身就是这个 bug 的成因，别再长回来。"""
        from pathlib import Path

        repo = Path(__file__).resolve().parents[2]
        for module in self.ENTRY_POINTS:
            path = repo / "AgenticArxiv" / (module.replace(".", "/") + ".py")
            source = path.read_text(encoding="utf-8")
            with self.subTest(module=module):
                self.assertNotIn(
                    "import tools.pdf_download_tool",
                    source,
                    f"{module} 又在手写工具模块清单了，改用 tools.bootstrap",
                )


class PpoAvailabilityTest(unittest.TestCase):
    """PPO 对当前 TRL 不可用时，必须响亮失败而不是静默消失。

    TRL 从 0.9 起弃用、并在后续版本移除了经典 PPO trainer
    （`PPOTrainer` / `PPOConfig` / `AutoModelForCausalLMWithValueHead`）。
    requirements.txt 只写了 `trl>=0.28.0`，所以 README 里作为「阶段4」宣传的这条
    路径在受支持的版本上根本跑不起来 —— 这正是需要被说出来的那类不一致。
    """

    def test_import_either_succeeds_or_explains_itself(self):
        import importlib
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        try:
            importlib.import_module("rl.train_ppo")
        except SystemExit as exc:
            message = str(exc)
            for expected in ("PPOTrainer", "trl.experimental.ppo", "GRPO"):
                with self.subTest(fragment=expected):
                    self.assertIn(expected, message)
        except ImportError as exc:  # pragma: no cover - 未受控的失败形态
            self.fail(
                "train_ppo 抛出了不带解释的 ImportError，应当改成 SystemExit 并说明"
                f"受支持的替代路径: {exc}"
            )


class VerificationModelDeviceTest(unittest.TestCase):
    """阶段验证必须在训练设备上生成。

    `StageVerifier` 里的三个 verify_* 原本用裸 `from_pretrained` 加载产物模型，
    模型落在 CPU；`CanaryEvaluator` 又是从参数上读设备，于是整道闸门在 CPU 上
    做生成。1.5B 模型跑 8×256 token 要几十分钟一个阶段 —— 闸门慢到这个程度就
    没人愿意留着它，这正是它当初被默认关掉的原因之一。
    """

    def _loaded(self, cuda: bool):
        from unittest.mock import MagicMock, patch

        from rl import stage_verifier

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "</s>"

        with patch.object(stage_verifier.torch.cuda, "is_available", return_value=cuda), \
             patch.object(
                 stage_verifier.AutoModelForCausalLM, "from_pretrained", return_value=model
             ), patch.object(
                 stage_verifier.AutoTokenizer, "from_pretrained", return_value=tokenizer
             ):
            loaded_model, loaded_tokenizer = stage_verifier._load_verification_model(
                "/tmp/fake-model"
            )
        # 助手返回的是 .to() 的结果，断言要看被调用的那个对象
        return model, loaded_model, loaded_tokenizer

    def test_model_is_moved_to_the_cuda_device_when_available(self):
        loaded, _moved, _tokenizer = self._loaded(cuda=True)
        loaded.to.assert_called_once_with("cuda")

    def test_cpu_only_environment_does_not_try_to_move(self):
        loaded, _moved, _tokenizer = self._loaded(cuda=False)
        loaded.to.assert_not_called()

    def test_missing_pad_token_falls_back_to_eos(self):
        _loaded, _moved, tokenizer = self._loaded(cuda=True)
        self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)


class StageVerifierTest(unittest.TestCase):
    """阶段验证器的逻辑测试（不需要真实模型）。"""

    def test_report_passed_formatting(self):
        from rl.stage_verifier import VerificationReport
        report = VerificationReport(
            stage="sft", model_path="/tmp/model",
            passed=True, metrics={"parse_rate": 0.8},
            thresholds={"parse_rate": 0.3},
        )
        summary = report.summary()
        self.assertIn("PASS", summary)
        self.assertIn("SFT", summary)
        self.assertIn("0.8", summary)

    def test_report_failed_formatting(self):
        from rl.stage_verifier import VerificationReport
        report = VerificationReport(
            stage="dpo", model_path="/tmp/model",
            passed=False, metrics={"mean_reward": -0.5},
            thresholds={"mean_reward": -0.3},
            failures=["mean_reward too low"],
        )
        summary = report.summary()
        self.assertIn("FAIL", summary)
        self.assertIn("DPO", summary)
        self.assertIn("too low", summary)

    def test_thresholds_configurable(self):
        from rl.stage_verifier import StageVerifier
        v = StageVerifier(
            sft_min_parse_rate=0.5,
            dpo_min_reward=0.0,
            grpo_min_reward=0.1,
        )
        self.assertEqual(v.thresholds["sft"]["parse_rate"], 0.5)
        self.assertEqual(v.thresholds["dpo"]["mean_reward"], 0.0)
        self.assertEqual(v.thresholds["grpo"]["mean_reward"], 0.1)

    def test_verify_sft_model_load_failure(self):
        """模型路径不存在时返回 failed report 而非抛异常。"""
        from rl.stage_verifier import StageVerifier
        v = StageVerifier()
        report = v.verify_sft(model_path="/nonexistent/path/model")
        self.assertFalse(report.passed)
        self.assertIn("模型加载失败", report.failures[0])

    def test_verify_dpo_model_load_failure(self):
        from rl.stage_verifier import StageVerifier
        v = StageVerifier()
        report = v.verify_dpo(model_path="/nonexistent/path/model")
        self.assertFalse(report.passed)
        self.assertIn("模型加载失败", report.failures[0])

    def test_verify_grpo_model_load_failure(self):
        from rl.stage_verifier import StageVerifier
        v = StageVerifier()
        report = v.verify_grpo(model_path="/nonexistent/path/model")
        self.assertFalse(report.passed)
        self.assertIn("模型加载失败", report.failures[0])

    def test_save_report_writes_file(self):
        import tempfile
        import json
        from pathlib import Path
        from rl.stage_verifier import StageVerifier, VerificationReport

        report = VerificationReport(
            stage="sft", model_path="/tmp/model",
            passed=True, metrics={"parse_rate": 0.8},
            thresholds={"parse_rate": 0.3},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            StageVerifier.save_report(report, out)
            report_path = out / "verification_report.json"
            self.assertTrue(report_path.exists())
            data = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(data["stage"], "sft")
            self.assertTrue(data["passed"])
            self.assertEqual(data["metrics"]["parse_rate"], 0.8)


class TrainingTaskSetTest(unittest.TestCase):
    """GRPO 训到哪些任务上。

    此前写死 get_all_tasks()，也就是 benchmark/tasks.py 那 8 条冒烟任务 ——
    59 条的完整基准集和 data/splits/v1.json 的切分都在，训练脚本一条都没用上。
    """

    def test_default_still_uses_the_smoke_subset(self):
        """默认值不动：换了会让此前所有训练曲线不可比。"""
        self.assertEqual(len(_load_tasks("default", None)), len(get_all_tasks()))

    def test_expanded_uses_the_full_benchmark_set(self):
        self.assertEqual(len(_load_tasks("expanded", None)), len(get_expanded_tasks()))

    def test_rl_train_split_keeps_only_the_middle_band(self):
        """两端的任务同一 prompt 采样出的奖励一致，组内方差为零，没有梯度。"""
        chosen = {t["id"] for t in _load_tasks("expanded", "rl_train")}
        self.assertEqual(chosen, set(load_split("rl_train")))
        self.assertLess(len(chosen), len(get_expanded_tasks()))

    def test_split_against_the_wrong_task_set_fails_loudly(self):
        """切分按完整任务集划定；配上冒烟集会静默少掉大半训练数据。"""
        with self.assertRaises(SystemExit) as ctx:
            _load_tasks("default", "rl_train")
        self.assertIn("不在当前任务集中", str(ctx.exception))

    def test_task_ids_come_back_in_a_deterministic_order(self):
        first = [t["id"] for t in _load_tasks("expanded", "train")]
        self.assertEqual(first, sorted(first))
        self.assertEqual(first, [t["id"] for t in _load_tasks("expanded", "train")])


if __name__ == "__main__":
    unittest.main()
