"""训练脚本的两个「把静默失败变成响亮失败」的守卫。

两个失败模式都不会让训练崩：跑完、保存 checkpoint、日志看着正常，
只是模型什么都没学到。
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from rl.precision import precision_flags
from rl.train_grpo import RewardVarianceGuard
from rl.train_sft import _check_lengths, _messages_of


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


class RewardVarianceGuardTest(unittest.TestCase):
    def _fire(self, guard, n, **logs):
        control = SimpleNamespace(should_training_stop=False)
        for _ in range(n):
            guard.on_log(Mock(), Mock(), control, logs=dict(logs))
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
        self.assertEqual(self._flags(cuda=False, bf16=False), {})

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


if __name__ == "__main__":
    unittest.main()
