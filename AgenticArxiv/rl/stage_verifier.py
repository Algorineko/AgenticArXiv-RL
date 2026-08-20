"""Post-stage verification: validates that each training stage produces
a model that meets minimum quality thresholds.

Usage:
    from rl.stage_verifier import StageVerifier

    verifier = StageVerifier()
    report = verifier.verify_sft(model_path, tokenizer)
    if not report.passed:
        print(report.summary())
        raise SystemExit(1)
    verifier.save_report(report, Path("outputs/sft/final"))
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from rl.grpo_reward import parse_react_action, synthesize_trajectory
from rl.reward import RewardCalculator

# 工具导入（触发注册）
import tools.arxiv_tool  # noqa: F401
import tools.cache_status_tool  # noqa: F401
import tools.pdf_download_tool  # noqa: F401
import tools.pdf_translate_tool  # noqa: F401


@dataclass
class VerificationReport:
    """单次阶段验证的结果。"""

    stage: str  # "sft", "dpo", "grpo"
    model_path: str
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        lines = [f"{status} {self.stage.upper()} 阶段验证: {self.model_path}"]
        for k, v in self.metrics.items():
            t = self.thresholds.get(k, 0)
            flag = "✅" if v >= t else "❌"
            lines.append(f"  {flag} {k}: {v:.4f} (阈值: {t})")
        if self.failures:
            lines.append(f"  失败: {', '.join(self.failures)}")
        return "\n".join(lines)


class StageVerifier:
    """在每个训练阶段结束后验证模型质量。

    阈值参考：
    - SFT parse_rate=0.3：至少 30% 的输出能解析为合法工具调用
    - DPO mean_reward=-0.3：应优于随机猜测
    - GRPO mean_reward=-0.2：应展现学习进展

    所有阈值均可通过构造参数覆盖。
    """

    def __init__(
        self,
        sft_min_parse_rate: float = 0.3,
        dpo_min_reward: float = -0.3,
        grpo_min_reward: float = -0.2,
    ):
        self.thresholds = {
            "sft": {"parse_rate": sft_min_parse_rate},
            "dpo": {"mean_reward": dpo_min_reward},
            "grpo": {"mean_reward": grpo_min_reward},
        }

    # ------------------------------------------------------------------
    # SFT 验证
    # ------------------------------------------------------------------

    def verify_sft(
        self,
        model_path: str | Path,
        tokenizer: PreTrainedTokenizer | None = None,
        num_samples: int = 8,
    ) -> VerificationReport:
        """验证 SFT 模型能否产出可解析的 ReAct 动作。

        加载模型，对 benchmark 任务生成 completion，统计 parse 成功率。
        使用 chat template（SFT 模型训练时的格式），而非裸 ReAct prompt。

        Args:
            model_path: 模型路径
            tokenizer: 可选外部 tokenizer；不传则从 model_path 加载
            num_samples: 生成样本数（每个任务一条）
        """
        model_path = str(model_path)
        print(f"🔍 验证 SFT 模型: {model_path}")

        try:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            return VerificationReport(
                stage="sft",
                model_path=model_path,
                passed=False,
                metrics={"parse_rate": 0.0},
                thresholds=self.thresholds["sft"],
                failures=[f"模型加载失败: {e}"],
            )

        prompts = self._sft_prompts(num_samples)
        parse_rate = _check_parse_rate(model, tokenizer, prompts)

        thresholds = self.thresholds["sft"]
        passed = parse_rate >= thresholds["parse_rate"]
        failures = []
        if not passed:
            failures.append(
                f"parse_rate={parse_rate:.2%} < {thresholds['parse_rate']:.0%}"
            )

        return VerificationReport(
            stage="sft",
            model_path=model_path,
            passed=passed,
            metrics={"parse_rate": round(parse_rate, 4)},
            thresholds=thresholds,
            failures=failures,
        )

    # ------------------------------------------------------------------
    # DPO 验证
    # ------------------------------------------------------------------

    def verify_dpo(
        self,
        model_path: str | Path,
        tokenizer: PreTrainedTokenizer | None = None,
        env: Any = None,
    ) -> VerificationReport:
        """验证 DPO 模型在 canary 任务上的平均奖励。

        Args:
            model_path: 模型路径
            tokenizer: 可选外部 tokenizer
            env: 可选 MockArxivEnv 用于工具执行
        """
        model_path = str(model_path)
        print(f"🔍 验证 DPO 模型: {model_path}")

        try:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            return VerificationReport(
                stage="dpo",
                model_path=model_path,
                passed=False,
                metrics={"mean_reward": -1.0},
                thresholds=self.thresholds["dpo"],
                failures=[f"模型加载失败: {e}"],
            )

        mean_reward = _check_canary_reward(model, tokenizer, env=env)

        thresholds = self.thresholds["dpo"]
        passed = mean_reward >= thresholds["mean_reward"]
        failures = []
        if not passed:
            failures.append(
                f"mean_reward={mean_reward:.4f} < {thresholds['mean_reward']}"
            )

        return VerificationReport(
            stage="dpo",
            model_path=model_path,
            passed=passed,
            metrics={"mean_reward": round(mean_reward, 4)},
            thresholds=thresholds,
            failures=failures,
        )

    # ------------------------------------------------------------------
    # GRPO 验证
    # ------------------------------------------------------------------

    def verify_grpo(
        self,
        model_path: str | Path,
        tokenizer: PreTrainedTokenizer | None = None,
        env: Any = None,
    ) -> VerificationReport:
        """验证 GRPO 模型在 canary 任务上的平均奖励。

        Args:
            model_path: 模型路径
            tokenizer: 可选外部 tokenizer
            env: 可选 MockArxivEnv 用于工具执行
        """
        model_path = str(model_path)
        print(f"🔍 验证 GRPO 模型: {model_path}")

        try:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            return VerificationReport(
                stage="grpo",
                model_path=model_path,
                passed=False,
                metrics={"mean_reward": -1.0},
                thresholds=self.thresholds["grpo"],
                failures=[f"模型加载失败: {e}"],
            )

        mean_reward = _check_canary_reward(model, tokenizer, env=env)

        thresholds = self.thresholds["grpo"]
        passed = mean_reward >= thresholds["mean_reward"]
        failures = []
        if not passed:
            failures.append(
                f"mean_reward={mean_reward:.4f} < {thresholds['mean_reward']}"
            )

        return VerificationReport(
            stage="grpo",
            model_path=model_path,
            passed=passed,
            metrics={"mean_reward": round(mean_reward, 4)},
            thresholds=thresholds,
            failures=failures,
        )

    # ------------------------------------------------------------------
    # 报告持久化
    # ------------------------------------------------------------------

    @staticmethod
    def save_report(report: VerificationReport, output_dir: Path) -> None:
        """保存验证报告到模型目录。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "verification_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "stage": report.stage,
                    "model_path": report.model_path,
                    "passed": report.passed,
                    "metrics": report.metrics,
                    "thresholds": report.thresholds,
                    "failures": report.failures,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"📋 验证报告已保存: {report_path}")

    # ------------------------------------------------------------------
    # SFT 测试 prompt 构造
    # ------------------------------------------------------------------

    @staticmethod
    def _sft_prompts(num_samples: int) -> List[str]:
        """构造 SFT 验证用的 chat prompt 列表。

        使用 benchmark 任务描述作为 user 消息，配上 system prompt。
        """
        from benchmark.tasks import get_all_tasks

        SYSTEM = (
            "你是 arXiv 论文检索 Agent。"
            "根据用户需求调用工具，以 JSON 格式返回动作："
            '{"name": "工具名", "arguments": {...}}'
        )

        tasks = get_all_tasks()
        # 取前 num_samples 个任务，不够就循环
        selected = (tasks * (1 + num_samples // max(1, len(tasks))))[:num_samples]

        prompts = []
        for task in selected:
            prompts.append(
                f"System: {SYSTEM}\n\nUser: {task['task']}\n\nAssistant:"
            )
        return prompts


# ------------------------------------------------------------------
# 内部辅助函数
# ------------------------------------------------------------------


def _check_parse_rate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: Sequence[str],
    max_new_tokens: int = 256,
    temperature: float = 0.3,
) -> float:
    """生成 completion 并统计 parse 成功率。

    一个 completion 被认为"可解析"当且仅当：
    - 包含合法的 JSON 对象（有 "name" 字段）
    - 或可被 parse_react_action 识别为 "call"
    """
    device = next(model.parameters()).device
    model.eval()

    parsed = 0
    total = 0
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.shape[1]

            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    num_return_sequences=2,
                    pad_token_id=pad_token_id,
                )
            except Exception:
                total += 2
                continue

            generated_ids = outputs[:, prompt_len:]
            completions = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for completion in completions:
                total += 1
                # 先尝试 ReAct 解析
                kind, action = parse_react_action(completion)
                if kind == "call":
                    parsed += 1
                    continue
                # 再尝试直接 JSON 解析（SFT 模型可能不输出 ReAct 前缀）
                try:
                    obj = json.loads(completion.strip())
                    if isinstance(obj, dict) and "name" in obj:
                        parsed += 1
                except (json.JSONDecodeError, TypeError):
                    pass

    model.train()
    return parsed / total if total > 0 else 0.0


def _check_canary_reward(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    env: Any = None,
    num_generations: int = 4,
) -> float:
    """在 canary 任务上评估模型，返回平均奖励。

    复用 CanaryEvaluator 以保持评估逻辑一致。
    """
    from rl.canary import DEFAULT_CANARY_TASK_IDS, CanaryEvaluator

    evaluator = CanaryEvaluator(
        model=model,
        tokenizer=tokenizer,
        canary_task_ids=DEFAULT_CANARY_TASK_IDS,
        reward_calc=RewardCalculator(),
        env=env,
        num_generations=num_generations,
    )
    result = evaluator.evaluate(step=0)
    return result.mean_reward