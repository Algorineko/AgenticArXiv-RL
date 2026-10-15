"""Post-stage verification: validates that each training stage produces
a model that meets minimum quality thresholds.

Usage:
    from agentic_arxiv.training.stage_verifier import StageVerifier

    verifier = StageVerifier()
    report = verifier.verify_sft(model_path, tokenizer)
    if not report.passed:
        print(report.summary())
        raise SystemExit(1)
    verifier.save_report(report, Path("outputs/sft/final"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

from agentic_arxiv.runtime.react_contract import build_react_messages, protocol_metadata
from agentic_arxiv.provenance import evaluation_protocol_relation, read_protocol_metadata

@dataclass
class VerificationReport:
    """单次阶段验证的结果。"""

    stage: str  # "sft", "dpo", "grpo"
    model_path: str
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    coverage: Dict[str, Any] = field(default_factory=dict)
    protocol: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        lines = [f"{status} {self.stage.upper()} 阶段验证: {self.model_path}"]
        for k, v in self.metrics.items():
            if k in self.thresholds:
                t = self.thresholds[k]
                flag = "✅" if v >= t else "❌"
                lines.append(f"  {flag} {k}: {v:.4f} (阈值: {t})")
            else:
                lines.append(f"  {k}: {v:.4f}（报告指标，无阶段阈值）")
        if self.coverage:
            lines.append(f"  覆盖: {self.coverage}")
        if self.failures:
            lines.append(f"  失败: {', '.join(self.failures)}")
        return "\n".join(lines)


def _load_verification_model(
    model_path: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    *,
    stage: str | None = None,
) -> tuple:
    """按明确产物契约离线加载全参/适配器，并保持原训练设备选择。"""
    import torch
    from agentic_arxiv.llm.local import TransformersLLMClient

    try:
        client = TransformersLLMClient(
            model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            artifact=True,
            allowed_stages=None if stage is None else {f"{stage}_full", f"{stage}_qlora"},
            expected_protocol=protocol_metadata(tokenizer),
        )
    except SystemExit as error:
        # 共享产物检查也服务 CLI；这里转换成可保存的失败报告。
        raise ValueError(str(error)) from error
    return client.model, tokenizer if tokenizer is not None else client.tokenizer


class StageVerifier:
    """在每个训练阶段结束后验证模型质量。

    阈值参考：
    - SFT parse_rate=0.3：完整 episode 中至少 30% 的模型决策可解析
    - DPO mean_reward=-0.3：最低冒烟质量门，不是优于 SFT 的证据
    - GRPO mean_reward=-0.2：应展现学习进展

    所有阈值均可通过构造参数覆盖。
    """

    def __init__(
        self,
        sft_min_parse_rate: float = 0.3,
        grpo_min_reward: float = -0.2,
        dpo_min_reward: float = -0.3,
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
        env: Any = None,
        max_iterations: int = 6,
    ) -> VerificationReport:
        """完整执行冒烟任务，分别报告决策解析率与真实任务完成率。

        SFT 原解析阈值不变，但解析率不再只测首步；它不是任务成功率。

        Args:
            model_path: 模型路径
            tokenizer: 可选外部 tokenizer；不传则从 model_path 加载
            num_samples: 冒烟任务选择数，不足时循环；每次选择运行两个 episode
        """
        model_path = str(model_path)
        print(f"🔍 验证 SFT 模型: {model_path}")

        try:
            model, tokenizer = _load_verification_model(model_path, tokenizer, stage="sft")
        except Exception as e:
            return VerificationReport(
                stage="sft",
                model_path=model_path,
                passed=False,
                metrics={"parse_rate": 0.0},
                thresholds=self.thresholds["sft"],
                failures=[f"模型加载失败: {e}"],
                coverage={"execution": "not-started", "scope": "smoke-only"},
                protocol=_evaluation_protocol(model_path, tokenizer),
            )

        from agentic_arxiv.tasks.tasks import get_all_tasks

        tasks = get_all_tasks()
        selected = (tasks * (1 + num_samples // max(1, len(tasks))))[:num_samples]
        if not selected:
            raise ValueError("SFT 验证必须至少选择一个任务")
        task_ids = [task["id"] for task in selected]
        try:
            result = _check_episode_result(
                model, tokenizer, task_ids=task_ids, env=env,
                num_generations=2, temperature=0.3, max_iterations=max_iterations,
                model_source=model_path,
            )
        except Exception as error:
            return self._evaluation_failure("sft", model_path, error, task_ids, max_iterations, tokenizer)
        parse_rate = result.parse_rate

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
            metrics=_episode_metrics(result),
            thresholds=thresholds,
            failures=failures,
            coverage=_episode_coverage(result),
            protocol=_evaluation_protocol(model_path, tokenizer),
        )

    # ------------------------------------------------------------------
    # GRPO 验证
    # ------------------------------------------------------------------

    def verify_grpo(
        self,
        model_path: str | Path,
        tokenizer: PreTrainedTokenizer | None = None,
        env: Any = None,
        max_iterations: int = 6,
    ) -> VerificationReport:
        """验证 GRPO 模型在既有 Canary 任务上的完整 episode 平均奖励。"""
        return self._verify_reward_stage("grpo", model_path, tokenizer, env, max_iterations)

    def verify_dpo(
        self,
        model_path: str | Path,
        tokenizer: PreTrainedTokenizer | None = None,
        env: Any = None,
        max_iterations: int = 6,
    ) -> VerificationReport:
        """验证可选 DPO 产物；范围与 GRPO 相同，不新增通过门槛。"""
        return self._verify_reward_stage("dpo", model_path, tokenizer, env, max_iterations)

    def _verify_reward_stage(self, stage, model_path, tokenizer, env, max_iterations):
        model_path = str(model_path)
        print(f"🔍 验证 {stage.upper()} 模型: {model_path}")

        try:
            model, tokenizer = _load_verification_model(model_path, tokenizer, stage=stage)
        except Exception as e:
            return VerificationReport(
                stage=stage,
                model_path=model_path,
                passed=False,
                metrics={"mean_reward": -1.0},
                thresholds=self.thresholds[stage],
                failures=[f"模型加载失败: {e}"],
                coverage={"execution": "not-started", "scope": "smoke-only"},
                protocol=_evaluation_protocol(model_path, tokenizer),
            )

        from agentic_arxiv.training.canary import DEFAULT_CANARY_TASK_IDS

        try:
            result = _check_episode_result(
                model, tokenizer, task_ids=DEFAULT_CANARY_TASK_IDS,
                env=env, max_iterations=max_iterations, model_source=model_path,
                artifact=stage == "dpo",
            )
        except Exception as error:
            return self._evaluation_failure(
                stage, model_path, error, DEFAULT_CANARY_TASK_IDS, max_iterations, tokenizer
            )
        mean_reward = result.mean_reward

        thresholds = self.thresholds[stage]
        passed = mean_reward >= thresholds["mean_reward"]
        failures = []
        if not passed:
            failures.append(
                f"mean_reward={mean_reward:.4f} < {thresholds['mean_reward']}"
            )

        return VerificationReport(
            stage=stage,
            model_path=model_path,
            passed=passed,
            metrics=_episode_metrics(result),
            thresholds=thresholds,
            failures=failures,
            coverage=_episode_coverage(result),
            protocol=_evaluation_protocol(model_path, tokenizer),
        )

    def _evaluation_failure(self, stage, model_path, error, task_ids, max_iterations, tokenizer):
        return VerificationReport(
            stage=stage, model_path=model_path, passed=False,
            metrics={"parse_rate": 0.0, "mean_reward": -1.0, "task_completion_rate": 0.0},
            thresholds=self.thresholds[stage], failures=[f"完整 episode 验证失败: {error}"],
            coverage={"task_ids": list(task_ids), "max_iterations": max_iterations,
                      "execution": "failed", "scope": "smoke-only"},
            protocol=_evaluation_protocol(model_path, tokenizer),
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
                    "coverage": report.coverage,
                    "protocol": report.protocol,
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
    def _sft_prompts(num_samples: int, tokenizer=None) -> List[str]:
        """仅供初始输入检查；共同消息经实际模板渲染，禁止裸 prompt 降级。"""
        from agentic_arxiv.tasks.tasks import get_all_tasks
        from agentic_arxiv.runtime.prompt_templates import format_tool_description
        from agentic_arxiv.tools.bootstrap import register_all_tools
        from agentic_arxiv.tools.tool_registry import registry

        register_all_tools()
        tasks = get_all_tasks()
        # 取前 num_samples 个任务，不够就循环
        selected = (tasks * (1 + num_samples // max(1, len(tasks))))[:num_samples]
        if tokenizer is None or not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise ValueError("验证输入需要实际 tokenizer/chat template，不回退裸 prompt")
        description = format_tool_description(registry.list_tools())
        return [tokenizer.apply_chat_template(
            build_react_messages(task["task"], description),
            tokenize=False, add_generation_prompt=True,
        ) for task in selected]


# ------------------------------------------------------------------
# 内部辅助函数
# ------------------------------------------------------------------


def _check_episode_result(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    *,
    task_ids,
    env: Any = None,
    num_generations: int = 4,
    temperature: float = 1.0,
    max_iterations: int = 6,
    model_source: str = "",
    artifact: bool = False,
):
    """全部阶段共用真实多轮执行，不从首个 completion 合成轨迹。"""
    from agentic_arxiv.training.canary import CanaryEvaluator
    from agentic_arxiv.tools.bootstrap import register_all_tools

    register_all_tools()
    evaluator = CanaryEvaluator(
        model=model,
        tokenizer=tokenizer,
        canary_task_ids=task_ids,
        env=env,
        num_generations=num_generations,
        temperature=temperature,
        max_iterations=max_iterations,
        model_source=model_source,
        artifact=artifact,
    )
    return evaluator.evaluate(step=0)


def _episode_metrics(result):
    return {"parse_rate": result.parse_rate, "mean_reward": result.mean_reward,
            "task_completion_rate": result.task_completion_rate,
            "false_finish_rate": result.false_finish_rate}


def _evaluation_protocol(model_path, tokenizer):
    """训练协议来自原 manifest；缺字段保持未知，不能从评测协议倒推。"""
    return evaluation_protocol_relation(
        read_protocol_metadata(Path(model_path)), protocol_metadata(tokenizer)
    )


def _episode_coverage(result):
    return {"scope": "smoke-only", "execution": "complete-episode",
            "task_ids": result.task_ids, "episode_count": result.episode_count,
            "num_generations_per_task": result.num_generations_per_task,
            "max_iterations": result.max_iterations,
            "temperature": result.temperature,
            "max_new_tokens": result.max_new_tokens,
            "scoring": result.scoring,
            "termination_counts": result.termination_counts,
            "strict_success_count": result.strict_success_count,
            "false_finish_count": result.false_finish_count,
            "parsed_decision_count": result.parsed_decision_count,
            "model_decision_count": result.model_decision_count,
            "task_completion_basis": "is_strict_success; FINISH is not success",
            "false_finish_rate_basis": "same-episode metrics.false_finish / all episodes",
            "quality_claim": "smoke threshold only; not held-out improvement or superiority to SFT",
            "reference_oracle": result.reference_oracle,
            "parse_rate_basis": "all-model-decisions including token truncation; framework-only markers excluded"}
