"""Training canary: periodic evaluation on a fixed small task set during training.

During GRPO training, the model's reward can fluctuate. The canary provides
an early warning signal: if performance on a fixed set of tasks starts
degrading, training can be stopped before wasting more compute.

与 RewardVarianceGuard 的分工：
- RewardVarianceGuard：检测"组内奖励方差为 0"（梯度为零，训练空转）
- CanaryCallback：检测"模型在固定任务上性能退化"（学到错误策略）

Usage:
    from rl.canary import CanaryEvaluator, CanaryCallback

    evaluator = CanaryEvaluator(model, tokenizer, canary_task_ids, reward_calc, env)
    callback = CanaryCallback(evaluator, steps=50, min_reward=-0.5, patience=3)
    trainer.add_callback(callback)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import TrainerCallback

from benchmark.tasks import get_task_by_id
from rl.grpo_reward import synthesize_trajectory
from rl.reward import RewardCalculator

# 默认 canary 任务：一个简单搜索 + 一个复合任务，覆盖单步和多步场景
DEFAULT_CANARY_TASK_IDS = ["search_01", "composite_01"]


@dataclass
class CanaryResult:
    """单次 canary 评估的结果。"""

    step: int
    mean_reward: float
    task_completion_rate: float
    per_task_rewards: Dict[str, float]
    num_generations_per_task: int = 4


class CanaryEvaluator:
    """在固定 canary 任务集上评估模型。

    对每个任务生成多条 completion，计算 reward，汇总结果。
    复用 grpo_reward.py 的 synthesize_trajectory 和 RewardCalculator，
    确保评分标准与训练完全一致。
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        canary_task_ids: Sequence[str] | None = None,
        reward_calc: RewardCalculator | None = None,
        env: Any = None,
        num_generations: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.canary_task_ids = list(canary_task_ids or DEFAULT_CANARY_TASK_IDS)
        self.reward_calc = reward_calc or RewardCalculator()
        self.env = env
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # 加载任务定义
        self._tasks: Dict[str, Dict] = {}
        for tid in self.canary_task_ids:
            task = get_task_by_id(tid)
            if task is None:
                raise ValueError(f"Canary 任务不存在: {tid}")
            self._tasks[tid] = task

    @staticmethod
    def _build_prompt(task: Dict) -> str:
        """构造与训练时一致的 ReAct prompt（含工具描述与格式约束）。"""
        from agents.prompt_templates import format_tool_description, get_react_prompt
        from tools.tool_registry import registry

        tools_desc = format_tool_description(registry.list_tools())
        return get_react_prompt(task=task["task"], tools_description=tools_desc, history="")

    def evaluate(self, step: int = 0) -> CanaryResult:
        """运行一次 canary 评估。

        Returns:
            CanaryResult 包含聚合指标。
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        all_rewards: List[float] = []
        per_task: Dict[str, float] = {}
        completed: int = 0
        total: int = 0

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        with torch.no_grad():
            for tid, task in self._tasks.items():
                prompt = self._build_prompt(task)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                prompt_len = inputs.input_ids.shape[1]

                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        num_return_sequences=self.num_generations,
                        pad_token_id=pad_token_id,
                    )
                except Exception:
                    # 生成失败（如 OOM）→ 记录地板分
                    per_task[tid] = -1.0
                    all_rewards.extend([-1.0] * self.num_generations)
                    total += self.num_generations
                    continue

                # 只取新生成的 token（去掉 prompt 部分）
                generated_ids = outputs[:, prompt_len:]
                completions = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                task_rewards = []
                for completion in completions:
                    result = synthesize_trajectory(completion, env=self.env)
                    breakdown, _ = self.reward_calc.compute_reward_breakdown(task, result)
                    reward = float(breakdown.total)
                    task_rewards.append(reward)
                    if breakdown.outcome > 0:
                        completed += 1
                    total += 1

                per_task[tid] = (
                    sum(task_rewards) / len(task_rewards) if task_rewards else -1.0
                )
                all_rewards.extend(task_rewards)

        self.model.train()

        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else -1.0
        completion_rate = completed / total if total > 0 else 0.0

        return CanaryResult(
            step=step,
            mean_reward=round(mean_reward, 4),
            task_completion_rate=round(completion_rate, 4),
            per_task_rewards={k: round(v, 4) for k, v in per_task.items()},
            num_generations_per_task=self.num_generations,
        )


class CanaryCallback(TrainerCallback):
    """TRL 回调：每 N 步在固定 canary 任务上评估模型。

    若 canary 奖励连续 `patience` 次低于 `min_reward`，则停止训练。

    这捕获了 RewardVarianceGuard 覆盖不到的场景：模型学到了"钻空子"策略，
    奖励整体缓慢下降（组内方差仍 > 0，但绝对水平在退化）。
    """

    def __init__(
        self,
        evaluator: CanaryEvaluator,
        steps: int = 50,
        min_reward: float = -1.0,
        patience: int = 3,
    ):
        """
        Args:
            evaluator: CanaryEvaluator 实例（持有模型引用）
            steps: 每隔多少步评估一次
            min_reward: 奖励下限。默认 -1.0 表示禁用检查（-1 是理论最低分）
            patience: 连续低于阈值多少次后停止训练
        """
        self.evaluator = evaluator
        self.steps = steps
        self.min_reward = min_reward
        self.patience = patience
        self._below_streak = 0
        self._tripped = False
        self._results: List[CanaryResult] = []

    @property
    def tripped(self) -> bool:
        """是否已触发阈值停止。"""
        return self._tripped

    @property
    def results(self) -> List[CanaryResult]:
        """历次 canary 评估结果（按时间顺序）。"""
        return list(self._results)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.steps != 0:
            return

        result = self.evaluator.evaluate(step=state.global_step)
        self._results.append(result)

        # 打印 canary 报告
        print(
            f"\n🐤 Canary (step {result.step}): "
            f"mean_reward={result.mean_reward:.4f}  "
            f"completion_rate={result.task_completion_rate:.2%}  "
            f"per_task={result.per_task_rewards}"
        )

        # 阈值检查（min_reward == -1.0 时禁用）
        if self.min_reward <= -1.0:
            return

        if result.mean_reward < self.min_reward:
            self._below_streak += 1
            print(
                f"⚠️  Canary reward {result.mean_reward:.4f} 低于阈值 {self.min_reward} "
                f"({self._below_streak}/{self.patience})"
            )
            if self._below_streak >= self.patience:
                self._tripped = True
                control.should_training_stop = True
                recent = self._results[-self.patience:]
                print(
                    f"\n❌ Canary reward 连续 {self.patience} 次低于阈值 {self.min_reward}，"
                    f"训练已停止。\n"
                    f"   最近 canary 结果: {[(r.step, r.mean_reward) for r in recent]}\n"
                    f"   处理办法：\n"
                    f"     1) 换更小的学习率\n"
                    f"     2) 增大 beta（加强 KL 约束）\n"
                    f"     3) 检查任务是否全在难度谱两端\n"
                    f"   确实想继续跑，用 --min_canary_reward -1.0 跳过本检查。"
                )
        else:
            self._below_streak = 0