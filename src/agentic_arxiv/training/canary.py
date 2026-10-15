"""Training canary: periodic evaluation on a fixed small task set during training.

During GRPO training, the model's reward can fluctuate. The canary provides
an early warning signal: if performance on a fixed set of tasks starts
degrading, training can be stopped before wasting more compute.

与 RewardVarianceGuard 的分工：
- RewardVarianceGuard：检测"组内奖励方差为 0"（梯度为零，训练空转）
- CanaryCallback：检测"模型在固定任务上性能退化"（学到错误策略）

Usage:
    from agentic_arxiv.training.canary import CanaryEvaluator, CanaryCallback

    evaluator = CanaryEvaluator(model, tokenizer, canary_task_ids, reward_calc, env)
    callback = CanaryCallback(evaluator, steps=50, min_reward=-0.5, patience=3)
    trainer.add_callback(callback)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

from agentic_arxiv.config import workspace_path
from agentic_arxiv.evaluation.episode import evaluate_episodes
from agentic_arxiv.runtime.react_contract import build_react_messages
from agentic_arxiv.tasks.tasks import get_task_by_id
from agentic_arxiv.runtime.env import MockArxivEnv
from agentic_arxiv.runtime.multiturn_env import AgenticArxivMultiTurnEnv
from agentic_arxiv.scoring.reward import RewardCalculator
from agentic_arxiv.llm.client import TransformersLLMClient
from agentic_arxiv.training.callbacks import TrainerCallback

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
    parse_rate: float = 0.0
    task_ids: List[str] = field(default_factory=list)
    episode_count: int = 0
    max_iterations: int = 6
    termination_counts: Dict[str, int] = field(default_factory=dict)
    reference_oracle: str = "not_available"
    protocol: Dict[str, Any] = field(default_factory=dict)
    false_finish_rate: float = 0.0
    false_finish_count: int = 0
    strict_success_count: int = 0
    parsed_decision_count: int = 0
    model_decision_count: int = 0
    temperature: float = 1.0
    max_new_tokens: int = 256
    scoring: Dict[str, Any] = field(default_factory=dict)


class CanaryEvaluator:
    """在固定 canary 任务集上评估模型。

    每个采样独立执行完整 episode，只从实际轨迹计算奖励与完成率。
    默认仅离线回放，缺少输入明确失败，不自动调用在线工具。
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
        max_iterations: int = 6,
        environment_factory: Callable[[], AgenticArxivMultiTurnEnv] | None = None,
        model_source: str = "",
        artifact: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.canary_task_ids = list(canary_task_ids or DEFAULT_CANARY_TASK_IDS)
        self.reward_calc = reward_calc or RewardCalculator()
        self.env = env
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.environment_factory = environment_factory
        self.model_source = model_source
        self.artifact = artifact
        if num_generations < 1 or max_iterations < 1:
            raise ValueError("Canary 采样数和完整 episode 轮数必须为正数")

        # 加载任务定义
        self._tasks: List[Dict] = []
        for tid in self.canary_task_ids:
            task = get_task_by_id(tid)
            if task is None:
                raise ValueError(f"Canary 任务不存在: {tid}")
            self._tasks.append(task)

    @staticmethod
    def _build_messages(task: Dict) -> List[Dict]:
        """验证辅助输入也使用共同初始消息，不保留另一种上下文实现。"""
        from agentic_arxiv.runtime.prompt_templates import format_tool_description
        from agentic_arxiv.tools.tool_registry import registry

        tools_desc = format_tool_description(registry.list_tools())
        return build_react_messages(task["task"], tools_desc)

    def _new_environment(self) -> AgenticArxivMultiTurnEnv:
        if self.environment_factory is not None:
            return self.environment_factory()
        source = self.env
        if isinstance(source, AgenticArxivMultiTurnEnv):
            source = source.backend
        if source is None:
            snapshot = workspace_path("data/mock_arxiv_snapshot.json")
            return AgenticArxivMultiTurnEnv(snapshot_path=snapshot, mode="replay")
        if not isinstance(source, MockArxivEnv):
            raise TypeError("Canary env 必须是共同 episode 环境或 MockArxivEnv；也可显式传 factory")
        if source.mode != "replay":
            raise ValueError("Canary 默认离线回放，禁止自动联网后端；在线运行须显式传 environment_factory")
        # 快照和后端可变状态的绑定/隔离只由共同环境负责。
        return AgenticArxivMultiTurnEnv(backend=source, mode="replay")

    def evaluate(self, step: int = 0) -> CanaryResult:
        """运行一次 canary 评估。

        Returns:
            CanaryResult 包含聚合指标。
        """
        was_training = bool(self.model.training)
        self.model.eval()

        client = TransformersLLMClient.from_loaded(
            self.model, self.tokenizer, self.model_source, artifact=self.artifact,
        )
        reference_oracle = "not_available"

        try:
            tasks = self._tasks
            source = self.env.backend if isinstance(self.env, AgenticArxivMultiTurnEnv) else self.env
            snapshot = getattr(source, "snapshot_path", None) if source is not None else (
                workspace_path("data/mock_arxiv_snapshot.json")
            )
            if self.environment_factory is None and snapshot and Path(snapshot).is_file():
                from agentic_arxiv.tasks.reference import attach_expected_paper_ids

                tasks = attach_expected_paper_ids(tasks, snapshot)
                reference_oracle = "snapshot-gold-steps; evaluator-only"
            # 传入当前训练 step，保留奖励课程边界与训练目标的一致性。
            evaluation = evaluate_episodes(
                client, tasks, self._new_environment, self.reward_calc,
                num_generations=self.num_generations,
                max_iterations=self.max_iterations,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                training_step=step,
                session_prefix="canary",
            )
        finally:
            self.model.train(was_training)

        return CanaryResult(
            step=step,
            task_ids=list(self.canary_task_ids),
            reference_oracle=reference_oracle,
            **evaluation.__dict__,
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
        distributed=None,
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
        self.distributed = distributed
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

        def evaluate():
            return self.evaluator.evaluate(step=state.global_step)

        result = (self.distributed.run_on_main(evaluate)
                  if self.distributed is not None else evaluate())
        self._results.append(result)

        # 打印 canary 报告
        self._print(
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
            self._print(
                f"⚠️  Canary reward {result.mean_reward:.4f} 低于阈值 {self.min_reward} "
                f"({self._below_streak}/{self.patience})"
            )
            if self._below_streak >= self.patience:
                self._tripped = True
                control.should_training_stop = True
                recent = self._results[-self.patience:]
                self._print(
                    f"\n❌ Canary reward 连续 {self.patience} 次低于阈值 {self.min_reward}，"
                    f"训练已停止。\n"
                    f"   最近 canary 结果: {[(r.step, r.mean_reward) for r in recent]}\n"
                    f"   处理办法：\n"
                    f"     1) 换更小的学习率\n"
                    f"     2) 增大 beta（加强 KL 约束）\n"
                    f"     3) 检查任务是否全在难度谱两端\n"
                    f"   确实想继续跑，用 --min-canary-reward -1.0 跳过本检查。"
                )
        else:
            self._below_streak = 0

    def _print(self, message):
        if self.distributed is None or self.distributed.is_main:
            print(message)
