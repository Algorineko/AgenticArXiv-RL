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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import TrainerCallback

from agents.agent_engine import ReActAgent
from agents.react_contract import ParseKind, build_react_messages, parse_react_response, project_result
from benchmark.metrics import is_strict_success
from benchmark.tasks import get_task_by_id
from rl.env import MockArxivEnv
from rl.multiturn_env import AgenticArxivMultiTurnEnv
from rl.reward import RewardCalculator
from utils.llm_client import TransformersLLMClient

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
        from agents.prompt_templates import format_tool_description
        from tools.tool_registry import registry

        tools_desc = format_tool_description(registry.list_tools())
        return build_react_messages(task["task"], tools_desc)

    def _new_environment(self) -> AgenticArxivMultiTurnEnv:
        if self.environment_factory is not None:
            return self.environment_factory()
        source = self.env
        if isinstance(source, AgenticArxivMultiTurnEnv):
            source = source.backend
        if source is None:
            snapshot = Path(__file__).resolve().parents[2] / "data" / "mock_arxiv_snapshot.json"
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

        all_rewards: List[float] = []
        per_task: Dict[str, List[float]] = {}
        completed: int = 0
        total: int = 0
        parsed = decisions = 0
        termination_counts: Dict[str, int] = {}
        client = TransformersLLMClient.from_loaded(self.model, self.tokenizer, self.model_source)
        reference_oracle = "not_available"
        protocol: Dict[str, Any] = {}

        try:
            tasks = self._tasks
            source = self.env.backend if isinstance(self.env, AgenticArxivMultiTurnEnv) else self.env
            snapshot = getattr(source, "snapshot_path", None) if source is not None else (
                Path(__file__).resolve().parents[2] / "data" / "mock_arxiv_snapshot.json"
            )
            if self.environment_factory is None and snapshot and Path(snapshot).is_file():
                from benchmark.semantic_oracle import attach_expected_paper_ids

                tasks = attach_expected_paper_ids(tasks, snapshot)
                reference_oracle = "snapshot-gold-steps; evaluator-only"
            task_lookup = {task["id"]: task for task in tasks}
            for task in tasks:
                tid = task["id"]
                task_rewards = per_task.setdefault(tid, [])
                for trial in range(self.num_generations):
                    environment = self._new_environment()
                    agent = None
                    try:
                        agent = ReActAgent(
                            client, env=environment, max_iterations=self.max_iterations,
                            llm_extra={"temperature": self.temperature, "max_tokens": self.max_new_tokens},
                        )
                        result = agent.run(
                            task["task"], task_def=task, task_lookup=task_lookup,
                            session_id=f"canary_{tid}_{trial}"
                        )
                    finally:
                        if agent is not None:
                            agent.close()
                        else:
                            environment.close()
                    # 奖励带按 global step 变化的课程。Canary 必须使用当前训练
                    # step，否则跨过课程边界后会继续按 step=0 的早期权重打分，
                    # 与训练目标不可比。
                    breakdown, metrics = self.reward_calc.compute_reward_breakdown(
                        task, project_result(result), training_step=step
                    )
                    reward = float(breakdown.total)
                    task_rewards.append(reward)
                    if is_strict_success(metrics):
                        completed += 1
                    total += 1
                    all_rewards.append(reward)
                    reason = result.get("termination_reason") or metrics.termination_type
                    termination_counts[reason] = termination_counts.get(reason, 0) + 1
                    if not protocol:
                        protocol = dict(result.get("protocol_relation") or {})
                    for history_step in result["history"]:
                        # 纯框架标记不是模型决策；TOKEN_LIMIT 保留了实际采样文本，
                        # 即使它被框架截停，也须计入完整 episode 的解析率。
                        if history_step.get("framework_generated") and history_step.get("termination_reason") != "TOKEN_LIMIT":
                            continue
                        raw = history_step.get("raw_response")
                        if raw is None:
                            continue
                        decisions += 1
                        if parse_react_response(raw).kind != ParseKind.PARSE_ERROR:
                            parsed += 1
        finally:
            self.model.train(was_training)

        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else -1.0
        completion_rate = completed / total if total > 0 else 0.0

        return CanaryResult(
            step=step,
            mean_reward=round(mean_reward, 4),
            task_completion_rate=round(completion_rate, 4),
            per_task_rewards={k: round(sum(v) / len(v), 4) for k, v in per_task.items()},
            num_generations_per_task=self.num_generations,
            parse_rate=round(parsed / decisions, 4) if decisions else 0.0,
            task_ids=list(self.canary_task_ids),
            episode_count=total,
            max_iterations=self.max_iterations,
            termination_counts=termination_counts,
            reference_oracle=reference_oracle,
            protocol=protocol,
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
