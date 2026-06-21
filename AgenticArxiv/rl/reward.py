"""奖励计算器（复用 benchmark/metrics.py 的 verifiable 组件）

Verifiable Reward 设计：
- 任务成功: +1.0
- 工具调用准确: +0.5
- 解析错误: -0.2 per failure
- 工具执行失败: -0.3 per failure
- 超时: -0.5
- 错误终止: -1.0
- 不必要调用: -0.1 per call
"""

from typing import Dict, Any, Tuple
from benchmark.metrics import TaskMetrics, compute_metrics


class RewardCalculator:
    """基于 verifiable metrics 的奖励计算器"""

    def __init__(self):
        """初始化奖励权重"""
        self.weights = {
            "task_completed": 1.0,
            "tool_call_accurate": 0.5,
            "parse_failure_penalty": -0.2,
            "tool_exec_failure_penalty": -0.3,
            "force_stop_penalty": -0.5,
            "error_penalty": -1.0,
            "unnecessary_call_penalty": -0.1,
        }

    def compute_reward(
        self,
        task_def: Dict[str, Any],
        result: Dict[str, Any],
        agent_type: str = "regex",
        trial: int = 0,
        session_id: str = "rl_train",
    ) -> Tuple[float, TaskMetrics]:
        """计算 reward + 返回 TaskMetrics

        Args:
            task_def: 任务定义（来自 benchmark/tasks.py）
            result: Agent 执行结果（history/timing/token_usage/iteration_count）
            agent_type: Agent 类型（默认 "regex"）
            trial: 试验次数（默认 0）
            session_id: 会话 ID

        Returns:
            (reward, metrics) 元组
        """
        # 复用 benchmark/metrics.py 的 compute_metrics
        metrics = compute_metrics(task_def, result, agent_type, trial, session_id)

        # 基于 metrics 计算 reward
        reward = 0.0

        # 正向奖励
        if metrics.task_completed:
            reward += self.weights["task_completed"]

        if metrics.tool_call_accurate:
            reward += self.weights["tool_call_accurate"]

        # 负向惩罚
        reward += self.weights["parse_failure_penalty"] * metrics.parse_failures
        reward += self.weights["tool_exec_failure_penalty"] * metrics.tool_exec_failures

        # 终止类型惩罚
        if metrics.termination_type == "FORCE_STOP":
            reward += self.weights["force_stop_penalty"]
        elif metrics.termination_type == "ERROR":
            reward += self.weights["error_penalty"]

        # 不必要调用惩罚（调用了不在 expected_tools 中的工具）
        if metrics.expected_tools:
            extra_calls = len(metrics.tool_call_sequence) - len(metrics.expected_tools)
            if extra_calls > 0:
                reward += self.weights["unnecessary_call_penalty"] * extra_calls

        return reward, metrics

    def get_weights(self) -> Dict[str, float]:
        """返回当前奖励权重配置"""
        return self.weights.copy()

    def set_weights(self, new_weights: Dict[str, float]) -> None:
        """更新奖励权重配置

        Args:
            new_weights: 新的权重字典（部分更新）
        """
        self.weights.update(new_weights)


def compute_step_reward(
    step_dict: Dict[str, Any], metrics: TaskMetrics
) -> float:
    """计算单步奖励（可选，用于 step-wise RL）

    Args:
        step_dict: 单步数据（thought/action/observation）
        metrics: 当前累积的 metrics

    Returns:
        单步奖励
    """
    step_reward = 0.0

    # 解析失败惩罚
    if step_dict.get("parse_failed", False):
        step_reward -= 0.2

    # 工具执行失败惩罚（根据 observation 判断）
    observation = step_dict.get("observation", "")
    if "错误" in observation or "Error" in observation or "失败" in observation:
        step_reward -= 0.3

    # 正常执行奖励（小奖励，鼓励有效推进）
    if not step_dict.get("parse_failed", False) and "成功" in observation:
        step_reward += 0.1

    return step_reward
