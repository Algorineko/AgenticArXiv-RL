"""Trajectory 数据类 + JSONL 读写

轨迹记录格式：
- 每条轨迹包含多个 step（thought/action/observation）
- 最终包含 reward 和 metrics
- 以 JSONL 格式存储（每行一条完整轨迹）
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime


@dataclass
class TrajectoryStep:
    """单个执行步骤"""

    step: int
    thought: str
    action: str  # JSON 字符串 或 "FINISH"
    observation: str
    llm_latency_ms: int = 0
    tool_latency_ms: int = 0
    parse_failed: bool = False


@dataclass
class Trajectory:
    """完整轨迹"""

    task_id: str
    task: str  # 任务描述
    session_id: str
    steps: List[TrajectoryStep]
    final_reward: float
    metrics: Dict[str, Any]  # TaskMetrics 的字典形式
    timestamp: str
    model: str = ""
    termination_type: str = ""  # "FINISH" | "FORCE_STOP" | "ERROR"


def save_trajectory(traj: Trajectory, filepath: Path) -> None:
    """保存单条 trajectory 到 JSONL（追加模式）

    Args:
        traj: Trajectory 对象
        filepath: JSONL 文件路径
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # 转为字典（嵌套 dataclass 也会被正确转换）
    traj_dict = asdict(traj)

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(traj_dict, ensure_ascii=False) + "\n")


def load_trajectories(filepath: Path) -> List[Trajectory]:
    """从 JSONL 加载 trajectories

    Args:
        filepath: JSONL 文件路径

    Returns:
        Trajectory 对象列表
    """
    if not filepath.exists():
        return []

    trajectories = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            # 重建嵌套的 TrajectoryStep 对象
            steps_data = data.pop("steps", [])
            steps = [TrajectoryStep(**step) for step in steps_data]

            traj = Trajectory(steps=steps, **data)
            trajectories.append(traj)

    return trajectories


def create_trajectory(
    task_id: str,
    task: str,
    session_id: str,
    history: List[Dict[str, Any]],
    final_reward: float,
    metrics: Dict[str, Any],
    model: str = "",
    termination_type: str = "FINISH",
) -> Trajectory:
    """从 Agent 执行结果构造 Trajectory

    Args:
        task_id: 任务 ID
        task: 任务描述
        session_id: 会话 ID
        history: Agent 执行历史（list of dict）
        final_reward: 最终奖励
        metrics: TaskMetrics 字典
        model: 模型名称
        termination_type: 终止类型

    Returns:
        Trajectory 对象
    """
    steps = []
    for i, step_dict in enumerate(history):
        step = TrajectoryStep(
            step=i + 1,
            thought=step_dict.get("thought", ""),
            action=step_dict.get("action", ""),
            observation=step_dict.get("observation", ""),
            llm_latency_ms=step_dict.get("llm_latency_ms", 0),
            tool_latency_ms=step_dict.get("tool_latency_ms", 0),
            parse_failed=step_dict.get("parse_failed", False),
        )
        steps.append(step)

    return Trajectory(
        task_id=task_id,
        task=task,
        session_id=session_id,
        steps=steps,
        final_reward=final_reward,
        metrics=metrics,
        timestamp=datetime.now().isoformat(),
        model=model,
        termination_type=termination_type,
    )
