"""Lossless native-tool trace transport in the original Trajectory container.

The embedded native trace is authoritative. Legacy display fields contain no
inferred chain of thought and must not be treated as a reconstructed ReAct run.
"""
from copy import deepcopy
import json

from AgenticArxiv.rl.trajectory import Trajectory, TrajectoryStep


def to_trajectory(trace, task_prompt="", model=""):
    steps = []
    for step in trace["steps"]:
        steps.append(TrajectoryStep(
            step=step["step"] + 1,
            thought="",
            action=json.dumps(step["assistant"], ensure_ascii=False, sort_keys=True),
            observation=json.dumps(step["observations"], ensure_ascii=False, sort_keys=True),
        ))
    return Trajectory(
        task_id=trace["task_id"], task=task_prompt,
        session_id=trace["episode_id"], steps=steps,
        final_reward=float(bool(trace["verdict"]["success"])),
        metrics={"rsi_native_trace": deepcopy(trace),
                 "format": "rsi_native_trace_v1",
                 "reward_semantics": "independent terminal goal indicator; no legacy shaped reward"},
        timestamp=trace["started_utc"], model=model,
        termination_type=trace["stop"],
        reward_components={"independent_goal_success": bool(trace["verdict"]["success"])},
    )


def from_trajectory(trajectory):
    if trajectory.metrics.get("format") != "rsi_native_trace_v1":
        raise ValueError("Not a native RSI trace container")
    return deepcopy(trajectory.metrics["rsi_native_trace"])
