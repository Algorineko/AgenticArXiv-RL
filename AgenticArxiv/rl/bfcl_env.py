"""BFCL 任务域环境 (轨道 B, 第二环境外部效度): BfclMultiTurnEnv — submit_calls 元工具的确定性打分面。

契约同 rl/code_env.py (再上溯 rl/multiturn_env.py): reset(task_id) + 每工具
一方法。无需快照回放 (打分 = benchmark/tasks_bfcl.score_chain 纯字符串/
结构匹配, 天然确定性)。任务索引懒加载: 首次用到才读 parquet, 全池 200 条
按 id 建索引缓存 (factory 每 rollout 造一个实例, 不能每次都重读数据)。

与代码域的关键差异: 标准答案不是 expected_tools 工具序列, 而是 gold 调用
链本身。RewardCalculator 的 outcome/tool/argument 消费的是 expected_tools
任务声明, 对 gold_calls 一无所知 (代码域实测: 奖励是纯结构分, run_tests
的通过与否并不进入奖励), 所以 BFCL 的奖励由本模块的 BfclRewardCalculator
把 score_chain 五分量映射进同一套 [-1,1] 分量接口 —— 加权框架、课程与
tracker 日志口径与其它域同构, 训练管线无需感知域差异。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from benchmark.tasks_bfcl import get_bfcl_tasks, score_chain
from rl.reward import RewardBreakdown, RewardSchedule

_POOL_INDEX: Optional[Dict[str, Dict[str, Any]]] = None

#: submit_calls 的 calls 载荷解析失败时写进 observation 的前缀 (同域内错误惯例)
_CALLS_ERROR_PREFIX = "提交格式错误: "


def _task_index() -> Dict[str, Dict[str, Any]]:
    """全池按 id 建索引; 模块级缓存, 整个训练进程只读一次 parquet。"""
    global _POOL_INDEX
    if _POOL_INDEX is None:
        _POOL_INDEX = {task["id"]: task for task in get_bfcl_tasks()}
    return _POOL_INDEX


def _parse_calls_payload(calls: Any) -> Tuple[Optional[List[str]], str]:
    """submit_calls 的 calls 载荷 -> 调用字符串列表 (失败返回 (None, 原因))。

    协议: calls 是 JSON 数组字符串, 元素形如 'name(k=v, ...)'。模型把数组
    直接作为 list 提交时同样放行 —— 协议违规应体现在 format 分量上, 而不是
    在工具面抛异常把整条轨迹变成工具执行失败。
    """
    if isinstance(calls, str):
        try:
            calls = json.loads(calls)
        except (json.JSONDecodeError, TypeError):
            return None, "calls 不是合法 JSON"
    if not isinstance(calls, list):
        return None, "calls 须为 JSON 数组"
    return [str(item) for item in calls], ""


def _action_payload(action: Any) -> Optional[Dict[str, Any]]:
    """history 里的 action 字段 -> 工具调用 dict (终止动作/非法 JSON 返回 None)。"""
    if isinstance(action, dict):
        return action
    if not isinstance(action, str):
        return None
    try:
        parsed = json.loads(action)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class BfclMultiTurnEnv:
    """单元工具环境: agent 一次性提交完整调用链 -> score_chain 五分量打分。"""

    def __init__(self) -> None:
        self._task: Optional[Dict[str, Any]] = None

    # ---- 工具面 ----
    def submit_calls(self, calls: Any = "", task_id: str = "", **_: Any) -> Dict[str, Any]:
        if self._task is None or (task_id and task_id != self._task["id"]):
            spec = _task_index().get(task_id)
            if spec is None:
                return {"ok": False, "error": f"unknown task_id: {task_id}"}
            self._task = spec
        gold = list(self._task["gold_calls"])
        pred, error = _parse_calls_payload(calls)
        if pred is None:
            # 载荷非法按空提交计: score_chain 空链 -> format/outcome 全 0
            return {
                "ok": False,
                "error": f"{_CALLS_ERROR_PREFIX}{error}",
                "task_id": self._task["id"],
                **score_chain([], gold),
                "n_pred": 0,
                "n_gold": len(gold),
            }
        # 发射前决策(2026-10-27): expected(gold 回显)移出 observation —— 它会经
        # rollout 的 Observation 回填进入策略上下文, 存在"看到 gold→重提交"泄漏
        # 通道; 审计侧用 gold_calls 字段自 task 池核对即可, 不需要环境回显。
        return {
            "ok": True,
            "task_id": self._task["id"],
            **score_chain(pred, gold),
            "n_pred": len(pred),
            "n_gold": len(gold),
        }

    # ---- 环境契约 ----
    def reset(self, task_id: str) -> None:
        spec = _task_index().get(task_id)
        if spec is None:
            raise KeyError(f"unknown bfcl task: {task_id}")
        self._task = spec


class BfclRewardCalculator:
    """BFCL 域奖励: score_chain 五分量 -> 与 RewardCalculator 同构的加权总分。

    接口与 RewardCalculator 对齐 (compute_reward_breakdown / schedule /
    get_weights / set_weights), tracker / canary / 课程日志无需感知域差异。
    分量映射 (score_chain 的 [0,1] -> 分量口径的 [-1,1], 同 _argument_score):

        format   <- format    可解析调用语法比例
        tool     <- call_name 逐调用名匹配 (≈工具选择)
        argument <- call_arg  逐调用参数匹配
        process  <- process   链前缀正确比例
        outcome  <- outcome   全链精确匹配

    多次 submit_calls 按最后一次计: 提交后 Observation 带分量反馈, 重提交是
    环境契约内的合法动作, 最终答案以最后一次为准; 未提交或载荷非法等价于
    提交空链 (分量全 0, 映射后为下限)。disable_failure_caps 仅为 CLI 兼容
    保留 —— 纯字符串打分, 无执行失败可封顶。
    """

    def __init__(
        self,
        curriculum_steps: int = 30,
        early_correctness_scale: float = 1.0 / 3.0,
        weights: Optional[Dict[str, float]] = None,
        disable_failure_caps: bool = False,
    ) -> None:
        self.curriculum_steps = max(0, curriculum_steps)
        self.early_correctness_scale = early_correctness_scale
        self.disable_failure_caps = disable_failure_caps
        # 权重与 RewardCalculator 默认值一致, 保证跨域分量曲线可比
        self.weights = {
            "format": 1.0,
            "tool": 3.0,
            "argument": 2.0,
            "process": 1.0,
            "outcome": 3.0,
        }
        if weights:
            self.set_weights(dict(weights))

    def schedule(self, training_step: int = 0) -> RewardSchedule:
        # 与 RewardCalculator.schedule 同一门课程: 早期把权重压在结构分量上
        scale = (
            self.early_correctness_scale
            if training_step < self.curriculum_steps
            else 1.0
        )
        return RewardSchedule(
            format=self.weights["format"],
            tool=self.weights["tool"] * scale,
            argument=self.weights["argument"] * scale,
            process=self.weights["process"],
            outcome=self.weights["outcome"] * scale,
        )

    def get_weights(self) -> Dict[str, float]:
        return self.weights.copy()

    def set_weights(self, new_weights: Dict[str, float]) -> None:
        unknown = set(new_weights) - set(self.weights)
        if unknown:
            raise ValueError(f"Unknown reward weights: {sorted(unknown)}")
        if any(value < 0 for value in new_weights.values()):
            raise ValueError("Reward weights must be non-negative")
        self.weights.update(new_weights)

    def compute_reward_breakdown(
        self,
        task_def: Dict[str, Any],
        result: Dict[str, Any],
        agent_type: str = "regex",
        trial: int = 0,
        session_id: str = "rl_train",
        training_step: int = 0,
    ) -> Tuple[RewardBreakdown, None]:
        history = result.get("history", [])
        gold = list(task_def.get("gold_calls") or [])
        submitted: Optional[List[str]] = None
        for step in history:
            payload = _action_payload(step.get("action", ""))
            if not payload or str(payload.get("name", "")) != "submit_calls":
                continue
            args = payload.get("args")
            args = args if isinstance(args, dict) else {}
            calls, _ = _parse_calls_payload(args.get("calls"))
            # 「最后一次提交为准」; 载荷非法等价于提交了空链
            submitted = calls if calls is not None else []
        raw = score_chain(submitted if submitted is not None else [], gold)
        components = {
            "format": 2.0 * raw["format"] - 1.0,
            "tool": 2.0 * raw["call_name"] - 1.0,
            "argument": 2.0 * raw["call_arg"] - 1.0,
            "process": 2.0 * raw["process"] - 1.0,
            "outcome": 2.0 * raw["outcome"] - 1.0,
        }
        schedule = self.schedule(training_step)
        active = asdict(schedule)
        # BFCL 只有五个分量; result_quality/efficiency 权重恒为 0, 不进分母
        denominator = sum(abs(weight) for weight in active.values()) or 1.0
        total = sum(active[name] * components[name] for name in components) / denominator
        breakdown = RewardBreakdown(
            total=round(_clip(total), 6),
            format=round(components["format"], 6),
            tool=round(components["tool"], 6),
            argument=round(components["argument"], 6),
            process=round(components["process"], 6),
            outcome=round(components["outcome"], 6),
            weights=schedule,
        )
        # BFCL 无 TaskMetrics 语义 (termination/router/指代解析都是 arXiv 域
        # 概念); 调用方 (grpo_reward/canary) 只消费 breakdown。
        return breakdown, None

    def compute_reward(
        self,
        task_def: Dict[str, Any],
        result: Dict[str, Any],
        agent_type: str = "regex",
        trial: int = 0,
        session_id: str = "rl_train",
        training_step: int = 0,
    ) -> Tuple[float, None]:
        breakdown, _ = self.compute_reward_breakdown(
            task_def, result, agent_type, trial, session_id, training_step
        )
        return breakdown.total, None


def make_bfcl_environment_factory():
    def factory():
        return BfclMultiTurnEnv()
    return factory


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))
