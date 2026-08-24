"""训练可观测性：把奖励曲线从 console scrollback 搬到 TensorBoard / wandb。

三个训练脚本此前的日志后端是不一致的，而且两种方式都不对：

  - train_grpo.py 写死 ``report_to=[]``，任何后端都进不去；
  - train_sft.py / train_dpo.py 根本不设 ``report_to``，HF 会解析成 "all"，
    也就是「装了什么就启用什么」—— 同一份代码在本机和集群上行为不同，
    集群上恰好装了 wandb 就会在训练开始时卡住等 API key。

这里给三个阶段统一一个显式的 ``--report_to``，并且在后端没装时**直接报错**，
而不是静默地什么都不记（静默失败正是这个仓库反复踩过的坑）。

除了 TRL 自带的 reward / kl / grad_norm，本模块还负责把五个奖励分量单独记出来。
这不是锦上添花：``RewardCalculator`` 带一个 30 步的课程，前期把 tool/argument/
outcome 的权重压到 1/3，之后恢复满权重。也就是说**只看 total reward 无法判断
曲线上升是策略变强还是权重表在动**。五个分量各自恒在 [-1, 1] 且与权重无关，
配合同时记录的当前权重，才能把这两件事分开。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

# HF 的集成名 -> 需要装的包名。用于在启用前给出可操作的报错。
_BACKEND_REQUIREMENTS = {
    "tensorboard": "tensorboard",
    "wandb": "wandb",
    "mlflow": "mlflow",
    "comet_ml": "comet_ml",
    "neptune": "neptune",
    "clearml": "clearml",
    "dvclive": "dvclive",
    "swanlab": "swanlab",
    "trackio": "trackio",
}


def resolve_report_to(value: Optional[str]) -> List[str]:
    """把 ``--report_to`` 的取值规范成 HF 认识的列表，并校验后端可用。

    与 HF 的默认行为有意不同：HF 在 ``report_to=None`` 时解析成 "all"，
    等于「装了什么用什么」。这里要求显式声明，且——

      - ``"none"`` / ``""``  -> ``[]``，明确表示不记录；
      - ``"auto"``          -> 装了 tensorboard 就用，没装则退回 ``[]`` 并提示；
      - 具体后端名           -> 没装就抛错，不静默降级。

    Returns:
        传给 ``TrainingArguments.report_to`` 的列表。
    """
    raw = (value or "none").strip().lower()
    if raw in ("none", "no", "off", ""):
        return []

    if raw == "auto":
        if _backend_available("tensorboard"):
            return ["tensorboard"]
        print("ℹ️  --report_to auto：未安装 tensorboard，本次不记录训练曲线。"
              "   需要曲线请先 pip install tensorboard")
        return []

    names = [name.strip() for name in raw.split(",") if name.strip()]
    for name in names:
        if name not in _BACKEND_REQUIREMENTS:
            raise SystemExit(
                f"❌ 未知的 --report_to 后端: {name}\n"
                f"   可选: none / auto / {' / '.join(sorted(_BACKEND_REQUIREMENTS))}"
            )
        if not _backend_available(name):
            package = _BACKEND_REQUIREMENTS[name]
            raise SystemExit(
                f"❌ --report_to {name} 需要先安装 {package}: pip install {package}\n"
                "   （这里选择直接失败而不是静默跳过：训练跑完却没有任何曲线，"
                "比启动时报错难查得多）"
            )
    return names


def _backend_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(_BACKEND_REQUIREMENTS.get(name, name)) is not None


class RewardComponentTracker:
    """收集每条 rollout 的奖励分量与轨迹健康度，汇总进训练日志。

    走 TRL ``GRPOTrainer._metrics`` 这个口子：TRL 在 ``log()`` 里会把它按 key
    取均值、并进 ``logs``、再分发给所有 report_to 后端，最后清空。也就是说
    这里只管 append，聚合与清空都由 TRL 负责，梯度累积下的多次调用会被正确
    平均成一个点。

    ``bind`` 失败（TRL 改了内部结构）时对象退化成 no-op —— 训练不该因为
    日志记不了就崩，但会打印一次提示，避免变成又一处静默失效。
    """

    #: 与 RewardBreakdown 对齐的五个分量
    COMPONENTS = ("format", "tool", "argument", "process", "outcome")

    def __init__(self) -> None:
        self._sink: Optional[Dict[str, List[float]]] = None
        self._pending: Dict[str, List[float]] = {}
        self.bound = False

    # --- 接线 -------------------------------------------------------------
    def bind(self, trainer: Any) -> bool:
        """接到 trainer 的指标缓冲区。返回是否接上。"""
        metrics = getattr(trainer, "_metrics", None)
        if not isinstance(metrics, Mapping) or "train" not in metrics:
            print("⚠️  当前 TRL 版本没有可用的 _metrics 缓冲区，"
                  "奖励分量曲线本次不会被记录（训练本身不受影响）")
            return False
        self._sink = metrics["train"]
        self.bound = True
        # bind 之前 record 的数据不丢
        for key, values in self._pending.items():
            self._sink[key].extend(values)
        self._pending.clear()
        return True

    def _emit(self, key: str, value: float) -> None:
        if self.bound:
            self._sink[key].append(float(value))
        else:
            self._pending.setdefault(key, []).append(float(value))

    # --- 采集 -------------------------------------------------------------
    def record(self, breakdown: Any, trajectory: Optional[Mapping[str, Any]] = None) -> None:
        """记录一条 rollout 的奖励分量、当前课程权重与轨迹健康度。"""
        for name in self.COMPONENTS:
            value = getattr(breakdown, name, None)
            if value is not None:
                self._emit(f"reward_components/{name}", value)

        weights = getattr(breakdown, "weights", None)
        if weights is not None:
            # 同时记权重，否则无法区分「策略变强」与「课程把权重放开了」
            for name, weight in asdict(weights).items():
                self._emit(f"reward_weights/{name}", weight)

        if trajectory is not None:
            self._record_trajectory(trajectory)

    def _record_trajectory(self, trajectory: Mapping[str, Any]) -> None:
        history = trajectory.get("history") or []
        if not history:
            return
        stats = trajectory_health(history)
        for key, value in stats.items():
            self._emit(f"rollout/{key}", value)


def trajectory_health(history: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """从一条 ReAct 轨迹里抽出多轮 rollout 的健康度指标。

    多轮采样落地后，``reward`` 掉下去可能是策略退化，也可能是环境侧出了问题
    （比如工具一直报错、或者模型压根没学会收尾而是每次都跑满 max_turns）。
    这几个量把这两类原因分开。

    Returns:
        - ``turns``：这条轨迹用了几步
        - ``finished``：是否以 FINISH 正常收尾（0/1）。持续为 0 说明模型在
          「跑满轮数」而不是「决定结束」，此时 reward 低与工具能力无关。
        - ``parse_error_rate``：解析失败步数占比
        - ``tool_error_rate``：工具调用步中执行失败的占比
    """
    from rl.grpo_reward import TOOL_ERROR_PREFIX

    turns = len(history)
    actions = [str(step.get("action", "")) for step in history]
    parse_errors = sum(
        1 for step, action in zip(history, actions)
        if step.get("parse_failed") or action == "PARSE_ERROR"
    )
    tool_steps = [
        step for step, action in zip(history, actions)
        if action not in ("FINISH", "PARSE_ERROR")
    ]
    tool_errors = sum(
        1 for step in tool_steps
        if str(step.get("observation", "")).startswith(TOOL_ERROR_PREFIX)
    )
    return {
        "turns": float(turns),
        "finished": 1.0 if actions and actions[-1] == "FINISH" else 0.0,
        "parse_error_rate": parse_errors / turns,
        "tool_error_rate": (tool_errors / len(tool_steps)) if tool_steps else 0.0,
    }


def describe_logging(report_to: Sequence[str], logging_dir: Optional[str]) -> str:
    """训练启动时打印一行人能看懂的说明（含查看命令）。"""
    if not report_to:
        return ("📉 未启用训练曲线记录（--report_to none）。"
                "要看 reward / kl / 各奖励分量，用 --report_to tensorboard")
    if "tensorboard" in report_to and logging_dir:
        return (f"📈 训练曲线 -> {report_to}，日志目录 {logging_dir}\n"
                f"   查看: tensorboard --logdir {logging_dir}")
    return f"📈 训练曲线 -> {report_to}"
