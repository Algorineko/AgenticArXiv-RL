"""trl 的旧版 torch 兼容层。

TRL 的 GRPOTrainer 在内部无条件地
``from torch.distributed.fsdp import FSDPModule``，而 ``FSDPModule`` 只
在 torch>=2.6 才存在。因此在使用 torch<2.6（例如本仓库本地 RTX 4050 +
torch 2.5.x）时，仅仅 ``from trl import ...`` 就会直接崩溃：:

    ImportError: cannot import name 'FSDPModule' from 'torch.distributed.fsdp'

单卡训练（大多数用户的场景）根本不需要真正的 FSDP 分布式打包，所以这里
在导入 trl 之前补一个最小可用的占位类，让 GRPO 的
``prepare_fsdp`` 能正常通过 while isinstance 判断、正常走非 FSDP 分支。

用法：在 ``from trl import ...`` 之前 ``import agentic_arxiv.training.trl_compat``。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# 目标：早期是 torch.distributed.fsdp.FSDPModule
_ATTACH_TARGET: str = "torch.distributed.fsdp"


def ensure_fsdp_compat() -> bool:
    """在缺 FSDPModule 时注入占位类，返回是否真的注入了兜底。"""
    import importlib

    try:
        fsdp = importlib.import_module(_ATTACH_TARGET)
    except ImportError:
        # torch 太老连 fsdp 模块都没有，直接放弃兜底，让后续 import 报它自己的错
        return False

    if hasattr(fsdp, "FSDPModule"):
        return False  # torch>=2.6，什么都不用做

    class _FSDPModulePlaceholder:
        """仅用于让 trl 的 isinstance / 类型标注通过的最小占位。"""

        def __init__(self, *args, **kwargs):
            pass

    setattr(fsdp, "FSDPModule", _FSDPModulePlaceholder)
    logger.warning(
        "torch<2.6 未提供 FSDPModule，已注入单卡兼容占位。"
        "本仓库的单卡训练不需要真实 FSDP；若你需要多卡 FSDP，请升级到 torch>=2.6。"
    )
    return True


# 模块导入即生效：任何训练脚本 import 本模块后，再 import trl 就不会崩
ensure_fsdp_compat()


MANAGED_GRPO_TRL_VERSION = "0.29.1"


def require_managed_grpo_interfaces(trainer, *, installed_version=None):
    """固定版本的私有接口不匹配时拒绝受管恢复，绝不丢弃生成缓存。"""
    from importlib.metadata import version
    import inspect

    installed = installed_version or version("trl")
    if installed != MANAGED_GRPO_TRL_VERSION:
        raise ValueError(f"受管 GRPO 只支持 trl={MANAGED_GRPO_TRL_VERSION}，实际为 {installed}")
    required = ("_step", "_buffered_inputs", "_metrics", "_logs", "_total_train_tokens",
                "_current_train_step_time", "num_iterations", "ref_model")
    missing = [name for name in required if not hasattr(trainer, name)]
    if missing:
        raise ValueError(f"TRL GRPO 恢复接口缺失: {missing}")
    for name, parameters in {
        "_prepare_inputs": ("inputs", "generation_batch"),
        "training_step": ("num_items_in_batch",),
        "_get_train_sampler": (),
        "_load_from_checkpoint": ("resume_from_checkpoint",),
    }.items():
        method = getattr(trainer, name, None)
        if not callable(method):
            raise ValueError(f"TRL GRPO 恢复接口缺失: {name}")
        signature = inspect.signature(method)
        if parameters and not any(key in signature.parameters for key in parameters):
            raise ValueError(f"TRL GRPO 恢复接口签名变化: {name}{signature}")
    if getattr(trainer, "use_vllm", False) or getattr(trainer.args, "sync_ref_model", False):
        raise ValueError("受管 GRPO 只支持无 vLLM、原始冻结参考的固定版本路径")


def _state_tree(value, *, device=None):
    """缓存完整保留所有字段；张量保存到 CPU，恢复到当前 rank 的设备。"""
    import copy
    import torch
    from collections import deque

    if torch.is_tensor(value):
        return value.detach().to(device or "cpu").clone()
    if isinstance(value, dict):
        return {key: _state_tree(item, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [_state_tree(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_state_tree(item, device=device) for item in value)
    if isinstance(value, deque):
        return deque((_state_tree(item, device=device) for item in value), maxlen=value.maxlen)
    return copy.deepcopy(value)


def _reference_parameters(trainer):
    """只枚举实际 KL 参考参数，不把可训练策略混入参考身份。"""
    if trainer.beta == 0:
        return []
    if trainer.ref_model is not None:
        return list(trainer.ref_model.named_parameters())
    model = trainer.accelerator.unwrap_model(trainer.model)
    if "ref" not in getattr(model, "peft_config", {}):
        raise ValueError("受管 GRPO 缺少原始冻结 ref adapter")
    return [(name, parameter) for name, parameter in model.named_parameters() if ".ref." in name]


def reference_weight_fingerprint(trainer):
    """不采样、不改变随机流，记录实际冻结参考的权重内容。"""
    import hashlib
    import torch

    digest = hashlib.sha256()
    parameters = _reference_parameters(trainer)
    if trainer.beta != 0 and not parameters:
        raise ValueError("受管 GRPO 的冻结参考参数为空")
    optimizer_ids = {id(item) for group in getattr(getattr(trainer, "optimizer", None),
                                                "param_groups", []) for item in group["params"]}
    for name, parameter in parameters:
        if parameter.requires_grad or id(parameter) in optimizer_ids:
            raise ValueError(f"GRPO 参考必须冻结且排除优化器: {name}")
        value = parameter.detach().contiguous().cpu()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {"kind": "not_applicable" if trainer.beta == 0 else "frozen_reference",
            "sha256": digest.hexdigest(), "parameter_count": len(parameters)}


class GRPOResumeState:
    """TRL 0.29.1 每 rank 附加状态；标准 Trainer 仍负责数据跳过和随机状态。"""

    def __init__(self, trainer, execution, reference_identity, *, tracker=None, auditor=None,
                 guard=None, canary=None, curriculum_steps=0):
        self.trainer = trainer
        self.execution = execution
        self.reference_identity = reference_identity
        self.reference_weights = reference_weight_fingerprint(trainer)
        self.tracker = tracker
        self.auditor = auditor
        self.guard = guard
        self.canary = canary
        self.curriculum_steps = curriculum_steps
        self.restored_step = None
        self.restored_planned_steps = None
        self.inherited_audit = None
        self.sampler = None
        self.sampler_epoch_start = None
        self.pending_sampler_state = None

    def install_policy_adapter_restore(self):
        """补齐固定 Trainer 对根 default 与子目录 ref 共存时的策略恢复。"""
        from pathlib import Path
        from types import MethodType
        from agentic_arxiv.training.checkpoint import preserve_adapter_restore_contract

        trainer = self.trainer
        if trainer.beta == 0 or trainer.ref_model is not None:
            return
        original = trainer._load_from_checkpoint

        def load_checkpoint(instance, resume_from_checkpoint, model=None):
            target = instance.model if model is None else model
            if (set(getattr(target, "peft_config", {})) != {"default", "ref"}
                    or target.active_adapters != ["default"]):
                raise ValueError("受管 GRPO adapter 恢复要求活动 default 策略和冻结 ref")
            root = Path(resume_from_checkpoint)
            if not any((root / name).is_file()
                       for name in ("adapter_model.safetensors", "adapter_model.bin")):
                raise ValueError("受管 GRPO checkpoint 缺少根 default 策略权重")
            # Transformers 4.57.6 发现 ref/ 后只加载子目录，跳过根 default。
            # 保留标准恢复时序，并在优化器恢复和首次 rollout 之前补载策略。
            with preserve_adapter_restore_contract(instance, target):
                result = original(resume_from_checkpoint, model=model)
                loaded = target.load_adapter(str(root), "default", is_trainable=True,
                                             local_files_only=True, autocast_adapter_dtype=False)
                if loaded.missing_keys or loaded.unexpected_keys:
                    raise ValueError("受管 GRPO checkpoint 的 default 策略权重不完整或不匹配")
                if reference_weight_fingerprint(instance) != self.reference_weights:
                    raise ValueError("GRPO 恢复后原始冻结参考权重不一致")
            return result

        trainer._load_from_checkpoint = MethodType(load_checkpoint, trainer)

    def install_sampler_tracking(self):
        """旁路记录 RepeatSampler 的局部 RNG，不改变索引、组或迭代次数。"""
        from types import MethodType

        original = self.trainer._get_train_sampler
        controller = self

        class TrackedSampler:
            def __init__(self, sampler):
                self.sampler = sampler
                controller.sampler = sampler

            def __getattr__(self, name):
                # Accelerate 仍须看到原 sampler.generator 并执行原有 rank 同步。
                return getattr(self.sampler, name)

            def __len__(self):
                return len(self.sampler)

            def __iter__(self):
                generator = getattr(self.sampler, "generator", None)
                if generator is not None:
                    if controller.pending_sampler_state is not None:
                        generator.set_state(controller.pending_sampler_state)
                        controller.pending_sampler_state = None
                    controller.sampler_epoch_start = generator.get_state().clone()
                yield from self.sampler

        def tracked_get_sampler(trainer, *args, **kwargs):
            return TrackedSampler(original(*args, **kwargs))

        self.trainer._get_train_sampler = MethodType(tracked_get_sampler, self.trainer)

    def _layout(self):
        args = self.trainer.args
        return {"steps_per_generation": args.steps_per_generation,
                "num_iterations": self.trainer.num_iterations,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "num_generations": self.trainer.num_generations}

    def _validate_position(self, state):
        step = state["microstep"]
        layout = self._layout()
        if type(step) is not int or step < 0:
            raise ValueError("GRPO 微步位置非法")
        window = layout["steps_per_generation"] * layout["num_iterations"]
        if min(layout.values()) < 1:
            raise ValueError("GRPO 生成窗口配置非法")
        if state["generation_position"] != step % window or state["slice_position"] != step % layout["steps_per_generation"]:
            raise ValueError("GRPO 生成缓存消费位置不一致")
        cached = state["buffered_inputs"]
        if step % window and cached is None:
            raise ValueError("GRPO 生成窗口尚未消费完，禁止丢弃缓存重新采样")
        if cached is not None and (not isinstance(cached, list) or len(cached) != layout["steps_per_generation"]
                                   or any(not isinstance(item, dict) for item in cached)):
            raise ValueError("GRPO 生成缓存切片数量或结构不一致")
        if type(state["global_step"]) is not int or not 0 <= state["global_step"] <= step:
            raise ValueError("GRPO 逻辑步数与微步位置不一致")
        if (type(state.get("planned_steps")) is not int or state["planned_steps"] < 1
                or state["planned_steps"] < state["global_step"]):
            raise ValueError("GRPO 缺少有效的原始核定优化步数预算")

    def state_dict(self):
        trainer = self.trainer
        layout = self._layout()
        state = {"schema_version": 1, "trl_version": MANAGED_GRPO_TRL_VERSION,
                 "rank": self.execution.rank, "world_size": self.execution.world_size,
                 "reference_identity": self.reference_identity,
                 "reference_weights": self.reference_weights,
                 "curriculum_steps": self.curriculum_steps,
                 "requested_num_train_epochs": trainer.args.num_train_epochs,
                 "requested_max_steps": trainer.args.max_steps,
                 "planned_steps": trainer.state.max_steps,
                 "global_step": int(trainer.state.global_step), "microstep": trainer._step,
                 "layout": layout,
                 "generation_position": trainer._step % (layout["steps_per_generation"] * layout["num_iterations"]),
                 "slice_position": trainer._step % layout["steps_per_generation"],
                 "buffered_inputs": _state_tree(trainer._buffered_inputs),
                 "metrics": _state_tree(trainer._metrics), "logs": _state_tree(trainer._logs),
                 "total_train_tokens": trainer._total_train_tokens,
                 "current_train_step_time": trainer._current_train_step_time,
                 "sampler": self._sampler_state(),
                 "guard": self.guard.state_dict() if self.guard is not None else None,
                 "canary": self.canary.state_dict() if self.canary is not None else None,
                 "tracker": ({key: _state_tree(getattr(self.tracker, key)) for key in
                              ("_pending", "_task_rewards", "_task_group_stds")}
                             if self.tracker is not None else None),
                 "audit": self.auditor.summary() if self.auditor is not None else None,
                 "inherited_audit": self.inherited_audit}
        self._validate_position(state)
        if reference_weight_fingerprint(trainer) != self.reference_weights:
            raise ValueError("GRPO 冻结参考权重发生变化")
        return state

    def _sampler_state(self):
        if self.sampler is None:
            # 未进行任何数据消费时无排列状态；非零步必须有真实 sampler 记录。
            if self.trainer._step:
                raise ValueError("GRPO 采样器尚未接线，不能发布完整恢复状态")
            return {"schema_version": 1, "shuffle": None, "epoch": 0.0,
                    "epoch_start": None, "generator_current": None}
        generator = getattr(self.sampler, "generator", None)
        return {"schema_version": 1, "shuffle": bool(self.sampler.shuffle),
                "epoch": float(self.trainer.state.epoch or 0.0),
                "epoch_start": _state_tree(self.sampler_epoch_start),
                "generator_current": generator.get_state().clone() if generator is not None else None}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or type(state.get("schema_version")) is not int
                or type(state.get("rank")) is not int or type(state.get("world_size")) is not int):
            raise ValueError("GRPO 恢复状态版本或 rank 类型非法")
        expected = {"schema_version": 1, "trl_version": MANAGED_GRPO_TRL_VERSION,
                    "rank": self.execution.rank, "world_size": self.execution.world_size,
                    "layout": self._layout(), "reference_identity": self.reference_identity,
                    "reference_weights": self.reference_weights, "curriculum_steps": self.curriculum_steps}
        expected.update(requested_num_train_epochs=self.trainer.args.num_train_epochs,
                        requested_max_steps=self.trainer.args.max_steps)
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"GRPO 恢复状态不一致: {key}")
        self._validate_position(state)
        sampler = state["sampler"]
        if type(sampler.get("schema_version")) is not int or sampler.get("schema_version") != 1:
            raise ValueError("未知的 GRPO 采样位置状态版本")
        if sampler["shuffle"]:
            if sampler["epoch_start"] is None or sampler["generator_current"] is None:
                raise ValueError("GRPO 缺少局部采样器随机状态")
            epoch = sampler["epoch"]
            # 完整 epoch 的下一批需下一排列；epoch 中途则重建同一排列后由
            # Trainer skip_first_batches 跳过已经提交的批次，不重新采样策略。
            self.pending_sampler_state = _state_tree(
                sampler["generator_current"] if epoch > 0 and float(epoch).is_integer()
                else sampler["epoch_start"])
        for name, callback in (("guard", self.guard), ("canary", self.canary)):
            if (state[name] is None) != (callback is None):
                raise ValueError(f"GRPO 恢复保护配置不一致: {name}")
            if callback is not None:
                callback.load_state_dict(state[name])
        trainer = self.trainer
        trainer._step = state["microstep"]
        trainer._buffered_inputs = _state_tree(state["buffered_inputs"], device=trainer.accelerator.device)
        for mode in ("train", "eval"):
            # tracker 持有 train sink 引用，必须原地更新，不能替换其字典。
            trainer._metrics[mode].clear()
            trainer._metrics[mode].update(_state_tree(state["metrics"][mode]))
        trainer._logs.clear()
        trainer._logs.update(_state_tree(state["logs"]))
        # 恢复 defaultdict 的工厂；checkpoint 不序列化 TRL 的局部 lambda。
        from collections import defaultdict, deque
        trainer._logs["rewards"] = defaultdict(
            lambda: deque(maxlen=trainer.args.generation_batch_size), trainer._logs["rewards"])
        trainer._total_train_tokens = state["total_train_tokens"]
        trainer._current_train_step_time = state["current_train_step_time"]
        if (state["tracker"] is None) != (self.tracker is None):
            raise ValueError("GRPO 恢复统计配置不一致")
        if self.tracker is not None:
            for key, value in state["tracker"].items():
                setattr(self.tracker, key, _state_tree(value))
        self.inherited_audit = {"committed_global_step": state["global_step"],
                                "source_summary": state["audit"],
                                "previous": state.get("inherited_audit")}
        self.restored_step = state["global_step"]
        self.restored_planned_steps = state["planned_steps"]

    def validate_loaded_state(self):
        """on_train_begin 已加载标准进度，但尚未 rollout 或优化。"""
        if self.restored_step is not None and self.trainer.state.global_step != self.restored_step:
            raise ValueError("GRPO 标准 TrainerState 与附加状态逻辑步数不一致")
        if (self.restored_planned_steps is not None
                and self.trainer.state.max_steps != self.restored_planned_steps):
            raise ValueError("GRPO 标准 TrainerState 与原始核定预算不一致")
        if reference_weight_fingerprint(self.trainer) != self.reference_weights:
            raise ValueError("GRPO 恢复后原始冻结参考权重不一致")
