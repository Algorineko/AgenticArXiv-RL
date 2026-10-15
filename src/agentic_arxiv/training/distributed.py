"""训练进程拓扑、DDP 设备绑定与主进程副作用边界。

导入不初始化 CUDA 或进程组；通信只发生在 Trainer 已准备好之后。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable


@dataclass(frozen=True)
class TrainingContext:
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    @classmethod
    def from_environment(cls) -> TrainingContext:
        names = ("WORLD_SIZE", "RANK", "LOCAL_RANK")
        if not any(name in os.environ for name in names):
            return cls()
        if set(names) - os.environ.keys():
            raise SystemExit("分布式启动需要完整的 WORLD_SIZE/RANK/LOCAL_RANK；请使用 torchrun 或 accelerate launch")
        try:
            size, rank, local = (int(os.environ[name]) for name in names)
        except ValueError as exc:
            raise SystemExit("WORLD_SIZE/RANK/LOCAL_RANK 必须为整数") from exc
        if size < 1 or not 0 <= rank < size or local < 0:
            raise SystemExit("非法的分布式进程拓扑：需要 WORLD_SIZE>=1、0<=RANK<WORLD_SIZE、LOCAL_RANK>=0")
        return cls(size, rank, local)

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def prepare_device(self, *, qlora: bool, allow_sharded: bool = False) -> None:
        """量化权重加载前选择本地设备，不能使用跨机器的全局 rank。"""
        sharded = any(os.environ.get(name, "").lower() == "true" for name in (
            "ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED",
        ))
        if sharded and (qlora or not allow_sharded):
            raise SystemExit("当前多卡 QLoRA/多轮 GRPO 仅支持 DDP，不支持 FSDP/DeepSpeed 分片启动")
        import torch

        if torch.cuda.is_available():
            if self.local_rank >= torch.cuda.device_count():
                raise SystemExit(f"LOCAL_RANK={self.local_rank} 超出当前可见 GPU 数量")
            torch.cuda.set_device(self.local_rank)
        elif qlora:
            raise SystemExit("QLoRA 需要 CUDA；当前没有可用 GPU")

    def trainer_kwargs(self, *, stage: str) -> dict:
        if not self.is_distributed:
            return {}
        values = {"ddp_find_unused_parameters": False, "ddp_broadcast_buffers": False}
        if stage == "sft":
            # 由 Trainer 汇总整个累积窗口、所有 rank 的非 -100 labels；
            # loss 仍交给模型/Trainer，不能再按每卡样本均值二次归一化。
            values["average_tokens_across_devices"] = True
        return values

    def validate_trainer(self, trainer, *, stage: str, allow_sharded: bool = False) -> None:
        accelerator = trainer.accelerator
        if (accelerator.num_processes != self.world_size
                or accelerator.process_index != self.rank):
            raise SystemExit("Trainer/Accelerate 的实际进程拓扑与启动环境不一致")
        backend = str(getattr(accelerator.distributed_type, "value", accelerator.distributed_type))
        if self.is_distributed and not allow_sharded and backend not in {"MULTI_GPU", "MULTI_CPU"}:
            raise SystemExit(f"当前训练仅支持 DDP；实际分布式后端为 {backend}")
        if stage == "sft" and self.is_distributed:
            if not trainer.args.average_tokens_across_devices or not (
                trainer.model_accepts_loss_kwargs or getattr(trainer, "compute_loss_func", None) is not None
            ):
                raise SystemExit("DDP SFT 需要支持 num_items_in_batch 的 token 归一化 loss")
        if stage == "grpo" and getattr(trainer, "use_vllm", False):
            raise SystemExit("当前独立多轮 rollout 不支持 vLLM 跨 rank 生成同步")

    def effective_batch_size(self, batch_size: int, grad_accum: int) -> int:
        if batch_size < 1 or grad_accum < 1:
            raise SystemExit("batch_size 与 grad_accum 必须至少为 1")
        return self.world_size * batch_size * grad_accum

    def metadata(self) -> dict:
        return {"world_size": self.world_size, "rank": self.rank,
                "local_rank": self.local_rank, "placement": "one-process-per-device",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    def resource_overrides(self) -> dict:
        return {"world_size": {"baseline": 1, "requested": self.world_size}} if self.is_distributed else {}

    def _process_group(self):
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError("分布式通信必须在 Trainer 初始化进程组之后执行")
        if dist.get_world_size() != self.world_size or dist.get_rank() != self.rank:
            raise RuntimeError("实际进程组与 TrainingContext 不一致")
        return dist

    def gather(self, value: Any) -> list:
        """按全局 rank 顺序收集，保持 TRL generation 分组顺序。"""
        if not self.is_distributed:
            return [value]
        values = [None] * self.world_size
        self._process_group().all_gather_object(values, value)
        return values

    def run_on_main(self, operation: Callable[[], Any]) -> Any:
        """所有 rank 同时调用；仅主进程执行副作用并广播结果/异常。"""
        if not self.is_distributed:
            return operation()
        outcome = [None]
        original_error = None
        if self.is_main:
            try:
                outcome[0] = {"ok": True, "result": operation()}
            except BaseException as exc:
                original_error = exc
                outcome[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._process_group().broadcast_object_list(outcome, src=0)
        if not outcome[0]["ok"]:
            if original_error is not None:
                raise original_error
            raise RuntimeError("主进程操作失败：" + outcome[0]["error"])
        return outcome[0]["result"]

    def memory_summary(self) -> dict:
        import torch

        cuda = torch.cuda.is_available()
        local = {"rank": self.rank, "local_rank": self.local_rank,
                 "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
                 "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if cuda else None}
        records = self.gather(local)
        return {"per_rank": records, "peak_allocated_gib": max(
            (entry["peak_allocated_gib"] for entry in records if entry["peak_allocated_gib"] is not None),
            default=None,
        )}


def grpo_generation_batch_size(batch_size: int, grad_accum: int,
                               num_generations: int, context: TrainingContext) -> int:
    """分组约束针对全局生成批量，不要求每个 rank 拥有完整组。"""
    size = context.effective_batch_size(batch_size, grad_accum)
    if num_generations < 2 or size % num_generations:
        raise SystemExit(
            f"GRPO 全局生成批量 {size}（每卡 batch={batch_size} × "
            f"world_size={context.world_size} × grad_accum={grad_accum}）"
            f"必须能被 num_generations={num_generations} 整除，且 num_generations>=2；"
            "不会自动增大每卡 batch"
        )
    return size
