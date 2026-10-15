"""固定版本 SFT/DPO 恢复时保留 Accelerate 跳批加载器的 epoch。"""

from __future__ import annotations

from contextlib import contextmanager
from functools import update_wrapper, wraps
from importlib.metadata import version
import inspect
from types import FunctionType, MethodType

from agentic_arxiv.training.checkpoint import CheckpointError, PINNED_VERSIONS


PINNED_DATALOADER_VERSIONS = {**PINNED_VERSIONS, "accelerate": "1.11.0"}


@contextmanager
def preserve_managed_dataloader_epoch(trainer, *, enabled: bool):
    """仅在受管 SFT/DPO 的 train 作用域内修复跳批加载器的 epoch 丢失。"""
    if not enabled:
        yield
        return

    installed = {name: version(name) for name in PINNED_DATALOADER_VERSIONS}
    if installed != PINNED_DATALOADER_VERSIONS:
        raise CheckpointError(
            f"受管数据消费恢复要求固定依赖 {PINNED_DATALOADER_VERSIONS}，实际 {installed}"
        )

    from accelerate import skip_first_batches
    from accelerate.data_loader import DataLoaderShard
    from transformers import Trainer
    from trl import DPOTrainer, SFTTrainer

    original = getattr(trainer, "_inner_training_loop", None)
    loop = Trainer._inner_training_loop
    expected_arguments = (
        "self", "batch_size", "args", "resume_from_checkpoint", "trial", "ignore_keys_for_eval",
    )
    if (not isinstance(trainer, (SFTTrainer, DPOTrainer))
            or not isinstance(original, MethodType) or original.__self__ is not trainer
            or original.__func__ is not loop or not isinstance(loop, FunctionType)
            or tuple(inspect.signature(loop).parameters) != expected_arguments
            or loop.__globals__.get("skip_first_batches") is not skip_first_batches
            or "skip_first_batches" not in loop.__code__.co_names
            or tuple(inspect.signature(skip_first_batches).parameters) != ("dataloader", "num_batches")):
        raise CheckpointError("不支持的 SFT/DPO 数据消费恢复接口: _inner_training_loop/skip_first_batches")

    @wraps(skip_first_batches)
    def skip_preserving_epoch(dataloader, num_batches=0):
        epoch = getattr(dataloader, "iteration", None)
        if not isinstance(dataloader, DataLoaderShard) or type(epoch) is not int or epoch < 0:
            raise CheckpointError("受管数据消费恢复要求具有有效 epoch 的 DataLoaderShard")
        skipped = skip_first_batches(dataloader, num_batches=num_batches)
        if not isinstance(skipped, DataLoaderShard):
            raise CheckpointError("受管数据消费恢复未得到 DataLoaderShard")
        # Accelerate 1.11.0 新建的 shard 默认从 epoch 0 开始；其 __iter__
        # 会覆盖共享 SeedableRandomSampler 的 epoch，必须在迭代前承接原值。
        skipped.set_epoch(epoch)
        return skipped

    # 保留标准训练循环的原代码对象，仅为当前实例隔离这一项全局依赖；
    # 不修改 Transformers 模块、同进程的其他 Trainer 或训练循环源码。
    scoped_globals = {**loop.__globals__, "skip_first_batches": skip_preserving_epoch}
    scoped_loop = FunctionType(
        loop.__code__, scoped_globals, loop.__name__, loop.__defaults__, loop.__closure__,
    )
    update_wrapper(scoped_loop, loop)
    scoped_loop.__kwdefaults__ = loop.__kwdefaults__
    missing = object()
    previous = trainer.__dict__.get("_inner_training_loop", missing)
    trainer._inner_training_loop = MethodType(scoped_loop, trainer)
    try:
        yield
    finally:
        if previous is missing:
            del trainer._inner_training_loop
        else:
            trainer._inner_training_loop = previous
