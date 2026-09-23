"""混合精度开关的唯一来源。

原本 train_sft / train_dpo / train_grpo 各自带一份 `_precision_flags`，
其中两份写成「只要有 CUDA 就开 fp16」。这在 bf16 权重的模型上是硬崩：

    NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
                         not implemented for 'BFloat16'

fp16 的 GradScaler 不接受 bf16 梯度。现代基座模型（Qwen2.5 等）默认就是
bf16 权重，所以这条路在任何 bf16 显卡上都走不通。
"""


def precision_flags() -> dict:
    """训练器的精度参数：CUDA 上优先 bf16，退回 fp16；CPU / Apple MPS 不开混合精度。"""
    import torch

    if not torch.cuda.is_available():
        # Transformers 5.x requires CPU training to be selected explicitly;
        # an empty dict can otherwise leave the default bf16 path enabled.
        return {"use_cpu": True}
    if torch.cuda.is_bf16_supported():
        return {"bf16": True}
    return {"fp16": True}


def is_distributed_launch() -> bool:
    """True when this process belongs to an ``accelerate launch`` / torchrun run.

    ``accelerate launch`` and ``torchrun`` export ``LOCAL_RANK`` (and friends)
    into every worker.  Checking the environment rather than
    ``torch.distributed.is_initialized()`` matters because the trainer
    initialises the process group *after* the argument parsing this guard runs
    in the middle of.
    """
    import os

    return "LOCAL_RANK" in os.environ or "RANK" in os.environ


def pin_single_gpu(config) -> None:
    """Force single-GPU training when this process is *not* part of a launch.

    transformers wraps the model in ``torch.nn.DataParallel`` whenever
    ``CUDA_VISIBLE_DEVICES`` exposes more than one GPU, the model is not
    quantized, and no distributed launch is in effect (i.e. every non-QLoRA
    path: DPO, PPO, OPD). Trainer-managed DataParallel replicates the model for
    each forward, and DPOTrainer's interleaved policy/reference forwards crash
    inside ``broadcast_coalesced`` with a hard segfault on current torch builds
    (verified with torch 2.11.0+cu130, transformers 5.14.1, trl 0.29.1).

    Under ``accelerate launch`` the accelerator owns placement and the trainer
    never takes the DataParallel branch, so pinning ``_n_gpu`` there would
    actively break the run by hiding the other ranks' devices.  The guard
    therefore only applies to single-process invocations, which remain the
    default (see README「多卡支持」).
    """
    if is_distributed_launch():
        return
    config._n_gpu = 1


def assert_quantization_is_single_process(qlora: bool) -> None:
    """Refuse QLoRA under a multi-process launch instead of mis-training.

    The QLoRA path loads with ``device_map={"": 0}``, which pins every rank to
    ``cuda:0``; DDP then has all ranks computing on the same device and
    gradients from a single card.  That is a silent correctness bug, so fail
    loudly with the supported alternative.
    """
    if qlora and is_distributed_launch():
        raise SystemExit(
            "❌ QLoRA 不支持多进程启动：device_map={'': 0} 会把每个 rank 都钉在 "
            "cuda:0，DDP 实际只在单卡上训练。\n"
            "   多卡请改用 --no-qlora（1.5B 全参微调在 64GB 卡上余量充足），"
            "或不要用 accelerate launch。"
        )
