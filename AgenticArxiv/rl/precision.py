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
