"""trl 的旧版 torch 兼容层。

TRL 的 DPOTrainer / GRPOTrainer / PPOTrainer 在内部无条件地
``from torch.distributed.fsdp import FSDPModule``，而 ``FSDPModule`` 只
在 torch>=2.6 才存在。因此在使用 torch<2.6（例如本仓库本地 RTX 4050 +
torch 2.5.x）时，仅仅 ``from trl import ...`` 就会直接崩溃：:

    ImportError: cannot import name 'FSDPModule' from 'torch.distributed.fsdp'

单卡训练（大多数用户的场景）根本不需要真正的 FSDP 分布式打包，所以这里
在导入 trl 之前补一个最小可用的占位类，让 DPO / GRPO / PPO 的
``prepare_fsdp`` 能正常通过 while isinstance 判断、正常走非 FSDP 分支。

用法：在 ``from trl import ...`` 之前 ``import AgenticArxiv.rl.trl_compat``。
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
