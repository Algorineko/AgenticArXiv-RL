"""GRPO 训练脚本（使用 TRL GRPOTrainer）

GRPO（Group Relative Policy Optimization）：
- 目标：用 verifiable reward 在线训练
- 优势：无需 reward model、无需 value model（比 PPO 更轻量）
- 数据：**不需要预先生成**。GRPO 是在线算法，只要 prompt + 可验证奖励，
        数据集直接由 benchmark/tasks.py 派生

奖励定义见 rl/grpo_reward.py：TRL 原生执行多轮 tool-calling，把每轮环境
observation 插回上下文，并将完整轨迹交给 RewardCalculator 打分。

训练中每 N 步自动运行 canary 评估（在固定小任务集上验证模型未退化），
训练结束后可选运行阶段验证（检查模型是否达到最低质量阈值）。

使用方式：
    python -m AgenticArxiv.rl.train_grpo
    python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --num_generations 8
    python -m AgenticArxiv.rl.train_grpo --canary_steps 20 --min_canary_reward -0.3
"""

import argparse
import dataclasses
import json
import math
import os
import random
import sys

import torch
from pathlib import Path
from typing import Any, Dict, List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent

#: 历史实验一律跑在这个 KL 系数上。`--dapo` 预设只在调用方没有显式改过它时
#: 才把它归零（DAPO 不用 KL 正则），否则会把用户传的值默默吃掉。
DEFAULT_BETA = 0.04

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(PACKAGE_ROOT))

# 奖励计算要执行工具，会话状态走内存，不依赖数据库
os.environ.setdefault("STORE_BACKEND", "memory")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from rl import trl_compat  # noqa: F401  (torch<2.6 时兜底 FSDPModule，须在 trl 之前)
from trl import GRPOConfig, GRPOTrainer

# 工具集从 tools/bootstrap.py 的注册表来，不在这里手写模块清单：
# 手写的那份漏掉过 paper_content / paper_summary / paper_figures，后果是
# prompt 里的工具列表少了三个，而策略的动作空间缩小是**静默**的。
from tools.bootstrap import require_all_tools

from benchmark.splits import DEFAULT_SPLIT_PATH
from benchmark.task_io import load_tasks
from benchmark.semantic_oracle import attach_expected_paper_ids
from rl.canary import CanaryEvaluator, CanaryCallback
from rl.dynamic_sampling import (
    DynamicSamplingDistributedError,
    assert_dynamic_sampling_single_process,
    make_dynamic_sampling_rollout_func,
)
from rl.grpo_reward import (
    SUPPORTED_TOOL_NAMES,
    build_prompt_dataset,
    load_mock_env,
    make_grpo_reward_fn,
    make_multiturn_rollout_func,
)
from rl.multiturn_env import make_environment_factory
from rl.rollout_audit import RolloutAuditWriter
from rl.observability import (
    RewardComponentTracker,
    describe_logging,
    resolve_report_to,
)
from rl.reward import RewardCalculator
from rl.stage_verifier import StageVerifier
from rl.train_sft import (
    _assert_lora_only_trainable,
    _build_qlora_configs,
    _qlora_runtime_guard,
    _trainable_parameter_stats,
)

DEFAULT_SNAPSHOT = REPO_ROOT / "data" / "mock_arxiv_snapshot.json"


def _precision_flags():
    """见 rl/precision.py：CUDA 上优先 bf16，退回 fp16；CPU / MPS 不开混合精度。"""
    from rl.precision import precision_flags
    return precision_flags()


def _build_reward_calculator(
    curriculum_steps: int, weights_override: Optional[str] = None,
    disable_failure_caps: bool = False,
    task_set: str = "default",
) -> Any:
    """构造本次运行唯一的奖励配置，并拒绝含糊的负步数。

    weights_override: JSON 对象，逐分量覆盖权重（E4 奖励消融用），
    如 '{"outcome": 0.0}'；分量名以 RewardCalculator.weights 为准。
    """
    if curriculum_steps < 0:
        raise SystemExit("❌ reward_curriculum_steps 不能为负数")
    if task_set == "bfcl":
        # BFCL 轨道: 标准答案是 gold 调用链, 五分量来自 score_chain 而非
        # expected_tools 任务声明; 加权框架/课程与其它域同构 (rl/bfcl_env.py)。
        from AgenticArxiv.rl.bfcl_env import BfclRewardCalculator
        calc = BfclRewardCalculator(curriculum_steps=curriculum_steps)
    else:
        calc = RewardCalculator(curriculum_steps=curriculum_steps)
    if weights_override:
        try:
            parsed = json.loads(weights_override)
        except ValueError as exc:
            raise SystemExit(f"❌ --reward_weights 不是合法 JSON: {exc}")
        if not isinstance(parsed, dict) or not parsed:
            raise SystemExit("❌ --reward_weights 须为非空对象, 如 '{\"outcome\": 0.0}'")
        calc.set_weights({k: float(v) for k, v in parsed.items()})
        print(f"⚖️  奖励分量覆盖（E4 消融）: {parsed}")
    if disable_failure_caps:
        calc.disable_failure_caps = True
        print("⚖️  非补偿失败封顶已禁用（E4 消融）")
    return calc


def _load_tasks(task_set: str, split: Optional[str]):
    """选择训练任务集（任务池与切分过滤见 benchmark.task_io.load_tasks）。

    默认仍是 benchmark/tasks.py 的 8 条冒烟任务 —— 换默认值会让此前所有
    训练曲线不可比，所以要用完整任务集必须显式 --task_set expanded。

    `--split rl_train` 取的是 train 中成功率处于中间带的那部分：GRPO 的
    优势是组内相对的，成功率贴近 0 或 1 的任务，同一 prompt 采样出的轨迹
    奖励一致，组内方差为零、优势为零、不产生任何梯度。把它们放进训练集
    是纯粹的空转，还会把 frac_reward_zero_std 顶到 1 触发方差守卫。
    """
    tasks = load_tasks(task_set, split)
    if not split and task_set != "expanded":
        print(
            f"⚠️  正在用 benchmark/tasks.py 的 {len(tasks)} 条冒烟任务训练。"
            "完整任务集用 --task_set expanded；\n"
            "    只训练有梯度的那部分用 --task_set expanded "
            f"--split rl_train（{DEFAULT_SPLIT_PATH.name}）"
        )
    return tasks


def _gold_completion_tokens(tokenizer, tasks) -> int:
    """标准动作渲染成 ReAct 文本后的最大 token 数。

    用于校验 max_completion_length：设小了模型永远吐不出完整动作，
    奖励恒为下限、组内方差为 0、梯度为 0，而日志上只表现为
    completions/clipped_ratio=1，很容易被误读成「模型太差」。
    """
    import json as _json
    lengths = [0]
    for task in tasks:
        if task.get("gold_calls"):
            # bfcl 任务没有 expected_tools: 标准动作是整条 gold 链的
            # submit_calls 元提交, 长度按完整 JSON 数组字符串估, 否则守卫
            # 对这一域形同虚设（截断的提交永远拿不到 outcome 分）。
            action = _json.dumps(
                {
                    "name": "submit_calls",
                    "args": {
                        "calls": _json.dumps(task["gold_calls"], ensure_ascii=False)
                    },
                },
                ensure_ascii=False,
            )
            text = f"Thought: xxx\nAction: {action}"
            lengths.append(len(tokenizer(text)["input_ids"]))
            continue
        for name in task.get("expected_tools", []) or []:
            text = f'Thought: xxx\nAction: {_json.dumps({"name": name, "args": {}}, ensure_ascii=False)}'
            lengths.append(len(tokenizer(text)["input_ids"]))
    return max(lengths)


def _global_step(state) -> int:
    """当前步数，取不到就按 0（视作还在开局宽限期内）。

    宽限期只用来区分「开局就没梯度」和「练到没梯度」，判错的代价是
    多报一次故障；为它让训练崩掉才是更坏的结果。
    """
    try:
        return int(getattr(state, "global_step", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _adapter_base_model(adapter_path: Path, override: Optional[str] = None) -> str:
    """Resolve the immutable base model behind an SFT LoRA adapter."""
    if override:
        base = Path(override)
        return str(base.resolve()) if base.exists() else override
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return ""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    value = str(config.get("base_model_name_or_path") or "")
    if not value:
        raise SystemExit(f"❌ adapter_config 未声明 base_model_name_or_path: {config_path}")
    base = Path(value)
    if base.exists():
        return str(base.resolve())
    if "/" in value and not value.startswith(("outputs/", "./", "/")):
        return value
    raise SystemExit(
        f"❌ SFT adapter 引用的基座模型不存在: {value}\n"
        "   adapter 换机器后绝对路径可能失效，请显式传 --base_model"
    )


def _load_policy(
    model_path: str,
    *,
    qlora: bool,
    base_model: Optional[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    """Load either an existing SFT adapter or a fresh policy for GRPO."""
    import torch

    local = Path(model_path)
    is_adapter = local.is_dir() and (local / "adapter_config.json").exists()
    arch_model = (
        _adapter_base_model(local, base_model) if is_adapter else (
            str(Path(base_model).resolve()) if base_model and Path(base_model).exists()
            else base_model or model_path
        )
    )
    from utils.llm_client import _is_vlm_model_path

    is_vlm = _is_vlm_model_path(str(arch_model))
    if is_vlm and qlora:
        print("⚠️ VLM 策略在 DCU 上没有可用的 bitsandbytes（das 构建），改走 bf16 LoRA")
        qlora = False
    if not qlora:
        if is_vlm:
            # 27B 级 VLM 单卡装不下，双卡 device_map；LoRA 目标必须含视觉-语言投影层
            from transformers import AutoModelForImageTextToText

            from rl.train_vlm_sft import resolve_lora_targets

            model_cls = AutoModelForImageTextToText
            print(f"🌐 GRPO VLM 策略（bf16, device_map=auto）: {arch_model}")
            n_vis = torch.cuda.device_count()
            max_mem = {i: "56GiB" for i in range(n_vis)}
            policy = model_cls.from_pretrained(
                arch_model, dtype=torch.bfloat16, device_map="auto",
                max_memory=max_mem,
                local_files_only=Path(arch_model).exists(),
            )
            if is_adapter:
                from peft import PeftModel

                policy = PeftModel.from_pretrained(policy, model_path, is_trainable=True)
            else:
                from peft import LoraConfig, get_peft_model

                target_modules = resolve_lora_targets(policy, "language+projector")
                policy = get_peft_model(policy, LoraConfig(
                    r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                    target_modules=target_modules, task_type="CAUSAL_LM",
                ))
            policy.config.use_cache = False
            # bf16 VLM 路径不走 QLoRA 的 4-bit 断言（is_loaded_in_4bit 恒为 False）
            trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
            total = sum(p.numel() for p in policy.parameters())
            print(f"🔧 参数统计: trainable={trainable:,}, model_parameters={total:,}, "
                  f"trainable_ratio={100.0 * trainable / max(total, 1):.4f}%")
            return policy
        print("⚠️  --no-qlora：加载完整精度策略，不属于单张4090冻结方案")
        print("⚠️  --no-qlora：加载完整精度策略，不属于单张4090冻结方案")
        policy = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=local.exists()
        )
        policy.config.use_cache = False
        return policy

    if os.getenv("AA_LORA_BF16") == "1":
        # bf16-LoRA: 全精度冻结基座 + LoRA 适配器（无 bitsandbytes 依赖; 8B 级单卡可行）
        from peft import LoraConfig, get_peft_model
        source = _adapter_base_model(local, base_model) if is_adapter else (
            str(Path(base_model).resolve()) if base_model and Path(base_model).exists()
            else base_model or model_path
        )
        print(f"🧩 GRPO bf16-LoRA: bf16 冻结基座 + LoRA r={lora_r}\n   base={source}")
        base = AutoModelForCausalLM.from_pretrained(
            source, dtype=torch.bfloat16, device_map={"": 0},
            low_cpu_mem_usage=True, local_files_only=Path(source).exists(),
        )
        base.config.use_cache = False
        if hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        policy = (
            PeftModel.from_pretrained(base, model_path, is_trainable=True)
            if is_adapter
            else get_peft_model(base, LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                task_type="CAUSAL_LM",
            ))
        )
        policy.config.use_cache = False
        trainable = sum(pp.numel() for pp in policy.parameters() if pp.requires_grad)
        total = sum(pp.numel() for pp in policy.parameters())
        print(f"🔧 参数统计: trainable={trainable:,}, model_parameters={total:,}, "
              f"trainable_ratio={100.0 * trainable / max(total, 1):.4f}%")
        return policy

    _qlora_runtime_guard()
    quantization, fresh_lora = _build_qlora_configs(
        lora_r, lora_alpha, lora_dropout
    )
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

    source = _adapter_base_model(local, base_model) if is_adapter else (
        str(Path(base_model).resolve()) if base_model and Path(base_model).exists()
        else base_model or model_path
    )
    source_path = Path(source)
    print(
        "🪶 GRPO QLoRA: 4-bit NF4 + double quant + BF16 compute\n"
        f"   base={source}\n"
        f"   adapter={model_path if is_adapter else '(new)'}"
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = AutoModelForCausalLM.from_pretrained(
        source,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=source_path.exists(),
    )
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=False
    )
    policy = (
        PeftModel.from_pretrained(base, model_path, is_trainable=True)
        if is_adapter
        else get_peft_model(base, fresh_lora)
    )
    policy.config.use_cache = False
    trainable, total, ratio = _assert_lora_only_trainable(policy)
    print(
        f"🔧 参数统计: trainable={trainable:,}, model_parameters={total:,}, "
        f"trainable_ratio={ratio:.4f}%"
    )
    return policy


class RewardVarianceGuard(TrainerCallback):
    """组内奖励方差为 0 时中止训练。

    GRPO 的梯度来自同一 prompt 采样出的若干条轨迹之间的奖励差异。
    若一组内所有样本拿到同一个奖励，优势归零，这一步不产生任何梯度。
    最常见的成因是基座模型还吐不出可解析的动作 —— 每条采样都落到解析
    失败的地板分，方差恒为 0。

    这种失败是静默的：训练照常跑完、loss 看着像在收敛（其实只是 KL 项）、
    checkpoint 照常保存，唯一的迹象是日志里 frac_reward_zero_std 恒为 1，
    而这一栏很容易被忽略。这里把它变成一次响亮的失败。
    """

    def __init__(self, patience: int = 5, grace_steps: int = 20):
        self.patience = patience
        # Keep the constructor argument for caller/manifest compatibility.
        # Quality, rather than elapsed steps, distinguishes a full-score window
        # from uniform failure. Early saturation is possible after good SFT.
        self.grace_steps = grace_steps
        self.streak = 0
        self.ceiling_streak = 0
        self.tripped = False
        self.converged = False
        self.stop_step = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        frac = logs.get("frac_reward_zero_std")
        std = logs.get("reward_std")
        if frac is None and std is None:
            return
        dead = (frac is not None and float(frac) >= 1.0) or (
            frac is None and std is not None and float(std) == 0.0
        )
        if not dead:
            self.streak = 0
            self.ceiling_streak = 0
            return

        mean = logs.get("reward")
        try:
            value = float(mean)
            clean = (
                float(logs.get("rollout/finished", 1.0)) >= 1.0
                and float(logs.get("rollout/parse_error_rate", 0.0)) == 0.0
                and float(logs.get("rollout/tool_error_rate", 0.0)) == 0.0
            )
            full_score = math.isfinite(value) and value >= 1.0 - 1e-6 and clean
        except (TypeError, ValueError):
            full_score = False
        self.ceiling_streak = self.ceiling_streak + 1 if full_score else 0
        self.streak += 1
        if self.streak < self.patience:
            return

        control.should_training_stop = True
        mean = logs.get("reward")
        step = _global_step(state)
        self.stop_step = step
        if self.ceiling_streak >= self.patience:
            # Every group in this zero-variance window has full reward and no
            # execution failure. Preserve the policy even during early steps;
            # this describes the sampled window, not global convergence.
            self.converged = True
            print(
                f"\n⏹  截至第 {step} 步已连续 {self.streak} 次记录到组内奖励方差为 0"
                + (f"（奖励恒为 {float(mean):.4f}）" if mean is not None else "")
                + "，当前采样窗口已饱和，提前停止并保存。\n"
                "   请先做固定评测确认是否真正收敛；若仍需继续训练，应重新划中间带"
                "（成功率会随训练漂移，旧的 rl_train 会逐渐变成 ceiling）或扩充任务集。"
            )
            return

        self.tripped = True
        print(
            f"\n❌ 连续 {self.streak} 步组内奖励方差为 0"
            + (f"（奖励恒为 {float(mean):.4f}）" if mean is not None else "")
            + "，优势全零，训练不产生任何梯度。\n"
            "   最常见原因：基座模型还产不出可解析的 ReAct 动作，每条采样都落到解析失败的地板分。\n"
            "   处理办法：\n"
            "     1) 先跑 SFT 让模型学会输出格式，再用 --model outputs/sft/final 起 GRPO\n"
            "     2) 换更强的基座模型\n"
            "     3) 若确认是任务全在难度谱两端（全对或全错），换一批成功率居中的任务\n"
            "   确实想继续跑，用 --allow_zero_variance 跳过本检查。"
        )


class ZeroVarianceNoiseInjector:
    """R1.6「方差直接操纵」实验：向零方差组的奖励注入均匀噪声。

    GRPO 的优势来自同一 prompt 若干条轨迹之间的奖励差异；组内奖励完全一致
    （std==0）的组优势恒为 0，这一步不产生任何梯度。no_format 配置的训练
    崩溃正是这个机制：format 分量归零后，绝大多数组的全部 rollout 落到同
    一个语义地板分。

    因果闭环要求证明方向而非仅仅相关：保持奖励语义不变（注入的噪声期望为
    0，不引入系统性偏置，也不改变任务判定），仅对零方差组叠加
    ``eps_i ~ Uniform(-delta, +delta)``，恢复组内相对排序信号。若学习随之
    恢复，则「零方差 -> 学习崩溃」的因果方向成立；若仍不恢复，同样排除
    「方差不足是唯一瓶颈」的备择解释。

    注入只发生在返回给 trainer 的奖励上：RewardTracker / rollout 审计记录
    的仍是干净奖励，注入规模由本类的 history / summary 单独提供。
    """

    def __init__(self, delta: float, num_generations: int, seed: int):
        self.delta = float(delta)
        self.num_generations = int(num_generations)
        # 独立 RNG：噪声序列只由本旗标与 seed 决定，不挤占 trainer 的采样流
        self.rng = random.Random(seed)
        self.batches = 0
        self.groups_total = 0
        self.groups_injected = 0
        self.samples_total = 0
        self.samples_injected = 0
        self.history: List[Dict[str, Any]] = []

    def inject(self, task_ids, rewards: List[float], step: int) -> List[float]:
        n = len(rewards)
        if n == 0:
            return list(rewards)
        if task_ids is None or len(task_ids) != n:
            # 退路：TRL 按 prompt 连续重复 num_generations 次排布，按位置分块
            keys = [i // max(self.num_generations, 1) for i in range(n)]
        else:
            keys = list(task_ids)
        groups: Dict[Any, List[int]] = {}
        for i, key in enumerate(keys):
            groups.setdefault(key, []).append(i)
        noisy = list(rewards)
        n_groups_injected = 0
        n_samples_injected = 0
        for members in groups.values():
            values = [rewards[i] for i in members]
            if max(values) - min(values) > 1e-12:
                continue  # 组内已有相对信号，保持原奖励不动
            n_groups_injected += 1
            n_samples_injected += len(members)
            for i in members:
                noisy[i] = rewards[i] + self.rng.uniform(-self.delta, self.delta)
        self.batches += 1
        self.groups_total += len(groups)
        self.groups_injected += n_groups_injected
        self.samples_total += n
        self.samples_injected += n_samples_injected
        self.history.append({
            "step": step,
            "groups": len(groups),
            "groups_injected": n_groups_injected,
            "samples": n,
            "samples_injected": n_samples_injected,
        })
        print(
            f"🎲 [zero_variance_noise] step={step} "
            f"注入组 {n_groups_injected}/{len(groups)}"
            f"（{100.0 * n_groups_injected / max(len(groups), 1):.1f}%） "
            f"注入样本 {n_samples_injected}/{n}"
        )
        return noisy

    def summary(self) -> Dict[str, Any]:
        return {
            "delta": self.delta,
            "batches": self.batches,
            "groups_total": self.groups_total,
            "groups_injected": self.groups_injected,
            "group_injection_rate": (
                self.groups_injected / self.groups_total if self.groups_total else 0.0
            ),
            "samples_total": self.samples_total,
            "samples_injected": self.samples_injected,
            "sample_injection_rate": (
                self.samples_injected / self.samples_total if self.samples_total else 0.0
            ),
        }

    def save_stats(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"summary": self.summary(), "history": self.history},
                ensure_ascii=False, indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        return path


def _wrap_reward_fn_with_zero_variance_noise(reward_fn, injector: ZeroVarianceNoiseInjector):
    """把注入器接在 reward_fn 与 GRPOTrainer 之间（奖励语义不变，只加噪声）。"""

    def noisy_reward_fn(*args, **kwargs):
        rewards = reward_fn(*args, **kwargs)
        task_ids = kwargs.get("task_id")
        if isinstance(task_ids, str):
            task_ids = [task_ids]
        step = int(getattr(kwargs.get("trainer_state"), "global_step", 0) or 0)
        return injector.inject(task_ids, list(rewards), step)

    # TRL 用 __name__ 作为奖励分量指标前缀，必须保持不变
    noisy_reward_fn.__name__ = getattr(reward_fn, "__name__", "grpo_reward_fn")
    return noisy_reward_fn


# ---------------------------------------------------------------------------
# V 系列优势干预（第四轮审稿裁决实验）
#   permute: 非 tie 组内置换优势 —— 保幅度/频率、毁与轨迹的对应（gpt#2 对照表）
#   avspo:   AVSPO 忠实移植 —— 虚拟样本只进归一化统计、无策略梯度项、
#            坍塌组内真实样本保持相同优势（He et al., ICML 2026, arXiv:2605.21125；
#            r_obs<0 的带符号推广为本仓披露的适配，原文只定义 r_obs>0 与 ==0）
# 干预点：TRL 算完 advantages 之后、写进 loss 之前（_generate_and_score_completions
# 的输出 dict）；mode=none 且未开插桩时与 GRPOTrainer 完全等价。
# ---------------------------------------------------------------------------
_REWARD_RECORDER: Dict[str, Any] = {"last": None}

def _wrap_reward_fn_with_folr(reward_fn, first_turn_credit: float = 0.25):
    """FOLR 移植 (TSPO 忠实家族检验, GLM#4): 首个正确工具调用 turn 获部分奖励.

    TSPO 的 FOLR 把部分奖励分配给目标首次正确出现的 turn (质量相关 by construction).
    我们在标准奖励之上, 对每个 rollout 找到首个"期望工具+正确参数"的 turn, 给该 turn
    的动作一个加性 credit —— 与期望 paper-id 的 gold chain replay 对齐.
    实现层: 借用 reward_fn 的 history 通道 (auditor 已记录逐turn工具调用); 简化版:
    对首个调用期望工具名的 turn 给 credit (参数精确性已由标准奖励的 argument 分量度量).
    """
    def folr_reward_fn(*args, **kwargs):
        rewards = reward_fn(*args, **kwargs)
        try:
            task_ids = kwargs.get("task_id")
            if isinstance(task_ids, str):
                task_ids = [task_ids]
            steps = kwargs.get("steps")  # gold 链: [(tool, args), ...]
            histories = kwargs.get("histories")
            if not steps or not histories:
                return rewards
            out = list(rewards)
            for idx, (hist, gold) in enumerate(zip(histories, steps or [[]] * len(out))):
                expected_first = gold[0][0] if gold else None
                if not expected_first or not hist:
                    continue
                for turn in hist:
                    act = turn.get("action") or ""
                    if expected_first in act:
                        out[idx] = out[idx] + first_turn_credit
                        break
            return out
        except Exception:
            return rewards
    folr_reward_fn.__name__ = getattr(reward_fn, "__name__", "grpo_reward_fn")
    return folr_reward_fn


def _wrap_reward_fn_with_recorder(reward_fn):
    """记录 reward_fn 最近一次的逐 completion 奖励（干预与插桩的组结构数据源）。
    必须包在噪声注入之后，记录的才是 TRL 实际参与归一化的奖励。"""

    def recorded_reward_fn(*args, **kwargs):
        rewards = reward_fn(*args, **kwargs)
        try:
            _REWARD_RECORDER["last"] = [float(x) for x in rewards]
        except (TypeError, ValueError):
            _REWARD_RECORDER["last"] = None
        return rewards

    recorded_reward_fn.__name__ = getattr(reward_fn, "__name__", "grpo_reward_fn")
    return recorded_reward_fn


class InterventionGRPOTrainer(GRPOTrainer):
    """GRPOTrainer + 优势级干预与逐组奖励插桩（类属性在 main() 实例化前配置）。"""

    varm_mode = "none"        # none / permute / avspo
    varm_seed = 42
    varm_dump = False         # 逐组奖励向量插桩（group_rewards.jsonl）
    avspo_alpha = 0.5         # K = max(1, min(G, ⌈G·ACR^α⌉))
    avspo_anchor = 0.1        # r_obs == 0 时的虚拟锚点
    avspo_eta = 0.01          # τ_adapt 自适应步长
    avspo_tau0 = 0.5          # τ_adapt 初值，夹在 [0.1, 0.9]
    avspo_tau_sigma = 1e-6    # 组坍塌判定阈值（AVSPO 的 τ）
    avspo_eps = 1e-4          # 数值稳定项（与 TRL 归一化一致）
    avspo_clip_positive = False  # 扩展消融: 负 r_obs 按 0-anchor 处理 (GLM#7)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._iv_rng = torch.Generator(device="cpu")
        self._iv_rng.manual_seed(int(self.varm_seed))
        self._tau_adapt = float(self.avspo_tau0)
        self._prev_J = None
        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._stats_fh = open(out_dir / "intervention_stats.jsonl", "a", encoding="utf-8")
        self._dump_fh = (
            open(out_dir / "group_rewards.jsonl", "a", encoding="utf-8")
            if self.varm_dump else None
        )
        if self.varm_mode != "none" or self.varm_dump:
            print(
                f"🧪 [advantage_intervention] mode={self.varm_mode} seed={self.varm_seed}"
                + (f" alpha={self.avspo_alpha} anchor={self.avspo_anchor} "
                   f"eta={self.avspo_eta} tau0={self.avspo_tau0}"
                   if self.varm_mode == "avspo" else "")
                + (f" dump=group_rewards.jsonl" if self.varm_dump else "")
            )

    def _generate_and_score_completions(self, *args, **kwargs):
        output = super()._generate_and_score_completions(*args, **kwargs)
        if self.varm_mode != "none" or self._dump_fh is not None:
            try:
                self._intervene(output)
            except Exception:  # 干预失败不允许拖垮训练，但必须可见可查
                import traceback
                traceback.print_exc()
        return output

    def _intervene(self, output):
        if not getattr(self.model, "training", False):
            return  # eval 批次（evaluate/canary 走独立评估器）不改写
        adv = output.get("advantages")
        rec = _REWARD_RECORDER.get("last")
        if adv is None or not rec:
            return
        G = int(self.num_generations)
        n = int(adv.shape[0])
        if n == 0 or n % G or len(rec) != n:
            print(f"⚠️ [intervention] 结构不齐 adv={n} rec={len(rec)} G={G}，跳过本步")
            return
        step = int(self.state.global_step)
        adv_g = adv.view(-1, G)
        rw = torch.tensor(rec, dtype=torch.float64).view(-1, G)
        collapsed = rw.std(dim=1, unbiased=False) < self.avspo_tau_sigma  # AVSPO σ_ℛ 总体口径
        acr = float(collapsed.float().mean())
        J = float(rw.mean())
        mode, gate, K, n_touched = self.varm_mode, False, 0, 0

        if mode == "permute":
            # 非 tie 组（有信号组）内置换：幅度分布不变，与轨迹的对应被打断。
            # tie 组优势全零，置换为恒等，天然跳过。
            for j in torch.nonzero(~collapsed).flatten().tolist():
                perm = torch.randperm(G, generator=self._iv_rng)
                adv_g[j] = adv_g[j][perm.to(adv_g.device)]
                n_touched += 1
        elif mode == "rloo":
            # RLOO 估计器对照 (GLM III.1): A_i = r_i − mean(r_其他) = G/(G−1)·(r_i−r̄)
            # 无 std 归一化。tie 组天然全零（与 GRPO 同为并列静默）；幅度口径为奖励原尺度
            # （与 GRPO 的 O(1) 归一化优势不同，等效步长差异在论文中披露）。
            adv_g[:] = ((rw - rw.mean(dim=1, keepdim=True)) * (G / (G - 1.0))).to(adv_g.dtype)
            n_touched = int(rw.shape[0])
        elif mode == "zero":
            # 非 tie 组优势置零：纯移除信号，不新增任何随机更新（GPT#3 四臂矩阵的归零对照）
            for j in torch.nonzero(~collapsed).flatten().tolist():
                adv_g[j] = 0.0
                n_touched += 1
        elif mode == "swap":
            # RSPO 式组内成对交换 (GLM C2): 保留幅度与排序结构, 仅交换两两配对的内容
            # 检验 P2 预言: 结构化交换若无质量相关 -> 应无害或有害 (可证伪)
            for j in torch.nonzero(~collapsed).flatten().tolist():
                row = adv_g[j].clone()
                order = torch.argsort(rw[j], descending=True)
                idx = order.to(adv_g.device)
                adv_g[j][idx[0]], adv_g[j][idx[-1]] = row[idx[-1]], row[idx[0]]  # 最优<->最差交换
                if G > 2:
                    mid = G // 2
                    adv_g[j][idx[mid]], adv_g[j][idx[mid - 1]] = row[idx[mid - 1]], row[idx[mid]]
                n_touched += 1
        elif mode == "reverse":
            # 非 tie 组优势按奖励秩反向分配：系统性反相关（最优轨迹拿最差优势）
            for j in torch.nonzero(~collapsed).flatten().tolist():
                order = torch.argsort(rw[j], descending=True)
                vals = adv_g[j][order.to(adv_g.device)].clone()  # 按轨迹好坏排序的优势值
                adv_g[j][order.to(adv_g.device)] = vals.flip(0).to(adv_g[j].dtype)
                n_touched += 1
        elif mode == "avspo":
            gate = acr > self._tau_adapt
            if gate and acr > 0:
                K = max(1, min(G, math.ceil(G * (acr ** self.avspo_alpha))))
                for j in torch.nonzero(collapsed).flatten().tolist():
                    r_obs = float(rw[j].max())
                    if self.avspo_clip_positive and r_obs < 0:
                        r_obs = 0.0  # 原文二元设定无负值; clip变体把负组按r=0 anchor处理
                    if r_obs == 0.0:
                        v = [self.avspo_anchor * (K - k + 1) / K
                             for k in range(1, K + 1)]
                    else:
                        # r_obs>0 原式 v_k=r_obs(1−k/(K+1))；r_obs<0 时同一式给出
                        # (r_obs,0) 内的虚拟值 → 共同优势为负（整组压低），符号推广
                        v = [r_obs * (1.0 - k / (K + 1)) for k in range(1, K + 1)]
                    aug = torch.tensor([r_obs] * G + v, dtype=torch.float64)
                    mu, sd = float(aug.mean()), float(aug.std(unbiased=False))
                    adv_g[j].fill_((r_obs - mu) / (sd + self.avspo_eps))
                    n_touched += 1
            # Eq.10: τ_adapt ← τ_adapt + η·sign(ΔJ)·(ACR − τ_adapt)，夹 [0.1, 0.9]
            if self._prev_J is not None:
                dj = J - self._prev_J
                s = 1.0 if dj > 0 else (-1.0 if dj < 0 else 0.0)
                self._tau_adapt = min(
                    0.9, max(0.1, self._tau_adapt + self.avspo_eta * s * (acr - self._tau_adapt))
                )
            self._prev_J = J

        self._stats_fh.write(json.dumps({
            "step": step, "mode": mode, "acr": round(acr, 4),
            "tau_adapt": round(self._tau_adapt, 4), "gate": bool(gate), "K": K,
            "n_touched": n_touched, "n_groups": int(rw.shape[0]), "J": round(J, 4),
        }, ensure_ascii=False) + "\n")
        self._stats_fh.flush()
        if self._dump_fh is not None:
            self._dump_fh.write(json.dumps({
                "step": step, "mode": mode, "rewards": rw.tolist(),
                "collapsed": collapsed.tolist(),
            }, ensure_ascii=False) + "\n")
            self._dump_fh.flush()


#: DAPO 论文的推荐取值（clip-higher 上界、overlong filtering 开关、KL 系数）。
DAPO_PRESET = {
    "loss_type": "dapo",
    "epsilon_high": 0.28,
    "mask_truncated_completions": True,
    "beta": 0.0,
}


def resolve_dapo_options(
    *,
    dapo: bool,
    loss_type: Optional[str],
    epsilon_high: Optional[float],
    mask_truncated_completions: Optional[bool],
    beta: float,
) -> tuple:
    """Fill in the DAPO preset without overwriting anything set explicitly.

    ``--dapo --epsilon_high 0.2`` must keep 0.2.  ``beta`` is the exception: it
    has no "unset" sentinel, so it only drops to zero when the caller left it at
    the project default, which is what ``DEFAULT_BETA`` is for.
    """
    if not dapo:
        return loss_type, epsilon_high, mask_truncated_completions, beta
    return (
        loss_type or DAPO_PRESET["loss_type"],
        DAPO_PRESET["epsilon_high"] if epsilon_high is None else epsilon_high,
        (
            DAPO_PRESET["mask_truncated_completions"]
            if mask_truncated_completions is None
            else mask_truncated_completions
        ),
        DAPO_PRESET["beta"] if beta == DEFAULT_BETA else beta,
    )


def resolve_dynamic_sampling_option(
    *, dapo: bool, dynamic_sampling: Optional[bool]
) -> bool:
    """DAPO enables dynamic sampling unless the user explicitly overrides it."""

    if dynamic_sampling is None:
        return bool(dapo)
    return bool(dynamic_sampling)


def main(
    model: str = "outputs/dpo/final",
    output_dir: str = "outputs/grpo",
    epochs: int = 1,
    max_steps: int = -1,
    batch_size: int = 4,
    grad_accum: int = 1,
    lr: float = 1e-5,
    beta: float = DEFAULT_BETA,
    reward_curriculum_steps: int = 30,
    reward_weights: Optional[str] = None,
    disable_failure_caps: bool = False,
    zero_variance_noise: float = 0.0,
    advantage_intervention: str = "none",
    avspo_alpha: float = 0.5,
    avspo_anchor: float = 0.1,
    avspo_eta: float = 0.01,
    avspo_tau0: float = 0.5,
    dump_group_rewards: bool = False,
    folr_first_turn_credit: float = 0.0,
    avspo_clip_positive: bool = False,
    num_generations: int = 4,
    max_completion_length: int = 256,
    max_turns: int = 4,
    temperature: float = 1.0,
    loss_type: str = None,
    epsilon: float = None,
    epsilon_high: float = None,
    mask_truncated_completions: bool = None,
    importance_sampling_level: str = None,
    scale_rewards: str = None,
    dapo: bool = False,
    dynamic_sampling: bool = None,
    dynamic_sampling_std_threshold: float = 0.0,
    dynamic_sampling_max_resamples: int = 32,
    seed: int = 42,
    snapshot: str = None,
    task_set: str = "default",
    split: str = None,
    allow_zero_variance: bool = False,
    canary_steps: int = 50,
    min_canary_reward: float = -1.0,
    canary_patience: int = 3,
    verify: bool = True,
    report_to: str = "none",
    run_name: str = None,
    qlora: bool = True,
    base_model: str = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    gradient_checkpointing: bool = True,
    optim: str = "paged_adamw_8bit",
    save_steps: int = 0,
    save_total_limit: int = 2,
    resume_from_checkpoint: str = None,
    save_rollout_traces: bool = False,
    rollout_trace_path: str = None,
    rollout_trace_max_samples: int = 0,
):
    # 工具没注册齐就直接失败：模型不会因为工具列表不全而报错，
    # 它会编工具名，训练照跑、奖励照记，只是那些工具对应的任务永远做不成。
    require_all_tools("GRPO 训练")

    # 先校验日志后端再加载模型：参数写错时应立刻失败，而不是等模型加载完
    backends = resolve_report_to(report_to)
    dapo_options = resolve_dapo_options(
        dapo=dapo,
        loss_type=loss_type,
        epsilon_high=epsilon_high,
        mask_truncated_completions=mask_truncated_completions,
        beta=beta,
    )
    loss_type, epsilon_high, mask_truncated_completions, beta = dapo_options
    dynamic_sampling = resolve_dynamic_sampling_option(
        dapo=dapo, dynamic_sampling=dynamic_sampling
    )
    if dapo:
        print(
            f"🧬 DAPO 预设: loss_type={loss_type}, epsilon_high={epsilon_high}, "
            f"mask_truncated_completions={mask_truncated_completions}, beta={beta}, "
            f"dynamic_sampling={dynamic_sampling}"
        )
    if not math.isfinite(dynamic_sampling_std_threshold) or dynamic_sampling_std_threshold < 0:
        raise SystemExit("❌ dynamic_sampling_std_threshold 必须是有限的非负数")
    if dynamic_sampling_max_resamples < 0:
        raise SystemExit("❌ dynamic_sampling_max_resamples 不能为负数")
    if dynamic_sampling and num_generations < 2:
        raise SystemExit("❌ dynamic sampling 要求 num_generations 至少为 2")
    if dynamic_sampling:
        try:
            # Fail before loading tokenizer/model when launched with torchrun or
            # accelerate. The wrapper repeats this check against the actual
            # Accelerator at rollout time as a second line of defence.
            assert_dynamic_sampling_single_process()
        except DynamicSamplingDistributedError as exc:
            raise SystemExit(f"❌ {exc}") from exc
    if rollout_trace_max_samples < 0:
        raise SystemExit("❌ rollout_trace_max_samples 不能为负数")
    if not math.isfinite(zero_variance_noise) or zero_variance_noise < 0:
        raise SystemExit("❌ zero_variance_noise 必须是有限的非负数（0 表示禁用）")
    if save_steps < 0:
        raise SystemExit("❌ save_steps 不能为负数")
    if save_steps > 0 and save_total_limit < 1:
        raise SystemExit("❌ 启用周期 checkpoint 时 save_total_limit 必须至少为 1")
    resolved_resume_checkpoint = None
    if resume_from_checkpoint:
        resume_path = Path(resume_from_checkpoint)
        if not resume_path.is_absolute():
            resume_path = REPO_ROOT / resume_path
        if not resume_path.exists():
            raise SystemExit(f"❌ resume_from_checkpoint 不存在: {resume_path}")
        resolved_resume_checkpoint = str(resume_path)
    logging_dir = str(REPO_ROOT / output_dir / "logs")

    model_path = REPO_ROOT / model
    if model_path.exists():
        resolved = str(model_path)
    elif "/" in model and not model.startswith(("outputs/", "./", "/")):
        resolved = model                    # 形如 org/name 的 HF 仓库
    else:
        raise SystemExit(
            f"❌ 未找到本地模型 {model_path}\n"
            f"请先运行 train_sft / train_dpo，或用 --model 指定其它检查点"
        )

    # 先验证任务与快照，再加载模型。语义标准答案损坏时应在占用显存前失败。
    tasks = _load_tasks(task_set, split)
    snapshot_path = Path(snapshot) if snapshot else DEFAULT_SNAPSHOT
    if task_set not in ("code", "bfcl") and not snapshot_path.exists():
        raise SystemExit(
            f"❌ 多轮 GRPO 需要离线快照 {snapshot_path}\n"
            "   请先运行: python -m AgenticArxiv.rl.build_snapshot"
        )
    if task_set not in ("code", "bfcl"):
        tasks = attach_expected_paper_ids(tasks, snapshot_path)
    # bfcl 任务没有 expected_tools 字段: 标准链是单次 submit_calls 元提交,
    # 工具步数按 1 计（submit_calls + FINISH），与代码域 run_tests 同构。
    required_turns = max(
        (
            (1 if task_set == "bfcl" else len(t.get("expected_tools") or [])) + 1
            for t in tasks
        ),
        default=1,
    )
    if max_turns < required_turns:
        raise SystemExit(
            f"❌ max_turns={max_turns} 小于当前任务标准链所需的 {required_turns} "
            "（工具步骤 + FINISH），最长任务永远无法完成"
        )

    # --- 数据集：先验证身份元数据，再加载 tokenizer / 模型 ---
    # 数据或 datasets 版本不兼容时应在占用 GPU 前失败。
    rows = build_prompt_dataset(tasks)
    train_dataset = Dataset.from_list(rows)
    expected_hidden_ids = [row["task_id"] for row in rows]
    actual_hidden_ids = [
        row["prompt"][0].get("_task_id")
        for row in train_dataset
    ]
    if actual_hidden_ids != expected_hidden_ids:
        raise SystemExit(
            "❌ datasets 转换后隐藏 task_id 元数据丢失，"
            "不能安全区分相同文本、不同 setup 的 GRPO 任务"
        )
    # ``task_id`` also lives in each structured prompt as rollout-only metadata.
    # Keep this text lookup solely as a compatibility fallback for unique
    # prompts. Duplicate visible prompts are valid when their hidden setup is
    # different, so they must not be forced into a lossy text -> id mapping.
    prompt_task_ids = {}
    ambiguous_prompt_texts = set()
    for row in rows:
        prompt_text = row["prompt"][0]["content"]
        hidden_task_id = row["prompt"][0].get("_task_id")
        if hidden_task_id != row["task_id"]:
            raise SystemExit(
                "❌ GRPO prompt 缺少可信的隐藏 task_id 元数据: "
                f"row={row['task_id']}, hidden={hidden_task_id}"
            )
        if prompt_text in ambiguous_prompt_texts:
            continue
        previous = prompt_task_ids.get(prompt_text)
        if previous is not None and previous != row["task_id"]:
            prompt_task_ids.pop(prompt_text)
            ambiguous_prompt_texts.add(prompt_text)
        else:
            prompt_task_ids[prompt_text] = row["task_id"]
    if ambiguous_prompt_texts:
        print(
            f"🪪 检测到 {len(ambiguous_prompt_texts)} 个重复可见 prompt；"
            "使用不会进入模型 token 的隐藏 task_id 选择各自 setup"
        )
    print(f"📚 任务数: {len(tasks)}（GRPO 为在线算法，无需预生成轨迹数据）")

    print(f"📦 加载 tokenizer: {resolved}")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved, local_files_only=Path(resolved).exists()
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = _load_policy(
        resolved,
        qlora=qlora,
        base_model=base_model,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # --- 生成长度守卫 ---
    need = _gold_completion_tokens(tokenizer, tasks)
    if max_completion_length < need:
        raise SystemExit(
            f"❌ max_completion_length={max_completion_length} 小于标准动作所需的 {need} tokens，"
            f"模型不可能生成完整动作，奖励会恒为下限、梯度为 0。\n"
            f"请改用 --max_completion_length {need + 64}"
        )
    if task_set in ("code", "bfcl"):
        env = None  # 代码/BFCL 域: canary/verifier 快照语义不适用, 旁路
    else:
        env = load_mock_env(snapshot_path)
    if task_set == "code":
        from AgenticArxiv.rl.code_env import make_code_environment_factory
        environment_factory = make_code_environment_factory()
    elif task_set == "bfcl":
        from AgenticArxiv.rl.bfcl_env import make_bfcl_environment_factory
        environment_factory = make_bfcl_environment_factory()
    else:
        environment_factory = make_environment_factory(snapshot_path)
    print(f"🗂  使用独立离线环境执行多轮工具调用: {snapshot_path}")
    # structured completions 已携带真实 tool observations，不在 reward 端重复执行。
    tracker = RewardComponentTracker()
    reward_calc = _build_reward_calculator(
        reward_curriculum_steps, reward_weights, disable_failure_caps, task_set
    )
    initial_weights = reward_calc.schedule(training_step=0)
    print(
        "⚖️  奖励课程: "
        f"curriculum_steps={reward_curriculum_steps}, initial_weights={initial_weights}"
    )
    auditor = None
    resolved_trace_path = None
    if save_rollout_traces or rollout_trace_path:
        resolved_trace_path = (
            Path(rollout_trace_path)
            if rollout_trace_path
            else REPO_ROOT / output_dir / "rollout_traces.jsonl"
        )
        if not resolved_trace_path.is_absolute():
            resolved_trace_path = REPO_ROOT / resolved_trace_path
        auditor = RolloutAuditWriter(
            resolved_trace_path,
            num_generations=num_generations,
            allowed_tools=SUPPORTED_TOOL_NAMES,
            max_samples=rollout_trace_max_samples,
        )
        limit_text = (
            str(rollout_trace_max_samples)
            if rollout_trace_max_samples
            else "不限"
        )
        print(
            f"🔎 原始 rollout 审计: {resolved_trace_path} "
            f"(保存上限={limit_text})"
        )
    tasks_by_id = {t["id"]: t for t in tasks}
    reward_fn = make_grpo_reward_fn(
        tasks_by_id,
        env=None,
        reward_calc=reward_calc,
        tracker=tracker,
        auditor=auditor,
    )
    noise_injector = None
    if zero_variance_noise > 0:
        # 只包住正式训练奖励；dynamic sampling 的无副作用预打分保持干净，
        # 噪声只应影响优势计算，不应决定哪些组被接受进 batch。
        noise_injector = ZeroVarianceNoiseInjector(
            zero_variance_noise, num_generations=num_generations, seed=seed
        )
        reward_fn = _wrap_reward_fn_with_zero_variance_noise(reward_fn, noise_injector)
        print(
            f"🎲 [zero_variance_noise] R1.6 方差直接操纵: delta={zero_variance_noise}, "
            f"seed={seed} —— 组内奖励 std==0 的组注入 Uniform(-δ,+δ) 噪声"
        )
    if folr_first_turn_credit > 0:
        reward_fn = _wrap_reward_fn_with_folr(reward_fn, folr_first_turn_credit)
        print(f"FOLR 移植: first-turn credit={folr_first_turn_credit} (TSPO 家族忠实检验)")
    if advantage_intervention != "none" or dump_group_rewards:
        # 记录器必须最后包（噪声之后），保证捕获的是 TRL 实际归一化的奖励
        reward_fn = _wrap_reward_fn_with_recorder(reward_fn)
        InterventionGRPOTrainer.varm_mode = advantage_intervention
        InterventionGRPOTrainer.varm_seed = seed
        InterventionGRPOTrainer.varm_dump = bool(dump_group_rewards)
        InterventionGRPOTrainer.avspo_alpha = avspo_alpha
        InterventionGRPOTrainer.avspo_anchor = avspo_anchor
        InterventionGRPOTrainer.avspo_eta = avspo_eta
        InterventionGRPOTrainer.avspo_tau0 = avspo_tau0
        InterventionGRPOTrainer.avspo_clip_positive = avspo_clip_positive
    rollout_func = make_multiturn_rollout_func(
        environment_factory,
        max_turns=max_turns,
        tasks_by_id=tasks_by_id,
        prompt_task_ids=prompt_task_ids,
    )
    if dynamic_sampling:
        # Preview scoring has no tracker/auditor. Rejected candidates therefore
        # leave no formal training record; TRL still calls the normal reward_fn
        # exactly once for the accepted batch.
        preview_reward_fn = make_grpo_reward_fn(
            tasks_by_id,
            env=None,
            reward_calc=reward_calc,
            tracker=None,
            auditor=None,
        )
        rollout_func = make_dynamic_sampling_rollout_func(
            rollout_func,
            prompt_rows=rows,
            preview_reward_func=preview_reward_fn,
            reward_std_threshold=dynamic_sampling_std_threshold,
            max_resample_attempts=dynamic_sampling_max_resamples,
            seed=seed,
        )
        print(
            "🎲 DAPO dynamic sampling: "
            f"std>{dynamic_sampling_std_threshold}, "
            f"max_resamples={dynamic_sampling_max_resamples}, seed={seed}"
        )

    # GRPO 要求生成批量能被 num_generations 整除
    if batch_size % num_generations != 0:
        batch_size = num_generations
        print(f"  调整 per_device_train_batch_size = {batch_size}（需被 num_generations 整除）")

    cfg_kwargs = {
        "output_dir": str(REPO_ROOT / output_dir),
        "num_train_epochs": epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        # 27B 级策略的 ref model 再要一份等量权重，2×64GiB 放不下：
        # 大模型走 beta=0（无 KL，DAPO/Dr.GRPO 同款口径）
        "beta": 0.0 if (
            beta > 0 and getattr(policy, "num_parameters", lambda: 0)() > 15_000_000_000
        ) else beta,
        "num_generations": num_generations,
        "max_completion_length": max_completion_length,
        "temperature": temperature,
        "seed": seed,
        "data_seed": seed,
        "logging_steps": 1,
        "logging_first_step": True,
        "save_strategy": "steps" if save_steps > 0 else "no",
        "save_steps": save_steps if save_steps > 0 else 500,
        "save_total_limit": save_total_limit,
        "optim": ("adamw_torch_fused" if os.getenv("AA_LORA_BF16") == "1"
                  else (optim if qlora else "adamw_torch_fused")),
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_kwargs": (
            {"use_reentrant": False} if gradient_checkpointing else None
        ),
        "report_to": backends,
        "run_name": run_name or Path(output_dir).name,
        **_precision_flags(),
    }

    # ---- DAPO 系选项（README P3） ----
    # 这三项 TRL 已经原生支持，无需 fork GRPOTrainer：clip-higher 走
    # epsilon_high、overlong filtering 走 mask_truncated_completions、
    # token-level loss 走 loss_type。默认全部保持 TRL 现值，只有显式指定
    # （或用 --dapo 预设）才改变行为 —— 历史实验的可比性不能被默认值悄悄改掉。
    #
    # Dynamic sampling 走公开 rollout_func 的 extra-fields 通道，由包装器在
    # 正式 reward 调用前完成无副作用预打分与补采，不覆写 TRL 私有方法。
    if loss_type is not None:
        cfg_kwargs["loss_type"] = loss_type
    if epsilon_high is not None:
        cfg_kwargs["epsilon_high"] = epsilon_high
    if epsilon is not None:
        cfg_kwargs["epsilon"] = epsilon
    if mask_truncated_completions is not None:
        cfg_kwargs["mask_truncated_completions"] = mask_truncated_completions
    # ---- GSPO / Dr.GRPO（Roadmap P1） ----
    # 序列级重要性采样走 importance_sampling_level=sequence（GSPO）；
    # Dr.GRPO 完整配方是 --loss_type dr_grpo --scale_rewards none（全局常数
    # 归一替代组内 std 归一，去长度偏置）。与 --dapo 预设正交可组合。
    # 同样遵守「默认不动、显式才改」，保证历史实验可比性。
    if importance_sampling_level is not None:
        cfg_kwargs["importance_sampling_level"] = importance_sampling_level
    if scale_rewards is not None:
        cfg_kwargs["scale_rewards"] = scale_rewards
    # GRPOConfig 的字段在 TRL 各版本间有增删，按实际安装版本过滤，
    # 避免因为一个参数名不存在就整个训练起不来
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k in cfg_kwargs if k not in valid)
    if dropped:
        print(f"  提示：当前 TRL 不支持这些 GRPOConfig 参数，已忽略 -> {dropped}")
    config = GRPOConfig(**{k: v for k, v in cfg_kwargs.items() if k in valid})

    # 单进程时钉住单卡（避免 Trainer 走 DataParallel 的段错误）；
    # accelerate launch 下 accelerator 自己管设备，这里自动跳过。
    from rl.precision import assert_quantization_is_single_process, pin_single_gpu

    assert_quantization_is_single_process(qlora)
    pin_single_gpu(config)

    print(describe_logging(backends, logging_dir if backends else None))
    print(f"🚀 开始 GRPO 训练（每个 prompt 采样 {num_generations} 条，规则奖励）")
    trainer = InterventionGRPOTrainer(
        model=policy,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        rollout_func=rollout_func,
    )
    # 必须在 trainer 建好之后接线。即使 report_to=none，TRL 仍会把 _metrics
    # 汇总到控制台；report_to 只控制是否转发到 TensorBoard/wandb，不应决定
    # 是否采集指标。此前的条件绑定还会让 _pending 在长训练中持续增长。
    tracker.bind(trainer)

    # --- Canary 回调：每 N 步在固定任务上评估，检测性能退化 ---
    canary_cb = None
    if canary_steps > 0:
        canary_evaluator = CanaryEvaluator(
            model=policy,
            tokenizer=tokenizer,
            # Canary 与训练必须使用同一奖励课程，否则监控值和训练目标不可比。
            # 例外: canary 任务集固定为 arXiv 冒烟任务，BFCL 臂的 score_chain
            # 分量口径对这些任务恒为下限 —— 直接传入会把 canary 变成常量
            # -1.0 的死监控并误触 min_reward 熔断。这里与代码域同口径，用
            # 结构分监控策略是否退化。
            reward_calc=(
                RewardCalculator(curriculum_steps=reward_curriculum_steps)
                if task_set == "bfcl"
                else reward_calc
            ),
            env=env,
            num_generations=num_generations,
            max_new_tokens=max_completion_length,
            temperature=temperature,
        )
        canary_cb = CanaryCallback(
            evaluator=canary_evaluator,
            steps=canary_steps,
            min_reward=min_canary_reward,
            patience=canary_patience,
        )
        trainer.add_callback(canary_cb)
        print(f"🐤 Canary: 每 {canary_steps} 步评估，阈值={min_canary_reward}，patience={canary_patience}")

    guard = None
    if not allow_zero_variance:
        guard = RewardVarianceGuard()
        trainer.add_callback(guard)
    try:
        train_result = trainer.train(
            resume_from_checkpoint=resolved_resume_checkpoint
        )
    except BaseException:
        # JSONL is flushed after every reward batch. Also persist its compact
        # summary on OOM, Ctrl-C, or trainer failure so the last healthy batch
        # remains diagnosable instead of disappearing with the failed run.
        if auditor is not None:
            failed_summary_path = auditor.save_summary(
                resolved_trace_path.with_suffix(".summary.json")
            )
            print(f"🔎 训练中断，已保留 rollout 审计: {failed_summary_path}")
        raise

    # 放在方差/canary 退出检查之前：即使保护器提前停止，也保留诊断数据。
    reward_summary_path = tracker.save_task_summary(
        REPO_ROOT / output_dir / "reward_probe_summary.json"
    )
    print(f"📊 逐任务奖励统计: {reward_summary_path}")
    noise_summary = None
    noise_stats_path = None
    if noise_injector is not None:
        noise_summary = noise_injector.summary()
        noise_stats_path = noise_injector.save_stats(
            REPO_ROOT / output_dir / "zero_variance_noise_stats.json"
        )
        print(
            f"🎲 [zero_variance_noise] 注入统计: 组 "
            f"{noise_summary['groups_injected']}/{noise_summary['groups_total']}"
            f"（{100.0 * noise_summary['group_injection_rate']:.1f}%）、样本 "
            f"{noise_summary['samples_injected']}/{noise_summary['samples_total']}"
            f"（{100.0 * noise_summary['sample_injection_rate']:.1f}%）"
            f" -> {noise_stats_path}"
        )
    rollout_audit_summary_path = None
    rollout_audit_summary = None
    if auditor is not None:
        rollout_audit_summary_path = auditor.save_summary(
            resolved_trace_path.with_suffix(".summary.json")
        )
        rollout_audit_summary = auditor.summary()
        print(
            f"🔎 rollout 审计摘要: {rollout_audit_summary_path} | "
            f"已保存={rollout_audit_summary['saved_samples']}, "
            f"异常样本={rollout_audit_summary['anomalous_samples']}"
        )

    if guard is not None and guard.tripped:
        raise SystemExit(1)
    if canary_cb is not None and canary_cb.tripped:
        raise SystemExit(1)

    final_dir = REPO_ROOT / output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if qlora:
        required_adapter_files = (
            final_dir / "adapter_config.json",
            final_dir / "adapter_model.safetensors",
        )
        missing = [str(path) for path in required_adapter_files if not path.exists()]
        if missing:
            raise SystemExit(f"❌ GRPO完成但adapter文件缺失: {missing}")
    import torch
    actual_steps = int(getattr(trainer.state, "global_step", 0) or 0)
    stopped_early = max_steps > 0 and actual_steps < max_steps
    if guard is not None and guard.converged:
        stop_reason = "reward_variance_window_saturated"
    elif canary_cb is not None and canary_cb.tripped:
        stop_reason = "canary_threshold"
    elif stopped_early:
        stop_reason = "trainer_stopped_early"
    else:
        stop_reason = "max_steps_or_epochs_completed"
    manifest = {
        "stage": "grpo_qlora" if qlora else "grpo_full",
        "source_model": resolved,
        "base_model": base_model,
        "task_set": task_set,
        "split": split,
        "task_count": len(tasks),
        "train_loss": float(train_result.training_loss),
        "learning_rate": lr,
        "beta": beta,
        "dapo": dapo,
        "loss_type": loss_type,
        "epsilon_high": epsilon_high,
        "mask_truncated_completions": mask_truncated_completions,
        "importance_sampling_level": importance_sampling_level,
        "scale_rewards": scale_rewards,
        "dynamic_sampling": {
            "enabled": dynamic_sampling,
            "reward_std_threshold": dynamic_sampling_std_threshold,
            "max_resample_attempts": dynamic_sampling_max_resamples,
            "single_process_only": True,
        },
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "temperature": temperature,
        "frozen_policy_probe": lr == 0.0,
        "max_steps": max_steps,
        "requested_max_steps": max_steps,
        "actual_steps": actual_steps,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "reward_variance_guard": (
            {
                "enabled": True,
                "converged": guard.converged,
                "tripped": guard.tripped,
                "patience": guard.patience,
                "grace_steps": guard.grace_steps,
                "stop_step": guard.stop_step,
            }
            if guard is not None
            else {"enabled": False}
        ),
        "reward_curriculum_steps": reward_curriculum_steps,
        "zero_variance_noise": zero_variance_noise,
        "zero_variance_noise_stats": (
            {
                "stats_path": str(noise_stats_path),
                **noise_summary,
            }
            if (noise_injector is not None and noise_summary is not None)
            else None
        ),
        "initial_reward_weights": dataclasses.asdict(initial_weights),
        "rollout_audit": (
            {
                "trace_path": str(resolved_trace_path),
                "summary_path": str(rollout_audit_summary_path),
                "seen_samples": rollout_audit_summary["seen_samples"],
                "saved_samples": rollout_audit_summary["saved_samples"],
                "anomalous_samples": rollout_audit_summary["anomalous_samples"],
            }
            if rollout_audit_summary is not None
            else None
        ),
        "num_generations": num_generations,
        "max_turns": max_turns,
        "max_completion_length": max_completion_length,
        "seed": seed,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit if save_steps > 0 else None,
        "resumed_from_checkpoint": resolved_resume_checkpoint,
        "peak_vram_gib": (
            torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        ),
    }
    trainable, model_parameters, trainable_ratio = _trainable_parameter_stats(trainer.model)
    manifest.update({
        "trainable_parameters": trainable,
        "model_parameters": model_parameters,
        "trainable_ratio_percent": trainable_ratio,
    })
    if not math.isfinite(manifest["train_loss"]):
        raise SystemExit(f"❌ GRPO training_loss不是有限数: {manifest['train_loss']}")
    (final_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"✅ GRPO 训练完成，模型已保存: {final_dir}")

    # --- 阶段验证：训练后检查模型是否达到最低质量阈值 ---
    if verify:
        print(f"\n🔍 运行 GRPO 阶段验证...")
        verifier = StageVerifier()
        report = verifier.verify_grpo(
            model_path=str(final_dir),
            tokenizer=tokenizer,
            env=env,
        )
        verifier.save_report(report, final_dir)
        print(report.summary())
        if not report.passed:
            raise SystemExit(
                f"\n❌ GRPO 阶段验证失败，模型未达到最低质量阈值。\n"
                f"   检查 {final_dir / 'verification_report.json'} 了解详情。\n"
                f"   跳过验证用 --no-verify。"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GRPO 训练")
    p.add_argument("--model", default="outputs/dpo/final")
    p.add_argument("--output_dir", default="outputs/grpo")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--disable_failure_caps", action="store_true",
        help="E4 消融：禁用非补偿失败封顶（hard_invalid/false_finish/efficiency 三处截断）")
    p.add_argument(
        "--reward_weights", default=None,
        help="JSON 逐分量覆盖奖励权重（E4 消融），分量名见 RewardCalculator.weights，"
             '如 {"outcome": 0.0}',
    )
    p.add_argument(
        "--reward_curriculum_steps", type=int, default=30,
        help="奖励课程持续步数。默认30保持历史实验兼容；SFT后直接优化工具、参数和结果时"
             "显式传0，从第一步启用完整语义权重。该值会写入training_manifest.json",
    )
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument(
        "--zero_variance_noise", type=float, default=0.0,
        help="R1.6 因果闭环实验：对组内奖励完全一致（std==0）的组，向每个成员"
             "奖励叠加 eps~Uniform(-δ,+δ)（δ=本值，建议 0.05）。噪声期望为 0，"
             "不改奖励语义，只恢复组内相对信号；0 表示禁用。注入统计写入 "
             "output_dir/zero_variance_noise_stats.json",
    )
    p.add_argument(
        "--allow_zero_variance", action="store_true",
        help="跳过「组内奖励方差为 0」检查。默认开启该检查：方差为 0 时"
             "GRPO 不产生任何梯度，训练会静默空转",
    )
    p.add_argument("--max_completion_length", type=int, default=256)
    p.add_argument(
        "--max_turns", type=int, default=4,
        help="每条 rollout 最多执行多少轮工具调用；环境 observation 不计入策略 loss",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--loss_type", default=None,
        choices=["grpo", "dapo", "bnpo", "dr_grpo", "cispo", "sapo", "luspo"],
        help="token 级损失的归一化口径。'dapo' 按全局有效 token 数归一化，"
             "消除长度偏置（DAPO 论文）。默认不传，沿用当前 TRL 的默认值",
    )
    p.add_argument(
        "--epsilon", type=float, default=None,
        help="重要性比裁剪下界（PPO-clip）。TRL 默认 0.2；设大值（如 10）等价关闭裁剪，"
             "用于无裁剪对照实验。默认不传，沿用 TRL 现值",
    )
    p.add_argument(
        "--epsilon_high", type=float, default=None,
        help="重要性比的上界（clip-higher）。DAPO 论文建议 0.28；"
             "默认不传，上下界取同一个 epsilon",
    )
    p.add_argument(
        "--mask_truncated_completions",
        action=argparse.BooleanOptionalAction, default=None,
        help="把撞上 max_completion_length 的截断轨迹从 loss 里剔除（DAPO overlong "
             "filtering）。默认不传，沿用 TRL 现值",
    )
    p.add_argument(
        "--importance_sampling_level",
        choices=["token", "sequence"], default=None,
        help="重要性采样的粒度：token（默认）或 sequence（GSPO，Qwen3 提出的"
             "序列级重要性采样——整条 completion 用一个比值，缓解长序列上"
             "token 级比值的方差爆炸）。默认不传，沿用 TRL 现值",
    )
    p.add_argument(
        "--scale_rewards",
        choices=["group", "none", "batch"], default=None,
        help="优势的归一化方式：group（组内 std 归一，默认）/ none（不归一，"
             "Dr.GRPO 配方需配合 --loss_type dr_grpo 一起用）/ batch。"
             "默认不传，沿用 TRL 现值",
    )
    p.add_argument(
        "--dapo", action="store_true",
        help="DAPO 预设：loss_type=dapo、epsilon_high=0.28、"
             "mask_truncated_completions=True、beta=0、dynamic_sampling=True。"
             "显式给出的单项参数优先于该预设",
    )
    p.add_argument(
        "--dynamic_sampling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="按 prompt 过滤组内奖励标准差不足的 rollout，并用新 prompt 补满 batch。"
             "默认仅随 --dapo 开启；当前只支持单进程",
    )
    p.add_argument(
        "--dynamic_sampling_std_threshold",
        type=float,
        default=0.0,
        help="接受 reward 组所需的最小样本标准差（严格大于该值）",
    )
    p.add_argument(
        "--dynamic_sampling_max_resamples",
        type=int,
        default=32,
        help="单个 generation batch 最多补采多少个新 prompt；耗尽时明确失败",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="训练与 rollout 采样种子；对比两个策略的原始轨迹时必须保持一致",
    )
    p.add_argument(
        "--advantage_intervention",
        choices=["none", "permute", "zero", "reverse", "swap", "avspo", "rloo"], default="none",
        help="V 系列优势干预（第四轮审稿裁决实验）: permute=非tie组内置换优势"
             "（保幅度毁内容）; avspo=AVSPO 忠实移植（虚拟样本只进归一化统计，"
             "He et al. ICML 2026, arXiv:2605.21125）; rloo=组相对估计器对照"
             "（A_i=G/(G-1)·(r_i-r̄)，无 std 归一化，回答并列静默是 GRPO 家族还是组相对估计的病）",
    )
    p.add_argument(
        "--avspo_alpha", type=float, default=0.5,
        help="AVSPO K 自适应指数: K = max(1, min(G, ceil(G·ACR^α)))",
    )
    p.add_argument(
        "--avspo_anchor", type=float, default=0.1,
        help="AVSPO r_obs == 0 时的虚拟锚点奖励",
    )
    p.add_argument(
        "--avspo_eta", type=float, default=0.01,
        help="AVSPO τ_adapt 自适应步长（Eq.10）",
    )
    p.add_argument(
        "--avspo_tau0", type=float, default=0.5,
        help="AVSPO τ_adapt 初值（运行中夹在 [0.1, 0.9]）",
    )
    p.add_argument(
        "--folr_first_turn_credit", type=float, default=0.0,
        help="FOLR 移植: 首个正确工具调用 turn 的加性 credit (TSPO 家族忠实检验), 0=禁用",
    )
    p.add_argument(
        "--avspo_clip_positive", action="store_true",
        help="AVSPO 扩展消融: 负 tie 组按原文 r=0 anchor 处理 (clip-to-positive)",
    )
    p.add_argument(
        "--dump_group_rewards", action="store_true",
        help="逐组逐步奖励向量写 output_dir/group_rewards.jsonl"
             "（tie 组成分析插桩，可与干预正交使用）",
    )
    p.add_argument(
        "--qlora", action=argparse.BooleanOptionalAction, default=True,
        help="使用4-bit QLoRA继续训练SFT adapter（单4090默认方案）",
    )
    p.add_argument(
        "--base_model", default=None,
        help="SFT adapter对应的基座模型；默认读取adapter_config.json",
    )
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True,
    )
    p.add_argument("--optim", default="paged_adamw_8bit")
    p.add_argument(
        "--save_steps", type=int, default=0,
        help="每隔多少个 optimizer step 保存可恢复 checkpoint；0 表示只保存最终模型",
    )
    p.add_argument(
        "--save_total_limit", type=int, default=2,
        help="最多保留多少个周期 checkpoint（最终模型不计入）",
    )
    p.add_argument(
        "--resume_from_checkpoint", default=None,
        help="从指定 checkpoint 继续模型、优化器和调度器状态；需保持原 output_dir 与训练参数",
    )
    p.add_argument("--snapshot", default=None)
    p.add_argument(
        "--task_set", choices=["default", "expanded", "code", "bfcl"], default="default",
        help="default=benchmark/tasks.py 的 8 条冒烟任务（保持现状）；"
             "expanded=完整基准集；code=36 条参数化代码任务；"
             "bfcl=BFCL-V3 multi_turn_base 200 任务（PREREG_BFCL 第二环境轨）。"
             "改默认值会让既有训练曲线不可比，故需显式指定",
    )
    p.add_argument(
        "--split", default=None, metavar="[FILE:]NAME",
        help=f"只用某一份切分训练，如 rl_train（用 {DEFAULT_SPLIT_PATH.name}）或 "
             "data/splits/v2.json:train。rl_train 只含成功率中间带的任务——"
             "GRPO 的梯度来自组内奖励方差，两端的任务不产生梯度。需配合 --task_set expanded",
    )
    p.add_argument(
        "--canary_steps", type=int, default=50,
        help="每隔多少步运行 canary 评估（0 表示禁用）",
    )
    p.add_argument(
        "--min_canary_reward", type=float, default=-1.0,
        help="Canary 奖励下限。低于此值连续 patience 次则停止训练。"
             "默认 -1.0（理论最低分）表示禁用检查",
    )
    p.add_argument(
        "--canary_patience", type=int, default=3,
        help="Canary 奖励连续低于阈值多少次后停止训练",
    )
    p.add_argument(
        "--verify", action=argparse.BooleanOptionalAction, default=True,
        help="训练结束后运行阶段验证（默认开启）",
    )
    p.add_argument(
        "--report_to", default="none",
        help="训练曲线记到哪：none / auto / tensorboard / wandb（可逗号分隔）。"
             "记录 reward、kl、grad_norm、frac_reward_zero_std，"
             "以及 format/tool/argument/process/outcome 五个奖励分量与当前课程权重",
    )
    p.add_argument(
        "--run_name", default=None,
        help="本次运行在 TensorBoard / wandb 里的名字，默认取 output_dir 末段",
    )
    p.add_argument(
        "--save_rollout_traces", action="store_true",
        help="保存每条 GRPO 原始生成、工具轨迹、奖励分解和格式异常。"
             "默认写到 output_dir/rollout_traces.jsonl",
    )
    p.add_argument(
        "--rollout_trace_path", default=None,
        help="自定义 rollout 审计 JSONL 路径；传入该参数会自动启用审计",
    )
    p.add_argument(
        "--rollout_trace_max_samples", type=int, default=0,
        help="最多写入多少条原始 rollout（0=全部）；摘要统计始终覆盖全部样本",
    )
    main(**vars(p.parse_args()))
