"""策略侧多模态 VLM SFT（plan.md S2：27B 级端侧模型的混合数据后训练）。

由 ``train_vlm_figure_qa.py`` 扩展而来，差异点：

- 数据是 ``scripts/build_vlm_policy_sft_mix.py`` 产出的混合集
  （``messages`` + ``images``（relpath 列表）+ ``source``），文本与图文行统一；
- ``--device_map auto``：单进程把权重均分到两张卡（``HIP_VISIBLE_DEVICES=2,3``），
  Trainer 检测 ``hf_device_map`` 自动放弃 DataParallel/搬移；
- LoRA 目标按预设**运行时解析**（``language`` / ``language+projector`` /
  ``all-linear``），必须覆盖视觉-语言投影层——只挂语言层会静默漏配大部分
  多模态路径；MoE expert 层默认排除，``--include_experts`` 消融用；
- 大模型**默认 adapter-only**（``--no-merge`` 默认开），merge 需显式
  ``--merge`` 且仅限单卡小模型；保存窗走 :mod:`rl.save_guard` 的跨进程锁 +
  page cache 逐出 + settle。

用法::

    HIP_VISIBLE_DEVICES=2,3 python -m AgenticArxiv.rl.train_vlm_sft \\
        --model <本地基座目录> --data data/vlm/vlm_policy_sft_v1.jsonl \\
        --output_dir outputs/vlm_policy_sft --device_map auto
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Mapping

from PIL import Image  # noqa: E402  # 提前加载，避免 dataloader worker 内重复初始化

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

LANGUAGE_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PROJECTOR_PATTERNS = ("merger", "projector", "multi_modal")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filter_dataclass_kwargs(config_cls, kwargs: Dict[str, Any]):
    supported = {field.name for field in dataclasses.fields(config_cls)}
    dropped = sorted(set(kwargs) - supported)
    return {key: value for key, value in kwargs.items() if key in supported}, dropped


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"数据集为空: {path}")
    return rows


def load_image(relpath: str, image_root: Path, *, max_side: int = 1024) -> Image.Image | None:
    """全解码加载；截断图先按 PIL 截断容错重试，仍失败返回 None（调用方剔除该行）。"""
    from PIL import ImageFile

    path = (image_root / relpath).resolve()
    try:
        image = Image.open(path)
        image.load()
        image = image.convert("RGB")
    except OSError:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            image = Image.open(path).convert("RGB")
        except Exception:  # noqa: BLE001 - 坏图按缺失处理
            return None
    longest = max(image.size)
    if longest > max_side:
        scale = max_side / longest
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.LANCZOS,
        )
    return image


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """工具轨迹是纯字符串 content、图文行是分段 content；arrow 列必须同型。"""
    out = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        out.append({"role": message["role"], "content": content})
    return out


def build_hf_dataset(rows: List[Dict[str, Any]], *, image_root: Path):
    from datasets import Dataset

    records = []
    dropped = 0
    for row in rows:
        relpaths = row.get("images") or []
        images = [img for img in (load_image(rel, image_root) for rel in relpaths) if img is not None]
        if relpaths and not images:
            dropped += 1  # 整行图文全坏：剔除而不是让 Trainer 在数小时后炸掉
            continue
        records.append(
            {
                "images": images,
                "messages": _normalize_messages(row["messages"]),
            }
        )
    if dropped:
        print(f"⚠️ 剔除坏图整行: {dropped} 行")
    return Dataset.from_list(records)


def resolve_lora_targets(model, preset: str, *, include_experts: bool) -> List[str]:
    """按预设扫描 named_modules，返回精确模块名列表（跨架构稳）。"""
    import torch.nn as nn

    names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

    def is_expert(name: str) -> bool:
        return "experts" in name

    def is_lm_head(name: str) -> bool:
        return name.endswith("lm_head") or "embed_tokens" in name

    if preset == "language":
        selected = [
            name for name in names
            if name.endswith(LANGUAGE_PROJECTIONS)
            and not is_lm_head(name)
            and (include_experts or not is_expert(name))
        ]
    elif preset == "language+projector":
        selected = [
            name for name in names
            if (
                name.endswith(LANGUAGE_PROJECTIONS)
                or any(pattern in name for pattern in PROJECTOR_PATTERNS)
            )
            and not is_lm_head(name)
            and (include_experts or not is_expert(name))
        ]
    elif preset == "all-linear":
        selected = [name for name in names if not is_lm_head(name)]
    else:  # pragma: no cover - argparse 限定
        raise ValueError(f"未知 LoRA 预设: {preset}")
    if not selected:
        raise SystemExit(f"LoRA 预设 {preset} 没有命中任何模块；检查模型结构")
    return sorted(selected)


class MaskedLMHeadSFTTrainer:
    """只对 label≠-100 的位置计算 lm_head/loss 的 SFTTrainer。

    248k 词表上 TRL 默认的全序列 logits（bf16→fp32→CE）要吃 ~10GB 级
    瞬时显存；本数据集的监督密度只有 ~5%（识图 target ~40 token、总结
    ~250、工具轨迹 argument 数十字），把 lm_head 限制在目标位置后词表
    维度开销降两个数量级。任何一步失败都自动回退 TRL 默认路径。
    """

    def __new__(cls, *args, **kwargs):  # pragma: no cover - 见 install()
        raise TypeError("use install()")

    @staticmethod
    def install(trainer_cls):
        class _Impl(trainer_cls):
            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                try:
                    loss = self._sliced_loss(model, inputs)
                    if loss is not None:
                        return (loss, {"loss": loss}) if return_outputs else loss
                except Exception as exc:  # noqa: BLE001 - 结构不符时回退默认路径
                    if not getattr(self, "_sliced_loss_warned", False):
                        print(f"⚠️ sliced lm_head loss 不可用（{exc}），回退 TRL 默认 loss")
                        self._sliced_loss_warned = True
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            def _sliced_loss(self, model, inputs):
                import torch

                labels = inputs.get("labels")
                if labels is None:
                    raise AttributeError("no labels")
                base = model.get_base_model() if hasattr(model, "get_base_model") else model
                backbone = base.model
                lm_head = base.get_output_embeddings()
                if backbone is None or lm_head is None:
                    raise AttributeError("backbone or lm_head missing")
                model_inputs = {
                    k: v for k, v in inputs.items()
                    if k not in ("labels",) and hasattr(v, "to")
                }
                outputs = backbone(**model_inputs)
                hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                shift_hidden = hidden[:, :-1, :]
                shift_labels = labels[:, 1:].to(shift_hidden.device)
                mask = shift_labels != -100
                if mask.sum() == 0:
                    raise ValueError("empty target positions")
                sel_hidden = shift_hidden[mask]
                sel_labels = shift_labels[mask]
                logits = lm_head(sel_hidden).float()
                loss = torch.nn.functional.cross_entropy(
                    logits, sel_labels, ignore_index=-100
                )
                return loss

        return _Impl


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="策略侧多模态 VLM SFT")
    parser.add_argument("--model", required=True, help="本地 VLM 目录")
    parser.add_argument("--data", default="data/vlm/vlm_policy_sft_v1.jsonl")
    parser.add_argument("--data-manifest", default="", help="数据 manifest（存在则强校验 sha256）")
    parser.add_argument("--image-root", default="AgenticArxiv")
    parser.add_argument("--output_dir", default="outputs/vlm_policy_sft")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=3e-5, help="强基座+短轨迹，防过拟合（勿用 1e-4 档）")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", choices=["language", "language+projector", "all-linear"],
                        default="language+projector")
    parser.add_argument("--include_experts", action="store_true", help="MoE expert 层纳入 LoRA（消融）")
    parser.add_argument("--no-lora", action="store_true", help="全参微调（27B 级别在双卡上放不下）")
    parser.add_argument("--device_map", choices=["single", "auto"], default="single")
    parser.add_argument("--max_memory", default="56GiB,56GiB", help="device_map=auto 时每卡上限")
    parser.add_argument("--attn_implementation", default="sdpa",
                        help="DCU 上用 sdpa；不要开 flash_attention_2（das 构建内核过旧）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=0, help="调试：只取前 N 条")
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--run_name", default="vlm_policy_sft")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_strategy", default="epoch")
    parser.add_argument("--merge", action="store_true", help="合并 LoRA（仅单卡小模型！大模型必爆宿主内存）")
    parser.add_argument("--settle_seconds", type=int, default=120, help="保存锁内 settle 秒数")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import SFTConfig, SFTTrainer

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"数据集不存在: {data_path}（先跑 scripts/build_vlm_policy_sft_mix.py）")
    if args.data_manifest:
        manifest_path = Path(args.data_manifest)
        if manifest_path.exists():
            frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
            actual = _sha256_file(data_path)
            if frozen.get("output_sha256") != actual:
                raise SystemExit(
                    f"数据 manifest 校验失败: {manifest_path} 记录 {frozen.get('output_sha256')} != 实际 {actual}"
                )
        else:
            raise SystemExit(f"--data-manifest 指定的文件不存在: {manifest_path}")

    image_root = Path(args.image_root).resolve()
    rows = load_rows(data_path)
    random.Random(args.seed).shuffle(rows)
    if args.num_samples:
        rows = rows[: args.num_samples]
    sources: Dict[str, int] = {}
    multimodal = 0
    for row in rows:
        sources[row.get("source", "unknown")] = sources.get(row.get("source", "unknown"), 0) + 1
        multimodal += 1 if row.get("images") else 0
    print(f"🧾 训练样本: {len(rows)} | 来源: {sources} | 图文行: {multimodal}")

    model_dir = Path(args.model).expanduser()
    load_kwargs: Dict[str, Any] = dict(
        dtype=torch.bfloat16,
        local_files_only=model_dir.exists(),
        attn_implementation=args.attn_implementation,
    )
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=model_dir.exists())
    if args.device_map == "auto":
        caps = {i: cap.strip() for i, cap in enumerate(args.max_memory.split(","))}
        print(f"🗺️ device_map=auto, max_memory={caps}")
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir), device_map="auto", max_memory=caps, **load_kwargs
        )
        assert getattr(model, "hf_device_map", None), "device_map=auto 但模型没有 hf_device_map"
    else:
        model = AutoModelForImageTextToText.from_pretrained(str(model_dir), **load_kwargs).to("cuda")
    model.config.use_cache = False

    from rl.save_guard import cgroup_memory_gib, evict_model_dir_page_cache

    # 装载读入的权重会把等量 page cache 留在宿主 cgroup 配额里，读完立逐（实测 63.9/64GiB 险情）
    evicted = evict_model_dir_page_cache(model_dir)
    print(f"🧹 page cache evicted: {evicted} files | cgroup now {cgroup_memory_gib()} GiB")

    if not args.no_lora:
        from peft import LoraConfig, get_peft_model

        target_modules = resolve_lora_targets(
            model, args.lora_targets, include_experts=args.include_experts
        )
        print(f"🎯 LoRA 目标模块: {len(target_modules)} 个（预设 {args.lora_targets}）")
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    else:
        try:
            model.enable_input_require_grads()
        except Exception:  # pragma: no cover - 取决于模型实现
            pass

    dataset = build_hf_dataset(rows, image_root=image_root)
    print(f"🖼️ 数据集列: {dataset.column_names} | 行数: {len(dataset)}")

    output_dir = Path(args.output_dir)
    from rl.precision import pin_single_gpu, precision_flags

    config_kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=1,
        save_only_model=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=args.report_to,
        run_name=args.run_name,
        seed=args.seed,
        remove_unused_columns=False,
        packing=False,
        dataloader_num_workers=0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )
    config_kwargs, dropped = _filter_dataclass_kwargs(SFTConfig, config_kwargs)
    if dropped:
        print(f"⚠️  当前 TRL 版本不支持以下 SFTConfig 参数，已忽略: {dropped}")
    config_kwargs.update(precision_flags())
    config = SFTConfig(**config_kwargs)
    # device_map=auto 时 Trainer 看到 hf_device_map 不搬移不加 DP；单卡路径按仓库惯例钉住。
    pin_single_gpu(config)

    trainer_cls = MaskedLMHeadSFTTrainer.install(SFTTrainer)
    trainer = trainer_cls(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=processor,
    )

    from rl.save_guard import cgroup_memory_gib, guarded_save

    train_result = trainer.train()

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    guarded_save(
        lambda: (trainer.save_model(str(final_dir)), processor.save_pretrained(str(final_dir))),
        lock_path=output_dir / "save.lock",
        out_dir=final_dir,
        settle_seconds=args.settle_seconds,
    )

    merged = False
    if not args.no_lora and args.merge:
        if args.device_map == "auto":
            raise SystemExit("拒绝在 device_map=auto 的多卡大模型上 merge（宿主重组权重必爆 cgroup）")
        merged_model = trainer.model.merge_and_unload()
        merged_model.config.use_cache = True
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        guarded_save(
            lambda: (merged_model.save_pretrained(str(merged_dir), safe_serialization=True),
                     processor.save_pretrained(str(merged_dir))),
            lock_path=output_dir / "save.lock",
            out_dir=merged_dir,
            settle_seconds=args.settle_seconds,
        )
        merged = True
        print(f"🔗 已合并 LoRA 权重: {merged_dir}")

    peak_gib = None
    if torch.cuda.is_available():
        peak_gib = round(torch.cuda.max_memory_allocated() / 2**30, 1)

    manifest = {
        "stage": "vlm_policy_sft",
        "base_model": str(model_dir),
        "data": data_path.as_posix(),
        "data_sha256": _sha256_file(data_path),
        "train_rows": len(rows),
        "source_counts": sources,
        "multimodal_rows": multimodal,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "lora": None if args.no_lora else {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "preset": args.lora_targets,
            "include_experts": args.include_experts,
            "resolved_target_count": None if args.no_lora else len(target_modules),
        },
        "merged": merged,
        "seed": args.seed,
        "train_loss": train_result.training_loss,
        "trainable_parameters": trainable,
        "trainable_ratio_percent": round(100.0 * trainable / total, 4) if total else None,
        "peak_vram_gib": peak_gib,
        "cgroup_end_gib": cgroup_memory_gib(),
        "output": str(final_dir),
        "versions": {
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "trl": version("trl"),
            "peft": version("peft"),
        },
    }
    manifest_path = output_dir / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # 每步日志（ETA/平台判定数据源）
    log_path = output_dir / "training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as handle:
        for entry in trainer.state.log_history:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print("VLM_POLICY_SFT_OK")
    print(f"train_loss:  {train_result.training_loss:.4f}")
    print(f"trainable:   {trainable} ({manifest['trainable_ratio_percent']}%)")
    print(f"peak VRAM:   {peak_gib} GiB | cgroup end: {manifest['cgroup_end_gib']} GiB")
    print(f"manifest:    {manifest_path}")


if __name__ == "__main__":
    main()
