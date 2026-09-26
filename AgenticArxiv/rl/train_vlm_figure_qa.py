"""FigureQA 后训练：在本地 VLM 上做 (图, 问题, caption 证据) 的监督微调。

这是 T5 ``analyze_figure`` 的 env 侧 VLM（README「🧰 工具集演进设计」）的
训练脚本；策略模型仍是纯文本小模型，本脚本训练的模型只作为快照录制
后端（``FIGURE_ANALYSIS_BACKEND=vlm``），不进入策略梯度。

数据来自 ``scripts/build_figure_qa_dataset.py``（caption 监督、按论文切分）。
训练走 TRL 的视觉语言路径：数据集带 ``messages`` + ``images`` 列，
``processing_class`` 传 processor，collator 在 batch 内完成图像处理。
默认 LoRA（r=32，仅语言层），``--no-lora`` 可切全参。

用法::

    python -m AgenticArxiv.rl.train_vlm_figure_qa \
        --model <本地 VLM 目录> --data data/vlm/figureqa_seed.jsonl \
        --output_dir outputs/vlm_figureqa --epochs 3
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

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from PIL import Image  # noqa: E402

from tools.figure_analysis_tool import VLM_MAX_IMAGE_SIDE  # noqa: E402

LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


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


def load_image(row: Mapping[str, Any], *, image_root: Path) -> Image.Image:
    """Load and downscale exactly like the snapshot recorder does."""
    relpath = row.get("image_relpath")
    if not relpath:
        raise ValueError("row is missing image_relpath")
    path = (image_root / relpath).resolve()
    image = Image.open(path).convert("RGB")
    longest = max(image.size)
    if longest > VLM_MAX_IMAGE_SIDE:
        scale = VLM_MAX_IMAGE_SIDE / longest
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.LANCZOS,
        )
    return image


def build_hf_dataset(rows: List[Dict[str, Any]], *, image_root: Path):
    from datasets import Dataset

    records = []
    for row in rows:
        records.append(
            {
                "images": [load_image(row, image_root=image_root)],
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["target"]},
                ],
            }
        )
    return Dataset.from_list(records)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FigureQA 后训练（T5 env 侧 VLM）")
    parser.add_argument("--model", required=True, help="本地 VLM 目录（如 Qwen3-VL-4B-Instruct）")
    parser.add_argument("--data", default="data/vlm/figureqa_seed.jsonl")
    parser.add_argument("--image-root", default="AgenticArxiv")
    parser.add_argument("--output_dir", default="outputs/vlm_figureqa")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--no-lora", action="store_true", help="全参微调（显存显著更高）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=0, help="调试：只取前 N 条训练样本")
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--run_name", default="vlm_figureqa")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_strategy", default="epoch")
    parser.add_argument("--no-merge", action="store_true", help="只保存 LoRA adapter，不合并全量权重")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import SFTConfig, SFTTrainer

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"数据集不存在: {data_path}（先运行 scripts/build_figure_qa_dataset.py）")
    image_root = Path(args.image_root).resolve()
    rows = load_rows(data_path)
    train_rows = [row for row in rows if row.get("split") == "train"]
    if not train_rows:
        raise SystemExit("数据集没有 train 划分")

    random.Random(args.seed).shuffle(train_rows)
    if args.num_samples:
        train_rows = train_rows[: args.num_samples]

    from collections import Counter

    counts = Counter(row["question"] for row in train_rows)
    print(f"🧾 训练样本: {len(train_rows)} | 每类问题: {dict(sorted(counts.items()))}")

    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=Path(args.model).exists()
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=Path(args.model).exists(),
    )
    model.config.use_cache = False

    if not args.no_lora:
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(LORA_TARGET_MODULES),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    else:
        try:
            model.enable_input_require_grads()
        except Exception:  # pragma: no cover - 取决于模型实现
            pass

    dataset = build_hf_dataset(train_rows, image_root=image_root)
    print(f"🖼️ 数据集列: {dataset.column_names} | 行数: {len(dataset)}")

    output_dir = Path(args.output_dir)
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
        save_total_limit=2,
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
    from rl.precision import pin_single_gpu, precision_flags

    config_kwargs.update(precision_flags())
    config = SFTConfig(**config_kwargs)
    # 视觉塔在 Trainer 自动启用的 DataParallel 空 replica 上会崩
    # （Qwen3-VL 位置编码 shape 失配），按仓库惯例钉单卡。
    pin_single_gpu(config)

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=processor,
    )
    train_result = trainer.train()

    # 在合并前统计可训练参数，合并后 adapter 会被卸下。
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    merged = False
    if not args.no_lora and not args.no_merge:
        merged_model = trainer.model.merge_and_unload()
        merged_model.config.use_cache = True
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
        processor.save_pretrained(str(merged_dir))
        merged = True
        print(f"🔗 已合并 LoRA 权重: {merged_dir}")
    peak_gib = None
    if torch.cuda.is_available():
        peak_gib = round(torch.cuda.max_memory_allocated() / 2**30, 1)

    manifest = {
        "stage": "vlm_figure_qa_lora" if not args.no_lora else "vlm_figure_qa_full",
        "base_model": args.model,
        "data": data_path.as_posix(),
        "data_sha256": _sha256_file(data_path),
        "train_rows": len(train_rows),
        "question_counts": dict(sorted(counts.items())),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "lora": None if args.no_lora else {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": list(LORA_TARGET_MODULES),
        },
        "merged": merged,
        "seed": args.seed,
        "train_loss": train_result.training_loss,
        "trainable_parameters": trainable,
        "trainable_ratio_percent": round(100.0 * trainable / total, 4) if total else None,
        "peak_vram_gib": peak_gib,
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
    print("VLM_FIGURE_QA_TRAIN_OK")
    print(f"train_loss:  {train_result.training_loss:.4f}")
    print(f"trainable:   {trainable} ({manifest['trainable_ratio_percent']}%)")
    print(f"peak VRAM:   {peak_gib} GiB")
    print(f"manifest:    {manifest_path}")


if __name__ == "__main__":
    main()
