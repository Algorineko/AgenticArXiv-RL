"""DCU 上的 VLM 训练探针（S0 闸门工具）。

在选定卡上对一个本地 VLM 做「加载 → 真实 figureqa 图 forward → N 步 LoRA
backward → greedy 生成」的最小闭环，产出峰值显存 / 吞吐 / loss 的量化报告，
供 plan.md S0 阶段的 GO/NO-GO 判定使用。

运行环境（覆盖层方案，见 local_docs/plan.md）::

    export HCU_ROOT=$HOME/env/hcu_root
    source $HCU_ROOT[LOCAL_PATH_WITHHELD]   # set +u 先行
    export PYTHONHOME=$HCU_ROOT/usr
    export PYTHONPATH=$HOME/env/pip_overlay_tf514:$HCU_ROOT/usr/local/lib/python3.10/dist-packages
    export HIP_VISIBLE_DEVICES=2            # 选卡用 HIP，不用 CUDA

用法::

    python scripts/probe_vlm_dcu.py --model ~/data/models/Qwen3-VL-4B-Instruct --steps 8
    python scripts/probe_vlm_dcu.py --model ~/data/models/Qwen3.8-27B --device_map auto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "AgenticArxiv"))

from PIL import Image  # noqa: E402

from tools.figure_analysis_tool import VLM_MAX_IMAGE_SIDE  # noqa: E402


def cgroup_used_gib() -> float | None:
    import os

    path = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if not os.access(path, os.R_OK):
        path = Path("/sys/fs/cgroup/memory.current")
    try:
        return round(int(path.read_text()) / 2**30, 2)
    except OSError:
        return None


def load_probe_rows(data_path: Path, image_root: Path, n: int):
    rows = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                continue
            relpath = row["image_relpath"]
            path = (image_root / relpath).resolve()
            if not path.exists():
                continue
            image = Image.open(path).convert("RGB")
            longest = max(image.size)
            if longest > VLM_MAX_IMAGE_SIDE:
                scale = VLM_MAX_IMAGE_SIDE / longest
                image = image.resize(
                    (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                    Image.LANCZOS,
                )
            rows.append({"messages": [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": row["prompt"]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": row["target"]},
                ]},
            ], "image": image})
            if len(rows) >= n:
                break
    if not rows:
        raise SystemExit(f"没有可用的 train 样本（图片缺失?）: {data_path}")
    return rows


def build_batch(processor, rows):
    """构造 teacher-forced 训练 batch：prompt 段 label 置 -1，只在 target 上计 loss。"""
    prompts = [
        processor.apply_chat_template(
            [r["messages"][0]], add_generation_prompt=True, tokenize=False
        )
        for r in rows
    ]
    fulls = [processor.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    images = [[r["image"]] for r in rows]
    prompt_enc = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    full_enc = processor(text=fulls, images=images, return_tensors="pt", padding=True)
    labels = full_enc["input_ids"].clone()
    labels[full_enc["attention_mask"] == 0] = -100
    for i, p_ids in enumerate(prompt_enc["input_ids"]):
        p_len = (prompt_enc["attention_mask"][i] == 1).sum()
        labels[i, :p_len] = -100
    return {**{k: v for k, v in full_enc.items() if k != "labels"}, "labels": labels}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="VLM DCU 训练探针")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/vlm/figureqa_seed.jsonl")
    parser.add_argument("--image-root", default="AgenticArxiv")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--device_map", choices=["single", "auto"], default="single")
    parser.add_argument("--max_memory", default="56GiB,56GiB", help="device_map=auto 时每卡上限")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gen_tokens", type=int, default=32)
    parser.add_argument("--out", default="", help="可选：报告 JSON 输出路径")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    report: dict = {"model": args.model, "device_map": args.device_map}
    report["versions"] = {
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "trl": version("trl"),
        "peft": version("peft"),
    }
    try:
        import fla  # noqa: F401

        report["fla"] = fla.__version__
    except ImportError:
        report["fla"] = None
    report["cgroup_before_gib"] = cgroup_used_gib()
    print(f"📦 versions: {report['versions']} | fla: {report['fla']}")
    print(f"🧱 cgroup host: {report['cgroup_before_gib']} GiB")

    model_dir = Path(args.model).expanduser()
    load_kwargs = dict(dtype=torch.bfloat16, local_files_only=model_dir.exists())
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=model_dir.exists())
    t0 = time.time()
    if args.device_map == "auto":
        caps = {i: cap for i, cap in enumerate(args.max_memory.split(","))}
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir), device_map="auto", max_memory=caps,
            **load_kwargs,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(str(model_dir), **load_kwargs).to("cuda")
    report["load_seconds"] = round(time.time() - t0, 1)
    model.config.use_cache = False
    attn = getattr(model.config, "_attn_implementation", None)
    report["attn_implementation"] = attn
    print(f"⚡ loaded in {report['load_seconds']}s | attn: {attn}")
    # 装载读入的 50GB page cache 计入宿主 cgroup 配额且对已上卡权重无用，读完立逐
    try:
        from rl.save_guard import evict_model_dir_page_cache

        evicted = evict_model_dir_page_cache(model_dir)
        report["page_cache_evicted_files"] = evicted
        print(f"🧹 evicted page cache for {evicted} model files")
    except Exception as exc:  # noqa: BLE001 - 逐出失败只降级不致命
        print(f"⚠️ page cache eviction skipped: {exc}")

    rows = load_probe_rows(Path(args.data), Path(args.image_root).resolve(), args.samples)
    # 大词表（~250k）模型的 logits+CE 在 fp32 上会膨胀数 GB，探针一律单样本 forward
    batch = build_batch(processor, rows[:1])
    batch = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in batch.items()}

    t0 = time.time()
    with torch.no_grad():
        out = model(**batch)
    report["forward_seconds"] = round(time.time() - t0, 2)
    report["forward_loss"] = round(float(out.loss), 4)
    print(f"🔤 forward loss: {report['forward_loss']} ({report['forward_seconds']}s)")

    from peft import LoraConfig, get_peft_model

    target_names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                      target_modules=list(target_names), task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    report["trainable"] = trainable
    report["trainable_ratio_percent"] = round(100.0 * trainable / total, 4)
    model.print_trainable_parameters()

    # 27B 级模型无梯度检查点时，训练步的全量激活会顶爆权重所在卡
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, foreach=False
    )
    t0 = time.time()
    losses = []
    for step in range(args.steps):
        b = build_batch(processor, [rows[step % 2]])
        b = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in b.items()}
        loss = model(**b).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(round(float(loss), 4))
    dt = time.time() - t0
    report["train_steps"] = args.steps
    report["train_seconds"] = round(dt, 2)
    report["train_samples_per_sec"] = round(args.steps / dt, 3)
    report["train_losses"] = losses
    report["peak_vram_gib_per_device"] = [
        round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in range(torch.cuda.device_count())
    ]
    report["peak_vram_reserved_gib_per_device"] = [
        round(torch.cuda.max_memory_reserved(i) / 2**30, 1) for i in range(torch.cuda.device_count())
    ]
    print(f"🏋️ {args.steps} steps in {dt:.1f}s ({report['train_samples_per_sec']} samples/s)")
    print(f"   losses: {losses}")
    print(f"   VRAM alloc/reserved per device: {report['peak_vram_gib_per_device']} / {report['peak_vram_reserved_gib_per_device']}")

    model.eval()
    model.config.use_cache = True
    gen_rows = [rows[0]["messages"][0]]
    prompt = processor.apply_chat_template(gen_rows, add_generation_prompt=True, tokenize=False)
    enc = processor(text=[prompt], images=[[rows[0]["image"]]], return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=args.gen_tokens, do_sample=False)
    gen_dt = time.time() - t0
    text = processor.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    report["gen_seconds"] = round(gen_dt, 2)
    report["gen_tokens_per_sec"] = round(args.gen_tokens / gen_dt, 2)
    report["gen_sample"] = text[0][:200]
    print(f"💬 greedy gen: {report['gen_tokens_per_sec']} tok/s | head: {report['gen_sample']!r}")
    report["cgroup_after_gib"] = cgroup_used_gib()
    print(f"🧱 cgroup host after: {report['cgroup_after_gib']} GiB")

    ok = (
        report["forward_loss"] == report["forward_loss"]  # not NaN
        and len(losses) == args.steps
        and all(loss == loss for loss in losses)
    )
    report["ok"] = bool(ok)
    print(("✅ PROBE_OK" if ok else "❌ PROBE_FAILED"))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"📄 report: {out_path}")


if __name__ == "__main__":
    main()
