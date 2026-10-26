"""RLCD next-tool 判别头训练器（手写循环，单卡）。

RLCD（arXiv:2307.12950 精神）在判别式选择上的落地：
``L = L_CE(z/τ, gold) + λ·Σ_neg[-log σ(β·(z[gold] − z[neg] − margin))]``。
正样本来自 RLVR 确定性标签（expected_tools 重算），替代论文中的
preference model + PPO；负样本是失败轨迹分歧步的真实错误工具。

双臂对照：``--lambda 0`` 为臂 A（纯 CE 的 router SFT 基线）。两臂共用
同一数据行流与启动时由 seed 显式生成的 permutation（sha256 写入两侧
manifest），除对比项外逐位一致。

DCU 纪律：sdpa attention（禁 flash_attn2）、AdamW ``foreach=False``
（vendor das 构建的 foreach 内核坑）、装载后逐模型文件 page cache
（宿主 cgroup 64GiB 墙）、保存走 :func:`rl.save_guard.guarded_save`。

用法（单卡，选卡用 HIP）::

    HIP_VISIBLE_DEVICES=3 python -m AgenticArxiv.rl.train_jev_rlcd \
        --model outputs/sft/final --data data/rlcd/rlcd_v1.jsonl \
        --output_dir outputs/jev_rlcd_armA --lambda 0
    HIP_VISIBLE_DEVICES=3 python -m AgenticArxiv.rl.train_jev_rlcd ... \
        --output_dir outputs/jev_rlcd_armB --lambda 0.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))

from routing.local_prompt import (  # noqa: E402
    CANDIDATE_SUFFIX_PREFIX,
    ROUTER_PROMPT_VERSION,
    build_router_prompt,
    candidate_suffix_texts,
)
from rl import jev_scoring  # noqa: E402
from rl.save_guard import evict_model_dir_page_cache, guarded_save  # noqa: E402

DEFAULT_MODEL = "outputs/sft/final"
DEFAULT_DATA = "data/rlcd/rlcd_v1.jsonl"
TRAIN_SPLITS = ("train",)
EVAL_SPLITS = ("dev", "iid_test", "ood_test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()


def load_rows(data_path: Path) -> Dict[str, List[dict]]:
    rows = {"train": [], "eval": []}
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        bucket = "train" if row.get("split") in TRAIN_SPLITS else "eval"
        if bucket == "eval" and row.get("split") not in EVAL_SPLITS:
            continue
        rows[bucket].append(row)
    if not rows["train"]:
        raise SystemExit(f"数据集没有 train 行: {data_path}")
    return rows


def load_registry_tools() -> List[Dict[str, Any]]:
    from tools.bootstrap import require_all_tools

    require_all_tools("Jev RLCD trainer")
    from tools.tool_registry import registry

    return registry.list_tools()


def expected_index(names: List[str], tool: str) -> int:
    try:
        return names.index(tool)
    except ValueError:
        raise SystemExit(f"gold 工具 {tool!r} 不在候选集 {names} 中")


class RouterDataset:
    """预 tokenize 的决策点集合（prompt 由 local_prompt 现拼，不入库）。"""

    def __init__(
        self,
        rows: List[dict],
        *,
        tokenizer: Any,
        tools: List[Dict[str, Any]],
        max_state_chars: int,
        desc: str,
    ) -> None:
        self.rows = rows
        self.names = [
            str(tool.get("name") or "") for tool in tools
        ] + ["FINISH"]
        suffix_texts = candidate_suffix_texts(self.names)
        self.suffix_ids = jev_scoring.tokenize_suffixes(tokenizer, suffix_texts)
        self.name_starts = jev_scoring.name_token_starts(
            tokenizer, suffix_texts, CANDIDATE_SUFFIX_PREFIX
        )
        self.tokenizer = tokenizer
        self.tools = tools
        self.max_state_chars = max_state_chars
        self.desc = desc
        self.prompt_ids: List[List[int]] = []
        self.gold_idx: List[int] = []
        self.neg_idx: List[List[int]] = []
        skipped = 0
        for row in rows:
            prompt = build_router_prompt(
                task=str(row.get("task_text") or ""),
                state_text=str(row.get("state_text") or ""),
                tools=tools,
                max_state_chars=max_state_chars,
            )
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
            )
            if isinstance(encoded, (list, tuple)):
                ids = list(encoded)
            else:  # tf5 返回 BatchEncoding（UserDict，不是 dict 子类）
                ids = list(encoded["input_ids"])
            if not ids or len(ids) > 8192:
                skipped += 1
                continue
            self.prompt_ids.append(list(ids))
            self.gold_idx.append(expected_index(self.names, str(row["gold_tool"])))
            negatives = []
            for tool in row.get("negative_tools") or []:
                if tool in self.names and tool != row["gold_tool"]:
                    negatives.append(self.names.index(tool))
            self.neg_idx.append(negatives[:2])
        if skipped:
            print(f"⚠️ {desc}: 跳过 {skipped} 条（prompt 超 8192 token）")

    def __len__(self) -> int:
        return len(self.prompt_ids)

    def sample(self, index: int, device: Any) -> Dict[str, Any]:
        import torch

        prompt_ids = torch.tensor(
            self.prompt_ids[index], dtype=torch.long, device=device
        )
        suffix_ids, suffix_lens = jev_scoring.build_suffix_batch(
            self.suffix_ids, device
        )
        return {
            "prompt_ids": prompt_ids,
            "suffix_ids": suffix_ids,
            "suffix_lens": suffix_lens,
            "name_starts": self.name_starts,
            "gold": self.gold_idx[index],
            "negatives": self.neg_idx[index],
        }


def evaluate(
    model: Any,
    dataset: RouterDataset,
    indices: List[int],
    device: Any,
    *,
    tau: float,
    scoring: str,
    agg: str,
) -> Dict[str, Any]:
    """冻结评测点上的 micro-accuracy / ECE / 逐类混淆。"""
    import torch

    model.eval()
    correct = 0
    confidence_all: List[float] = []
    accuracy_all: List[int] = []
    confusion: Dict[str, Dict[str, int]] = {}
    for index in indices:
        batch = dataset.sample(index, device)
        with torch.no_grad():
            z = jev_scoring.score_candidates(
                model,
                batch["prompt_ids"],
                batch["suffix_ids"],
                batch["suffix_lens"],
                agg=agg,
                scoring=scoring,
                name_starts=batch["name_starts"],
            )
        selected, confidence, _ = jev_scoring.decision_from_z(
            z, dataset.names, tau
        )
        gold_name = dataset.names[batch["gold"]]
        hit = int(selected == gold_name)
        correct += hit
        accuracy_all.append(hit)
        confidence_all.append(confidence)
        confusion.setdefault(gold_name, {}).setdefault(selected, 0)
        confusion[gold_name][selected] += 1
    model.train()
    n = max(1, len(indices))
    # 10-bin ECE：各箱 |平均置信度 − 平均准确率| 的样本量加权
    bins: List[List[Tuple[float, int]]] = [[] for _ in range(10)]
    for conf, acc in zip(confidence_all, accuracy_all):
        bins[min(9, int(conf * 10))].append((conf, acc))
    ece = 0.0
    for bucket in bins:
        if bucket:
            mean_conf = sum(conf for conf, _ in bucket) / len(bucket)
            mean_acc = sum(acc for _, acc in bucket) / len(bucket)
            ece += len(bucket) / n * abs(mean_conf - mean_acc)
    return {
        "n": len(indices),
        "accuracy": correct / n,
        "mean_confidence": sum(confidence_all) / n,
        "ece": ece,
        "confusion": confusion,
    }


def cosine_lr(step: int, total: int, base_lr: float, warmup: int, floor_ratio: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def main(argv: Optional[List[str]] = None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rl.precision import trainable_model_load_kwargs

    parser = argparse.ArgumentParser(description="RLCD next-tool 判别头训练")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lambda", dest="lambda_rlcd", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--agg", choices=["mean", "sum"], default="mean")
    parser.add_argument("--scoring", choices=["cache", "batch"], default="cache")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--lr-floor-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=1000, help="每 N 个样本评一次")
    parser.add_argument("--eval-patience", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0, help=">0 时冒烟跑固定步")
    parser.add_argument("--max-state-chars", type=int, default=12000)
    parser.add_argument("--gold-repeat", type=int, default=2)
    parser.add_argument("--save-lock-settle", type=int, default=120)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data) if Path(args.data).exists() else REPO_ROOT / args.data
    model_path = Path(args.model)
    resolved_model = str(model_path.resolve()) if model_path.exists() else args.model
    local_only = model_path.exists()

    tools = load_registry_tools()
    tokenizer = AutoTokenizer.from_pretrained(resolved_model, local_files_only=local_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    buckets = load_rows(data_path)
    train_rows = list(buckets["train"])
    gold_rows = [r for r in train_rows if r.get("source") == "gold_replay"]
    train_rows = train_rows + gold_rows * max(0, args.gold_repeat - 1)
    train_set = RouterDataset(
        train_rows, tokenizer=tokenizer, tools=tools,
        max_state_chars=args.max_state_chars, desc="train",
    )
    eval_set = RouterDataset(
        buckets["eval"], tokenizer=tokenizer, tools=tools,
        max_state_chars=args.max_state_chars, desc="eval",
    )
    eval_indices = list(range(len(eval_set)))
    print(
        f"📦 train rows: {len(train_set)} (含 gold ×{args.gold_repeat}) | "
        f"eval points: {len(eval_set)} | 候选 {len(train_set.names)} 类"
    )
    if args.inspect_only:
        print("INSPECT_ONLY_OK")
        return

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        local_files_only=local_only,
        attn_implementation="sdpa",
        **trainable_model_load_kwargs(),
    )
    model.config.use_cache = False
    model.to("cuda")
    try:
        evicted = evict_model_dir_page_cache(model_path if local_only else resolved_model)
        print(f"🧹 evicted page cache for {evicted} model files")
    except Exception as exc:  # noqa: BLE001 - 逐出失败只降级
        print(f"⚠️ page cache eviction skipped: {exc}")
    model.train()

    rng = random.Random(args.seed)
    permutation = list(range(len(train_set)))
    rng.shuffle(permutation)
    permutation_sha = hashlib.sha256(
        json.dumps(permutation).encode()
    ).hexdigest()[:16]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        foreach=False,
    )
    micro_total = len(permutation) * args.epochs
    if args.max_steps:
        micro_total = min(micro_total, args.max_steps * args.grad_accum)
    optim_total = max(1, micro_total // args.grad_accum)

    log_path = output_dir / "train_log.jsonl"
    log_handle = log_path.open("w", encoding="utf-8")

    def log_event(event: Dict[str, Any]) -> None:
        log_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        log_handle.flush()

    best = {"score": -1.0, "state": None, "eval": None}
    no_improve = 0
    micro = 0
    loss_ce_avg = loss_c_avg = grad_avg = 0.0
    eval_history: List[Dict[str, Any]] = []
    started = time.time()
    device = torch.device("cuda")

    def run_eval(tag: str) -> Dict[str, Any]:
        nonlocal best, no_improve
        result = evaluate(
            model, eval_set, eval_indices, device,
            tau=args.tau, scoring=args.scoring, agg=args.agg,
        )
        result["tag"] = tag
        result["micro"] = micro
        eval_history.append(result)
        log_event({"type": "eval", **{k: v for k, v in result.items() if k != "confusion"},
                   "confusion_top": {k: dict(sorted(v.items(), key=lambda x: -x[1])[:3])
                                     for k, v in result["confusion"].items()}})
        score = result["accuracy"] - 0.3 * result["ece"]
        if score > best["score"] + 1e-6:
            best["score"] = score
            best["eval"] = result
            best["state"] = {
                k: v.detach().to("cpu", copy=True)
                for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
        print(
            f"📊 eval[{tag}] acc={result['accuracy']:.4f} "
            f"conf={result['mean_confidence']:.4f} ece={result['ece']:.4f} "
            f"(best={best['score']:.4f}, patience {no_improve}/{args.eval_patience})"
        )
        return result

    print(f"🚂 开始训练：micro={micro_total} optim={optim_total} λ={args.lambda_rlcd} "
          f"τ={args.tau} β={args.beta} scoring={args.scoring}")
    run_eval("step0")
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        for pos_in_epoch, index in enumerate(permutation):
            if args.max_steps and micro >= args.max_steps * args.grad_accum:
                done = True
                break
            batch = train_set.sample(index, device)
            z = jev_scoring.score_candidates(
                model,
                batch["prompt_ids"],
                batch["suffix_ids"],
                batch["suffix_lens"],
                agg=args.agg,
                scoring=args.scoring,
                name_starts=batch["name_starts"],
            )
            loss_ce = jev_scoring.router_ce_loss(z, args.tau, batch["gold"])
            loss_c = z.new_zeros(())
            if args.lambda_rlcd > 0 and batch["negatives"]:
                loss_c = jev_scoring.router_contrastive_loss(
                    z, args.beta, batch["gold"], batch["negatives"], args.margin
                )
            loss = loss_ce + args.lambda_rlcd * loss_c
            if not torch.isfinite(loss):
                raise SystemExit(
                    f"❌ loss 非有限值（micro={micro}）：CE={float(loss_ce)} "
                    f"contrastive={float(loss_c)}——立即终止，不静默重启"
                )
            (loss / args.grad_accum).backward()

            loss_ce_avg += float(loss_ce)
            loss_c_avg += float(loss_c)
            micro += 1
            if micro % args.grad_accum == 0:
                optim_step = micro // args.grad_accum
                lr = cosine_lr(
                    optim_step, optim_total, args.lr,
                    args.warmup_steps, args.lr_floor_ratio,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.clip,
                    )
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                grad_avg += grad_norm
                log_event({
                    "type": "train", "micro": micro, "optim": optim_step,
                    "loss_ce": round(loss_ce_avg / args.grad_accum, 6),
                    "loss_contrastive": round(loss_c_avg / args.grad_accum, 6),
                    "grad_norm": round(grad_avg / args.grad_accum, 4),
                    "lr": lr, "epoch": epoch,
                })
                loss_ce_avg = loss_c_avg = grad_avg = 0.0
                if optim_step % 20 == 0:
                    peak = max(
                        torch.cuda.max_memory_allocated(i) / 2**30
                        for i in range(torch.cuda.device_count())
                    )
                    elapsed = time.time() - started
                    print(
                        f"step {optim_step}/{optim_total} | "
                        f"micro {micro}/{micro_total} | {elapsed/60:.1f}min | "
                        f"peak_vram {peak:.1f}GiB"
                    )
            if micro % args.eval_every == 0:
                run_eval(f"micro{micro}")
                if no_improve >= args.eval_patience:
                    print(f"⏹ 连续 {args.eval_patience} 个 eval 点无改善，提前停")
                    done = True
                    break
    run_eval("final")

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"🏆 载入历史最优 checkpoint（score={best['score']:.4f}）")
    peak_vram = max(
        torch.cuda.max_memory_allocated(i) / 2**30
        for i in range(torch.cuda.device_count())
    )

    def _save():
        model.save_pretrained(output_dir / "final")
        tokenizer.save_pretrained(output_dir / "final")
        return list((output_dir / "final").rglob("*"))

    guarded_save(_save, lock_path=output_dir / "save.lock",
                 out_dir=output_dir / "final", settle_seconds=args.save_lock_settle)
    log_handle.close()

    manifest = {
        "stage": "jev_rlcd_router",
        "arm": "B_rlcd" if args.lambda_rlcd > 0 else "A_ce",
        "base_model": resolved_model,
        "data": str(data_path),
        "data_sha256_head": sha256_file(data_path),
        "prompt_version": ROUTER_PROMPT_VERSION,
        "lambda_rlcd": args.lambda_rlcd,
        "tau": args.tau,
        "beta": args.beta,
        "margin": args.margin,
        "agg": args.agg,
        "scoring": args.scoring,
        "lr": args.lr,
        "epochs": args.epochs,
        "grad_accum": args.grad_accum,
        "clip": args.clip,
        "seed": args.seed,
        "gold_repeat": args.gold_repeat,
        "train_rows": len(train_set),
        "eval_points": len(eval_set),
        "permutation_sha256_16": permutation_sha,
        "eval_history": [
            {k: v for k, v in e.items() if k != "confusion"} for e in eval_history
        ],
        "best_score": best["score"],
        "best_eval": {k: v for k, v in (best["eval"] or {}).items() if k != "confusion"},
        "peak_vram_gib": round(peak_vram, 2),
        "wall_clock_min": round((time.time() - started) / 60, 1),
        "candidates": train_set.names,
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"JEV_RLCD_TRAIN_OK arm={'B' if args.lambda_rlcd > 0 else 'A'} "
          f"best_acc={best['eval']['accuracy'] if best['eval'] else 'n/a'} "
          f"out={output_dir / 'final'}")


if __name__ == "__main__":
    main()
