#!/usr/bin/env python3
"""总结偏好对生成（S4 DPO 数据）：SFT 模型自采样 k 条，规则评分定 chosen/rejected。

与 ``build_summary_sft_data.py``（SFT 自蒸馏）同一套过滤维度，但这里要的不是
「合格样本」而是「同 prompt 下好与坏的对比」：

- 每篇论文 × 风格 × 预算采样 k 条（temp 0.7，种子互异）；
- 软评分（越大越好）：词预算达标度 + 风格形状分 + 接地率 − 抄袭率；
- chosen = 合格样本中最高分；rejected = 最低分；两者同分或 chosen 不合格则丢弃；
- 支持分片（``--shard i --num-shards n``）双卡并行，输出按分片落盘后合并。

用法::

    HIP_VISIBLE_DEVICES=0 python scripts/generate_summary_dpo_data.py \
        --model <base> --adapter <sft adapter> --shard 0 --num-shards 2
    HIP_VISIBLE_DEVICES=1 python scripts/generate_summary_dpo_data.py \
        --model <base> --adapter <sft adapter> --shard 1 --num-shards 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_summary_sft_data import (  # noqa: E402
    BUDGETS,
    STYLES,
    STYLE_INSTRUCTIONS,
    content_words,
    extract_abstract,
    ngrams,
)
from build_figure_qa_v2_dataset import clean_text  # noqa: E402

_COMPLETENESS = 0.9  # 词数达到预算的 90% 才算吃满预算（太短扣分）
_ABBREV_SAFE_RE = re.compile(r"\b(e\.g|i\.e|etc|vs|cf|Fig|Figs|Eq|Eqs|Tab|Sec|al)\.", re.IGNORECASE)


def style_shape_score(target: str, style: str) -> float:
    if style == "tldr":
        guarded = _ABBREV_SAFE_RE.sub(r"\1<DOT>", target.strip())
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", guarded) if s.strip()]
        return 1.0 if len(sentences) == 1 else (0.4 if len(sentences) == 2 else 0.0)
    if style == "bullet":
        n = sum(1 for line in target.splitlines() if line.strip().startswith("-"))
        return min(n, 5) / 5
    headers = target.count("##")
    return min(headers, 4) / 4


def soft_score(target: str, source: str, abstract: str, style: str, budget: int,
               max_gram_overlap: float = 0.6) -> float:
    """连续评分：预算利用 + 风格形状 + 接地 − 抄袭惩罚。"""
    words = target.split()
    if len(words) < 5:
        return -1.0
    budget_use = min(len(words), budget * _COMPLETENESS) / (budget * _COMPLETENESS)
    shape = style_shape_score(target, style)
    src_grams = ngrams(source)
    grams = ngrams(target)
    copy_pen = (len(grams & src_grams) / len(grams)) if grams else 0.0
    abstract_words = content_words(abstract)
    tw = {w for w in re.findall(r"[a-z]{3,}", target.lower()) if w not in content_words("")}
    ground = (len(tw & abstract_words) / len(tw)) if tw else 0.0
    return round(0.35 * budget_use + 0.25 * shape + 0.25 * ground - 0.35 * copy_pen, 4)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="总结偏好对生成（SFT 模型自采样）")
    parser.add_argument("--model", required=True, help="基座模型目录")
    parser.add_argument("--adapter", required=True, help="SFT LoRA adapter 目录")
    parser.add_argument("--snapshot", default="data/mock_arxiv_snapshot.json")
    parser.add_argument("--seed-data", default="data/vlm/figureqa_seed.jsonl")
    parser.add_argument("--output", default="data/dpo/summary_dpo.jsonl")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-papers", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    records: Dict[str, Dict[str, Any]] = {}
    for record in snapshot.get("get_paper_content", {}).values():
        r = record.get("result") if isinstance(record, dict) else None
        if isinstance(r, dict) and not r.get("section") and r.get("content"):
            records.setdefault(r["paper_id"], r)
    paper_ids = sorted(records)
    if args.num_shards > 1:
        paper_ids = paper_ids[args.shard::args.num_shards]
    if args.max_papers:
        paper_ids = paper_ids[: args.max_papers]
    print(f"📄 shard {args.shard}/{args.num_shards}: {len(paper_ids)} papers", flush=True)

    val_papers = set()
    with Path(args.seed_data).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and json.loads(line).get("split") == "val":
                val_papers.add(json.loads(line)["paper_id"])

    model_dir = Path(args.model).expanduser()
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=model_dir.exists())
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, local_files_only=model_dir.exists()
    ).to("cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.config.use_cache = True
    try:
        from rl.save_guard import evict_model_dir_page_cache

        evict_model_dir_page_cache(model_dir)
    except Exception:  # noqa: BLE001
        pass

    def sample(prompt: str, budget: int, seed: int) -> str:
        try:
            text = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False,
            )
        except TypeError:
            text = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                add_generation_prompt=True, tokenize=False,
            )
        enc = processor(text=[text], return_tensors="pt").to("cuda")
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=int(budget * 2.2) + 24,
                                 do_sample=args.temperature > 0,
                                 temperature=max(args.temperature, 1e-4), top_p=0.9)
        decoded = processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        if "</think>" in decoded:
            decoded = decoded.split("</think>", 1)[1].strip()
        return decoded

    rows: List[Dict[str, Any]] = []
    kept = pairs = 0
    for index, paper_id in enumerate(paper_ids, 1):
        record = records[paper_id]
        source = record["content"][:6000]
        abstract = extract_abstract(record["content"])
        split = "val" if paper_id in val_papers else "train"
        for style in STYLES:
            for budget in BUDGETS:
                prompt = (
                    "You are building a paper-reading assistant. Read the arXiv paper "
                    f"content below and write a {style} summary in at most {budget} words.\n\n"
                    f"{STYLE_INSTRUCTIONS[style]}\n\n"
                    f"Paper content:\n{source}\n\nWrite only the summary."
                )
                scored: List[Tuple[float, str]] = []
                for k in range(args.k):
                    target = sample(prompt, budget, seed=1000 + k)
                    scored.append((soft_score(target, source, abstract, style, budget), target))
                scored.sort(key=lambda t: t[0], reverse=True)
                best_score, best = scored[0]
                worst_score, worst = scored[-1]
                if best == worst or best_score <= worst_score or not best:
                    continue
                base_prompt = (
                    "You are building a paper-reading assistant. Read the arXiv paper "
                    f"content below and write a {style} summary in at most {budget} words.\n\n"
                    f"{STYLE_INSTRUCTIONS[style]}\n\nPaper content:\n<<PAPER>>\n\nWrite only the summary."
                )
                rows.append({
                    "paper_id": paper_id,
                    "style": style,
                    "budget": budget,
                    "prompt": prompt,
                    "chosen": best,
                    "rejected": worst,
                    "score_chosen": best_score,
                    "score_rejected": worst_score,
                    "split": split,
                    "source": "summary_dpo",
                })
                pairs += 1
                if best_score > 0:
                    kept += 1
        if index % 5 == 0 or index == len(paper_ids):
            print(f"  [{index}/{len(paper_ids)}] pairs {pairs} (chosen 合格 {kept})", flush=True)
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with (output_path.parent / f"{output_path.stem}.shard{args.shard}.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (output_path.parent / f"{output_path.stem}.shard{args.shard}.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("SUMMARY_DPO_SHARD_OK")
    print(f"pairs: {pairs} | chosen 合格率: {round(kept/max(pairs,1),3)}")
    print(f"output: {output_path.parent / f'{output_path.stem}.shard{args.shard}.jsonl'}")


if __name__ == "__main__":
    main()
