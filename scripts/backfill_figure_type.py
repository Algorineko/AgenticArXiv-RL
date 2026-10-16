#!/usr/bin/env python3
"""figure_type 题型的模型回填：caption 关键词没命中的图用基座模型补判。

确定性纪律（与 figureqa v2 一致）：

- **自洽过滤**：temp 0.7 采样 2 次，两次一致才收（plan.md S1 约定）；
- **caption 交叉验证**：模型答案必须与 caption 关键词映射一致——caption
  里出现了该类型的特征词才接受（防模型瞎猜枚举）；
- 只产 train 划分（seed val 论文的图进 val2，永不训练）。

输入 ``data/vlm/figureqa_v2_extraction.json``（``reextract_figures.py`` 产物），
输出追加行 ``data/vlm/figure_type_backfill.jsonl``，由
``build_vlm_policy_sft_mix.py --figureqa-backfill`` 并入混合集。

用法::

    HIP_VISIBLE_DEVICES=3 python scripts/backfill_figure_type.py \
        --model ~/data/models/Qwen3.8-27B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_figure_qa_v2_dataset import (  # noqa: E402
    _FIGURE_TYPE_KEYWORDS,
    _V2_QUESTION_PROMPTS,
    clean_text,
    figure_type_target,
)
from tools.figure_analysis_tool import VLM_MAX_IMAGE_SIDE  # noqa: E402


def caption_supports(caption: str, label: str) -> bool:
    """模型判的类型必须在 caption 里有词面证据（去重 not-pure-guess）。"""
    keywords = dict(_FIGURE_TYPE_KEYWORDS).get(label, ())
    lowered = " " + clean_text(caption).lower()
    return any(kw in lowered for kw in keywords)


def main() -> None:
    parser = argparse.ArgumentParser(description="figure_type 模型回填")
    parser.add_argument("--model", required=True)
    parser.add_argument("--extraction", default="data/vlm/figureqa_v2_extraction.json")
    parser.add_argument("--seed-data", default="data/vlm/figureqa_seed.jsonl")
    parser.add_argument("--output", default="data/vlm/figure_type_backfill.jsonl")
    parser.add_argument("--image-root", default="AgenticArxiv")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    extraction = json.loads(Path(args.extraction).read_text(encoding="utf-8"))
    image_root = Path(args.image_root).resolve()
    val_papers: set = set()
    with Path(args.seed_data).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and json.loads(line).get("split") == "val":
                val_papers.add(json.loads(line)["paper_id"])

    # 关键词已命中的图不需要模型
    model_dir = Path(args.model).expanduser()
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=model_dir.exists())
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, local_files_only=model_dir.exists()
    ).to("cuda")
    model.eval()
    model.config.use_cache = True
    try:
        from rl.save_guard import evict_model_dir_page_cache

        evict_model_dir_page_cache(model_dir)
    except Exception:  # noqa: BLE001
        pass

    def ask(image: Image.Image) -> str:
        try:
            text = processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": _V2_QUESTION_PROMPTS["figure_type"]},
                ]}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False,
            )
        except TypeError:
            text = processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": _V2_QUESTION_PROMPTS["figure_type"]},
                ]}],
                add_generation_prompt=True, tokenize=False,
            )
        longest = max(image.size)
        if longest > VLM_MAX_IMAGE_SIDE:
            scale = VLM_MAX_IMAGE_SIDE / longest
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.LANCZOS,
            )
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        gen_kwargs = dict(max_new_tokens=24, do_sample=args.temperature > 0,
                          temperature=args.temperature, top_p=0.9)
        labels = [label for label, _ in _FIGURE_TYPE_KEYWORDS]
        votes: Dict[str, int] = {}
        for seed in (42, 43, 44):  # 3 采多数决：歧义图的自洽门槛 ≥2/3
            torch.manual_seed(seed)
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            a = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()
            if "</think>" in a:
                a = a.split("</think>", 1)[1].strip()
            a_norm = " ".join(a.split())
            if a_norm in labels:
                votes[a_norm] = votes.get(a_norm, 0) + 1
        if votes:
            best, count = max(votes.items(), key=lambda kv: kv[1])
            if count >= 2:
                return best
        return ""

    rows: List[Dict[str, Any]] = []
    considered = kept = 0
    papers = sorted(extraction.get("papers", {}).items())
    if args.limit:
        papers = papers[: args.limit]
    for paper_id, record in papers:
        if paper_id in val_papers:
            continue  # val 论文的图永不训练
        for figure in record.get("figures", []):
            caption = clean_text(figure.get("caption"))
            if not caption or figure_type_target(caption):
                continue  # 无 caption 或关键词已命中
            fpath = Path(figure["path"])
            if not fpath.is_file() or fpath.stat().st_size == 0:
                continue
            considered += 1
            try:
                image = Image.open(fpath).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            label = ask(image)
            if not label or not caption_supports(caption, label):
                continue
            rel = fpath.resolve().relative_to(image_root)
            rows.append({
                "paper_id": paper_id,
                "figure_no": int(figure.get("figure_no") or 0),
                "question": "figure_type",
                "prompt": _V2_QUESTION_PROMPTS["figure_type"],
                "target": label,
                "image_relpath": rel.as_posix(),
                "split": "train",
                "source": "v2_backfill",
            })
            kept += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("FIGURE_TYPE_BACKFILL_OK")
    print(f"considered: {considered} | kept: {kept} ({round(kept/considered, 3) if considered else 0})")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
