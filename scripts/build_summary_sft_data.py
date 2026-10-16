#!/usr/bin/env python3
"""论文总结 SFT 数据：基座模型自蒸馏（教师=将做 SFT 的同一个基座）。

零外部 API 依赖（本机 .env 无通用 LLM key）。从离线快照取
``get_paper_content`` 的全文记录，对每篇论文 × 3 风格 × 3 词预算生成
abstractive 总结，再过硬规则过滤器（全部确定性、落 manifest）：

- 词预算达标：``len(target.split()) <= budget``
- 风格形状：tldr 单句 / bullet ≥3 点 / structured ≥3 个 ``##`` 节
- 抽象度：总结与源文的 4-gram 重叠率 < ``--max-gram-overlap``（防抄）
- 接地：总结内容词出现在 abstract 的比例 ≥ ``--min-grounding``（防幻觉）

风格与预算口径与 ``tools/paper_summary_tool.py`` 一致
（tldr/structured/bullet × 60/120/250）。seed val 论文只产 ``val2``
（评测用，永不训练）。

用法::

    python scripts/build_summary_sft_data.py \
        --model <本地基座目录> --output data/vlm/paper_summary_sft.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))

STYLES = ("tldr", "bullet", "structured")
BUDGETS = (60, 120, 250)

STYLE_INSTRUCTIONS = {
    "tldr": "The summary must be a single sentence capturing the paper's core contribution.",
    "bullet": "The summary must be 3-5 bullet points, each starting with '- '.",
    "structured": "The summary must have 3-4 short sections, each starting with a '## ' header.",
}

STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    this to we with our these those than then thus using used can may which""".split()
)

_CONTENT_CUTOFF = 6000
_ABSTRACT_RE = re.compile(r"\n\s*Abstract\s*\n(.*?)(?=\n\s*(?:1\s+Introduction|1\.?\s+INTRODUCTION|Index Terms|Keywords|CCS Concepts)\b|\Z)",
                          re.IGNORECASE | re.DOTALL)
# 切句前先护住缩写点，避免 "e.g." / "Fig. 3" 被当成句界
_ABBREV_SAFE_RE = re.compile(
    r"\b(e\.g|i\.e|etc|vs|cf|Fig|Figs|Eq|Eqs|Tab|Sec|al)\." , re.IGNORECASE)


def content_words(text: str) -> Set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOPWORDS}


def ngrams(text: str, n: int = 4) -> Set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def extract_abstract(content: str) -> str:
    match = _ABSTRACT_RE.search(content)
    return match.group(1).strip() if match else content[:1500]


def passes_filters(target: str, source: str, abstract: str, style: str, budget: int,
                   max_gram_overlap: float, min_grounding: float,
                   budget_tolerance: float = 1.1) -> str | None:
    """Return a rejection reason, or None when the summary passes all hard rules."""
    words = target.split()
    if not (5 <= len(words) <= budget * budget_tolerance):
        return f"word_budget({len(words)}>{budget})"
    if style == "tldr":
        guarded = _ABBREV_SAFE_RE.sub(r"\1<DOT>", target.strip())
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", guarded) if s.strip()]
        if len(sentences) > 2:
            return "tldr_not_single_sentence"
    if style == "bullet" and sum(1 for line in target.splitlines() if line.strip().startswith("-")) < 3:
        return "bullet_lt3"
    if style == "structured" and target.count("##") < 3:
        return "structured_lt3_sections"
    src_grams = ngrams(source)
    grams = ngrams(target)
    if grams and len(grams & src_grams) / len(grams) >= max_gram_overlap:
        return f"copying({len(grams & src_grams) / len(grams):.2f})"
    abstract_words = content_words(abstract)
    target_words = {w for w in re.findall(r"[a-z]{3,}", target.lower()) if w not in STOPWORDS}
    if target_words and len(target_words & abstract_words) / len(target_words) < min_grounding:
        return "hallucination_low_grounding"
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="基座自蒸馏论文总结 SFT 数据")
    parser.add_argument("--model", required=True, help="本地基座目录（与后续 SFT 同一基座）")
    parser.add_argument("--snapshot", default="data/mock_arxiv_snapshot.json")
    parser.add_argument("--seed-data", default="data/vlm/figureqa_seed.jsonl")
    parser.add_argument("--output", default="data/vlm/paper_summary_sft.jsonl")
    parser.add_argument("--max-papers", type=int, default=0, help="调试：只取前 N 篇")
    parser.add_argument("--max-new-tokens", type=int, default=0, help="默认按预算×2.2 自动")
    parser.add_argument("--max-gram-overlap", type=float, default=0.6)
    parser.add_argument("--min-grounding", type=float, default=0.6)
    parser.add_argument("--content-cutoff", type=int, default=_CONTENT_CUTOFF)
    parser.add_argument("--out", default="", help="可选：manifest 之外的运行报告 JSON")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    snapshot_path = Path(args.snapshot)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    records: Dict[str, Dict[str, Any]] = {}
    for key, record in snapshot.get("get_paper_content", {}).items():
        result = record.get("result") if isinstance(record, dict) else None
        if not isinstance(result, dict) or result.get("section"):
            continue
        paper_id = result.get("paper_id")
        if paper_id and paper_id not in records and result.get("content"):
            records[paper_id] = result
    paper_ids = sorted(records)
    if args.max_papers:
        paper_ids = paper_ids[: args.max_papers]
    print(f"📚 papers with full content: {len(paper_ids)}")

    val_papers: Set[str] = set()
    with Path(args.seed_data).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("split") == "val":
                    val_papers.add(row["paper_id"])

    model_dir = Path(args.model).expanduser()
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=model_dir.exists())
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, local_files_only=model_dir.exists()
    ).to("cuda")
    model.eval()
    model.config.use_cache = True
    try:  # 装载读入的 50GB page cache 计入宿主 cgroup 配额，读完立逐
        sys.path.insert(0, str(REPO_ROOT / "AgenticArxiv"))
        from rl.save_guard import evict_model_dir_page_cache

        print(f"🧹 evicted page cache: {evict_model_dir_page_cache(model_dir)} files", flush=True)
    except Exception:  # noqa: BLE001
        pass

    def generate(prompt: str, budget: int) -> str:
        try:
            text = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                add_generation_prompt=True, tokenize=False,
                enable_thinking=False,  # Qwen3.5 thinking 会淹没短总结的预算与形状
            )
        except TypeError:  # 模板不支持该开关
            text = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                add_generation_prompt=True, tokenize=False,
            )
        enc = processor(text=[text], return_tensors="pt").to("cuda")
        max_new = args.max_new_tokens or int(budget * 2.2) + 24
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        decoded = processor.batch_decode(
            out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        # 残余思维链剥离（skip_special_tokens 只去标签不去正文）
        if "</think>" in decoded:
            decoded = decoded.split("</think>", 1)[1].strip()
        elif decoded.startswith("<think>"):
            return ""
        return decoded

    rows: List[Dict[str, Any]] = []
    rejections: Dict[str, int] = {}
    total, kept = 0, 0
    # 断点续跑：输出已存在的 (paper, style, budget) 直接继承，不再生成
    output_path = Path(args.output)
    done: Set[tuple] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    rows.append(row)
                    done.add((row["paper_id"], row["style"], row["budget"]))
                    kept += 1
        print(f"♻️ 续跑：已有 {len(done)} 条，跳过对应生成")
    for index, paper_id in enumerate(paper_ids, 1):
        record = records[paper_id]
        source = record["content"][: args.content_cutoff]
        abstract = extract_abstract(record["content"])
        split = "val2" if paper_id in val_papers else "train"
        for style in STYLES:
            for budget in BUDGETS:
                total += 1
                if (paper_id, style, budget) in done:
                    continue
                prompt = (
                    "You are building a paper-reading assistant. Read the arXiv paper "
                    f"content below and write a {style} summary in at most {budget} words.\n\n"
                    f"{STYLE_INSTRUCTIONS[style]}\n\n"
                    f"Paper content:\n{source}\n\nWrite only the summary."
                )
                target = generate(prompt, budget)
                reason = passes_filters(
                    target, source, abstract, style, budget,
                    args.max_gram_overlap, args.min_grounding,
                )
                if reason:
                    head = re.sub(r"\W+", "", reason)[:12]
                    rejections[head] = rejections.get(head, 0) + 1
                    continue
                kept += 1
                rows.append({
                    "paper_id": paper_id,
                    "question": "summary",
                    "prompt": prompt,
                    "target": target,
                    "style": style,
                    "budget": budget,
                    "split": split,
                    "source": "v2_summary",
                })
        if index % 5 == 0 or index == len(paper_ids):
            print(f"  [{index}/{len(paper_ids)}] kept {kept}/{total}", flush=True)
        # 逐篇落盘：长跑中断后 resume 粒度 = 单篇（9 个目标）
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    manifest = {
        "version": 1,
        "kind": "paper_summary_sft",
        "model": str(model_dir),
        "teacher": "base-self-distillation-greedy",
        "snapshot": snapshot_path.as_posix(),
        "snapshot_sha256": sha256_file(snapshot_path),
        "output": output_path.as_posix(),
        "output_sha256": sha256_file(output_path),
        "filters": {
            "max_gram_overlap": args.max_gram_overlap,
            "min_grounding": args.min_grounding,
            "style_rules": STYLE_INSTRUCTIONS,
            "budgets": list(BUDGETS),
        },
        "summary": {
            "papers": len(paper_ids),
            "generated": total,
            "kept": kept,
            "kept_ratio": round(kept / total, 3) if total else None,
            "rejections": rejections,
            "rows_val2": sum(1 for row in rows if row["split"] == "val2"),
        },
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY_SFT_DATASET_OK")
    print(f"kept {kept}/{total} ({manifest['summary']['kept_ratio']}) | rejections: {rejections}")
    print(f"output: {output_path}")
    if args.out:
        Path(args.out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
