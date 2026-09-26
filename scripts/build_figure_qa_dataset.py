#!/usr/bin/env python3
"""从离线快照构建 FigureQA 后训练数据集（T5 的 env 侧 VLM）。

监督目标完全来自论文作者写的 caption，不用任何第三方模型生成标签：

- ``describe``：caption 全文（去掉 ``Figure N:`` 前缀、折叠空白）
- ``axes`` / ``trend``：caption 中命中坐标轴/趋势线索词的整句（与
  ``tools.figure_analysis_tool`` 的抽取式后端共用同一套线索词与切句逻辑，
  保证「训练目标 = 该问题的确定性文本证据」）

问题文案直接 import 工具模块的 ``_QUESTION_PROMPTS``，确保训练与快照录制
（``FIGURE_ANALYSIS_BACKEND=vlm``）看到的是同一条指令。按论文切分
train/val，避免同一篇论文的近重复图跨集泄漏。

用法::

    python scripts/build_figure_qa_dataset.py \
        --snapshot data/mock_arxiv_snapshot.json \
        --output data/vlm/figureqa_seed.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))

# 单一事实来源：与快照录制、在线调用共享同一套问题文案与线索词。
from tools.figure_analysis_tool import (  # noqa: E402
    _AXES_HINTS,
    _QUESTION_PROMPTS,
    _TREND_HINTS,
    _matching_sentences,
    _sentences,
    FIGURE_QUESTIONS,
)

FIGURE_PREFIX_RE = re.compile(r"^(?:figure|fig\.?)\s*\d+\s*[:.\-]?\s*", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")

# 监督目标遵守工具提示词的长度约定（describe 1–2 句、axes/trend 1 句），
# 同时匹配录制端的 96 token 生成预算。
TARGET_MAX_SENTENCES = {"describe": 2, "axes": 1, "trend": 1}


def clean_text(text: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "")).strip()


def _cap_sentences(text: str, limit: int) -> str:
    sentences = _sentences(text)
    if not sentences:
        return clean_text(text)
    return " ".join(sentences[:limit])


def describe_target(caption: str) -> str:
    cleaned = FIGURE_PREFIX_RE.sub("", clean_text(caption))
    return _cap_sentences(cleaned, TARGET_MAX_SENTENCES["describe"])


def evidence_target(caption: str, question: str) -> str:
    hints = _AXES_HINTS if question == "axes" else _TREND_HINTS
    joined = clean_text(" ".join(_matching_sentences(caption, hints)))
    joined = FIGURE_PREFIX_RE.sub("", joined)
    return _cap_sentences(joined, TARGET_MAX_SENTENCES[question])


def build_target(caption: str, question: str) -> str:
    """Return the supervision target, or "" when the caption carries none."""
    if question == "describe":
        return describe_target(caption)
    return evidence_target(caption, question)


def is_readable_image(path: Path) -> bool:
    """A figure is usable only when the file exists, is non-empty and decodes."""
    from PIL import Image

    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with Image.open(path) as image:
            image.load()
    except Exception:  # noqa: BLE001 - 损坏的图片按不可用处理
        return False
    return True


def iter_t4_figures(snapshot: Mapping[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (paper_id, figure) for every T4 figure that declares an image path."""
    extraction = snapshot.get("extract_paper_figures")
    if not isinstance(extraction, dict):
        raise ValueError("Snapshot has no extract_paper_figures (T4) records")
    for record in extraction.values():
        result = record.get("result") if isinstance(record, dict) else None
        if not isinstance(result, dict):
            continue
        paper_id = result.get("paper_id")
        for figure in result.get("figures") or []:
            if paper_id and figure.get("path"):
                yield str(paper_id), figure


def iter_usable_figures(snapshot: Mapping[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (paper_id, figure) for figures whose image file exists and decodes."""
    for paper_id, figure in iter_t4_figures(snapshot):
        if is_readable_image(Path(figure["path"])):
            yield paper_id, figure


def split_papers(paper_ids: Sequence[str], *, seed: int, val_fraction: float) -> Tuple[List[str], List[str]]:
    """Deterministic paper-level train/val split."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    ordered = sorted(set(paper_ids))
    import random

    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction))
    val = sorted(shuffled[:val_count])
    train = sorted(shuffled[val_count:])
    return train, val


def build_rows(
    snapshot: Mapping[str, Any],
    *,
    seed: int,
    val_fraction: float,
    image_root: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build dataset rows plus a stats/summary dict (no absolute paths inside)."""
    figures: List[Tuple[str, Dict[str, Any]]] = []
    unreadable = 0
    for paper_id, figure in iter_t4_figures(snapshot):
        if is_readable_image(Path(figure["path"])):
            figures.append((paper_id, figure))
        else:
            unreadable += 1
    papers = [paper_id for paper_id, _ in figures]
    train_papers, val_papers = split_papers(papers, seed=seed, val_fraction=val_fraction)
    train_set, val_set = set(train_papers), set(val_papers)

    rows: List[Dict[str, Any]] = []
    counts: Dict[str, Dict[str, int]] = {
        "train": {q: 0 for q in FIGURE_QUESTIONS},
        "val": {q: 0 for q in FIGURE_QUESTIONS},
    }
    skipped_no_target = 0
    for paper_id, figure in figures:
        caption = clean_text(figure.get("caption"))
        split = "train" if paper_id in train_set else "val" if paper_id in val_set else None
        if split is None:  # pragma: no cover - defensive
            continue
        image_path = Path(figure["path"]).resolve()
        try:
            rel = image_path.relative_to(image_root)
        except ValueError as exc:
            raise ValueError(f"figure path outside image root: {image_path}") from exc
        for question in FIGURE_QUESTIONS:
            target = build_target(caption, question)
            if not target:
                skipped_no_target += 1
                continue
            rows.append(
                {
                    "paper_id": paper_id,
                    "figure_no": int(figure.get("figure_no") or 0),
                    "question": question,
                    "prompt": _QUESTION_PROMPTS[question],
                    "target": target,
                    "image_relpath": rel.as_posix(),
                    "split": split,
                }
            )
            counts[split][question] += 1

    summary = {
        "figures_usable": len(figures),
        "figures_unreadable": unreadable,
        "papers": len(set(papers)),
        "rows": len(rows),
        "skipped_no_target": skipped_no_target,
        "counts": counts,
        "train_papers": train_papers,
        "val_papers": val_papers,
    }
    return rows, summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 FigureQA 后训练数据集（T5 env 侧 VLM）")
    parser.add_argument("--snapshot", default="data/mock_arxiv_snapshot.json")
    parser.add_argument("--output", default="data/vlm/figureqa_seed.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--image-root",
        default="AgenticArxiv",
        help="图片路径必须位于该目录内；相对路径以它为基准写入数据行",
    )
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        raise SystemExit(f"快照不存在: {snapshot_path}")
    image_root = Path(args.image_root).resolve()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    rows, summary = build_rows(
        snapshot, seed=args.seed, val_fraction=args.val_fraction, image_root=image_root
    )
    if not rows:
        raise SystemExit("没有生成任何样本：检查 T4 记录与图片文件是否存在")

    output_path = Path(args.output)
    write_jsonl(output_path, rows)

    manifest = {
        "version": 1,
        "kind": "figure_qa_seed",
        "snapshot": snapshot_path.as_posix(),
        "snapshot_sha256": sha256_file(snapshot_path),
        "output": output_path.as_posix(),
        "output_sha256": sha256_file(output_path),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "target_rules": {
            "describe": "first up-to-2 sentences of the caption with the leading 'Figure N:' prefix stripped",
            "axes": "first caption sentence matching the axes hints of tools.figure_analysis_tool",
            "trend": "first caption sentence matching the trend hints of tools.figure_analysis_tool",
            "sentence_caps": TARGET_MAX_SENTENCES,
        },
        "prompt_source": "tools.figure_analysis_tool._QUESTION_PROMPTS",
        "summary": summary,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("FIGURE_QA_DATASET_OK")
    print(f"papers:            {summary['papers']} (train {len(summary['train_papers'])} / val {len(summary['val_papers'])})")
    print(f"figures usable:    {summary['figures_usable']} (unreadable: {summary['figures_unreadable']})")
    print(f"rows:              {summary['rows']} (skipped without target: {summary['skipped_no_target']})")
    print(f"per question:      {summary['counts']}")
    print(f"output:            {output_path}")


if __name__ == "__main__":
    main()
