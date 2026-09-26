#!/usr/bin/env python3
"""FigureQA 前后对比评测：在留出论文上比较两个 VLM 的回答质量。

评测路径与快照录制完全同构（复用 ``tools.figure_analysis_tool._vlm_answer``：
相同的缩放、提示词、贪心解码与 96 token 预算），指标为
ROUGE-1 / ROUGE-L（对监督目标）与空回答/异常率。目标是 caption 证据，
是代理指标而非人工评判——优势看相对变化，不看绝对值。

用法::

    python scripts/eval_figure_qa_vlm.py --model <VLM 目录> --tag base --output artifacts/vlm_figureqa_eval/base.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))

from tools.figure_analysis_tool import _vlm_answer  # noqa: E402

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def _f_measure(overlap: int, pred_len: int, ref_len: int) -> float:
    if pred_len == 0 or ref_len == 0:
        return 0.0
    precision = overlap / pred_len
    recall = overlap / ref_len
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_1(pred: str, ref: str) -> float:
    pred_tokens, ref_tokens = tokenize(pred), tokenize(ref)
    ref_counts = collections.Counter(ref_tokens)
    overlap = sum(min(count, ref_counts[token]) for token, count in collections.Counter(pred_tokens).items())
    return _f_measure(overlap, len(pred_tokens), len(ref_tokens))


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for index, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(pred: str, ref: str) -> float:
    pred_tokens, ref_tokens = tokenize(pred), tokenize(ref)
    return _f_measure(_lcs_length(pred_tokens, ref_tokens), len(pred_tokens), len(ref_tokens))


def evaluate_row(model_path: str, row: Mapping[str, Any], image_root: Path) -> Dict[str, Any]:
    figure = {"path": str((image_root / row["image_relpath"]).resolve())}
    os.environ["VLM_MODEL_PATH"] = model_path
    try:
        answer = _vlm_answer(figure, row["question"])
        error = None
    except Exception as exc:  # noqa: BLE001 - 记录并继续，评测不因单条中断
        answer = ""
        error = f"{type(exc).__name__}: {exc}"
    return {
        "paper_id": row["paper_id"],
        "figure_no": row["figure_no"],
        "question": row["question"],
        "target": row["target"],
        "answer": answer,
        "rouge1_f": round(rouge_1(answer, row["target"]), 4),
        "rougeL_f": round(rouge_l(answer, row["target"]), 4),
        "error": error,
    }


def aggregate(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    per_question: Dict[str, Any] = {}
    for question in sorted({row["question"] for row in results}):
        subset = [row for row in results if row["question"] == question]
        n = len(subset)
        per_question[question] = {
            "n": n,
            "rouge1_f": round(sum(row["rouge1_f"] for row in subset) / n, 4),
            "rougeL_f": round(sum(row["rougeL_f"] for row in subset) / n, 4),
            "empty_or_error": sum(1 for row in subset if row["error"] or not row["answer"].strip()),
            "avg_answer_tokens": round(sum(len(tokenize(row["answer"])) for row in subset) / n, 1),
        }
    n = len(results)
    return {
        "n": n,
        "rouge1_f": round(sum(row["rouge1_f"] for row in results) / n, 4) if n else 0.0,
        "rougeL_f": round(sum(row["rougeL_f"] for row in results) / n, 4) if n else 0.0,
        "empty_or_error": sum(1 for row in results if row["error"] or not row["answer"].strip()),
        "per_question": per_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FigureQA VLM 留出评测")
    parser.add_argument("--model", required=True, help="本地 VLM 目录（base 或合并后的权重）")
    parser.add_argument("--tag", default="model")
    parser.add_argument("--data", default="data/vlm/figureqa_seed.jsonl")
    parser.add_argument("--image-root", default="AgenticArxiv")
    parser.add_argument("--output", default=None, help="结果 JSON 路径")
    parser.add_argument("--limit", type=int, default=0, help="调试：只评测前 N 条")
    args = parser.parse_args()

    data_path = Path(args.data)
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_rows = [row for row in rows if row.get("split") == "val"]
    if args.limit:
        val_rows = val_rows[: args.limit]
    if not val_rows:
        raise SystemExit("没有 val 样本")

    image_root = Path(args.image_root).resolve()
    results = []
    for index, row in enumerate(val_rows, start=1):
        result = evaluate_row(args.model, row, image_root)
        results.append(result)
        print(
            f"[{index}/{len(val_rows)}] {result['paper_id']} fig{result['figure_no']} "
            f"{result['question']}: ROUGE-L={result['rougeL_f']}"
        )

    payload = {
        "tag": args.tag,
        "model": args.model,
        "data": data_path.as_posix(),
        "aggregate": aggregate(results),
        "results": results,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"output: {output_path}")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
