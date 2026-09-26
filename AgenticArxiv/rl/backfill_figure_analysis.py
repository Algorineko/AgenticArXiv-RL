"""使用已有 T4 快照补录 T5 图表分析结果（抽取式或本地 VLM）。

在仓库根目录运行::

    # 抽取式（默认）：只用 caption，不读图片、不加载权重
    python -m AgenticArxiv.rl.backfill_figure_analysis \
        --snapshot data/mock_arxiv_snapshot.json

    # 本地 VLM 录制（按环境变量选择后端，需图片文件存在）
    FIGURE_ANALYSIS_BACKEND=vlm VLM_MODEL_PATH=<本地 VLM 目录> \
    python -m AgenticArxiv.rl.backfill_figure_analysis \
        --snapshot data/mock_arxiv_snapshot.json --force --paper-id 2608.14546v1

默认只补录缺失条目；``--force`` 会覆盖已有条目（例如把抽取式答案换成
VLM 答案），``--paper-id`` 可把范围限制在若干篇论文（VLM 录制耗时较长）。
本命令不请求 arXiv，也不读取 PDF。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.env import MockArxivEnv  # noqa: E402
from tools.figure_analysis_tool import (  # noqa: E402
    FIGURE_QUESTIONS,
    analysis_backend,
    build_extractive_analysis_result,
    build_vlm_analysis_result,
)

BUILDERS = {
    "extractive": build_extractive_analysis_result,
    "vlm": build_vlm_analysis_result,
}


@dataclass(frozen=True)
class BackfillStats:
    figures: int
    added: int
    existing: int
    overwritten: int = 0
    failed: int = 0


def backfill_entries(
    snapshot: Mapping[str, Any],
    *,
    backend: str = "extractive",
    paper_ids: Optional[Sequence[str]] = None,
    force: bool = False,
) -> Tuple[Dict[str, Any], BackfillStats]:
    """返回补录后的快照，不修改输入映射。

    ``backend`` 选 ``extractive`` / ``vlm``；``paper_ids`` 只处理这些论文；
    ``force`` 时覆盖已有条目（失败时保留原条目并计数）。
    """
    if not isinstance(snapshot, dict):
        raise ValueError("Snapshot root must be a JSON object")
    if backend not in BUILDERS:
        raise ValueError(f"Unsupported backend: {backend!r}")

    extraction = snapshot.get("extract_paper_figures")
    if not isinstance(extraction, dict) or not extraction:
        raise ValueError("Snapshot has no extract_paper_figures (T4) records")

    current = snapshot.get("analyze_figure", {})
    if not isinstance(current, dict):
        raise ValueError("Snapshot analyze_figure (T5) records must be an object")

    selected = set(paper_ids) if paper_ids else None
    seen_paper_ids = set()
    builder = BUILDERS[backend]

    updates: Dict[str, Dict[str, Any]] = {}
    figures_seen = existing = overwritten = failed = 0
    for record_key, record in extraction.items():
        if not isinstance(record, dict) or not isinstance(record.get("result"), dict):
            raise ValueError(f"Invalid T4 record: {record_key}")
        result = record["result"]
        paper_id = result.get("paper_id")
        figures = result.get("figures")
        count = result.get("count")
        if not isinstance(paper_id, str) or not paper_id.strip():
            raise ValueError(f"T4 record has no paper_id: {record_key}")
        try:
            key_paper_id = json.loads(record_key)["paper_id"]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid T4 paper key: {record_key}") from exc
        if key_paper_id != paper_id:
            raise ValueError(f"T4 paper key does not match result: {record_key}")
        if not isinstance(figures, list) or type(count) is not int or count != len(figures):
            raise ValueError(f"T4 figure count is inconsistent: {record_key}")
        if selected is not None and paper_id not in selected:
            continue
        seen_paper_ids.add(paper_id)

        for figure_no, figure in enumerate(figures, start=1):
            if (
                not isinstance(figure, dict)
                or type(figure.get("figure_no")) is not int
                or figure["figure_no"] != figure_no
                or not isinstance(figure.get("path"), str)
                or not figure["path"]
                or not isinstance(figure.get("caption"), (str, type(None)))
            ):
                raise ValueError(f"Invalid T4 figure {figure_no}: {record_key}")
            figures_seen += 1
            for question in FIGURE_QUESTIONS:
                key = MockArxivEnv._figure_analysis_key(
                    {"figure_no": figure_no, "question": question},
                    resolved_paper_id=paper_id,
                )
                if key in current and not force:
                    existing += 1
                    continue
                try:
                    observation = builder(paper_id, figure, question)
                except Exception as exc:  # noqa: BLE001 - 单图失败不中断整批录制
                    failed += 1
                    print(
                        f"⚠️  {paper_id} figure {figure_no} [{question}] 录制失败: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                entry = {
                    "args": {"ref": 1, "figure_no": figure_no, "question": question},
                    "result": observation,
                }
                if key in updates and updates[key] != entry:
                    raise ValueError(f"Conflicting T4 figures for paper {paper_id}")
                if key in current:
                    overwritten += 1
                updates[key] = entry

    if selected is not None:
        missing = sorted(selected - seen_paper_ids)
        if missing:
            raise ValueError(f"--paper-id 未匹配任何 T4 记录: {missing}")
    if not figures_seen:
        raise ValueError("T4 records contain no figures to analyse")

    upgraded = dict(snapshot)
    upgraded["analyze_figure"] = {**current, **updates}
    return upgraded, BackfillStats(
        figures_seen,
        len(updates) - overwritten,
        existing,
        overwritten=overwritten,
        failed=failed,
    )


def backfill_snapshot(
    path: Path,
    *,
    backend: Optional[str] = None,
    paper_ids: Optional[Sequence[str]] = None,
    force: bool = False,
) -> BackfillStats:
    """原子写入快照；校验失败时保持原文件不变。

    未显式给出 ``backend`` 时沿用 ``FIGURE_ANALYSIS_BACKEND`` 环境变量，
    与在线调用保持一致。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot does not exist: {path}")
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Snapshot is not valid JSON: {path}") from exc

    resolved_backend = backend or analysis_backend()
    upgraded, stats = backfill_entries(
        snapshot, backend=resolved_backend, paper_ids=paper_ids, force=force
    )
    if not stats.added and not stats.overwritten:
        return stats

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(upgraded, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return stats


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--paper-id", action="append", default=None,
        help="只处理这些论文（可重复；VLM 录制建议先用它限制范围）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="覆盖已有 T5 条目（例如把抽取式答案替换为 VLM 答案）",
    )
    args = parser.parse_args(argv)
    stats = backfill_snapshot(
        args.snapshot, paper_ids=args.paper_id, force=args.force
    )
    print(
        f"T5 backfill [{analysis_backend()}]: {stats.figures} figures, "
        f"{stats.added} added, {stats.existing} already present, "
        f"{stats.overwritten} overwritten, {stats.failed} failed"
    )
    # 部分失败（缺个别图片文件）不致命，逐条已在上面打印；全军覆没才算错误。
    if stats.failed and not (stats.added + stats.overwritten):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
