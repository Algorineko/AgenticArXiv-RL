"""Backfill extractive T5 observations from an existing T4 snapshot.

Usage from the repository root::

    python -m AgenticArxiv.rl.backfill_figure_analysis \
        --snapshot data/mock_arxiv_snapshot.json

No arXiv request, PDF read, image read, or model inference is performed.  A
VLM snapshot still needs to be recorded with the real VLM backend.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.env import MockArxivEnv  # noqa: E402
from tools.figure_analysis_tool import (  # noqa: E402
    FIGURE_QUESTIONS,
    analysis_backend,
    build_extractive_analysis_result,
)


@dataclass(frozen=True)
class BackfillStats:
    figures: int
    added: int
    existing: int


def backfill_entries(snapshot: Mapping[str, Any]) -> Tuple[Dict[str, Any], BackfillStats]:
    """Return an upgraded snapshot without changing the input mapping."""
    if not isinstance(snapshot, dict):
        raise ValueError("Snapshot root must be a JSON object")

    extraction = snapshot.get("extract_paper_figures")
    if not isinstance(extraction, dict) or not extraction:
        raise ValueError("Snapshot has no extract_paper_figures (T4) records")

    current = snapshot.get("analyze_figure", {})
    if not isinstance(current, dict):
        raise ValueError("Snapshot analyze_figure (T5) records must be an object")

    additions: Dict[str, Dict[str, Any]] = {}
    figures_seen = existing = 0
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
                if key in current:
                    existing += 1
                    continue
                observation = build_extractive_analysis_result(
                    paper_id, figure, question
                )
                entry = {
                    "args": {"ref": 1, "figure_no": figure_no, "question": question},
                    "result": observation,
                }
                if key in additions and additions[key] != entry:
                    raise ValueError(f"Conflicting T4 figures for paper {paper_id}")
                additions[key] = entry

    if not figures_seen:
        raise ValueError("T4 records contain no figures to analyse")

    upgraded = dict(snapshot)
    upgraded["analyze_figure"] = {**current, **additions}
    return upgraded, BackfillStats(figures_seen, len(additions), existing)


def backfill_snapshot(path: Path) -> BackfillStats:
    """Upgrade one snapshot atomically; leave it untouched on validation errors."""
    if analysis_backend() != "extractive":
        raise ValueError("Backfill supports FIGURE_ANALYSIS_BACKEND=extractive only")

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot does not exist: {path}")
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Snapshot is not valid JSON: {path}") from exc

    upgraded, stats = backfill_entries(snapshot)
    if not stats.added:
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
    args = parser.parse_args(argv)
    stats = backfill_snapshot(args.snapshot)
    print(
        f"T5 extractive backfill: {stats.figures} figures, "
        f"{stats.added} added, {stats.existing} already present"
    )


if __name__ == "__main__":
    main()
