"""Opt-in translated-reading tasks bound to an explicitly supplied snapshot."""

import json
from pathlib import Path
from typing import List

from benchmark.task_spec import Step, TaskSpec


def get_translated_specs(snapshot_path: Path) -> List[TaskSpec]:
    """Derive tasks only for translated documents present in a recorded search pool."""
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    specs = []
    for entry in snapshot.get("get_translated_paper_content", {}).values():
        paper_id = entry["result"]["paper_id"]
        search = None
        for tool_name in ("get_recently_submitted_cs_papers", "search_arxiv_papers"):
            for pool in snapshot.get(tool_name, {}).values():
                result = pool.get("result")
                if isinstance(result, list) and any(
                    isinstance(paper, dict) and paper.get("id") == paper_id for paper in result
                ):
                    args = {key: value for key, value in pool.get("args", {}).items()
                            if key != "session_id"}
                    if "max_results" in args:
                        args["max_results"] = len(result)
                    search = Step(tool_name, args)
                    break
            if search is not None:
                break
        if search is None:
            raise ValueError(f"Translated paper {paper_id} is absent from recorded search pools")
        translate = Step("translate_arxiv_pdf", {"ref": paper_id})
        for budget in (64, 128, 256):
            read = Step("get_translated_paper_content", {
                "ref": paper_id, "offset": 0, "max_chars": budget})
            for cached in (True, False):
                mode = "cached" if cached else "chain"
                instruction = "读取已完成的译文" if cached else "先翻译，再读取译文"
                specs.append(TaskSpec(
                    id=f"translated_{paper_id.replace('/', '_')}_{mode}_{budget}",
                    task=f"{instruction}：论文 {paper_id} 的首段，最多 {budget} 个字符。只读取首段，勿声称已读完整篇。",
                    steps=(read,) if cached else (translate, read),
                    setup=(search, translate) if cached else (search,),
                    category="translated_reading", template=f"translated_{mode}",
                    requires_offline=True, max_iterations=4,
                ))
    if not specs:
        raise ValueError("Snapshot has no translated documents; backfill completed mono PDFs first.")
    return specs
