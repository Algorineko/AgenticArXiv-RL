"""TRL multi-turn environment adapter for AgenticArxiv.

Each GRPO generation receives an independent instance.  Public methods are
exposed to ``GRPOTrainer(environment_factory=...)`` as native tools; ``reset``
creates a fresh session so search/download/cache state never leaks between
rollouts.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

from models.schemas import Paper
from models.store_memory import MemoryStore
from rl.env import MockArxivEnv


class AgenticArxivMultiTurnEnv:
    """Independent, replayable arXiv environment for one multi-turn rollout."""

    def __init__(self, snapshot_path: Optional[Path] = None):
        self.backend = MockArxivEnv(
            snapshot_path=Path(snapshot_path) if snapshot_path else None,
            mode="replay" if snapshot_path else "auto",
            offline_download=True,
        )
        self.store = MemoryStore()
        self.session_id = ""

    def reset(self, task_id: str = "", **_: Any) -> str:
        """Reset per-rollout state and return optional initial observation."""
        self.store = MemoryStore()
        suffix = task_id or "task"
        self.session_id = f"grpo_{suffix}_{uuid.uuid4().hex[:10]}"
        return "环境已重置；请根据任务调用工具，完成后直接给出最终回答。"

    def get_recently_submitted_cs_papers(
        self, aspect: str = "*", days: int = 7, max_results: int = 50
    ) -> list[dict[str, Any]]:
        """Search recent computer-science papers.

        Args:
            aspect: arXiv CS suffix such as AI, LG, CL, CV, or *.
            days: Search window in days, from 1 to 30.
            max_results: Maximum number of papers, from 1 to 100.

        Returns:
            Paper metadata dictionaries used by later tools in this rollout.
        """
        result = self.backend.execute_tool(
            "get_recently_submitted_cs_papers",
            {"aspect": aspect, "days": days, "max_results": max_results},
        )
        papers = [Paper(**item) for item in result]
        self.store.set_last_papers(self.session_id, papers)
        return result

    def download_arxiv_pdf(self, ref: str | int = 1) -> dict[str, Any]:
        """Download a paper selected from the latest search results.

        Args:
            ref: One-based result index, arXiv id, or title fragment.

        Returns:
            Download status and local path. Training uses an offline PDF stub.
        """
        paper = self.store.resolve_paper(self.session_id, ref)
        if paper is None:
            raise ValueError("未找到论文；请先搜索，再按序号、ID 或标题下载")
        self.store.set_last_active_paper_id(self.session_id, paper.id)
        return {
            "paper_id": paper.id,
            "pdf_url": paper.pdf_url,
            "status": "READY",
            "offline": True,
        }

    def translate_arxiv_pdf(self, ref: str | int = 1) -> dict[str, Any]:
        """Translate a paper selected from the latest search results.

        Args:
            ref: One-based result index, arXiv id, or title fragment.

        Returns:
            Deterministic translation status used during RL training.
        """
        paper = self.store.resolve_paper(self.session_id, ref)
        if paper is None:
            raise ValueError("未找到论文；请先搜索，再按序号、ID 或标题翻译")
        self.store.set_last_active_paper_id(self.session_id, paper.id)
        return {"paper_id": paper.id, "status": "READY", "offline": True}

    def get_paper_cache_status(self, ref: str | int = 1) -> dict[str, Any]:
        """Inspect cached state for a paper in the current rollout.

        Args:
            ref: One-based result index, arXiv id, or title fragment.

        Returns:
            Whether the paper is known in the current rollout session.
        """
        paper = self.store.resolve_paper(self.session_id, ref)
        return {
            "known": paper is not None,
            "paper_id": paper.id if paper is not None else None,
        }


def make_environment_factory(snapshot_path: Path):
    """Return the zero-argument factory required by TRL GRPOTrainer."""
    path = Path(snapshot_path)
    if not path.exists():
        raise FileNotFoundError(
            f"多轮 GRPO 需要离线快照: {path}。"
            "请先运行 python -m AgenticArxiv.rl.build_snapshot"
        )

    def factory() -> AgenticArxivMultiTurnEnv:
        return AgenticArxivMultiTurnEnv(path)

    return factory
