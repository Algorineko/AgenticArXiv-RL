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
from rl.sandbox import RolloutSandbox


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
        self._downloaded: set[str] = set()
        self._translated: set[str] = set()
        # Keep a baseline before any task setup is applied.  TRL may call
        # reset() more than once on the same environment instance; restoring a
        # snapshot is stronger than merely clearing the current session.
        self._sandbox = RolloutSandbox(
            self,
            file_roots=self._artifact_roots(),
        )

    @staticmethod
    def _artifact_roots() -> tuple[Path, ...]:
        from config import settings

        return (
            Path(settings.pdf_raw_path),
            Path(settings.pdf_translated_path),
            Path(settings.figures_path),
        )

    def capture_state(self) -> dict[str, Any]:
        """Capture mutable state owned directly by this rollout environment."""
        return {
            "store": self.store.capture_state(),
            "session_id": self.session_id,
            "downloaded": set(self._downloaded),
            "translated": set(self._translated),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore a state returned by :meth:`capture_state`."""
        required = {"store", "session_id", "downloaded", "translated"}
        if set(state) != required:
            raise ValueError("invalid AgenticArxivMultiTurnEnv sandbox snapshot")

        self.store.restore_state(state["store"])
        self.session_id = str(state["session_id"])
        self._downloaded = set(state["downloaded"])
        self._translated = set(state["translated"])

    def reset(self, task_id: str = "", **_: Any) -> str:
        """Reset per-rollout state and return optional initial observation."""
        self._sandbox.reset()
        self._downloaded = set()
        self._translated = set()
        suffix = task_id or "task"
        self.session_id = f"grpo_{suffix}_{uuid.uuid4().hex[:10]}"
        return "环境已重置；请根据任务调用工具，完成后直接给出最终回答。"

    def get_recently_submitted_cs_papers(
        self,
        aspect: str = "*",
        days: int = 7,
        max_results: int = 50,
        output_path: Optional[str] = None,
        save_to_file: bool = True,
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
            {
                "aspect": aspect,
                "days": days,
                "max_results": max_results,
                # The RL environment never writes search results to disk, but
                # accepts the production contract so valid actions do not fail.
                "output_path": output_path,
                "save_to_file": save_to_file,
            },
        )
        papers = [Paper(**item) for item in result]
        self.store.set_last_papers(self.session_id, papers)
        return result

    def search_arxiv_papers(
        self, query: str, max_results: int = 10, days: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """Search arXiv by keyword, title, or author within this rollout.

        Args:
            query: Bare text or an ``all:``, ``ti:``, or ``au:`` query.
            max_results: Maximum number of papers to return.
            days: Optional submission-time window in days.
        """
        result = self.backend.execute_tool(
            "search_arxiv_papers",
            {"query": query, "max_results": max_results, "days": days},
        )
        papers = [
            Paper(**{key: value for key, value in item.items() if not key.startswith("_")})
            for item in result
        ]
        self.store.set_last_papers(self.session_id, papers)
        return result

    def download_arxiv_pdf(
        self, ref: str | int | None = 1, force: bool = False
    ) -> dict[str, Any]:
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
        existed = paper.id in self._downloaded
        self._downloaded.add(paper.id)
        return {
            "paper_id": paper.id,
            "pdf_url": paper.pdf_url,
            "status": "READY",
            "offline": True,
            "force": bool(force),
            "existed": existed and not force,
        }

    def get_paper_content(
        self,
        ref: str | int | None = 1,
        section: Optional[str] = None,
    ) -> dict[str, Any]:
        """Read paper text through the deterministic snapshot backend."""
        paper = self.store.resolve_paper(
            self.session_id,
            ref,
        )

        if paper is None:
            raise ValueError(
                "Paper not found; search for the paper and check the ref."
            )

        return self.backend.execute_tool(
            "get_paper_content",
            {
                "session_id": self.session_id,
                "ref": ref,
                "section": section,
                # Refs are session-local, while snapshot identity must remain
                # stable across independent rollouts.
                "_resolved_paper_id": paper.id,
            },
        )

    def summarize_paper(
        self,
        ref: str | int | None = 1,
        style: Optional[str] = None,
        max_words: Optional[int] = None,
    ) -> dict[str, Any]:
        """Summarise a downloaded paper through the deterministic backend.

        Args:
            ref: One-based result index, arXiv id, or title fragment.
            style: One of tldr / structured / bullet.
            max_words: Word budget; rounded up to the nearest supported value.

        Returns:
            Summary text plus the paper it was produced from.
        """
        paper = self.store.resolve_paper(self.session_id, ref)
        if paper is None:
            raise ValueError(
                "Paper not found; search for the paper and check the ref."
            )

        return self.backend.execute_tool(
            "summarize_paper",
            {
                "session_id": self.session_id,
                "ref": ref,
                "style": style,
                "max_words": max_words,
                # Refs are session-local, while snapshot identity must remain
                # stable across independent rollouts.
                "_resolved_paper_id": paper.id,
            },
        )

    def extract_paper_figures(
        self, ref: str | int | None = 1
    ) -> dict[str, Any]:
        """Extract a downloaded paper's figures through the snapshot backend.

        Args:
            ref: One-based result index, arXiv id, or title fragment.

        Returns:
            Extracted figure files with page numbers and captions.
        """
        paper = self.store.resolve_paper(self.session_id, ref)
        if paper is None:
            raise ValueError(
                "Paper not found; search for the paper and check the ref."
            )

        return self.backend.execute_tool(
            "extract_paper_figures",
            {
                "session_id": self.session_id,
                "ref": ref,
                "_resolved_paper_id": paper.id,
            },
        )

    def analyze_figure(
        self,
        ref: str | int | None = 1,
        figure_no: int = 1,
        question: Optional[str] = None,
    ) -> dict[str, Any]:
        """Answer a question about one figure through the snapshot backend.

        Args:
            ref: One-based result index, arXiv id, or title fragment.
            figure_no: 1-based figure index within the paper.
            question: One of describe / axes / trend.

        Returns:
            The environment's answer plus the figure it came from.
        """
        paper = self.store.resolve_paper(self.session_id, ref)
        if paper is None:
            raise ValueError(
                "Paper not found; search for the paper and check the ref."
            )

        return self.backend.execute_tool(
            "analyze_figure",
            {
                "session_id": self.session_id,
                "ref": ref,
                "figure_no": figure_no,
                "question": question,
                "_resolved_paper_id": paper.id,
            },
        )

    def translate_arxiv_pdf(
        self,
        ref: str | int | None = None,
        force: bool = False,
        service: Optional[str] = None,
        threads: Optional[int] = None,
        keep_dual: bool = False,
        paper_id: Optional[str] = None,
        pdf_url: Optional[str] = None,
        input_pdf_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Translate a paper selected from the latest search results.

        Args:
            ref: One-based result index, arXiv id, or title fragment.

        Returns:
            Deterministic translation status used during RL training.
        """
        target = paper_id if paper_id else ref
        paper = self.store.resolve_paper(self.session_id, target)
        if paper is None:
            raise ValueError("未找到论文；请先搜索，再按序号、ID 或标题翻译")
        self.store.set_last_active_paper_id(self.session_id, paper.id)
        self._downloaded.add(paper.id)
        self._translated.add(paper.id)
        return {
            "paper_id": paper.id,
            "status": "READY",
            "offline": True,
            "force": bool(force),
            "service": service,
            "threads": threads,
            "keep_dual": bool(keep_dual),
            "pdf_url": pdf_url,
            "input_pdf_path": input_pdf_path,
        }

    def get_paper_cache_status(
        self, ref: str | int | None = None, paper_id: Optional[str] = None
    ) -> dict[str, Any]:
        """Inspect cached state for a paper in the current rollout.

        Args:
            ref: One-based result index, arXiv id, or title fragment.

        Returns:
            Whether the paper is known in the current rollout session.
        """
        target = paper_id if paper_id else ref
        paper = self.store.resolve_paper(self.session_id, target)
        if paper is None:
            raise ValueError("未找到论文；请先搜索，再按序号、ID 或标题查询缓存")
        self.store.set_last_active_paper_id(self.session_id, paper.id)
        return {
            "paper_id": paper.id,
            "pdf": {"status": "READY"} if paper.id in self._downloaded else None,
            "translate": {"status": "READY"} if paper.id in self._translated else None,
            "pdf_ready": paper.id in self._downloaded,
            "translated_ready": paper.id in self._translated,
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
