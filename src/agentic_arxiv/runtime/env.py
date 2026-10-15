"""工具后端：显式 live、record 和只读 replay。

共同 episode 持有会话与产物所有权。独立快照构建仍可使用 record；
replay 缺记录时不会联网，翻译模拟由共同 episode 完成。
"""

import inspect
import json
import os
import copy
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional, Set

from agentic_arxiv.config import workspace_path
from agentic_arxiv.tools.tool_registry import registry
from agentic_arxiv.state.store import store
from loguru import logger as log

# 构造缓存 key 时忽略的易变字段
_VOLATILE_ARG_KEYS = {"session_id", "output_path", "save_to_file", "_resolved_paper_id"}

# 需要走快照回放的"网络型"工具
_RECENT_SEARCH_TOOL = "get_recently_submitted_cs_papers"
_KEYWORD_SEARCH_TOOL = "search_arxiv_papers"
_SEARCH_TOOLS = {_RECENT_SEARCH_TOOL, _KEYWORD_SEARCH_TOOL}
_PAPER_CONTENT_TOOL = "get_paper_content"
_PAPER_SUMMARY_TOOL = "summarize_paper"
_PAPER_FIGURES_TOOL = "extract_paper_figures"
_FIGURE_ANALYSIS_TOOL = "analyze_figure"
_TRANSLATED_CONTENT_TOOL = "get_translated_content"
DEFAULT_SNAPSHOT_TOOLS: Set[str] = set(_SEARCH_TOOLS) | {
    _PAPER_CONTENT_TOOL,
    _PAPER_SUMMARY_TOOL,
    _PAPER_FIGURES_TOOL,
    _FIGURE_ANALYSIS_TOOL,
    _TRANSLATED_CONTENT_TOOL,
}


class MockArxivEnv:
    """快照回放环境（用于快速、确定性、离线的 rollout）

    工作原理：
      1. record 模式：真实调用工具，把 (tool, key) → result 落盘为 snapshot
      2. replay 模式：只查 snapshot，miss 即报错（保证完全离线、可复现）
      3. live 模式：仅真实执行，不隐式命中或写入旧快照

    适用场景：
      - 快速 rollout（避免真实 arXiv API 调用，GRPO 采样吞吐的刚需）
      - 确定性重放（同样输入保证同样输出，实验可复现）
      - 离线训练（不依赖外部网络）
    """

    MODES = ("replay", "record", "live")

    def __init__(
        self,
        snapshot_path: Optional[Path] = None,
        mode: str = "live",
        offline_download: Optional[bool] = None,
        snapshot_tools: Optional[Set[str]] = None,
    ):
        """
        Args:
            snapshot_path: 快照文件路径（JSON）
            mode: replay | record | live
            offline_download: True 时 download_arxiv_pdf 走离线桩（不发 HTTP）
            snapshot_tools: 需要快照的工具名集合
        """
        if mode not in self.MODES:
            raise ValueError(f"mode 必须是 {self.MODES} 之一，收到 {mode!r}")

        self.snapshot_path = workspace_path(snapshot_path) if snapshot_path else None
        if mode == "replay" and self.snapshot_path and not self.snapshot_path.is_file():
            raise FileNotFoundError(f"离线快照不存在: {self.snapshot_path}")
        self.mode = mode
        self.offline_download = mode == "replay" if offline_download is None else offline_download
        self.snapshot_tools = snapshot_tools or set(DEFAULT_SNAPSHOT_TOOLS)
        self.snapshot: Dict[str, Dict[str, Any]] = {}
        # 只记录当前 rollout 进程内的离线下载状态。不能用磁盘文件是否存在
        # 判断，否则第二次复现实验会继承第一次留下的占位 PDF。
        self._offline_downloaded: Set[tuple] = set()

        # 统计信息，便于确认 rollout 真的没打外网
        self.stats = {
            "hit": 0,
            "miss": 0,
            "real_calls": 0,
            "offline_stubs": 0,
            "fallback": 0,
        }

        if self.snapshot_path and self.snapshot_path.exists():
            with open(self.snapshot_path, "r", encoding="utf-8") as f:
                self.snapshot = json.load(f)
            log.info(
                f"[MockArxivEnv] 载入快照 {self.snapshot_path} "
                f"({sum(len(v) for v in self.snapshot.values())} 条记录)"
            )

    # ---------- 主入口 ----------

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """执行工具（返回值与 registry.execute_tool 保持同一契约：list/dict/str）"""
        # ``_resolved_paper_id`` is private environment metadata used by the
        # isolated multi-turn store. It is never exposed to or passed into a tool.
        internal_args = dict(args or {})
        resolved_paper_id = internal_args.pop("_resolved_paper_id", None)
        self._validate_tool_arguments(tool_name, internal_args)
        args = internal_args

        # live 不从历史记录隐式命中，也不录制或合成失败替代结果。
        if self.mode == "live":
            self.stats["real_calls"] += 1
            result = self._execute_live(tool_name, args)
            self._sync_session_papers(tool_name, args, result)
            return result

        # 1) 离线下载桩
        if tool_name == "download_arxiv_pdf" and self.offline_download:
            self.stats["offline_stubs"] += 1
            return self._offline_download(args)

        if self.mode == "replay" and tool_name == "translate_arxiv_pdf":
            raise ValueError("离线翻译必须经由每轨迹共同环境执行，不能启动真实翻译")

        # 2) 非快照工具（纯本地，如缓存查询）→ 直接真实执行
        if tool_name not in self.snapshot_tools:
            if self.mode == "replay" and tool_name != "get_paper_cache_status":
                self.stats["miss"] += 1
                raise KeyError(f"[MockArxivEnv] replay 模式下快照缺失: tool={tool_name}")
            self.stats["real_calls"] += 1
            return self._execute_live(tool_name, args)

        # 3) 快照工具
        key = self._make_key(args)
        if tool_name == _KEYWORD_SEARCH_TOOL:
            key = self._keyword_search_key(args)
        elif tool_name == _PAPER_CONTENT_TOOL:
            key = self._paper_content_key(args, resolved_paper_id=resolved_paper_id)
        elif tool_name == _PAPER_SUMMARY_TOOL:
            key = self._paper_summary_key(args, resolved_paper_id=resolved_paper_id)
        elif tool_name == _PAPER_FIGURES_TOOL:
            key = self._paper_figures_key(args, resolved_paper_id=resolved_paper_id)
        elif tool_name == _FIGURE_ANALYSIS_TOOL:
            key = self._figure_analysis_key(args, resolved_paper_id=resolved_paper_id)
        elif tool_name == _TRANSLATED_CONTENT_TOOL:
            key = self._translated_content_key(args, resolved_paper_id=resolved_paper_id)
        tool_data = self.snapshot.get(tool_name, {})

        # record 模式必须每次都真打，否则派生逻辑会"帮倒忙"：
        # 后续 aspect 直接命中已有池子，快照里就只剩第一条记录。
        if self.mode == "record":
            self.stats["real_calls"] += 1
            result = self._execute_live(tool_name, args)
            self._add_to_snapshot(tool_name, key, args, result)
            self._sync_session_papers(tool_name, args, result)
            return result

        if key in tool_data:
            self.stats["hit"] += 1
            result = copy.deepcopy(tool_data[key]["result"])
            if tool_name == _KEYWORD_SEARCH_TOOL:
                result = self._limit_search_result(result, args)
            self._sync_session_papers(tool_name, args, result)
            return result

        # 3a) 搜索工具：按"论文池"语义派生，而不是死抠精确 key
        #     真实 API 下 max_results=5 / 7 / 10 都能正常返回，mock 也应如此，
        #     否则策略稍微改个参数就变成 KeyError，奖励信号会被污染成"全是工具失败"。
        if tool_name == _RECENT_SEARCH_TOOL:
            derived = self._derive_search_result(args, tool_data)
            if derived is not None:
                fallback = any(isinstance(item, dict) and item.get("_mock_env", {}).get("offline_fallback")
                               for item in derived)
                self.stats["fallback" if fallback else "hit"] += 1
                self._sync_session_papers(tool_name, args, derived)
                return copy.deepcopy(derived)

        if tool_name == _KEYWORD_SEARCH_TOOL:
            fallback = self._keyword_search_fallback(args, tool_data)
            if fallback is not None:
                # 降级回退池不是快照命中：只返回带标记的结果，不写会话记忆。
                # 共同 episode 检测到 offline_fallback 会报告"工具执行失败"，
                # 若这里把回退论文同步进 store，模型后续用 ref 操作会解析到
                # 无关论文——观测说失败、状态里却有"论文"，两者矛盾。
                # 共同 episode 会将标记回退转换为明确的失败观测。
                self.stats["fallback"] += 1
                return fallback

        if self.mode == "replay" and tool_name == _TRANSLATED_CONTENT_TOOL:
            out_of_range = self._translated_page_out_of_range(key, tool_data)
            if out_of_range is not None:
                raise out_of_range

        self.stats["miss"] += 1
        if self.mode == "replay":
            raise KeyError(
                f"[MockArxivEnv] replay 模式下快照缺失: tool={tool_name} key={key}。"
                f"请先运行 `agentic-arxiv data snapshot` 生成快照。"
            )

        raise RuntimeError("回放模式未命中；禁止自动切换为真实执行")

    @staticmethod
    def _execute_live(tool_name: str, args: Dict[str, Any]) -> Any:
        call_args = dict(args)
        signature = inspect.signature(registry.get_tool(tool_name)["func"])
        if "session_id" not in signature.parameters:
            call_args.pop("session_id", None)
        return registry.execute_tool(tool_name, call_args)

    @staticmethod
    def _validate_tool_arguments(tool_name: str, args: Dict[str, Any]) -> None:
        """让离线桩保留真实 Python 工具的参数边界。

        setup 会额外携带 session_id 以同步内存状态；对本身不接收它的搜索
        函数，这是框架元数据而不是模型参数，因此绑定前移除。其余未知参数
        （例如把 search_arxiv_papers 的 query 塞给 recent-search）必须报错，
        否则 replay 会给错误动作返回一个看似成功的 observation。
        """
        tool = registry.get_tool(tool_name)
        if tool is None:
            raise ValueError(f"工具 '{tool_name}' 未注册")
        func = tool["func"]
        signature = inspect.signature(func)
        call_args = dict(args or {})
        if "session_id" not in signature.parameters:
            call_args.pop("session_id", None)
        if tool_name == _KEYWORD_SEARCH_TOOL:
            query = " ".join(str(call_args.get("query") or "").split())
            if not query:
                raise ValueError("工具参数错误: query 不能为空")
        try:
            signature.bind(**call_args)
        except TypeError as exc:
            raise ValueError(f"工具参数错误: {exc}") from exc

    def _sync_session_papers(
        self, tool_name: str, args: Dict[str, Any], result: Any
    ) -> None:
        """若带 session_id 且为搜索工具，同步更新 store 中的 papers 缓存"""
        if tool_name in _SEARCH_TOOLS and isinstance(result, list):
            if any(isinstance(paper, dict) and paper.get("_mock_env", {}).get("offline_fallback")
                   for paper in result):
                return
            session_id = args.get("session_id")
            if session_id:
                try:
                    from agentic_arxiv.state.schemas import Paper
                    from agentic_arxiv.state.store import store

                    papers = [
                        Paper(
                            **{
                                key: value
                                for key, value in paper.items()
                                if not key.startswith("_")
                            }
                        )
                        if isinstance(paper, dict)
                        else paper
                        for paper in result
                    ]
                    store.set_last_papers(session_id, papers)
                except Exception:
                    pass

    # ---------- 搜索语义派生 ----------

    @staticmethod
    def _derive_search_result(args: Dict[str, Any], tool_data: Dict[str, Any]):
        """从已记录的论文池里按 aspect 取子集，模拟真实检索的参数语义。

        匹配优先级：同 aspect 的池 → "*" 全域池；任意池仅作为标记回退。
        返回 None 表示无可用池（交给上层走 miss 分支）。
        """
        if not tool_data:
            return None

        aspect = str(args.get("aspect", "*"))
        # 容错：模型可能写成 "cs.AI" 而 schema 要求 "AI"
        norm_aspect = aspect.split(".")[-1] if aspect.startswith("cs.") else aspect

        def pool_of(entry) -> list:
            r = entry.get("result")
            return r if isinstance(r, list) else []

        exact, wildcard, anypool = None, None, None
        for entry in tool_data.values():
            a = str(entry.get("args", {}).get("aspect", "*"))
            if anypool is None:
                anypool = pool_of(entry)
            if a == "*":
                wildcard = pool_of(entry)
            if a == norm_aspect:
                exact = pool_of(entry)

        pool = exact if exact else (wildcard if wildcard else anypool)
        if not pool:
            return None

        try:
            max_results = int(args.get("max_results", 50))
        except (TypeError, ValueError):
            max_results = 50
        max_results = max(1, min(max_results, len(pool)))
        selected = copy.deepcopy(pool[:max_results])
        if exact is None and wildcard is None:
            selected[0]["_mock_env"] = {
                "offline_fallback": True,
                "message": "领域未命中离线快照；任意回退池不代表查询匹配结果。",
            }
        return selected

    @staticmethod
    def _limit_search_result(result: Any, args: Dict[str, Any]) -> Any:
        """Apply the requested limit to a cached keyword-search result pool."""
        if not isinstance(result, list):
            return result
        try:
            max_results = int(args.get("max_results", 10))
        except (TypeError, ValueError):
            max_results = 10
        return result[: max(1, min(max_results, len(result)))]

    @classmethod
    def _keyword_search_key(cls, args: Dict[str, Any]) -> str:
        """Key keyword-search snapshots by a normalized query hash."""
        query = " ".join(str((args or {}).get("query", "")).split()).casefold()
        stable = {"query_sha256": sha256(query.encode("utf-8")).hexdigest()}
        days = (args or {}).get("days")
        if days is not None:
            stable["days"] = days
        return json.dumps(stable, sort_keys=True, ensure_ascii=False)

    @classmethod
    def _keyword_search_fallback(
        cls, args: Dict[str, Any], tool_data: Dict[str, Any]
    ) -> Optional[list]:
        """Return an explicitly marked, deterministic subset for an unseen query."""
        pool = []
        seen_ids = set()
        for key in sorted(tool_data):
            result = tool_data[key].get("result")
            if not isinstance(result, list):
                continue
            for paper in result:
                if not isinstance(paper, dict):
                    continue
                paper_id = str(paper.get("id", ""))
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)
                pool.append(paper)
        if not pool:
            return None

        query = " ".join(str((args or {}).get("query", "")).split()).casefold()
        start = int.from_bytes(sha256(query.encode("utf-8")).digest()[:8], "big") % len(
            pool
        )
        ordered = pool[start:] + pool[:start]
        selected = cls._limit_search_result(ordered, args)
        if not selected:
            return None

        marked = [dict(paper) for paper in selected]
        marked[0]["_mock_env"] = {
            "offline_fallback": True,
            "message": "关键词未命中离线快照；返回固定回退子集，不代表查询匹配结果。",
        }
        return marked

    # ---------- 离线下载桩 ----------

    def reset_runtime_state(self) -> None:
        """Clear mutable replay state between independent benchmark trials."""
        self._offline_downloaded.clear()
        self.stats = {key: 0 for key in self.stats}

    def capture_state(self) -> Dict[str, Any]:
        """Capture mutable runtime state for a rollout sandbox.

        The immutable snapshot itself is deliberately not copied.  Only
        counters and per-rollout download markers can change while a policy is
        running and therefore need restoration.
        """
        return {
            "_offline_downloaded": copy.deepcopy(self._offline_downloaded),
            "stats": copy.deepcopy(self.stats),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore a state returned by :meth:`capture_state`."""
        if set(state) != {"_offline_downloaded", "stats"}:
            raise ValueError("invalid MockArxivEnv sandbox snapshot")
        self._offline_downloaded = copy.deepcopy(state["_offline_downloaded"])
        self.stats = copy.deepcopy(state["stats"])

    def _offline_download(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """不发 HTTP 的 download_arxiv_pdf 替身，返回契约与真实工具一致。

        真实工具会：解析 ref → 下载 → 写 PdfAsset。
        这里保留前后两步（让 cache_status / 复合任务依然有意义），只跳过网络下载，
        改为写一个占位文件。
        """
        from datetime import datetime

        from agentic_arxiv.state.artifact_paths import artifact_path
        from agentic_arxiv.state.schemas import PdfAsset
        from agentic_arxiv.state.store import store

        session_id = args.get("session_id", "default")
        ref = args.get("ref", 1)

        if ref is None:
            paper_id = store.get_last_active_paper_id(session_id)
            if not paper_id:
                raise ValueError(
                    "未找到指代对象：请先下载/翻译/查状态某篇论文，或明确提供 ref（序号/id/标题）"
                )
            paper = store.resolve_paper(session_id, paper_id)
        else:
            paper = store.resolve_paper(session_id, ref)
            if paper is None:
                raise ValueError(
                    "未找到论文：请先调用 /arxiv/recent 写入 session 记忆，或检查 ref 是否正确"
                )
            paper_id = paper.id

        pdf_url = (
            paper.pdf_url if paper else None
        ) or f"https://arxiv.org/pdf/{paper_id}.pdf"
        store.set_last_active_paper_id(session_id, paper_id)

        from agentic_arxiv.state.artifact_paths import safe_artifact_name

        safe_id = safe_artifact_name(str(paper_id))
        raw_root = artifact_path("pdf_raw_path")
        os.makedirs(raw_root, exist_ok=True)
        local_path = os.path.join(raw_root, f"{safe_id}.pdf")

        download_key = (str(session_id), str(paper_id))
        existed = download_key in self._offline_downloaded
        self._offline_downloaded.add(download_key)

        stub_bytes = b"%PDF-1.4\n% offline stub for RL rollout\n"
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            # 最小合法 PDF 头，避免下游把它当成空文件
            with open(local_path, "wb") as f:
                f.write(stub_bytes)

        # 对外报告 mock 内容的固定大小，不读取可能由旧实验或真实下载留下的文件大小。
        size_bytes = len(stub_bytes)
        fixed_time = datetime(2000, 1, 1)
        stored_asset = store.upsert_pdf_asset(
            PdfAsset(
                paper_id=paper_id,
                pdf_url=pdf_url,
                local_path=local_path,
                status="READY",
                size_bytes=size_bytes,
                downloaded_at=fixed_time,
                updated_at=fixed_time,
                source="offline_simulation",
            )
        )
        # MemoryStore.upsert_pdf_asset 会写入当前时间；离线回放必须固定它，
        # 否则 cache-status Observation 会随墙上时钟变化并改变模型后续输出。
        stored_asset.updated_at = fixed_time

        return {
            "session_id": session_id,
            "paper_id": paper_id,
            "pdf_url": pdf_url,
            "local_path": local_path,
            "status": "READY",
            "existed": existed,
            "size_bytes": size_bytes,
            "sha256": None,
        }

    # ---------- 快照读写 ----------

    @staticmethod
    def _make_key(args: Dict[str, Any]) -> str:
        """构造参数 key（剔除易变字段后按键排序，保证跨 session 可命中）"""
        stable = {
            k: v
            for k, v in (args or {}).items()
            if k not in _VOLATILE_ARG_KEYS and v is not None
        }
        return json.dumps(stable, sort_keys=True, ensure_ascii=False)


    @staticmethod
    def _paper_content_key(
        args: Dict[str, Any], resolved_paper_id: Optional[str] = None
    ) -> str:
        """Key paper reads by resolved paper id, not session-local ``ref``.

        ``ref=1`` can point to different papers after different searches, so a
        raw argument key would replay the wrong paper.  Resolve the reference
        against the current session before building the stable snapshot key.
        """
        session_id = str((args or {}).get("session_id") or "default")
        ref = (args or {}).get("ref", 1)
        if resolved_paper_id:
            paper_id = str(resolved_paper_id)
        else:
            paper = store.resolve_paper(session_id, ref)
            if paper is None:
                raise ValueError("Paper not found; search for the paper and check the ref.")
            paper_id = paper.id
        section = (args or {}).get("section")
        if isinstance(section, str):
            section = section.strip().lower() or None
        return json.dumps(
            {"paper_id": paper_id, "section": section},
            sort_keys=True, ensure_ascii=False,
        )

    @staticmethod
    def _paper_summary_key(
        args: Dict[str, Any], resolved_paper_id: Optional[str] = None
    ) -> str:
        """Key summaries by (paper, style, bucketed budget).

        ``style`` and ``max_words`` are normalised here rather than in the tool
        so that an invalid argument is rejected deterministically *before* the
        snapshot lookup.  Otherwise replay mode would report a missing-snapshot
        KeyError for what is really a malformed action, and the two failure
        modes would be indistinguishable in the reward.
        """
        from agentic_arxiv.tools.paper_summary_tool import bucket_max_words, normalize_style

        session_id = str((args or {}).get("session_id") or "default")
        ref = (args or {}).get("ref", 1)
        if resolved_paper_id:
            paper_id = str(resolved_paper_id)
        else:
            paper = store.resolve_paper(session_id, ref)
            if paper is None:
                raise ValueError("Paper not found; search for the paper and check the ref.")
            paper_id = paper.id

        return json.dumps(
            {
                "paper_id": paper_id,
                "style": normalize_style((args or {}).get("style")),
                "max_words": bucket_max_words((args or {}).get("max_words")),
            },
            sort_keys=True,
            ensure_ascii=False,
        )

    @staticmethod
    def _paper_figures_key(
        args: Dict[str, Any], resolved_paper_id: Optional[str] = None
    ) -> str:
        """Key figure extraction by resolved paper id (no other arguments)."""
        session_id = str((args or {}).get("session_id") or "default")
        ref = (args or {}).get("ref", 1)
        if resolved_paper_id:
            paper_id = str(resolved_paper_id)
        else:
            paper = store.resolve_paper(session_id, ref)
            if paper is None:
                raise ValueError("Paper not found; search for the paper and check the ref.")
            paper_id = paper.id
        return json.dumps({"paper_id": paper_id}, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _figure_analysis_key(
        args: Dict[str, Any], resolved_paper_id: Optional[str] = None
    ) -> str:
        """Key figure analyses by (paper, figure_no, question).

        ``figure_no`` and ``question`` are validated here rather than in the
        tool so a malformed action fails deterministically *before* the
        snapshot lookup — otherwise replay would report a missing-snapshot
        KeyError for what is really a bad argument, and the reward could not
        tell the two apart.
        """
        from agentic_arxiv.tools.figure_analysis_tool import normalize_question, validate_figure_no

        session_id = str((args or {}).get("session_id") or "default")
        ref = (args or {}).get("ref", 1)
        if resolved_paper_id:
            paper_id = str(resolved_paper_id)
        else:
            paper = store.resolve_paper(session_id, ref)
            if paper is None:
                raise ValueError("Paper not found; search for the paper and check the ref.")
            paper_id = paper.id

        return json.dumps(
            {
                "paper_id": paper_id,
                "figure_no": validate_figure_no((args or {}).get("figure_no", 1)),
                "question": normalize_question((args or {}).get("question")),
            },
            sort_keys=True,
            ensure_ascii=False,
        )

    @staticmethod
    def _translated_content_key(
        args: Dict[str, Any], resolved_paper_id: Optional[str] = None
    ) -> str:
        """Key translation reads by (resolved paper, page).

        ``page`` is validated before the lookup, in the same order as the live
        tool, so a malformed page is a deterministic tool error rather than a
        snapshot miss.
        """
        from agentic_arxiv.tools.translated_content_tool import validate_page

        page = validate_page((args or {}).get("page", 1))
        session_id = str((args or {}).get("session_id") or "default")
        ref = (args or {}).get("ref", 1)
        if resolved_paper_id:
            paper_id = str(resolved_paper_id)
        else:
            paper = store.resolve_paper(session_id, ref)
            if paper is None:
                raise ValueError("Paper not found; search for the paper and check the ref.")
            paper_id = paper.id
        return json.dumps(
            {"paper_id": paper_id, "page": page},
            sort_keys=True,
            ensure_ascii=False,
        )

    @staticmethod
    def _translated_page_out_of_range(
        key: str, tool_data: Dict[str, Any]
    ) -> Optional[ValueError]:
        """A page past a recorded translation's end is the live tool's range error.

        Every recorded page carries ``total_pages``, so this is certain no matter
        how much of the paper was recorded; it must not read as a snapshot gap.
        """
        wanted = json.loads(key)
        for recorded_key, entry in tool_data.items():
            if json.loads(recorded_key).get("paper_id") != wanted["paper_id"]:
                continue
            total = int((entry.get("result") or {}).get("total_pages") or 0)
            if wanted["page"] > total > 0:
                return ValueError(
                    f"page must be between 1 and {total}; got {wanted['page']}"
                )
            return None
        return None

    def _add_to_snapshot(
        self, tool_name: str, key: str, args: Dict[str, Any], result: Any
    ) -> None:
        """添加到快照（供下次回放使用）"""
        self.snapshot.setdefault(tool_name, {})[key] = {
            "args": {
                k: v for k, v in (args or {}).items() if k not in _VOLATILE_ARG_KEYS
            },
            "result": result,
        }

    def save_snapshot(self) -> None:
        """保存快照到文件"""
        if not self.snapshot_path:
            raise ValueError("snapshot_path not set")

        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.snapshot_path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot, f, ensure_ascii=False, indent=2)
        n = sum(len(v) for v in self.snapshot.values())
        log.info(f"[MockArxivEnv] 快照已保存: {self.snapshot_path} ({n} 条)")

    def describe(self) -> str:
        s = self.stats
        return (
            f"mode={self.mode} hit={s['hit']} miss={s['miss']} "
            f"real_calls={s['real_calls']} offline_stubs={s['offline_stubs']} "
            f"fallback={s['fallback']}"
        )
