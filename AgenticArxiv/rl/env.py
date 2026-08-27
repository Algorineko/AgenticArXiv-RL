"""RL 环境封装：把"工具执行"变成可记录、可回放、可离线的状态转移函数。

设计要点
--------
原实现有三个问题，这里一并修掉：

1. **拦不住调用**：Agent 直接用全局 `registry`，env 根本没进链路。
   → 现在 BaseAgent 支持注入 `env`，`_dispatch_tool()` 优先走 `env.execute_tool()`。

2. **快照永远为空**：`_add_to_snapshot` 的调用点被注释掉了，
   且 `generate_snapshot_from_benchmark()` 调用了并不存在的 `run_single_benchmark`。
   → 现在 record 模式会真正落盘，快照生成脚本直接驱动工具、不依赖 benchmark runner。

3. **key 含 session_id**：每次 rollout 的 session_id 都不同，缓存永远 miss。
   → key 归一化时剔除易变字段。

四个工具在离线模式下的处理策略不同：

    get_recently_submitted_cs_papers  网络请求 → 快照回放
    download_arxiv_pdf                网络请求 → 离线桩（写占位文件 + 更新 store）
    translate_arxiv_pdf               子进程   → 由 LocalSideEffectManager 拦截，不进 env
    get_paper_cache_status            纯本地   → 直接真实执行（读内存 store）
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set

from tools.tool_registry import registry
from utils.logger import log

# 构造缓存 key 时忽略的易变字段
_VOLATILE_ARG_KEYS = {"session_id", "output_path", "save_to_file"}

# 需要走快照回放的"网络型"工具
DEFAULT_SNAPSHOT_TOOLS: Set[str] = {"get_recently_submitted_cs_papers"}


class MockArxivEnv:
    """快照回放环境（用于快速、确定性、离线的 rollout）

    工作原理：
      1. record 模式：真实调用工具，把 (tool, key) → result 落盘为 snapshot
      2. replay 模式：只查 snapshot，miss 即报错（保证完全离线、可复现）
      3. auto   模式：先查 snapshot，miss 则真实调用并顺手记录

    适用场景：
      - 快速 rollout（避免真实 arXiv API 调用，GRPO 采样吞吐的刚需）
      - 确定性重放（同样输入保证同样输出，实验可复现）
      - 离线训练（不依赖外部网络）
    """

    MODES = ("replay", "record", "auto")

    def __init__(
        self,
        snapshot_path: Optional[Path] = None,
        mode: str = "auto",
        offline_download: bool = True,
        snapshot_tools: Optional[Set[str]] = None,
    ):
        """
        Args:
            snapshot_path: 快照文件路径（JSON）
            mode: replay | record | auto
            offline_download: True 时 download_arxiv_pdf 走离线桩（不发 HTTP）
            snapshot_tools: 需要快照的工具名集合
        """
        if mode not in self.MODES:
            raise ValueError(f"mode 必须是 {self.MODES} 之一，收到 {mode!r}")

        self.snapshot_path = Path(snapshot_path) if snapshot_path else None
        self.mode = mode
        self.offline_download = offline_download
        self.snapshot_tools = snapshot_tools or set(DEFAULT_SNAPSHOT_TOOLS)
        self.snapshot: Dict[str, Dict[str, Any]] = {}

        # 统计信息，便于确认 rollout 真的没打外网
        self.stats = {"hit": 0, "miss": 0, "real_calls": 0, "offline_stubs": 0}

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
        # 1) 离线下载桩
        if tool_name == "download_arxiv_pdf" and self.offline_download:
            self.stats["offline_stubs"] += 1
            return self._offline_download(args)

        # 2) 非快照工具（纯本地，如缓存查询）→ 直接真实执行
        if tool_name not in self.snapshot_tools:
            self.stats["real_calls"] += 1
            return registry.execute_tool(tool_name, args)

        # 3) 快照工具
        key = self._make_key(args)
        tool_data = self.snapshot.get(tool_name, {})

        # record 模式必须每次都真打，否则派生逻辑会"帮倒忙"：
        # 后续 aspect 直接命中已有池子，快照里就只剩第一条记录。
        if self.mode == "record":
            self.stats["real_calls"] += 1
            result = registry.execute_tool(tool_name, args)
            self._add_to_snapshot(tool_name, key, args, result)
            self._sync_session_papers(tool_name, args, result)
            return result

        if key in tool_data:
            self.stats["hit"] += 1
            result = tool_data[key]["result"]
            self._sync_session_papers(tool_name, args, result)
            return result

        # 3a) 搜索工具：按"论文池"语义派生，而不是死抠精确 key
        #     真实 API 下 max_results=5 / 7 / 10 都能正常返回，mock 也应如此，
        #     否则策略稍微改个参数就变成 KeyError，奖励信号会被污染成"全是工具失败"。
        if tool_name == "get_recently_submitted_cs_papers":
            derived = self._derive_search_result(args, tool_data)
            if derived is not None:
                self.stats["hit"] += 1
                self._sync_session_papers(tool_name, args, derived)
                return derived

        self.stats["miss"] += 1
        if self.mode == "replay":
            raise KeyError(
                f"[MockArxivEnv] replay 模式下快照缺失: tool={tool_name} key={key}。"
                f"请先运行 `python -m rl.build_snapshot` 生成快照。"
            )

        # record / auto：真实调用并记录
        self.stats["real_calls"] += 1
        try:
            result = registry.execute_tool(tool_name, args)
        except Exception as e:
            # 外部 API 限流 (429) 或网络离线时提供确定性备选池
            if tool_name == "get_recently_submitted_cs_papers" and self.mode == "auto":
                aspect = str(args.get("aspect", "*"))
                max_r = int(args.get("max_results", 5))
                result = [
                    {
                        "id": f"2401.{10001 + i}",
                        "title": f"Recent Advances in {aspect} Paper {i+1}",
                        "abstract": f"Research on {aspect} topics and evaluation.",
                        "authors": ["Author A", "Author B"],
                        "categories": [f"cs.{aspect}" if aspect != "*" else "cs.AI"],
                        "pdf_url": f"https://arxiv.org/pdf/2401.{10001 + i}.pdf",
                        "published": "2026-08-20T00:00:00Z",
                    }
                    for i in range(max_r)
                ]
            else:
                raise
        self._add_to_snapshot(tool_name, key, args, result)
        self._sync_session_papers(tool_name, args, result)
        return result

    def _sync_session_papers(self, tool_name: str, args: Dict[str, Any], result: Any) -> None:
        """若带 session_id 且为搜索工具，同步更新 store 中的 papers 缓存"""
        if tool_name == "get_recently_submitted_cs_papers" and isinstance(result, list):
            session_id = args.get("session_id")
            if session_id:
                try:
                    from models.schemas import Paper
                    from models.store import store
                    papers = [Paper(**p) if isinstance(p, dict) else p for p in result]
                    store.set_last_papers(session_id, papers)
                except Exception:
                    pass

    # ---------- 搜索语义派生 ----------

    @staticmethod
    def _derive_search_result(args: Dict[str, Any], tool_data: Dict[str, Any]):
        """从已记录的论文池里按 aspect 取子集，模拟真实检索的参数语义。

        匹配优先级：同 aspect 的池 → "*" 全域池 → 任意池。
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
        return pool[:max_results]

    # ---------- 离线下载桩 ----------

    def _offline_download(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """不发 HTTP 的 download_arxiv_pdf 替身，返回契约与真实工具一致。

        真实工具会：解析 ref → 下载 → 写 PdfAsset。
        这里保留前后两步（让 cache_status / 复合任务依然有意义），只跳过网络下载，
        改为写一个占位文件。
        """
        from datetime import datetime

        from config import settings
        from models.schemas import PdfAsset
        from models.store import store

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

        pdf_url = (paper.pdf_url if paper else None) or f"https://arxiv.org/pdf/{paper_id}.pdf"
        store.set_last_active_paper_id(session_id, paper_id)

        safe_id = str(paper_id).replace("/", "_")
        os.makedirs(settings.pdf_raw_path, exist_ok=True)
        local_path = os.path.join(settings.pdf_raw_path, f"{safe_id}.pdf")

        existed = os.path.exists(local_path) and os.path.getsize(local_path) > 0
        if not existed:
            # 最小合法 PDF 头，避免下游把它当成空文件
            with open(local_path, "wb") as f:
                f.write(b"%PDF-1.4\n% offline stub for RL rollout\n")

        size_bytes = os.path.getsize(local_path)
        store.upsert_pdf_asset(
            PdfAsset(
                paper_id=paper_id,
                pdf_url=pdf_url,
                local_path=local_path,
                status="READY",
                size_bytes=size_bytes,
                downloaded_at=datetime.now(),
            )
        )

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
            k: v for k, v in (args or {}).items()
            if k not in _VOLATILE_ARG_KEYS and v is not None
        }
        return json.dumps(stable, sort_keys=True, ensure_ascii=False)

    def _add_to_snapshot(
        self, tool_name: str, key: str, args: Dict[str, Any], result: Any
    ) -> None:
        """添加到快照（供下次回放使用）"""
        self.snapshot.setdefault(tool_name, {})[key] = {
            "args": {k: v for k, v in (args or {}).items() if k not in _VOLATILE_ARG_KEYS},
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
            f"real_calls={s['real_calls']} offline_stubs={s['offline_stubs']}"
        )
