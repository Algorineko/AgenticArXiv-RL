"""普通推理、专家生成与 GRPO 共用的每轨迹工具执行环境。"""

from __future__ import annotations

import copy
import inspect
import json
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Optional, Sequence

from agentic_arxiv.config import workspace_path
from agentic_arxiv.state.schemas import Paper, PdfAsset, TranslateAsset
from agentic_arxiv.state.store import bind_store
from agentic_arxiv.state.store_memory import MemoryStore
from agentic_arxiv.runtime.env import MockArxivEnv
from agentic_arxiv.runtime.sandbox import RolloutSandbox
from agentic_arxiv.tools.bootstrap import require_all_tools
from agentic_arxiv.tools.tool_registry import registry
from agentic_arxiv.state.artifact_paths import bind_artifact_paths, search_output_path

_SEARCH_TOOLS = {"get_recently_submitted_cs_papers", "search_arxiv_papers"}
_PAPER_TOOLS = {
    "download_arxiv_pdf", "translate_arxiv_pdf", "get_paper_cache_status",
    "get_paper_content", "summarize_paper", "extract_paper_figures",
    "analyze_figure", "get_translated_content",
}


def _value_matches_schema(value: Any, spec: Mapping[str, Any]) -> bool:
    """覆盖现有十工具 schema 的参数边界，不扩展另一套注册机制。"""
    if isinstance(spec.get("anyOf"), list):
        return any(_value_matches_schema(value, item) for item in spec["anyOf"])
    expected = spec.get("type")
    if isinstance(expected, list):
        return any(_value_matches_schema(value, {**spec, "type": item}) for item in expected)
    checks = {
        "null": lambda x: x is None,
        "boolean": lambda x: isinstance(x, bool),
        "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
        "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
        "string": lambda x: isinstance(x, str),
        "object": lambda x: isinstance(x, dict),
        "array": lambda x: isinstance(x, list),
    }
    if expected in checks and not checks[expected](value):
        return False
    if "enum" in spec and value not in spec["enum"]:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            return False
        if "maximum" in spec and value > spec["maximum"]:
            return False
    return not isinstance(value, str) or len(value) >= int(spec.get("minLength", 0))


class AgenticArxivMultiTurnEnv:
    """每个 episode 独有 store、产物根和后端可变状态；快照只读共享。"""

    def __init__(self, snapshot_path: Optional[Path] = None, *, mode: Optional[str] = None,
                 backend: Any = None):
        require_all_tools("ReAct episode")
        path = workspace_path(snapshot_path) if snapshot_path else None
        self.mode = mode or getattr(backend, "mode", None) or ("replay" if path else "live")
        if self.mode not in {"replay", "live", "record"}:
            raise ValueError("episode mode 必须为 replay、live 或 record")
        if self.mode == "replay" and backend is None and path is None:
            raise FileNotFoundError("离线 episode 必须指定已有快照或显式回放后端")
        backend_path = path or getattr(backend, "snapshot_path", None)
        backend_path = workspace_path(backend_path) if backend_path else None
        if self.mode == "replay" and backend_path and not backend_path.is_file():
            raise FileNotFoundError(f"离线 episode 快照不存在: {backend_path}")
        if backend is None:
            self.backend = MockArxivEnv(path, mode=self.mode, offline_download=self.mode == "replay")
        elif isinstance(backend, MockArxivEnv):
            self.backend = copy.copy(backend)
            self.backend.mode = self.mode
            self.backend.offline_download = self.mode == "replay"
            self.backend.stats = dict(backend.stats)
            self.backend._offline_downloaded = set()
            if self.mode == "record":
                self.backend.snapshot = copy.deepcopy(backend.snapshot)
        else:
            self.backend = backend
        self.store = MemoryStore(make_dirs=False)
        self.session_id = ""
        self.setup_actions: list[dict[str, Any]] = []
        self.setup_context = ""
        self._closed = True
        self._allocate_artifacts()

    def _allocate_artifacts(self) -> None:
        self._temporary = TemporaryDirectory(prefix="aa-episode-")
        root = Path(self._temporary.name)
        self.artifact_paths = {name: root / name for name in (
            "pdf_raw_path", "pdf_translated_path", "pdf_translated_log_path",
            "figures_path", "search_path",
        )}
        for path in self.artifact_paths.values():
            path.mkdir()
        self._sandbox = RolloutSandbox(file_roots=self._artifact_roots())
        self._closed = False

    def _artifact_roots(self) -> tuple[Path, ...]:
        return tuple(self.artifact_paths.values())

    @contextmanager
    def bound(self):
        if self._closed:
            raise RuntimeError("episode 已关闭，请先 reset")
        with bind_store(self.store), bind_artifact_paths(self.artifact_paths):
            yield self

    def capture_state(self) -> dict[str, Any]:
        return {"store": self.store.capture_state(), "session_id": self.session_id,
                "setup_actions": copy.deepcopy(self.setup_actions), "setup_context": self.setup_context}

    def restore_state(self, state: dict[str, Any]) -> None:
        self.store.restore_state(state["store"])
        self.session_id = state["session_id"]
        self.setup_actions = copy.deepcopy(state["setup_actions"])
        self.setup_context = state["setup_context"]

    def reset(self, initialization_plan: Optional[Sequence[Mapping[str, Any]]] = None, *,
              session_id: Optional[str] = None) -> str:
        if self._closed:
            self._allocate_artifacts()
        self._sandbox.reset()
        self.store.reset()
        reset_backend = getattr(self.backend, "reset_runtime_state", None)
        if callable(reset_backend):
            reset_backend()
        self.session_id = session_id or f"episode_{uuid.uuid4().hex}"
        self.setup_actions = []
        self.setup_context = ""
        if initialization_plan is not None:
            return self.setup(initialization_plan)
        return ""

    def reset_runtime_state(self) -> None:
        self.reset()

    def setup(self, initialization_plan: Sequence[Mapping[str, Any]]) -> str:
        """按顺序执行明确初始化动作，不读取任务或把它们计入策略轨迹。"""
        for prepared in initialization_plan:
            action = {"name": prepared["name"], "args": dict(prepared.get("args") or {})}
            self.step(action)
            self.setup_actions.append({"name": action["name"], "args": {
                key: value for key, value in action["args"].items() if key != "session_id"
            }})
        self.setup_context = self.visible_state()
        return self.setup_context

    def visible_state(self) -> str:
        """仅显示真实候选和缓存事实，绝不从 task gold/评分字段构造上下文。"""
        papers = self.store.get_last_papers(self.session_id)
        active = self.store.get_last_active_paper_id(self.session_id)
        if not papers and not active:
            return ""
        state = {"papers": [{"ref": index, "id": paper.id, "title": paper.title}
                            for index, paper in enumerate(papers, 1)],
                 "last_active_paper_id": active, "cache": []}
        ids = list(dict.fromkeys([paper.id for paper in papers] + ([active] if active else [])))
        for paper_id in ids:
            pdf = self.store.get_pdf_asset(paper_id)
            translated = self.store.get_translate_asset(paper_id)
            if pdf or translated:
                state["cache"].append({"paper_id": paper_id,
                    "pdf_status": pdf.status if pdf else None,
                    "translate_status": translated.status if translated else None,
                    "source": (translated.source if translated else pdf.source)})
        return "当前会话状态（环境数据）：\n" + json.dumps(state, ensure_ascii=False, sort_keys=True)

    def step(self, action: Mapping[str, Any]) -> Any:
        if not isinstance(action, Mapping) or not isinstance(action.get("name"), str):
            raise ValueError("工具动作必须包含 name 和对象 args")
        name, raw_args = action["name"], action.get("args")
        if not isinstance(raw_args, dict):
            raise ValueError("工具 args 必须为 JSON 对象")
        if "session_id" in raw_args or any(key.startswith("_") for key in raw_args):
            raise ValueError("动作不得携带内部环境参数")
        args = dict(raw_args)
        if name == "search_arxiv_papers" and not str(args.get("query") or "").strip():
            raise ValueError("工具参数错误: query 不能为空")
        tool = registry.get_tool(name)
        if tool is None:
            raise ValueError(f"工具 '{name}' 未注册")
        schema = tool.get("parameters") or {}
        properties = schema.get("properties") or {}
        if properties:
            unknown = set(args) - set(properties)
            if unknown:
                raise ValueError(f"工具参数错误: 未知参数 {sorted(unknown)}")
            for required in schema.get("required") or []:
                if required == "session_id":
                    continue
                if required not in args:
                    raise ValueError(f"工具参数错误: 缺少参数 {required}")
            for key, value in args.items():
                if not _value_matches_schema(value, properties[key]):
                    raise ValueError(f"工具参数错误: {key} 的值无效")
        signature = inspect.signature(tool["func"])
        call_args = dict(args)
        if "session_id" in signature.parameters or name in _SEARCH_TOOLS:
            call_args["session_id"] = self.session_id
        bound_args = {key: value for key, value in call_args.items()
                      if key != "session_id" or "session_id" in signature.parameters}
        try:
            signature.bind(**bound_args)
        except TypeError as exc:
            raise ValueError(f"工具参数错误: {exc}") from exc
        with self.bound():
            if name == "get_recently_submitted_cs_papers" and args.get("save_to_file", True):
                search_output_path(args.get("output_path"), "")
            target = args.get("paper_id") or args.get("ref", None if name in {
                "translate_arxiv_pdf", "get_paper_cache_status"} else 1)
            paper = self.store.resolve_paper(self.session_id, target) if name in _PAPER_TOOLS else None
            if paper:
                self.store.set_last_active_paper_id(self.session_id, paper.id)
            if self.mode == "replay" and name in {"download_arxiv_pdf", "translate_arxiv_pdf"}:
                result = self._simulate_asset(name, args, paper)
            elif name == "get_paper_cache_status":
                result = registry.execute_tool(name, bound_args)
            else:
                result = self.backend.execute_tool(name, call_args)
            if result is None:
                raise RuntimeError(f"工具 {name} 未返回执行结果，不能记录为成功")
            if isinstance(result, list) and name in _SEARCH_TOOLS:
                if any(isinstance(item, dict) and item.get("_mock_env", {}).get("offline_fallback")
                       for item in result):
                    raise ValueError("关键词未命中离线快照；回退子集不是查询结果，候选状态未更新")
                papers = [Paper(**{key: value for key, value in item.items() if not key.startswith("_")})
                          if isinstance(item, dict) else item for item in result]
                self.store.set_last_papers(self.session_id, papers)
            if isinstance(result, dict) and result.get("paper_id"):
                paper_id = str(result["paper_id"])
                self.store.set_last_active_paper_id(self.session_id, paper_id)
                if self.mode == "replay" and name == "get_translated_content":
                    if self.store.get_translate_asset(paper_id) is None:
                        fixed = datetime(2000, 1, 1)
                        asset = self.store.upsert_translate_asset(TranslateAsset(
                            paper_id=paper_id, status="READY", translated_at=fixed,
                            source="offline_simulation"))
                        asset.updated_at = fixed
                if name == "download_arxiv_pdf" and result.get("status") == "READY":
                    if self.store.get_pdf_asset(paper_id) is None:
                        self.store.upsert_pdf_asset(PdfAsset(paper_id=paper_id,
                            pdf_url=result.get("pdf_url") or "", local_path=result.get("local_path") or "",
                            status="READY", size_bytes=result.get("size_bytes") or 0,
                            sha256=result.get("sha256"), source=result.get("source")))
                if name == "translate_arxiv_pdf" and result.get("status") == "READY":
                    if self.store.get_translate_asset(paper_id) is None:
                        self.store.upsert_translate_asset(TranslateAsset(paper_id=paper_id,
                            input_pdf_path=result.get("input_pdf_path") or "",
                            output_mono_path=result.get("output_pdf_path") or "", status="READY",
                            service=result.get("service"), threads=result.get("threads") or 0,
                            source=result.get("source")))
            return result

    def _simulate_asset(self, name: str, args: Mapping[str, Any], paper: Optional[Paper]) -> dict[str, Any]:
        paper_id = paper.id if paper else args.get("paper_id")
        input_path = args.get("input_pdf_path")
        if name == "translate_arxiv_pdf" and paper is None:
            if input_path:
                input_path = workspace_path(input_path)
                if not input_path.is_file():
                    raise FileNotFoundError(f"离线翻译指定的本地输入不存在: {input_path}")
                paper_id = paper_id or input_path.stem
            elif paper_id:
                # 显式 ID 可以绕过候选会话，但离线完成必须有已有资产或记录依据。
                recorded = self.store.get_pdf_asset(str(paper_id)) is not None
                for entries in getattr(self.backend, "snapshot", {}).values():
                    for entry in entries.values():
                        value = entry.get("result")
                        objects = value if isinstance(value, list) else [value]
                        recorded = recorded or any(isinstance(item, dict) and
                            str(item.get("paper_id") or item.get("id")) == str(paper_id)
                            for item in objects)
                if not recorded:
                    raise KeyError("离线翻译的 paper_id 没有已有资产或快照记录")
        if not paper_id:
            raise ValueError("未找到论文；请先搜索，再按序号、ID、标题或最近操作对象引用")
        paper_id = str(paper_id)
        pdf = self.store.get_pdf_asset(paper_id)
        translated = self.store.get_translate_asset(paper_id)
        force = bool(args.get("force", False))
        fixed = datetime(2000, 1, 1)
        asset = self.store.upsert_pdf_asset(PdfAsset(paper_id=paper_id,
            pdf_url=(paper.pdf_url if paper else args.get("pdf_url")) or "",
            status="READY", source="offline_simulation",
            downloaded_at=fixed))
        asset.updated_at = fixed
        result = {"paper_id": paper_id, "status": "READY", "offline": True,
                  "source": "offline_simulation", "force": force,
                  "existed": bool((translated if name == "translate_arxiv_pdf" else pdf) and not force)}
        if name == "download_arxiv_pdf":
            result.update(pdf_url=paper.pdf_url, local_path="", size_bytes=0, sha256=None)
        else:
            asset = self.store.upsert_translate_asset(TranslateAsset(paper_id=paper_id,
                status="READY", service=args.get("service"), threads=args.get("threads") or 0,
                translated_at=fixed, source="offline_simulation"))
            asset.updated_at = fixed
            result.update(service=args.get("service"), threads=args.get("threads"),
                keep_dual=bool(args.get("keep_dual", False)), output_pdf_path="",
                pdf_url=args.get("pdf_url"), input_pdf_path=args.get("input_pdf_path"))
        return result

    def execute_tool(self, name: str, args: dict[str, Any]) -> Any:
        return self.step({"name": name, "args": args})

    def close(self) -> None:
        if not self._closed:
            self._temporary.cleanup()
            self.store.reset()
            self._closed = True

    def get_recently_submitted_cs_papers(self, aspect="*", days=7, max_results=50,
                                         output_path=None, save_to_file=True):
        args = {"aspect": aspect, "days": days, "max_results": max_results, "save_to_file": save_to_file}
        if output_path is not None:
            args["output_path"] = output_path
        return self.execute_tool("get_recently_submitted_cs_papers", args)

    def search_arxiv_papers(self, query, max_results=10, days=None):
        return self.execute_tool("search_arxiv_papers", {"query": query, "max_results": max_results, "days": days})

    def download_arxiv_pdf(self, ref=1, force=False):
        return self.execute_tool("download_arxiv_pdf", {"ref": ref, "force": force})

    def get_paper_content(self, ref=1, section=None):
        return self.execute_tool("get_paper_content", {"ref": ref, "section": section})

    def summarize_paper(self, ref=1, style=None, max_words=None):
        args = {"ref": ref}
        if style is not None:
            args["style"] = style
        if max_words is not None:
            args["max_words"] = max_words
        return self.execute_tool("summarize_paper", args)

    def extract_paper_figures(self, ref=1):
        return self.execute_tool("extract_paper_figures", {"ref": ref})

    def analyze_figure(self, ref=1, figure_no=1, question=None):
        args = {"ref": ref, "figure_no": figure_no}
        if question is not None:
            args["question"] = question
        return self.execute_tool("analyze_figure", args)

    def get_translated_content(self, ref=1, page=1):
        return self.execute_tool("get_translated_content", {"ref": ref, "page": page})

    def translate_arxiv_pdf(self, ref=None, force=False, service=None, threads=None,
                            keep_dual=False, paper_id=None, pdf_url=None, input_pdf_path=None):
        args = {"ref": ref, "force": force, "keep_dual": keep_dual}
        args.update({key: value for key, value in {"service": service, "threads": threads,
            "paper_id": paper_id, "pdf_url": pdf_url, "input_pdf_path": input_pdf_path}.items() if value is not None})
        return self.execute_tool("translate_arxiv_pdf", args)

    def get_paper_cache_status(self, ref=None, paper_id=None):
        args = {"ref": ref}
        if paper_id is not None:
            args["paper_id"] = paper_id
        return self.execute_tool("get_paper_cache_status", args)


def make_environment_factory(snapshot_path: Path):
    path = workspace_path(snapshot_path)
    if not path.is_file():
        raise FileNotFoundError(f"多轮 GRPO 需要已有离线快照: {path}")
    return lambda: AgenticArxivMultiTurnEnv(path, mode="replay")
