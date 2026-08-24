# AgenticArxiv/benchmark/runner.py
"""Benchmark 运行器：驱动三种 Agent 执行标准化测试集。"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import glob

from utils.llm_client import get_env_llm_client, LLMClient
from utils.logger import log
from benchmark.tasks import get_task_by_id, get_dependency_chain, BENCHMARK_TASKS
from benchmark.metrics import TaskMetrics, extract_metrics
from config import settings as app_settings


@dataclass
class BenchmarkResult:
    task_id: str
    agent_type: str
    trial: int
    raw_result: Dict[str, Any]
    session_id: str = ""
    metrics: Optional[TaskMetrics] = None
    error: Optional[str] = None


class BenchmarkRunner:
    """对比测试三种 Agent 模式的性能和准确性"""

    AGENT_TYPES = ["regex", "mcp", "skill_cli"]

    def __init__(
        self,
        agent_types: Optional[List[str]] = None,
        repeat: int = 3,
        model: Optional[str] = None,
        session_prefix: Optional[str] = None,
        llm_extra: Optional[Dict[str, Any]] = None,
        offline: bool = False,
        snapshot: Optional[str] = None,
    ):
        self.agent_types = agent_types or self.AGENT_TYPES
        self.repeat = repeat
        self.model = model
        self.llm_extra = dict(llm_extra or {})
        self.offline = offline
        self.snapshot = snapshot
        self._env = None
        if session_prefix is None:
            from datetime import datetime
            session_prefix = f"bench_r{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_prefix = session_prefix
        self._llm_client: Optional[LLMClient] = None
        self._side_fx = None
        # 缓存已执行的依赖任务，避免重复
        self._dep_done: Dict[str, bool] = {}

    @property
    def llm_client(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = get_env_llm_client()
        return self._llm_client

    def run_all(self, tasks: Optional[List[Dict]] = None) -> List[BenchmarkResult]:
        tasks = tasks or BENCHMARK_TASKS
        results: List[BenchmarkResult] = []
        total = len(tasks) * len(self.agent_types) * self.repeat
        done = 0

        for task_def in tasks:
            for agent_type in self.agent_types:
                for trial in range(self.repeat):
                    done += 1
                    task_id = task_def["id"]
                    session_id = f"{self.session_prefix}_{task_id}_{agent_type}_{trial}"

                    log.info(f"[{done}/{total}] task={task_id} agent={agent_type} trial={trial}")
                    print(f"[{done}/{total}] task={task_id} agent={agent_type} trial={trial}", flush=True)

                    try:
                        # 确保依赖任务已执行
                        self._ensure_dependencies(task_def, session_id, agent_type)
                        self._apply_setup(task_def, session_id)

                        # 下载/翻译任务：清理上一次试验留下的文件和 DB 记录
                        if task_def.get("category") in ("download", "translate"):
                            self._cleanup_paper_artifacts(session_id)

                        agent = self._create_agent(
                            agent_type, task_def.get("max_iterations"))
                        raw = agent.run(
                            task=task_def["task"],
                            agent_model=self.model,
                            session_id=session_id,
                        )
                        metrics = extract_metrics(task_def, raw, agent_type, trial, session_id=session_id)
                        br = BenchmarkResult(
                            task_id=task_id,
                            agent_type=agent_type,
                            trial=trial,
                            raw_result=raw,
                            session_id=session_id,
                            metrics=metrics,
                        )
                    except Exception as e:
                        log.error(f"Benchmark 任务执行异常: {e}", exc_info=True)
                        br = BenchmarkResult(
                            task_id=task_id,
                            agent_type=agent_type,
                            trial=trial,
                            raw_result={},
                            session_id=session_id,
                            error=str(e),
                        )

                    results.append(br)
                    self._print_step_summary(br)

        return results

    def _side_effects(self):
        """Benchmark 沿用原行为（MySQL 落库 + SSE）；无数据库时自动降级到本地内存。

        离线模式必须用 LocalSideEffectManager：否则翻译任务仍会起线程调 pdf2zh、
        会话状态仍写 MySQL，"离线"就只离了一半。
        """
        if self._side_fx is None:
            if self.offline:
                from agents.side_effects import LocalSideEffectManager
                self._side_fx = LocalSideEffectManager()
            else:
                from agents.side_effects import default_side_effect_manager
                self._side_fx = default_side_effect_manager()
        return self._side_fx

    def _tool_env(self):
        """离线模式下用 MockArxivEnv 回放快照，取代真实 arXiv 请求。"""
        if not self.offline:
            return None
        if self._env is None:
            from pathlib import Path as _Path
            from rl.env import MockArxivEnv
            path = _Path(self.snapshot) if self.snapshot else (
                _Path(PROJECT_ROOT).parent / "data" / "mock_arxiv_snapshot.json")
            if not path.exists():
                raise SystemExit(
                    f"离线模式需要快照，但未找到: {path}\n"
                    f"请先运行: python -m AgenticArxiv.rl.build_snapshot"
                )
            self._env = MockArxivEnv(snapshot_path=path, mode="replay")
            log.info(f"[Benchmark] 离线模式，回放快照 {path}")
        return self._env

    def _create_agent(self, agent_type: str, max_iterations: Optional[int] = None):
        """按任务声明的轮数预算创建 Agent。

        Agent 默认 5 轮 = 最多 4 次工具调用 + 一次 FINISH。链更长的任务必须
        显式抬高，否则会被判成 FORCE_STOP —— 那是「预算不够」而不是「不会
        规划」，混在一起会让长链任务的失败率没法解读。
        """
        side_fx = self._side_effects()
        env = self._tool_env()
        kwargs = {"side_effect_mgr": side_fx, "env": env, "llm_extra": self.llm_extra}
        if max_iterations is not None:
            kwargs["max_iterations"] = max_iterations
        if agent_type == "mcp":
            from mcp_protocol.mcp_agent import MCPAgent
            return MCPAgent(self.llm_client, **kwargs)
        elif agent_type == "skill_cli":
            from skill_cli.skill_agent import SkillAgent
            return SkillAgent(self.llm_client, **kwargs)
        else:
            from agents.agent_engine import ReActAgent
            return ReActAgent(self.llm_client, **kwargs)

    @staticmethod
    def _cleanup_paper_artifacts(session_id: str):
        """清理 session 关联的 paper 下载/翻译状态，确保每次试验从干净状态开始"""
        from models.store import store
        papers = store.get_last_papers(session_id)
        if not papers:
            return
        for paper in papers:
            pid = paper.id
            # 清理 DB 记录
            store.delete_pdf_asset(pid)
            store.delete_translate_asset(pid)
            # 清理文件
            raw_pdf = os.path.join(app_settings.pdf_raw_path, f"{pid}.pdf")
            for f in [raw_pdf, raw_pdf + ".lock"]:
                if os.path.exists(f):
                    os.remove(f)
            for pattern in [f"{pid}-mono.pdf", f"{pid}-dual.pdf", f"{pid}-mono.pdf.lock"]:
                path = os.path.join(app_settings.pdf_translated_path, pattern)
                if os.path.exists(path):
                    os.remove(path)
            log_path = os.path.join(app_settings.pdf_translated_log_path, f"{pid}.pdf2zh.log")
            if os.path.exists(log_path):
                os.remove(log_path)

    def _apply_setup(self, task_def: Dict, session_id: str):
        """执行任务的 setup 动作，把会话状态铺好。

        与 depends_on 的区别：depends_on 会再跑一遍完整 Agent（含 LLM 调用），
        setup 直接调用工具铺状态。后者更适合"被测任务本身不该包含这些步骤"的场景 ——
        例如「下载标题含 X 的那篇」，论文列表应当是既有前提，
        而不是让被测 Agent 先检索一次（那样 expected_tools 就必须包含检索，
        ref 参数的正确性反而测不出来了）。
        """
        setup = task_def.get("setup")
        if not setup:
            return

        from models.schemas import Paper
        from tools.tool_registry import registry

        env = self._tool_env()
        side_fx = self._side_effects()
        for action in setup:
            args = dict(action.get("args") or {})
            args["session_id"] = session_id
            name = action["name"]
            if name == "translate_arxiv_pdf":
                side_fx.enqueue_translate(**args)
                continue
            result = env.execute_tool(name, args) if env else registry.execute_tool(name, args)
            if name == "get_recently_submitted_cs_papers" and isinstance(result, list) and result:
                side_fx.set_last_papers(session_id, [Paper(**p) for p in result])
            if isinstance(result, dict) and isinstance(result.get("paper_id"), str):
                side_fx.set_last_active_paper_id(session_id, result["paper_id"])
        log.info(f"  已铺设会话状态: {[a['name'] for a in setup]}")

    def _ensure_dependencies(self, task_def: Dict, session_id: str, agent_type: str):
        """如果任务有依赖，先执行依赖任务确保上下文（论文列表等）存在"""
        dep_id = task_def.get("depends_on")
        if not dep_id:
            return

        chain = get_dependency_chain(task_def["id"])
        # chain 包含从最早依赖到当前任务，去掉当前任务
        chain = chain[:-1]

        for dep_task_id in chain:
            dep_key = f"{dep_task_id}_{session_id}"
            if dep_key in self._dep_done:
                continue

            dep_task = get_task_by_id(dep_task_id)
            if dep_task is None:
                continue

            log.info(f"  执行依赖任务: {dep_task_id} (for {task_def['id']})")
            agent = self._create_agent(agent_type)
            agent.run(
                task=dep_task["task"],
                agent_model=self.model,
                session_id=session_id,
            )
            self._dep_done[dep_key] = True

    @staticmethod
    def _print_step_summary(br: BenchmarkResult):
        if br.error:
            print(f"  ERROR: {br.error[:100]}")
            return
        m = br.metrics
        if m is None:
            return
        status = "PASS" if m.task_completed else f"FAIL({m.termination_type})"
        tools = " -> ".join(m.tool_call_sequence) or "(none)"
        print(
            f"  {status} | {m.total_time_ms}ms "
            f"(LLM:{m.total_llm_ms} Tool:{m.total_tool_ms} OH:{m.framework_overhead_ms}) "
            f"| iter={m.iteration_count} | tools=[{tools}] "
            f"| accurate={m.tool_call_accurate}"
        )
