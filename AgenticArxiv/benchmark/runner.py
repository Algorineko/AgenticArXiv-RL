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
    ):
        self.agent_types = agent_types or self.AGENT_TYPES
        self.repeat = repeat
        self.model = model
        if session_prefix is None:
            from datetime import datetime
            session_prefix = f"bench_r{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_prefix = session_prefix
        self._llm_client: Optional[LLMClient] = None
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

                        # 下载/翻译任务：清理上一次试验留下的文件和 DB 记录
                        if task_def.get("category") in ("download", "translate"):
                            self._cleanup_paper_artifacts(session_id)

                        agent = self._create_agent(agent_type)
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

    def _create_agent(self, agent_type: str):
        if agent_type == "mcp":
            from mcp_protocol.mcp_agent import MCPAgent
            return MCPAgent(self.llm_client)
        elif agent_type == "skill_cli":
            from skill_cli.skill_agent import SkillAgent
            return SkillAgent(self.llm_client)
        else:
            from agents.agent_engine import ReActAgent
            return ReActAgent(self.llm_client)

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
