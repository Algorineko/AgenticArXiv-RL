"""Benchmark 运行器：使用共同 episode 执行文本 ReAct 评测。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agentic_arxiv.config import workspace_path
from agentic_arxiv.evaluation.episode import EvaluationCancelled, execute_episode
from agentic_arxiv.runtime.agent_engine import ReActAgent
from agentic_arxiv.scoring.metrics import TaskMetrics, extract_metrics, is_strict_success
from agentic_arxiv.tasks.tasks import BENCHMARK_TASKS
from agentic_arxiv.tasks.initialization import compile_initialization
from agentic_arxiv.runtime.multiturn_env import AgenticArxivMultiTurnEnv
from agentic_arxiv.llm.client import TransformersLLMClient, get_env_llm_client
from loguru import logger as log


@dataclass
class BenchmarkResult:
    task_id: str
    agent_type: str
    trial: int
    raw_result: Dict[str, Any]
    session_id: str = ""
    metrics: Optional[TaskMetrics] = None
    error: Optional[str] = None
    expected_paper_ids: Optional[List[Optional[str]]] = None


class BenchmarkRunner:
    """评测文本 ReAct；每个 task/trial 拥有独立状态和产物。"""

    AGENT_TYPES = ["regex"]
    LLM_BACKENDS = ("api", "transformers")

    def __init__(
        self, agent_types: Optional[List[str]] = None, repeat: int = 3,
        model: Optional[str] = None, session_prefix: Optional[str] = None,
        llm_extra: Optional[Dict[str, Any]] = None, offline: bool = False,
        snapshot: Optional[str] = None, llm_backend: str = "api",
        local_device: str = "auto", local_dtype: str = "auto",
        generation_seed: Optional[int] = None,
        cancel_requested: Callable[[], bool] | None = None,
        model_artifact: bool = False,
    ):
        if llm_backend not in self.LLM_BACKENDS:
            raise ValueError(f"llm_backend 必须是 {self.LLM_BACKENDS} 之一")
        self.agent_types = agent_types or list(self.AGENT_TYPES)
        if self.agent_types != self.AGENT_TYPES:
            raise ValueError("仅支持文本 ReAct；agent_types 必须为 ['regex']")
        if repeat < 1:
            raise ValueError("repeat 必须至少为 1")
        if snapshot and not offline:
            raise ValueError("snapshot 仅可用于显式 offline 模式")
        self.repeat = repeat
        self.model = model
        self.llm_extra = dict(llm_extra or {})
        self.offline = offline
        self.snapshot = snapshot
        self.llm_backend = llm_backend
        self.local_device = local_device
        self.local_dtype = local_dtype
        self.generation_seed = generation_seed
        self.cancel_requested = cancel_requested
        self.model_artifact = model_artifact
        if session_prefix is None:
            from datetime import datetime
            session_prefix = f"bench_r{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_prefix = session_prefix
        self._llm_client: Optional[Any] = None

    @property
    def llm_client(self) -> Any:
        if self._llm_client is None:
            if self.llm_backend == "transformers":
                if not self.model:
                    raise ValueError("transformers 后端需要本地模型路径")
                if not Path(self.model).exists():
                    raise FileNotFoundError(f"本地模型不存在: {self.model}")
                artifact_kwargs = {}
                if self.model_artifact:
                    from agentic_arxiv.runtime.react_contract import protocol_metadata
                    artifact_kwargs = {"artifact": True, "expected_protocol": protocol_metadata()}
                self._llm_client = TransformersLLMClient(
                    model=self.model, device=self.local_device,
                    dtype=self.local_dtype, seed=self.generation_seed,
                    **artifact_kwargs,
                )
            else:
                self._llm_client = get_env_llm_client()
        return self._llm_client

    def _snapshot_path(self) -> Path:
        return workspace_path(self.snapshot or "data/mock_arxiv_snapshot.json")

    def _tool_env(self) -> AgenticArxivMultiTurnEnv:
        if self.offline:
            path = self._snapshot_path()
            if not path.exists():
                raise FileNotFoundError(f"离线模式需要已有快照: {path}")
            return AgenticArxivMultiTurnEnv(path, mode="replay")
        return AgenticArxivMultiTurnEnv(mode="live")

    def _create_agent(self, agent_type: str, max_iterations: Optional[int] = None):
        if agent_type != "regex":
            raise ValueError("仅支持文本 ReAct")
        return ReActAgent(
            self.llm_client, env=self._tool_env(), llm_extra=self.llm_extra,
            max_iterations=max_iterations if max_iterations is not None else 6,
        )

    def _start_generation_stream(self, task_id: str, trial: int) -> None:
        reset = getattr(self.llm_client, "start_generation_stream", None)
        if callable(reset):
            reset(f"{task_id}:{trial}")

    def run_all(self, tasks: Optional[List[Dict]] = None) -> List[BenchmarkResult]:
        tasks = list(BENCHMARK_TASKS if tasks is None else tasks)
        if self.offline:
            from agentic_arxiv.tasks.reference import attach_expected_paper_ids
            tasks = attach_expected_paper_ids(tasks, self._snapshot_path())
        lookup = {task["id"]: task for task in tasks}
        results: List[BenchmarkResult] = []
        total = len(tasks) * self.repeat
        for task_def in tasks:
            for trial in range(self.repeat):
                if self.cancel_requested is not None and self.cancel_requested():
                    raise EvaluationCancelled(len(results), results)
                task_id = task_def["id"]
                session_id = f"{self.session_prefix}_{task_id}_regex_{trial}"
                print(f"[{len(results) + 1}/{total}] task={task_id} trial={trial}", flush=True)
                try:
                    self._start_generation_stream(task_id, trial)
                    # 环境真实初始化并过滤可见状态；setup 不进入策略 history。
                    raw = execute_episode(
                        self.llm_client, task_def, self._tool_env,
                        model=self.model, session_id=session_id,
                        max_iterations=(task_def["max_iterations"]
                                        if task_def.get("max_iterations") is not None else 6),
                        llm_extra=self.llm_extra,
                        initialization_plan=compile_initialization(task_def, task_lookup=lookup),
                    )
                    metrics = extract_metrics(task_def, raw, "regex", trial, session_id)
                    result = BenchmarkResult(
                        task_id, "regex", trial, raw, session_id, metrics=metrics,
                        expected_paper_ids=(list(task_def["expected_paper_ids"])
                                            if task_def.get("expected_paper_ids") is not None else None),
                    )
                except Exception as exc:
                    log.error(f"Benchmark 任务执行异常: {exc}", exc_info=True)
                    result = BenchmarkResult(
                        task_id, "regex", trial, {}, session_id, error=str(exc),
                    )
                results.append(result)
                self._print_step_summary(result)
        return results

    @staticmethod
    def _print_step_summary(result: BenchmarkResult):
        if result.error:
            print(f"  ERROR: {result.error[:100]}")
            return
        metrics = result.metrics
        if metrics is None:
            return
        status = "PASS" if is_strict_success(metrics) else (
            "FAIL(WRONG_RESULT)" if metrics.task_completed else
            f"FAIL({metrics.termination_type})"
        )
        tools = " -> ".join(metrics.tool_call_sequence) or "(none)"
        print(
            f"  {status} | {metrics.total_time_ms}ms "
            f"(LLM:{metrics.total_llm_ms} Tool:{metrics.total_tool_ms} "
            f"OH:{metrics.framework_overhead_ms}) "
            f"| iter={metrics.iteration_count} | tools=[{tools}] "
            f"| accurate={metrics.tool_call_accurate}"
        )
