"""从明确的本地 SFT 与训练切分采样可追溯的单步偏好。

此模块只编排共同运行时和只读评分，不加载训练器或评测编排层。
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agentic_arxiv.config import initialize_operation, workspace_path
from agentic_arxiv.contracts import copy_tool_result
from agentic_arxiv.data_pipeline.common import canonical_hash
from agentic_arxiv.data_pipeline.generate_sft_data import select_task_specs
from agentic_arxiv.llm.artifacts import artifact_metadata
from agentic_arxiv.provenance import (
    file_fingerprint, model_input_fingerprint, require_training_protocol,
    sha256_file, write_json, write_jsonl,
)
from agentic_arxiv.runtime.agent_engine import ReActAgent
from agentic_arxiv.runtime.env import MockArxivEnv
from agentic_arxiv.runtime.multiturn_env import AgenticArxivMultiTurnEnv
from agentic_arxiv.runtime.react_contract import protocol_metadata
from agentic_arxiv.scoring.reward import RewardCalculator
from agentic_arxiv.tasks.initialization import compile_initialization
from agentic_arxiv.tasks.reference import attach_expected_paper_ids
from agentic_arxiv.tasks.task_spec import build
from agentic_arxiv.tools.bootstrap import register_all_tools


class RecordingClient:
    """透明透传客户端；仅深拷贝实际请求/响应，不重放或改写消息。"""

    def __init__(self, client: Any):
        self.client = client
        self.requests: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def chat_completions(self, *args: Any, **kwargs: Any) -> Any:
        # 运行时使用关键字调用，保留位置参数亦不改变底层客户端的调用语义。
        messages = kwargs.get("messages", args[1] if len(args) > 1 else None)
        record = {
            "messages": copy.deepcopy(messages),
            "request_options": copy.deepcopy({key: value for key, value in kwargs.items()
                                               if key != "messages"}),
            "step_index": len(self.requests),
        }
        self.requests.append(record)
        try:
            response = self.client.chat_completions(*args, **kwargs)
        except Exception as exc:
            record["error"] = str(exc)
            raise
        record["response"] = copy.deepcopy(response)
        choices = response.get("choices") or [{}]
        record["finish_reason"] = choices[0].get("finish_reason")
        return response

    def bind_history(self, history: list[dict[str, Any]]) -> None:
        """把运行时的策略边界绑定到同次请求，绝不从原响应补造目标。"""
        for record in self.requests:
            index = record["step_index"]
            if index < len(history):
                record["policy_text"] = history[index].get("policy_text")


def batch_paths(output: Path) -> dict[str, Path]:
    return {"data": output, **{name: Path(str(output) + suffix) for name, suffix in (
        ("traces", ".traces.jsonl"), ("manifest", ".manifest.json"),
        ("report", ".report.json"),
    )}}


def _require_new_batch(paths: dict[str, Path]) -> None:
    for path in paths.values():
        # lexists 同时拒绝失效符号链接；最终发布仍使用独占 link，避免检查竞态覆盖。
        if os.path.lexists(path):
            raise FileExistsError(f"拒绝覆盖已有 DPO 批次文件: {path}")
        for parent in path.parents:
            if parent.exists():
                if not parent.is_dir():
                    raise NotADirectoryError(f"DPO 输出父路径不是目录: {parent}")
                break


def publish_batch(
    output: Path, rows: list[dict[str, Any]], traces: list[dict[str, Any]],
    report: dict[str, Any], manifest: dict[str, Any],
) -> dict[str, Path]:
    """先完成临时文件，再独占发布；成功 manifest 始终最后出现。

    中断留下的部分批次没有成功标记，训练审计必须拒绝；零对只交付诊断。
    """
    paths = batch_paths(output)
    _require_new_batch(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".dpo-batch-", dir=output.parent) as temporary:
        temp = {name: Path(temporary) / path.name for name, path in paths.items()}
        write_jsonl(temp["traces"], traces, overwrite=False)
        write_json(temp["report"], report, overwrite=False)
        names = ["traces", "report"]
        if rows:
            write_jsonl(temp["data"], rows, overwrite=False)
            completed = {**manifest, "status": "complete", "files": {
                name: {"path": paths[name].name, "sha256": sha256_file(temp[name]),
                       "count": len(rows) if name == "data" else len(traces)}
                for name in ("data", "traces")
            }}
            completed["report"] = {
                "path": paths["report"].name, "sha256": sha256_file(temp["report"]),
            }
            write_json(temp["manifest"], completed, overwrite=False)
            names = ["data", *names, "manifest"]
        for name in names:
            # 同一文件系统的 link 是原子且不覆盖的；临时目录清理只删临时硬链接。
            os.link(temp[name], paths[name])
    return paths


def _validate_options(
    num_trials: int, max_turns: int, max_new_tokens: int, seed: int,
    temperature: float, min_reward_gap: float, min_task_yield: float,
    reward_curriculum_steps: int,
) -> None:
    for name, value, lower in (("num_trials", num_trials, 2), ("max_turns", max_turns, 1),
                               ("max_new_tokens", max_new_tokens, 1),
                               ("reward_curriculum_steps", reward_curriculum_steps, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(f"{name} 必须是至少 {lower} 的整数")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed 必须是非负整数")
    for name, value in (("temperature", temperature), ("min_reward_gap", min_reward_gap),
                        ("min_task_yield", min_task_yield)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} 必须是有限数值")
    if temperature <= 0 or min_reward_gap < 0 or not 0 <= min_task_yield <= 1:
        raise ValueError("temperature 必须为正，min_reward_gap 非负，min_task_yield 位于 [0, 1]")


def generate_dpo_dataset(
    model: str = "outputs/sft_qlora/final",
    output: str = "data/dpo/dpo_react_messages_v3_train.jsonl",
    snapshot: str | None = None, task_set: str = "expanded", split: str | None = None,
    num_trials: int = 5, temperature: float = 0.8, seed: int = 42,
    min_reward_gap: float = 0.05, min_task_yield: float = 0.25,
    max_turns: int = 6, max_new_tokens: int = 256, reward_curriculum_steps: int = 30,
    base_model: str | None = None, *, workspace: Path | str | None = None,
) -> dict[str, Any]:
    """显式调用才采样；输入失败不联网、不创建输出，也不回退任务或模型。"""
    initialize_operation(workspace)
    _validate_options(num_trials, max_turns, max_new_tokens, seed, temperature,
                      min_reward_gap, min_task_yield, reward_curriculum_steps)
    if task_set != "expanded":
        raise ValueError("DPO 生成只接受 expanded 的显式版本化 :train 切分")
    specs, source_split = select_task_specs(task_set, split, workspace=workspace)
    if not specs:
        raise ValueError("DPO 请求的训练切分为空")
    output_path = workspace_path(output, workspace=workspace)
    _require_new_batch(batch_paths(output_path))
    snapshot_path = workspace_path(snapshot or "data/mock_arxiv_snapshot.json", workspace=workspace)
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"DPO 离线快照不存在: {snapshot_path}")
    snapshot_source = file_fingerprint(snapshot_path)
    snapshot_data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if sha256_file(snapshot_path) != snapshot_source["sha256"]:
        raise ValueError("DPO 快照在读取期间改变，拒绝采样")
    if not isinstance(snapshot_data, dict) or not all(isinstance(bucket, dict)
                                                      for bucket in snapshot_data.values()):
        raise ValueError(f"DPO 快照必须为工具到记录字典的映射: {snapshot_path}")
    model_path = workspace_path(model, workspace=workspace)
    resolved_base = base_model
    if base_model and workspace_path(base_model, workspace=workspace).is_dir():
        resolved_base = str(workspace_path(base_model, workspace=workspace))
    expected = protocol_metadata()
    artifact_metadata(str(model_path), base_model=resolved_base,
                      allowed_stages={"sft_qlora", "sft_full"},
                      expected_protocol=expected, local_base_only=True)
    tasks = build(specs)
    task_definition_hashes = {task["id"]: canonical_hash(task) for task in tasks}
    task_definitions_sha256 = canonical_hash(tasks)
    lookup = {task["id"]: task for task in tasks}
    plans = {task["id"]: compile_initialization(task, task_lookup=lookup) for task in tasks}
    source_model = model_input_fingerprint(model_path)
    if source_model.get("weights_status") != "known":
        raise ValueError(f"DPO 源 SFT 权重或索引分片不完整: {model_path}")
    split_source = file_fingerprint(workspace_path(split.rpartition(":")[0], workspace=workspace))
    sampling = {"num_trials": num_trials, "temperature": temperature, "seed": seed,
                "max_turns": max_turns, "max_new_tokens": max_new_tokens,
                "stream_key_format": "dpo:{task_id}:{trial_id}"}
    calculator = RewardCalculator(curriculum_steps=reward_curriculum_steps)
    scoring = {"curriculum_steps": reward_curriculum_steps, "training_step": 0,
               "early_correctness_scale": calculator.early_correctness_scale,
               "weights": calculator.get_weights(), "effective_weights": asdict(calculator.schedule(0)),
               "source": file_fingerprint(Path(__file__).parents[1] / "scoring" / "reward.py")}
    register_all_tools()
    # 标准答案仅在采样前的独立 replay 中准备，不进入策略消息或实际轨迹评分执行。
    tasks = attach_expected_paper_ids(tasks, snapshot_path, task_lookup=lookup)
    scoring["expected_paper_ids"] = {
        task["id"]: task.get("expected_paper_ids") for task in tasks
    }
    if sha256_file(snapshot_path) != snapshot_source["sha256"]:
        raise ValueError("DPO 快照在输入准备期间改变，拒绝采样")
    # 导入和实例化本地推理留到所有静态输入已通过校验之后。
    from agentic_arxiv.llm.local import TransformersLLMClient
    from agentic_arxiv.data_pipeline.dpo_pairs import build_dpo_pairs
    from agentic_arxiv.provenance import DPO_DATA_KIND, DPO_DATA_VERSION, DPO_PAIRING_VERSION

    client = TransformersLLMClient(str(model_path), seed=seed, artifact=True,
                                  base_model=resolved_base,
                                  allowed_stages={"sft_qlora", "sft_full"},
                                  expected_protocol=expected)
    actual_protocol = protocol_metadata(client.tokenizer)
    require_training_protocol(model_path, actual_protocol, source_label="DPO 源 SFT",
                              require_chat_template_hash=True)
    if not actual_protocol.get("chat_template_hash"):
        raise ValueError("DPO 源 SFT 的实际 tokenizer 模板不可确认")
    traces: list[dict[str, Any]] = []
    for task in tasks:
        tid = task["id"]
        identity = {"source_model": source_model, "source_split": source_split,
                    "protocol_metadata": actual_protocol, "snapshot": snapshot_source,
                    "initialization": plans[tid], "sampling": sampling,
                    "scoring": scoring, "task_definition_sha256": task_definition_hashes[tid]}
        for trial in range(num_trials):
            trace_id = f"dpo:{tid}:{trial}"
            recorder = RecordingClient(client)
            trace: dict[str, Any] = {"trace_id": trace_id, "task_id": tid, "trial_id": trial,
                                     "source_split": source_split, "identity": identity,
                                     "status": "invalid", "stream_key": trace_id,
                                     "requests": recorder.requests, "trajectory": {},
                                     "reward": None, "reward_components": {}}
            environment = None
            try:
                client.start_generation_stream(trace_id)
                # 每条轨迹独有后端运行态、store 和临时文件；快照内容只读复用。
                backend = MockArxivEnv(mode="replay", offline_download=True)
                backend.snapshot = snapshot_data
                environment = AgenticArxivMultiTurnEnv(mode="replay", backend=backend)
                agent = ReActAgent(recorder, env=environment, max_iterations=max_turns,
                                   llm_extra={"temperature": temperature,
                                              "max_tokens": max_new_tokens})
                result = agent.run(task["task"], agent_model=str(model_path),
                                   session_id=trace_id, initialization_plan=plans[tid])
                trace["trajectory"] = copy_tool_result(result)
                recorder.bind_history(result["history"])
                trace["environment_stats"] = dict(environment.backend.stats)
                if any(trace["environment_stats"].get(key, 0) for key in ("miss", "fallback", "real_calls")):
                    raise RuntimeError("实际回放存在快照缺失、后备响应或联网调用，轨迹不可可靠评分")
                if result.get("termination_reason") == "RUNTIME_ERROR" or any(
                    "error" in request for request in recorder.requests
                ):
                    raise RuntimeError("实际模型请求或运行时失败，不作为低分负例")
                breakdown, metrics = calculator.compute_reward_breakdown(
                    task, copy.deepcopy(result), trial=trial, session_id=trace_id, training_step=0,
                )
                if not math.isfinite(breakdown.total):
                    trace["failure_code"] = "nonfinite_score"
                    raise ValueError("实际轨迹评分不是有限数值")
                trace.update(status="completed", reward=float(breakdown.total),
                             reward_components=breakdown.to_dict(), metrics=asdict(metrics))
            except Exception as exc:
                trace["error"] = str(exc)
            finally:
                if environment is not None:
                    trace["environment_stats"] = dict(environment.backend.stats)
                    try:
                        environment.close()
                    except Exception as exc:
                        trace.update(status="invalid", cleanup_error=str(exc))
                traces.append(trace)
    rows, report = build_dpo_pairs(
        traces, task_ids=[task["id"] for task in tasks], trials_per_task=num_trials,
        min_reward_gap=min_reward_gap, yield_warning_threshold=min_task_yield,
    )
    report.update(status="complete" if rows else "failed", source_split=source_split,
                  sampling=sampling, scoring=scoring)
    manifest = {
        "kind": DPO_DATA_KIND, "version": DPO_DATA_VERSION, "pairing_version": DPO_PAIRING_VERSION,
        "protocol_metadata": actual_protocol, "source_model": source_model,
        "source_model_manifest": file_fingerprint(model_path / "training_manifest.json"),
        "source_split": source_split, "split_file": split_source,
        "task_ids": [task["id"] for task in tasks],
        "task_definitions_sha256": task_definitions_sha256,
        "task_definition_hashes": task_definition_hashes, "snapshot": snapshot_source,
        "initialization": {tid: {"plan": plan, "sha256": canonical_hash(plan)}
                           for tid, plan in plans.items()},
        "sampling": sampling, "scoring": scoring, "min_reward_gap": min_reward_gap,
        "min_task_yield": min_task_yield,
    }
    publish_batch(output_path, rows, traces, report, manifest)
    if not rows:
        raise RuntimeError(f"DPO 无有效偏好对，未发布训练包；诊断: {batch_paths(output_path)['report']}")
    if report["task_yield_rate"] < min_task_yield:
        print("[WARN] DPO 任务产出率低于警戒线；未降低阈值或追加采样")
    print(f"[DPO] 已发布 {len(rows)} 对；manifest: {batch_paths(output_path)['manifest']}")
    return report
