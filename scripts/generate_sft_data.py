"""从 benchmark tasks 生成 SFT 训练数据

SFT 数据格式：
- 支持 ReAct Prompt-Completion 格式与标准 Messages 格式
- 正确保留多轮任务的前置交互历史（Thought -> Action -> Observation），避免马尔可夫链破坏
- 与推理/GRPO 阶段的 ReAct Prompt 模板保持严格一致
- 支持离线快照环境回放与确定性专家轨迹生成（无需依赖外部 LLM API）

运行方式：
    python scripts/generate_sft_data.py
    python scripts/generate_sft_data.py --snapshot data/mock_arxiv_snapshot.json
    python scripts/generate_sft_data.py \
      --task_set expanded \
      --split data/splits/v3_81.json:train \
      --snapshot data/mock_arxiv_snapshot.json \
      --output data/sft/sft_react_messages_v3_seed.jsonl
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"

# 添加 AgenticArxiv 到 Python 路径
sys.path.insert(0, str(PACKAGE_ROOT))

# RL 路径不依赖数据库：会话状态走内存 store
os.environ.setdefault("STORE_BACKEND", "memory")

import tools.arxiv_tool  # noqa: F401
import tools.cache_status_tool  # noqa: F401
import tools.pdf_download_tool  # noqa: F401
import tools.pdf_translate_tool  # noqa: F401
from tools.bootstrap import register_all_tools
register_all_tools()

from agents.agent_engine import ReActAgent
from agents.prompt_templates import format_tool_description
from agents.react_contract import (
    append_react_turn, build_react_messages, make_history_step,
    parse_react_response, protocol_metadata,
)
from benchmark.metrics import extract_metrics, is_strict_success
from benchmark.splits import load_split
from benchmark.task_spec import build, reference_terminal_thought
from benchmark.tasks import BENCHMARK_SPECS
from rl.multiturn_env import AgenticArxivMultiTurnEnv
from rl.run_manifest import file_fingerprint
from tools.tool_registry import registry


def sample_fingerprint(messages: List[Dict[str, Any]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def format_step_action(action: Any) -> str:
    """统一将动作格式化为 JSON 字符串或 FINISH"""
    if isinstance(action, str):
        return action
    if isinstance(action, dict):
        return json.dumps(action, ensure_ascii=False)
    return str(action)


def build_sft_samples_from_history(
    task_text: str,
    tools_description: str,
    history: List[Dict[str, Any]],
    *,
    source_task_id: Optional[str] = None,
    source_split: Optional[str] = None,
    visible_setup_context: str = "",
    tokenizer: Any = None,
) -> List[Dict[str, Any]]:
    """从完整的执行历史构造各轮次训练样本。

    messages[:-1] 是共同历史前缀，messages[-1:] 是唯一当前 assistant 标签。
    历史动作、setup 和环境 Observation 由训练入口整体排除 loss。
    """
    samples = []
    accumulated_history: List[Dict[str, Any]] = []

    for step_index, step in enumerate(history):
        action = step.get("action", "")
        if not action or action in ("PARSE_ERROR", "ERROR", "FORCE_STOP"):
            break
        policy_text = step.get("policy_text")
        if not isinstance(policy_text, str) or not policy_text.strip():
            raise ValueError("SFT 当前协议轨迹缺少实际 policy_text，不支持旧上下文执行")
        parsed = parse_react_response(policy_text)
        if parsed.kind == "PARSE_ERROR":
            raise ValueError(f"SFT 当前响应格式错误: {parsed.error}")
        prompt = build_react_messages(
            task_text, tools_description, accumulated_history, visible_setup_context,
        )
        sample = {
            "messages": append_react_turn(prompt, policy_text),
            "protocol_metadata": protocol_metadata(tokenizer),
        }
        if source_task_id is not None:
            sample["source_task_id"] = source_task_id
        if source_split is not None:
            sample["source_split"] = source_split
        sample["trajectory_step"] = step_index
        sample["sample_sha256"] = sample_fingerprint(sample["messages"])
        samples.append(sample)

        accumulated_history.append(step)
        if parsed.kind == "FINISH":
            break

    return samples


def generate_deterministic_trajectories(
    task_specs: List[Any],
    env: Any,
    tools_description: str,
    *,
    source_split: str = "train",
) -> List[Dict[str, Any]]:
    """专家步骤使用共同 episode；标准答案仅存在于生成侧。"""
    sft_data = []
    task_lookup = {spec.id: spec.to_task() for spec in task_specs}
    for spec in task_specs:
        session_id = f"sft_spec_{spec.id}"
        task_def = spec.to_task()
        environment = env if isinstance(env, AgenticArxivMultiTurnEnv) else (
            AgenticArxivMultiTurnEnv(mode="replay", backend=env)
        )
        history = []
        try:
            try:
                setup_context = environment.reset(
                    task_def=task_def, session_id=session_id, task_lookup=task_lookup,
                )
            except Exception as exc:
                raise RuntimeError(f"SFT setup 执行失败: task={spec.id}") from exc
            for step_idx, step_spec in enumerate(spec.steps):
                tool_name = step_spec.tool
                # TaskSpec 的框架字段只在共同执行边界注入，不进入策略标签。
                args = {key: value for key, value in dict(step_spec.args or {}).items()
                        if key not in {"session_id", "_resolved_paper_id"}}
                thought = f"需要调用 {tool_name} 来处理任务步骤 {step_idx + 1}"
                action_dict = {"name": tool_name, "args": args}
                parsed = parse_react_response(
                    f"Thought: {thought}\nAction: {format_step_action(action_dict)}"
                )
                try:
                    result = environment.step(parsed.action)
                    if isinstance(result, dict) and (
                        result.get("error") or result.get("status") in {"FAILED", "ERROR"}
                    ):
                        raise ValueError(result.get("error") or result["status"])
                except Exception as exc:
                    raise RuntimeError(
                        f"SFT 专家步骤执行失败: task={spec.id}, step={step_idx}, "
                        f"tool={tool_name}, args={args}"
                    ) from exc
                history.append(make_history_step(parsed, result=result))

            terminal_thought = ("所有步骤均已成功执行，任务已完成" if spec.steps
                                else reference_terminal_thought(task_def))
            history.append(make_history_step(parse_react_response(
                f"Thought: {terminal_thought}\nAction: FINISH"
            )))
            metrics = extract_metrics(
                task_def, {"history": history}, agent_type="deterministic_expert",
                trial=0, session_id=session_id,
            )
            if not is_strict_success(metrics):
                raise RuntimeError(
                    f"SFT 专家轨迹未通过严格校验: task={spec.id}, "
                    f"termination={metrics.termination_type}, "
                    f"tools={metrics.tool_call_sequence}, arg_score={metrics.arg_score}, "
                    f"tool_failures={metrics.tool_exec_failures}"
                )
            sft_data.extend(build_sft_samples_from_history(
                spec.task, tools_description, history, source_task_id=spec.id,
                source_split=source_split, visible_setup_context=setup_context,
            ))
        finally:
            if environment is not env:
                environment.close()

    return sft_data


def select_task_specs(task_set: str, split: Optional[str]) -> tuple[List[Any], str]:
    """选择允许生成 SFT 的任务，并把防泄漏策略放在唯一入口。

    expanded 是正式实验任务集，必须显式使用某个版本文件的 train；裸 `train`
    会落到历史默认 v1，dev/iid/ood 则会污染评测，因此全部拒绝。
    """
    if task_set != "expanded":
        if split:
            raise SystemExit("--split 仅与 --task_set expanded 一起使用")
        return list(BENCHMARK_SPECS), "basic"

    path_part, separator, split_name = (split or "").rpartition(":")
    if not separator or not path_part or split_name != "train":
        raise SystemExit(
            "expanded SFT 数据生成必须显式指定版本化 train，"
            "例如 --split data/splits/v2_62.json:train；"
            "禁止使用全部扩展任务、裸 train、dev、iid_test 或 ood_test"
        )

    from benchmark.tasks_expanded import EXPANDED_SPECS

    wanted = set(load_split(split))
    by_id = {spec.id: spec for spec in EXPANDED_SPECS}
    missing = wanted - set(by_id)
    if missing:
        raise SystemExit(f"切分中存在 expanded 任务集没有的 ID: {sorted(missing)}")

    specs = [by_id[task_id] for task_id in sorted(wanted)]
    # 记录版本文件名而不是机器绝对路径，使数据血缘既明确又可移植。
    source_split = f"{Path(path_part).name}:{split_name}"
    return specs, source_split


def generate_sft_dataset(
    output: Optional[str] = None,
    snapshot: Optional[str] = None,
    use_llm: bool = False,
    task_set: str = "basic",
    split: Optional[str] = None,
    mode: str = "replay",
) -> None:
    """生成 SFT 专家数据集主函数"""
    output_path = (Path(output) if output else
                   REPO_ROOT / "data" / "sft" / "sft_react_messages_v3_seed.jsonl")
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise SystemExit(f"拒绝覆盖已有数据或 manifest: {output_path}")
    specs, source_split = select_task_specs(task_set, split)
    tools_desc = format_tool_description(registry.list_tools())
    snapshot_path = (
        Path(snapshot) if snapshot else REPO_ROOT / "data" / "mock_arxiv_snapshot.json"
    )
    env = AgenticArxivMultiTurnEnv(
        snapshot_path=snapshot_path if mode != "live" else None, mode=mode,
    )

    sft_data: List[Dict[str, Any]] = []

    print(
        f"[TASKS] 加载任务集 ({task_set}, split={source_split}): "
        f"{len(specs)} 个任务"
    )

    try:
        if not use_llm:
            print(f"[MODE] 使用确定性专家逻辑，工具模式={mode}...")
            sft_data = generate_deterministic_trajectories(
                specs, env, tools_desc, source_split=source_split
            )
        else:
            print(f"[MODE] 使用环境配置的 LLM，工具模式={mode}...")
            from utils.llm_client import get_env_llm_client

            agent = ReActAgent(get_env_llm_client(), env=env)
            tasks = build(specs)
            task_lookup = {task["id"]: task for task in tasks}
            for i, task_def in enumerate(tasks):
                print(f"[{i+1}/{len(tasks)}] 执行任务: {task_def['id']} - {task_def['task']}")
                try:
                    result = agent.run(
                        task_def["task"], session_id=f"sft_gen_{task_def['id']}",
                        task_def=task_def, task_lookup=task_lookup,
                    )
                    if result.get("history"):
                        metrics = extract_metrics(
                            task_def, result, agent_type="llm_expert", trial=0,
                            session_id=f"sft_gen_{task_def['id']}",
                        )
                        if not is_strict_success(metrics):
                            print(
                                f"   [REJECT] 非严格成功轨迹，不进入 SFT: "
                                f"termination={metrics.termination_type}, "
                                f"tools={metrics.tool_call_sequence}, "
                                f"arg_score={metrics.arg_score}"
                            )
                            continue
                        samples = build_sft_samples_from_history(
                            task_def["task"], tools_desc, result["history"],
                            source_task_id=task_def["id"], source_split=source_split,
                            visible_setup_context=result.get("setup_context", ""),
                            tokenizer=getattr(agent.llm_client, "tokenizer", None),
                        )
                        sft_data.extend(samples)
                        print(f"   [OK] 提取 {len(samples)} 条多轮训练样本")
                except Exception as exc:
                    print(f"   [ERROR] 执行出错: {exc}")
    finally:
        env.close()

    if not sft_data:
        raise SystemExit("没有生成任何通过校验的 SFT 样本，拒绝写出空数据集")

    fingerprints = {row["sample_sha256"] for row in sft_data}
    if len(fingerprints) != len(sft_data):
        raise SystemExit("SFT 专家数据含重复消息，拒绝写出")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    metadata = dict(sft_data[0]["protocol_metadata"])
    chat_hashes = {row["protocol_metadata"]["chat_template_hash"] for row in sft_data}
    if len(chat_hashes) != 1:
        metadata["chat_template_hash"] = None
    manifest = {
        "version": 2, "kind": "train_only_expert_seed",
        "task_set": task_set, "source_split": source_split,
        "source_task_ids": sorted({row["source_task_id"] for row in sft_data}),
        "split": file_fingerprint(Path(split.rpartition(":")[0]) if split else None),
        "snapshot": file_fingerprint(snapshot_path if mode != "live" else None),
        "mode": mode, "output": str(output_path),
        "output_sha256": file_fingerprint(output_path)["sha256"],
        "sample_rows": len(sft_data), "unique_sample_fingerprints": len(fingerprints),
        "protocol_metadata": metadata,
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    print(f"\n[DONE] SFT 数据生成完成：共 {len(sft_data)} 条样本 -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 SFT 专家数据集")
    parser.add_argument("--output", default=None, help="输出 JSONL 路径")
    parser.add_argument("--snapshot", default=None, help="离线快照文件路径")
    parser.add_argument(
        "--mode", choices=("replay", "live", "record"), default="replay",
        help="工具执行模式；默认离线且缺快照失败，在线必须显式选择 live",
    )
    parser.add_argument("--use_llm", action="store_true", help="使用外部 LLM API 生成（默认使用确定性专家）")
    parser.add_argument("--task_set", choices=["basic", "expanded"], default="basic", help="使用基础还是扩展任务集")
    parser.add_argument(
        "--split",
        default=None,
        metavar="PATH:NAME",
        help="expanded 正式数据必须显式指定版本化 train，例如 data/splits/v2_62.json:train",
    )
    args = parser.parse_args()

    generate_sft_dataset(
        output=args.output,
        snapshot=args.snapshot,
        use_llm=args.use_llm,
        task_set=args.task_set,
        split=args.split,
        mode=args.mode,
    )
