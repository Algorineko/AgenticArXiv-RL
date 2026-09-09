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
      --split data/splits/v2_62.json:train \
      --snapshot data/mock_arxiv_snapshot.json \
      --output data/sft/sft_v0_train.jsonl
"""

import argparse
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
from agents.prompt_templates import format_tool_description, get_react_prompt
from agents.side_effects import LocalSideEffectManager
from benchmark.metrics import extract_metrics, is_strict_success
from benchmark.splits import load_split
from benchmark.task_spec import build, reference_terminal_thought
from benchmark.tasks import BENCHMARK_SPECS, get_all_tasks
from rl.env import MockArxivEnv
from tools.tool_registry import registry

PAPER_SEARCH_ACTIONS = {
    "get_recently_submitted_cs_papers",
    "search_arxiv_papers",
}


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
) -> List[Dict[str, Any]]:
    """从完整的执行历史构造各轮次训练样本。

    关键设计：第 k 步的 prompt 必须包含前 k-1 步的 Thought/Action/Observation 历史，
    保持马尔可夫决策状态的完整性。
    """
    samples = []
    accumulated_history: List[Dict[str, Any]] = []

    for step_index, step in enumerate(history):
        action = step.get("action", "")
        if not action or action in ("PARSE_ERROR", "ERROR", "FORCE_STOP"):
            continue

        thought = step.get("thought", "")
        if not thought:
            thought = "分析当前状态并决定下一步动作" if action != "FINISH" else "任务已完成"

        # 格式化前置历史文本
        history_parts = []
        for prev in accumulated_history:
            prev_thought = prev.get("thought", "")
            prev_action = format_step_action(prev.get("action", ""))
            prev_obs = prev.get("observation", "")
            history_parts.append(
                f"Thought: {prev_thought}\nAction: {prev_action}\nObservation: {prev_obs}"
            )
        history_text = "\n\n".join(history_parts)

        # 构造与推理完全一致的 ReAct Prompt
        prompt = get_react_prompt(
            task=task_text,
            tools_description=tools_description,
            history=history_text,
        )

        action_str = format_step_action(action)
        # Assistant 输出包含思考与动作（或纯 Action JSON）
        assistant_content = f"Thought: {thought}\nAction: {action_str}"

        sample = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_content},
            ],
        }
        if source_task_id is not None:
            sample["source_task_id"] = source_task_id
        if source_split is not None:
            sample["source_split"] = source_split
        sample["trajectory_step"] = step_index
        samples.append(sample)

        accumulated_history.append(step)

    return samples


def execute_expert_tool(
    env: MockArxivEnv,
    side_effects: LocalSideEffectManager,
    tool_name: str,
    args: Dict[str, Any],
    session_id: str,
) -> Any:
    """按真实 Agent 的副作用语义执行专家动作。

    搜索工具本身不接收 session_id，因此不能靠把 session_id 塞进工具参数来建立
    会话记忆；BaseAgent 是在工具返回后通过 SideEffectManager 写入 papers。
    确定性生成器绕过了 BaseAgent，也必须显式完成同一状态转移。
    """
    call_args = dict(args)
    tool_def = registry.get_tool(tool_name)
    props = (tool_def or {}).get("parameters", {}).get("properties", {})
    if "session_id" in props:
        call_args["session_id"] = session_id

    if tool_name == "translate_arxiv_pdf":
        # session_id 是框架状态，不应出现在模型要学习的 Action 参数里。
        call_args.pop("session_id", None)
        handle = side_effects.enqueue_translate(session_id=session_id, **call_args)
        return {
            "task_id": handle.task_id,
            "paper_id": handle.paper_id,
            "status": handle.status,
        }

    result = env.execute_tool(tool_name, call_args)
    if tool_name in PAPER_SEARCH_ACTIONS and isinstance(result, list):
        fallback = next(
            (
                paper.get("_mock_env")
                for paper in result
                if isinstance(paper, dict)
                and isinstance(paper.get("_mock_env"), dict)
                and paper["_mock_env"].get("offline_fallback")
            ),
            None,
        )
        if fallback:
            raise RuntimeError(
                fallback.get("message", "关键词搜索只命中了离线回退池")
            )

        from models.schemas import Paper

        papers = [
            Paper(**{key: value for key, value in paper.items() if not key.startswith("_")})
            if isinstance(paper, dict) else paper
            for paper in result
        ]
        side_effects.set_last_papers(session_id, papers)
    return result


def generate_deterministic_trajectories(
    task_specs: List[Any],
    env: MockArxivEnv,
    tools_description: str,
    *,
    source_split: str = "train",
) -> List[Dict[str, Any]]:
    """根据 TaskSpec 标准答案和 MockArxivEnv 确定性执行构建专家轨迹。"""
    sft_data = []
    side_effects = LocalSideEffectManager()

    for spec in task_specs:
        # setdefault(STORE_BACKEND=memory) 无法覆盖调用者已有的环境变量；这里与
        # BenchmarkRunner 一样显式切到全新的 MemoryStore，保证每条专家轨迹隔离。
        from models.store import use_memory_store

        use_memory_store(reset=True)
        session_id = f"sft_spec_{spec.id}"
        history = []
        side_effects._translate_seq = 0

        reset_env = getattr(env, "reset_runtime_state", None)
        if callable(reset_env):
            reset_env()

        # 执行 setup 前置步骤以建立 session 状态（如 papers list）
        if getattr(spec, "setup", None):
            for setup_step in spec.setup:
                setup_args = dict(setup_step.args or {})
                try:
                    execute_expert_tool(
                        env, side_effects, setup_step.tool, setup_args, session_id
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"SFT setup 执行失败: task={spec.id}, "
                        f"tool={setup_step.tool}, args={setup_args}"
                    ) from exc

        # 如果没有 setup 但有 depends_on，确保先执行一次搜索以填充 session 状态
        if not getattr(spec, "setup", None) and getattr(spec, "depends_on", None):
            search_args = {"aspect": "AI", "days": 7, "max_results": 5}
            try:
                execute_expert_tool(
                    env,
                    side_effects,
                    "get_recently_submitted_cs_papers",
                    search_args,
                    session_id,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"SFT depends_on 前置搜索失败: task={spec.id}, args={search_args}"
                ) from exc

        task_def = spec.to_task()
        if not spec.steps:
            # blocked 任务：不调工具，并说明任务声明的具体阻塞原因。
            history.append({
                "thought": reference_terminal_thought(task_def),
                "action": "FINISH",
                "observation": "任务完成",
            })
        else:
            for step_idx, step_spec in enumerate(spec.steps):
                tool_name = step_spec.tool
                args = dict(step_spec.args or {})

                thought = f"需要调用 {tool_name} 来处理任务步骤 {step_idx + 1}"
                action_dict = {"name": tool_name, "args": args}

                try:
                    res = execute_expert_tool(
                        env, side_effects, tool_name, args, session_id
                    )
                    if isinstance(res, list):
                        obs = f"成功获取 {len(res)} 篇论文"
                    elif isinstance(res, dict):
                        obs = str(res)[:500]
                    else:
                        obs = str(res)[:500]
                except Exception as exc:
                    raise RuntimeError(
                        f"SFT 专家步骤执行失败: task={spec.id}, step={step_idx}, "
                        f"tool={tool_name}, args={args}"
                    ) from exc

                history.append({
                    "thought": thought,
                    "action": json.dumps(action_dict, ensure_ascii=False),
                    "observation": obs,
                })

            # 最后追加 FINISH
            history.append({
                "thought": "所有步骤均已成功执行，任务已完成",
                "action": "FINISH",
                "observation": "任务完成",
            })

        metrics = extract_metrics(
            task_def,
            {"history": history},
            agent_type="deterministic_expert",
            trial=0,
            session_id=session_id,
        )
        if not is_strict_success(metrics):
            raise RuntimeError(
                f"SFT 专家轨迹未通过严格校验: task={spec.id}, "
                f"termination={metrics.termination_type}, "
                f"tools={metrics.tool_call_sequence}, arg_score={metrics.arg_score}, "
                f"tool_failures={metrics.tool_exec_failures}"
            )

        samples = build_sft_samples_from_history(
            spec.task,
            tools_description,
            history,
            source_task_id=spec.id,
            source_split=source_split,
        )
        sft_data.extend(samples)

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
            "禁止使用全部 62 条、裸 train、dev、iid_test 或 ood_test"
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
) -> None:
    """生成 SFT 专家数据集主函数"""
    snapshot_path = (
        Path(snapshot) if snapshot else REPO_ROOT / "data" / "mock_arxiv_snapshot.json"
    )
    env = None
    if snapshot_path.exists():
        env = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")
        print(f"[SNAPSHOT] 使用离线快照环境: {snapshot_path}")
    else:
        print(f"[INFO] 未找到快照文件 {snapshot_path}，使用自动环境模式")
        env = MockArxivEnv(mode="auto")

    tools_desc = format_tool_description(registry.list_tools())
    sft_data: List[Dict[str, Any]] = []

    specs, source_split = select_task_specs(task_set, split)

    print(
        f"[TASKS] 加载任务集 ({task_set}, split={source_split}): "
        f"{len(specs)} 个任务"
    )

    if not use_llm:
        print("[MODE] 使用确定性专家逻辑与离线环境生成标准化 SFT 演示...")
        sft_data = generate_deterministic_trajectories(
            specs, env, tools_desc, source_split=source_split
        )
    else:
        print("[MODE] 使用环境配置的 LLM 生成专家轨迹...")
        from utils.llm_client import get_env_llm_client

        llm_client = get_env_llm_client()
        agent = ReActAgent(llm_client, side_effect_mgr=LocalSideEffectManager(), env=env)
        tasks = build(specs)

        for i, task_def in enumerate(tasks):
            print(f"[{i+1}/{len(tasks)}] 执行任务: {task_def['id']} - {task_def['task']}")
            try:
                result = agent.run(task_def["task"], session_id=f"sft_gen_{task_def['id']}")
                if result.get("history"):
                    metrics = extract_metrics(
                        task_def,
                        result,
                        agent_type="llm_expert",
                        trial=0,
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
                        task_def["task"],
                        tools_desc,
                        result["history"],
                        source_task_id=task_def["id"],
                        source_split=source_split,
                    )
                    sft_data.extend(samples)
                    print(f"   [OK] 提取 {len(samples)} 条多轮训练样本")
            except Exception as e:
                print(f"   [ERROR] 执行出错: {e}")

    if not sft_data:
        raise SystemExit("没有生成任何通过校验的 SFT 样本，拒绝写出空数据集")

    output_path = Path(output) if output else REPO_ROOT / "data" / "sft" / "sft_train.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[DONE] SFT 数据生成完成：共 {len(sft_data)} 条样本 -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 SFT 专家数据集")
    parser.add_argument("--output", default=None, help="输出 JSONL 路径")
    parser.add_argument("--snapshot", default=None, help="离线快照文件路径")
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
    )
