"""从 benchmark tasks 生成 SFT 训练数据

SFT 数据格式：
- 支持 ReAct Prompt-Completion 格式与标准 Messages 格式
- 正确保留多轮任务的前置交互历史（Thought -> Action -> Observation），避免马尔可夫链破坏
- 与推理/GRPO 阶段的 ReAct Prompt 模板保持严格一致
- 支持离线快照环境回放与确定性专家轨迹生成（无需依赖外部 LLM API）

运行方式：
    python scripts/generate_sft_data.py
    python scripts/generate_sft_data.py --snapshot data/mock_arxiv_snapshot.json
    python scripts/generate_sft_data.py --use_llm --task_set expanded
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
from benchmark.task_spec import build
from benchmark.tasks import BENCHMARK_SPECS, get_all_tasks
from rl.env import MockArxivEnv
from tools.tool_registry import registry


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
) -> List[Dict[str, Any]]:
    """从完整的执行历史构造各轮次训练样本。

    关键设计：第 k 步的 prompt 必须包含前 k-1 步的 Thought/Action/Observation 历史，
    保持马尔可夫决策状态的完整性。
    """
    samples = []
    accumulated_history: List[Dict[str, Any]] = []

    for step in history:
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

        samples.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_content},
            ]
        })

        accumulated_history.append(step)

    return samples


def generate_deterministic_trajectories(
    task_specs: List[Any],
    env: MockArxivEnv,
    tools_description: str,
) -> List[Dict[str, Any]]:
    """根据 TaskSpec 标准答案和 MockArxivEnv 确定性执行构建专家轨迹。"""
    sft_data = []

    for spec in task_specs:
        session_id = f"sft_spec_{spec.id}"
        history = []

        # 执行 setup 前置步骤以建立 session 状态（如 papers list）
        if getattr(spec, "setup", None):
            for setup_step in spec.setup:
                setup_args = dict(setup_step.args or {})
                setup_args["session_id"] = session_id
                try:
                    env.execute_tool(setup_step.tool, setup_args)
                except Exception:
                    pass

        # 如果没有 setup 但有 depends_on，确保先执行一次搜索以填充 session 状态
        if not getattr(spec, "setup", None) and getattr(spec, "depends_on", None):
            search_args = {"aspect": "AI", "days": 7, "max_results": 5, "session_id": session_id}
            try:
                env.execute_tool("get_recently_submitted_cs_papers", search_args)
            except Exception:
                pass

        if not spec.steps:
            # infeasible 任务：直接结束
            history.append({
                "thought": "该任务无法通过现有工具完成或参数无效",
                "action": "FINISH",
                "observation": "任务完成",
            })
        else:
            for step_idx, step_spec in enumerate(spec.steps):
                tool_name = step_spec.tool
                args = dict(step_spec.args or {})
                tool_def = registry.get_tool(tool_name)
                props = (tool_def or {}).get("parameters", {}).get("properties", {})
                if "session_id" in props:
                    args["session_id"] = session_id

                thought = f"需要调用 {tool_name} 来处理任务步骤 {step_idx + 1}"
                action_dict = {"name": tool_name, "args": args}

                try:
                    res = env.execute_tool(tool_name, args)
                    if isinstance(res, list):
                        obs = f"成功获取 {len(res)} 篇论文"
                    elif isinstance(res, dict):
                        obs = str(res)[:500]
                    else:
                        obs = str(res)[:500]
                except Exception as exc:
                    obs = f"工具执行异常: {exc}"

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

        samples = build_sft_samples_from_history(spec.task, tools_description, history)
        sft_data.extend(samples)

    return sft_data


def generate_sft_dataset(
    output: Optional[str] = None,
    snapshot: Optional[str] = None,
    use_llm: bool = False,
    task_set: str = "basic",
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

    if task_set == "expanded":
        try:
            from benchmark.tasks_expanded import EXPANDED_SPECS
            specs = EXPANDED_SPECS
        except ImportError:
            specs = BENCHMARK_SPECS
    else:
        specs = BENCHMARK_SPECS

    print(f"[TASKS] 加载任务集 ({task_set}): {len(specs)} 个任务")

    if not use_llm:
        print("[MODE] 使用确定性专家逻辑与离线环境生成标准化 SFT 演示...")
        sft_data = generate_deterministic_trajectories(specs, env, tools_desc)
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
                    samples = build_sft_samples_from_history(
                        task_def["task"], tools_desc, result["history"]
                    )
                    sft_data.extend(samples)
                    print(f"   [OK] 提取 {len(samples)} 条多轮训练样本")
            except Exception as e:
                print(f"   [ERROR] 执行出错: {e}")

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
    args = parser.parse_args()

    generate_sft_dataset(
        output=args.output,
        snapshot=args.snapshot,
        use_llm=args.use_llm,
        task_set=args.task_set,
    )

