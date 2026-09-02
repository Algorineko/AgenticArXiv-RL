"""GRPO 的可验证奖励（RLVR）。

TRL 的 GRPOTrainer 是「对同一 prompt 采样 N 条输出 → 逐条打分 → 组内相对优势」
的结构，它只负责生成**一步**，不会替你跑完整的 ReAct 循环。

兼容两条路径：旧的单步 completion 会补成最小轨迹；TRL 原生工具循环
产生的多轮 structured messages 会直接还原为完整 history，再交给项目已有的
`RewardCalculator.compute_reward_breakdown()` 打分，从而复用现成的奖励定义：

    模型输出  →  解析 Thought/Action
              →  用 MockArxivEnv 执行工具拿到 observation（离线、确定性）
              →  补一步 FINISH，构成完整轨迹
              →  compute_reward_breakdown().total

这样做的好处是**不引入第二套奖励标准**：format / tool / argument / process /
outcome 五个分量、`expected_tool_args` 的参数校验、严格工具序列匹配，
全都沿用 benchmark 与 rl/reward.py 里已有的实现。

注意：组内相对优势由 GRPOTrainer 内部计算，这里只返回标量奖励，
不要再调用 `compute_group_relative_advantages`（会重复归一化）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from benchmark.metrics import NON_TOOL_ACTIONS
from rl.reward import RewardCalculator

# 与 agents/agent_engine.py 的解析规则保持一致
_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\nAction:|$)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(.*?)(?=\nObservation:|$)", re.DOTALL)

# TRL 从 0.28.0 起才会在**非 vLLM** 路径上调用 rollout_func，签名也才是
# rollout_func(prompts, trainer)。更早的版本里它只在 use_vllm 且
# vllm_mode == "server" 时被调用 —— 默认配置下压根不会执行，多轮 rollout
# 静默失效，奖励退回 messages_to_trajectory（任何非空文本都算一步 FINISH）。
MIN_TRL_FOR_ROLLOUT_FUNC = "0.28.0"

# 工具执行失败时写进 observation 的前缀。抽成常量是为了让
# rl/observability.py 统计 tool_error_rate 时不必字面量匹配一段会漂的文案。
TOOL_ERROR_PREFIX = "工具执行失败: "


def parse_react_action(completion: str):
    """从模型输出中解析动作。

    返回 (kind, action)：
        ("finish", None)          模型输出 FINISH
        ("call",   {name, args})  合法工具调用
        ("parse_error", None)     无法解析
    """
    if not completion:
        return "parse_error", None

    match = _ACTION_RE.search(completion)
    if not match:
        return "parse_error", None

    action_text = match.group(1).strip()
    if not action_text:
        return "parse_error", None
    if action_text.upper().startswith(NON_TOOL_ACTIONS):
        return "finish", None

    json_match = re.search(r"(\{.*\})", action_text, re.DOTALL)
    if not json_match:
        return "parse_error", None
    try:
        parsed = json.loads(json_match.group(1))
    except json.JSONDecodeError:
        # prompt 明确要求严格 JSON，这里不做降级修复，否则格式惩罚会失效
        return "parse_error", None

    if not isinstance(parsed, dict) or not parsed.get("name"):
        return "parse_error", None
    # 模型常把结束写成 Action: {"name": "FINISH", "args": {}}。当成工具调用会让
    # rollout 拿到「未知工具」的 observation 继续跑满 max_turns，并给一条本已
    # 正确收尾的轨迹记上一次失败调用 —— 奖励因此与 benchmark 的判定不一致。
    if str(parsed["name"]).strip().upper() in NON_TOOL_ACTIONS:
        return "finish", None
    return "call", {"name": parsed["name"], "args": parsed.get("args") or {}}


def _thought_of(completion: str) -> str:
    text = completion or ""
    match = _THOUGHT_RE.search(text)
    if match:
        return match.group(1).strip()
    # 当 prompt 以 `\nThought:` 结尾时，模型生成可能不带 "Thought:" 前缀，而是直接输出思考内容
    if "Action:" in text:
        parts = text.split("Action:", 1)
        prefix = parts[0].strip()
        if prefix and not prefix.startswith("Observation:"):
            return prefix
    return ""


def synthesize_trajectory(
    completion: str,
    env: Any = None,
    session_id: str = "grpo",
) -> Dict[str, Any]:
    """把「模型生成的一步」补成一条可打分的最小完整轨迹。

    - 解析失败       → 记一步 PARSE_ERROR（parse_failed=True）
    - 直接 FINISH    → 只有一步 FINISH（没有任何工具调用）
    - 工具调用       → 执行取得 observation，再补一步 FINISH

    env 为 None 时不执行工具，observation 留空；此时奖励仅由
    格式 / 工具名 / 参数 决定，工具执行失败一项无法体现。
    """
    kind, action = parse_react_action(completion)
    thought = _thought_of(completion)

    if kind == "parse_error":
        return {
            "history": [{
                "thought": thought,
                "action": "PARSE_ERROR",
                "observation": "无法解析 Action",
                "parse_failed": True,
            }],
            "timing": {}, "token_usage": {}, "iteration_count": 1,
        }

    if kind == "finish":
        return {
            "history": [{"thought": thought, "action": "FINISH", "observation": "任务完成"}],
            "timing": {}, "token_usage": {}, "iteration_count": 1,
        }

    observation = ""
    if env is not None:
        args = dict(action.get("args") or {})
        args.setdefault("session_id", session_id)
        try:
            result = env.execute_tool(action["name"], args)
            if isinstance(result, list):
                observation = f"成功获取 {len(result)} 篇论文"
            else:
                observation = str(result)[:500]
        except Exception as exc:                      # noqa: BLE001
            observation = f"{TOOL_ERROR_PREFIX}{exc}"

    return {
        "history": [
            {
                "thought": thought,
                "action": json.dumps(action, ensure_ascii=False),
                "observation": observation,
            },
            {"thought": "", "action": "FINISH", "observation": "任务完成"},
        ],
        "timing": {}, "token_usage": {}, "iteration_count": 2,
    }


def _completion_text(completion: Any) -> str:
    """会话式数据集下 completion 是消息列表，标准格式下是字符串。"""
    if isinstance(completion, list):
        return "\n".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion or "")


def messages_to_trajectory(completion: Any) -> Dict[str, Any]:
    """Convert TRL native multi-turn messages into the project's history schema.

    Assistant tool-call messages become action steps and consume the following
    tool message as their observation.  The final assistant answer becomes a
    FINISH step.  This preserves the real Action -> Observation -> Action chain
    produced by GRPOTrainer's tool loop.
    """
    if not isinstance(completion, list):
        return synthesize_trajectory(_completion_text(completion))

    history: List[Dict[str, Any]] = []
    pending: List[int] = []
    for message in completion:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            if calls:
                for call in calls:
                    function = call.get("function", call) if isinstance(call, dict) else {}
                    name = function.get("name")
                    args = function.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    action = {"name": name, "args": args or {}}
                    history.append({
                        "thought": str(message.get("content") or ""),
                        "action": json.dumps(action, ensure_ascii=False),
                        "observation": "",
                    })
                    pending.append(len(history) - 1)
            elif str(message.get("content") or "").strip():
                history.append({
                    "thought": str(message.get("content") or ""),
                    "action": "FINISH",
                    "observation": "任务完成",
                })
        elif role == "tool" and pending:
            index = pending.pop(0)
            history[index]["observation"] = str(message.get("content") or "")

    if not history:
        return synthesize_trajectory(_completion_text(completion))
    return {
        "history": history,
        "timing": {},
        "token_usage": {},
        "iteration_count": len(history),
    }


def make_grpo_reward_fn(
    tasks_by_id: Dict[str, Dict[str, Any]],
    env: Any = None,
    reward_calc: Optional[RewardCalculator] = None,
    tracker: Any = None,
) -> Callable[..., List[float]]:
    """构造 TRL GRPOTrainer 用的 reward function。

    TRL 会把数据集中除 prompt/completion 之外的列按列名作为关键字参数传入，
    因此数据集需要带一个 `task_id` 列。

    Args:
        tracker: 可选的 `rl.observability.RewardComponentTracker`。只返回
            `breakdown.total` 会把五个分量丢掉，而课程会在前 30 步改变各分量
            的权重 —— 没有分量曲线就无法判断 reward 上升是策略变强还是权重
            表在动。传入 tracker 即把分量、权重与轨迹健康度一并记进训练日志。
    """
    calc = reward_calc or RewardCalculator()

    def grpo_reward_fn(completions=None, task_id=None, trainer_state=None,
                       trajectory_results=None,
                       **kwargs) -> List[float]:
        completions = completions or []
        ids = task_id or [None] * len(completions)

        # RewardCalculator 自带课程：训练早期把权重压在结构正确性上，
        # 后期才给语义正确性满权重。TRL 会把 trainer_state 传进来，
        # 顺手接上，让这套课程真正随训练推进（否则 training_step 恒为 0）。
        step = int(getattr(trainer_state, "global_step", 0) or 0)

        rewards: List[float] = []
        trajectories = trajectory_results or [None] * len(completions)
        for completion, tid, rollout_result in zip(completions, ids, trajectories):
            task_def = tasks_by_id.get(tid)
            if task_def is None:
                rewards.append(0.0)
                continue
            result = rollout_result or (
                messages_to_trajectory(completion)
                if isinstance(completion, list)
                else synthesize_trajectory(_completion_text(completion), env=env)
            )
            breakdown, _ = calc.compute_reward_breakdown(
                task_def, result, training_step=step
            )
            if tracker is not None:
                tracker.record(breakdown, result)
            rewards.append(float(breakdown.total))
        return rewards

    grpo_reward_fn.__name__ = "grpo_reward_fn"
    return grpo_reward_fn


def rollout_func_supported(trl_version: str) -> bool:
    """当前 TRL 是否会在非 vLLM 路径上调用 rollout_func。"""
    from packaging.version import Version

    return Version(trl_version) >= Version(MIN_TRL_FOR_ROLLOUT_FUNC)


def require_rollout_func_support() -> None:
    """TRL 太老就直接拦下，而不是让多轮 rollout 静默失效。

    这类失败特别难查：训练照常跑完、loss 也在动，只是每条 rollout 都退化成
    「一步 FINISH」，词沙拉也能拿到正分。宁可在启动时报错。
    """
    from importlib.metadata import version

    installed = version("trl")
    if not rollout_func_supported(installed):
        raise SystemExit(
            f"❌ 多轮 GRPO rollout 需要 trl >= {MIN_TRL_FOR_ROLLOUT_FUNC}，当前是 {installed}\n"
            f"   更早的版本只在 vLLM server 模式下调用 rollout_func，默认配置下多轮采样\n"
            f"   不会执行且不会报错 —— 奖励会退化成「任何非空输出都算完成」。\n"
            f"   升级: pip install -U 'trl>={MIN_TRL_FOR_ROLLOUT_FUNC}'"
        )


def make_multiturn_rollout_func(environment_factory, max_turns: int = 4):
    """Create a TRL custom rollout function for textual ReAct models.

    Every assistant turn is sampled from the current policy. Tool observations
    are appended to the completion token stream with ``env_mask=0``; generated
    policy tokens use ``env_mask=1`` and therefore participate in GRPO loss.
    The full history is forwarded to the reward function as an extra field.
    """
    require_rollout_func_support()
    max_turns = max(1, int(max_turns))

    def rollout_func(prompts, trainer):
        import copy

        tokenizer = trainer.processing_class
        # 不要再按 num_generations 展开：TRL 交进来的 prompts 已经是重复过的
        # （num_generations=2 时收到的是 2 条一模一样的 prompt）。再展开一次会让
        # 返回条数变成 N*G*G，与 TRL 期望的 N*G 对不上，在 shuffle_sequence_dict
        # 处炸成 IndexError。
        expanded_prompts = [copy.deepcopy(p) for p in prompts]
        prompt_ids = []
        for prompt in expanded_prompts:
            if isinstance(prompt, list):
                # transformers>=5 的 apply_chat_template(tokenize=True) 返回
                # BatchEncoding（含 "input_ids" 键），旧版返回纯 int 列表；统一取 token 序列。
                encoded = tokenizer.apply_chat_template(
                    prompt, tokenize=True, add_generation_prompt=True,
                )
                if "input_ids" in encoded:
                    ids = encoded["input_ids"]
                else:
                    ids = encoded
            else:
                ids = tokenizer(prompt)["input_ids"]
            prompt_ids.append(list(ids))

        environments = [environment_factory() for _ in expanded_prompts]
        trajectory_results = []
        completion_ids = [[] for _ in expanded_prompts]
        env_masks = [[] for _ in expanded_prompts]
        histories = [[] for _ in expanded_prompts]
        active = list(range(len(expanded_prompts)))
        full_ids = [list(ids) for ids in prompt_ids]

        # Dataset rows are expanded prompt-major, matching TRL's reward columns.
        for i, environment in enumerate(environments):
            environment.reset()

        for _turn in range(max_turns):
            if not active:
                break
            remaining = [trainer.max_completion_length - len(completion_ids[i]) for i in active]
            keep = [(i, r) for i, r in zip(active, remaining) if r > 0]
            if not keep:
                break
            active = [i for i, _ in keep]

            turn_ids, _, _ = trainer._generate_single_turn(
                [full_ids[i] for i in active], None, {}
            )
            next_active = []
            for batch_index, index in enumerate(active):
                budget = trainer.max_completion_length - len(completion_ids[index])
                # turn_ids 是 TRL 返回的完成 token；可能是 tensor，统一转成 int 列表，
                # 否则下一轮拼进 full_ids 后 torch.tensor() 无法处理
                generated = [int(t) for t in turn_ids[batch_index]][:budget]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                completion_ids[index].extend(generated)
                env_masks[index].extend([1] * len(generated))

                kind, action = parse_react_action(text)
                thought = _thought_of(text)
                if kind == "finish":
                    histories[index].append({
                        "thought": thought, "action": "FINISH", "observation": "任务完成"
                    })
                    continue
                if kind == "parse_error":
                    histories[index].append({
                        "thought": thought, "action": "PARSE_ERROR",
                        "observation": "无法解析 Action", "parse_failed": True,
                    })
                    continue

                args = dict(action.get("args") or {})
                try:
                    if action["name"] == "get_recently_submitted_cs_papers":
                        result = environments[index].get_recently_submitted_cs_papers(**args)
                    elif action["name"] == "search_arxiv_papers":
                        result = environments[index].search_arxiv_papers(**args)
                    elif action["name"] == "download_arxiv_pdf":
                        result = environments[index].download_arxiv_pdf(**args)
                    elif action["name"] == "translate_arxiv_pdf":
                        result = environments[index].translate_arxiv_pdf(**args)
                    elif action["name"] == "get_paper_cache_status":
                        result = environments[index].get_paper_cache_status(**args)
                    else:
                        raise ValueError(f"未知工具: {action['name']}")
                    observation = str(result)[:1000]
                except Exception as exc:  # noqa: BLE001
                    observation = f"{TOOL_ERROR_PREFIX}{exc}"

                histories[index].append({
                    "thought": thought,
                    "action": json.dumps(action, ensure_ascii=False),
                    "observation": observation,
                })
                suffix = f"\nObservation: {observation}\nThought:"
                suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
                suffix_ids = list(suffix_ids)[: max(0, trainer.max_completion_length - len(completion_ids[index]))]
                completion_ids[index].extend(suffix_ids)
                env_masks[index].extend([0] * len(suffix_ids))
                full_ids[index] = prompt_ids[index] + completion_ids[index]
                if len(completion_ids[index]) < trainer.max_completion_length:
                    next_active.append(index)
            active = next_active

        for history in histories:
            trajectory_results.append({
                "history": history,
                "timing": {},
                "token_usage": {},
                "iteration_count": len(history),
            })
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": None,
            "env_mask": env_masks,
            "trajectory_results": trajectory_results,
        }

    return rollout_func


def build_prompt_dataset(tasks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """由任务集直接生成 GRPO 数据集。

    GRPO 是在线算法：只需要 prompt 与可验证奖励，**不需要预先生成轨迹数据**，
    因此这里不像 SFT/DPO 那样依赖 scripts/generate_*_data.py。

    prompt 用的就是推理时真正送进模型的那段 ReAct prompt（含工具描述与格式约束），
    保证训练与推理的输入分布一致。
    """
    from agents.prompt_templates import format_tool_description, get_react_prompt
    from tools.tool_registry import registry

    tools_description = format_tool_description(registry.list_tools())
    rows = []
    for task in tasks:
        prompt = get_react_prompt(
            task=task["task"], tools_description=tools_description, history=""
        )
        rows.append({
            "prompt": [{"role": "user", "content": prompt}],
            "task_id": task["id"],
        })
    return rows


def build_multiturn_prompt_dataset(tasks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build conversational prompts for TRL native multi-turn tool calling."""
    rows = []
    for task in tasks:
        rows.append({
            "prompt": [
                {
                    "role": "system",
                    "content": (
                        "你是 arXiv 论文 Agent。根据任务选择工具；读取每次工具返回后再决定下一步。"
                        "需要多个工具时必须逐步执行，任务真正完成后给出简短最终回答。"
                    ),
                },
                {"role": "user", "content": task["task"]},
            ],
            "task_id": task["id"],
        })
    return rows


def load_mock_env(snapshot_path: Optional[Path] = None):
    """有快照就用 MockArxivEnv 离线执行工具，否则返回 None（不执行）。"""
    if snapshot_path is None or not Path(snapshot_path).exists():
        return None
    from rl.env import MockArxivEnv
    return MockArxivEnv(snapshot_path=Path(snapshot_path), mode="replay")
