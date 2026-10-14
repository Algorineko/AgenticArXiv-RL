"""GRPO 的独立多轮 token 采样与只读实际轨迹奖励。

解析、工具状态和观测使用共同 ReAct 契约。奖励回调不执行工具、不补
FINISH；组内相对优势只由 GRPOTrainer 计算，避免二次归一化。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from agentic_arxiv.contracts import (
    ParseKind,
    TOOL_ERROR_PREFIX,
    TerminationType,
    make_history_step,
    make_terminal_step,
)
from agentic_arxiv.runtime.react_contract import (
    append_react_turn,
    build_react_messages,
    parse_react_response,
    protocol_metadata,
    validate_react_messages,
)
from agentic_arxiv.scoring.reward import RewardCalculator
from agentic_arxiv.provenance import evaluation_protocol_relation, read_protocol_metadata
from agentic_arxiv.tasks.initialization import compile_initialization
from agentic_arxiv.tools.bootstrap import TOOL_MODULES

# TRL 从 0.28.0 起才会在**非 vLLM** 路径上调用 rollout_func，签名也才是
# rollout_func(prompts, trainer)。更早的版本里它只在 use_vllm 且
# vllm_mode == "server" 时被调用 —— 默认配置下压根不会执行，多轮 rollout
# 静默失效，因此必须在构造多轮采样器前检查版本。
MIN_TRL_FOR_ROLLOUT_FUNC = "0.28.0"

# 工具能力与审计名单由唯一注册入口的既有清单派生。
SUPPORTED_TOOL_NAMES = {name for names in TOOL_MODULES.values() for name in names}


def _dispatch_environment_tool(environment: Any, name: str, args: Mapping[str, Any]) -> Any:
    """Execute one allow-listed tool against a per-rollout environment."""
    tool_name = str(name)
    if tool_name not in SUPPORTED_TOOL_NAMES:
        raise ValueError(f"未知工具: {tool_name}")
    if callable(getattr(environment, "step", None)):
        return environment.step({"name": tool_name, "args": dict(args)})
    method = getattr(environment, tool_name, None)
    if method is None:
        raise ValueError(f"环境未实现工具: {tool_name}")
    clean_args = {
        key: value for key, value in dict(args or {}).items()
        if key != "session_id"
    }
    return method(**clean_args)


def _resolve_prompt_task_id(
    prompt: Any,
    prompt_task_ids: Mapping[str, str],
) -> Optional[str]:
    """Recover a task id without exposing it to the policy text.

    TRL versions differ in whether a custom rollout receives raw chat messages
    or an already-rendered string.  Exact lookup handles raw messages; the
    substring fallback handles a rendered chat template containing the same
    full ReAct prompt.
    """
    candidates: List[str] = []
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, Mapping):
                hidden = message.get("_task_id")
                if hidden:
                    return str(hidden)
                content = message.get("content")
                if isinstance(content, str):
                    candidates.append(content)
    elif isinstance(prompt, str):
        candidates.append(prompt)

    for text in candidates:
        direct = prompt_task_ids.get(text)
        if direct is not None:
            return str(direct)
        matches = {
            str(task_id)
            for known_text, task_id in prompt_task_ids.items()
            if known_text and known_text in text
        }
        if len(matches) == 1:
            return matches.pop()
    return None


def _strip_prompt_task_metadata(prompt: Any) -> Any:
    """Remove rollout-only task identity before tokenization.

    Some benchmark cases intentionally share the exact same visible user text
    while differing in hidden session setup.  ``_task_id`` travels inside the
    structured message only so the rollout can choose the right environment;
    stripping it here guarantees it never becomes a policy token even if a
    future chat template starts rendering unknown message fields.
    """
    if not isinstance(prompt, list):
        return prompt
    cleaned = []
    for message in prompt:
        if isinstance(message, Mapping):
            item = dict(message)
            item.pop("_task_id", None)
            cleaned.append(item)
        else:
            cleaned.append(message)
    return cleaned


def _prepare_rollout_environment(
    environment: Any,
    task_id: Optional[str],
    tasks_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Reset one rollout and apply task setup outside the policy trajectory.

    ``TaskSpec.setup`` describes pre-existing session state. BenchmarkRunner
    executes it before the Agent and does not score it as a model action. GRPO
    must do exactly the same; otherwise reference tasks are impossible while
    their reward oracle still expects a direct paper operation.
    """
    if not tasks_by_id:
        environment.reset()
        return []
    if not task_id or task_id not in tasks_by_id:
        raise ValueError(
            "GRPO rollout 无法将 prompt 映射回 task_id，"
            "不能安全执行任务 setup"
        )

    task_def = tasks_by_id[task_id]
    initialization_plan = compile_initialization(task_def, task_lookup=tasks_by_id)
    if callable(getattr(environment, "setup", None)):
        environment.reset(initialization_plan=initialization_plan)
        return [dict(action) for action in environment.setup_actions]

    # 保留已有测试环境的薄接口适配，不维护第二套真实环境状态。
    environment.reset(task_id=task_id)
    applied: List[Dict[str, Any]] = []
    for action in initialization_plan:
        name = str(action.get("name") or "")
        args = dict(action.get("args") or {})
        try:
            _dispatch_environment_tool(environment, name, args)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"任务 {task_id} 的 GRPO setup 执行失败: "
                f"tool={name}, args={args}, error={exc}"
            ) from exc
        applied.append({"name": name, "args": args})
    return applied


def parse_react_action(completion: str):
    """旧调用接口的薄适配；动作判定只在共同解析器中实现。"""
    parsed = parse_react_response(completion)
    return parsed.kind.value.lower(), parsed.action


def make_grpo_reward_fn(
    tasks_by_id: Dict[str, Dict[str, Any]],
    reward_calc: Optional[RewardCalculator] = None,
    tracker: Any = None,
    auditor: Any = None,
) -> Callable[..., List[float]]:
    """构造 TRL GRPOTrainer 用的 reward function。

    TRL 会把数据集中除 prompt/completion 之外的列按列名作为关键字参数传入，
    因此数据集需要带一个 `task_id` 列。

    Args:
        tracker: 可选的 `training.observability.RewardComponentTracker`。只返回
            `breakdown.total` 会把五个分量丢掉，而课程会在前 30 步改变各分量
            的权重 —— 没有分量曲线就无法判断 reward 上升是策略变强还是权重
            表在动。传入 tracker 即把分量、权重与轨迹健康度一并记进训练日志。
    """
    calc = reward_calc or RewardCalculator()

    def grpo_reward_fn(completions=None, task_id=None, trainer_state=None,
                       trajectory_results=None,
                       **kwargs) -> List[float]:
        completions = completions or []
        if task_id is None:
            ids = [None] * len(completions)
        elif isinstance(task_id, str):
            # 保持对单样本/手工调用的兼容；TRL 正常训练时传入的是列表。
            ids = [task_id] * len(completions)
        else:
            ids = list(task_id)
        if len(ids) != len(completions):
            raise ValueError(
                "GRPO reward 输入长度不一致: "
                f"completions={len(completions)}, task_ids={len(ids)}"
            )

        # RewardCalculator 自带课程：训练早期把权重压在结构正确性上，
        # 后期才给语义正确性满权重。TRL 会把 trainer_state 传进来，
        # 顺手接上，让这套课程真正随训练推进（否则 training_step 恒为 0）。
        step = int(getattr(trainer_state, "global_step", 0) or 0)

        rewards: List[float] = []
        breakdowns: List[Any] = []
        resolved_trajectories: List[Dict[str, Any]] = []
        if trajectory_results is None:
            raise ValueError("GRPO reward 缺少实际 trajectory_results，禁止合成轨迹降级")
        trajectories = list(trajectory_results)
        if len(trajectories) != len(completions):
            raise ValueError(
                "GRPO reward 输入长度不一致: "
                f"completions={len(completions)}, trajectories={len(trajectories)}"
            )
        for index, (tid, result) in enumerate(zip(ids, trajectories)):
            if not isinstance(result, Mapping) or not isinstance(result.get("history"), list):
                raise ValueError(f"GRPO reward 第 {index} 条实际轨迹缺少完整 history")
            if result.get("task_id") is not None and result["task_id"] != tid:
                raise ValueError(f"GRPO reward 第 {index} 条实际轨迹 task_id 不一致")
            history = result["history"]
            if any(not isinstance(item, Mapping) or "action" not in item for item in history):
                raise ValueError(f"GRPO reward 第 {index} 条实际轨迹 history 非法")
            last_action = str(history[-1].get("action") or "") if history else ""
            reason = result.get("termination_reason")
            if reason and reason not in {item.value for item in TerminationType} | {"MODEL_END"}:
                raise ValueError(f"GRPO reward 第 {index} 条实际轨迹终止原因非法")
            if reason in {"FINISH", "PARSE_ERROR"} and last_action != reason:
                raise ValueError(f"GRPO reward 第 {index} 条实际轨迹终止步骤缺失")
            if (
                last_action not in {"FINISH", "PARSE_ERROR", "ERROR", "FORCE_STOP"}
                and not result.get("termination_reason")
                and not result.get("clipped")
                and not result.get("reached_max_turns")
            ):
                raise ValueError(f"GRPO reward 第 {index} 条实际轨迹缺少终止原因")
        for completion, tid, rollout_result in zip(completions, ids, trajectories):
            task_def = tasks_by_id.get(tid)
            if task_def is None:
                rewards.append(0.0)
                breakdowns.append(None)
                resolved_trajectories.append(rollout_result)
                continue
            result = rollout_result
            breakdown, _ = calc.compute_reward_breakdown(
                task_def, result, training_step=step
            )
            if tracker is not None:
                tracker.record(breakdown, result)
            rewards.append(float(breakdown.total))
            breakdowns.append(breakdown)
            resolved_trajectories.append(result)
        if tracker is not None:
            tracker.record_group(ids, rewards)
        if auditor is not None:
            auditor.record_batch(
                completions=completions,
                task_ids=ids,
                trajectories=resolved_trajectories,
                breakdowns=breakdowns,
                rewards=rewards,
                training_step=step,
            )
        return rewards

    grpo_reward_fn.__name__ = "grpo_reward_fn"
    return grpo_reward_fn


def rollout_func_supported(trl_version: str) -> bool:
    """当前 TRL 是否会在非 vLLM 路径上调用 rollout_func。"""
    from packaging.version import Version

    return Version(trl_version) >= Version(MIN_TRL_FOR_ROLLOUT_FUNC)


def require_rollout_func_support() -> None:
    """TRL 太老就直接拦下，而不是让多轮 rollout 静默失效。

    旧实现会静默降级为合成 FINISH；现在缺少实际轨迹会拒绝评分。
    在训练开始前拦下不调用 rollout_func 的版本，避免走到奖励阶段才失败。
    """
    from importlib.metadata import version

    installed = version("trl")
    if not rollout_func_supported(installed):
        raise SystemExit(
            f"❌ 多轮 GRPO rollout 需要 trl >= {MIN_TRL_FOR_ROLLOUT_FUNC}，当前是 {installed}\n"
            f"   更早的版本只在 vLLM server 模式下调用 rollout_func，默认配置下多轮采样\n"
            f"   不会执行且不会报错；当前奖励回调会拒绝缺少实际轨迹的输入。\n"
            f"   升级: pip install -U 'trl>={MIN_TRL_FOR_ROLLOUT_FUNC}'"
        )


def _chat_template_ids(tokenizer: Any, messages: Sequence[Mapping[str, Any]]) -> List[int]:
    """只用当前指定模板编码消息，未知返回形状明确失败。"""
    validate_react_messages(messages)
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise ValueError("GRPO 多消息采样需要 tokenizer.apply_chat_template")
    encoded = apply_template(messages, tokenize=True, add_generation_prompt=True)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError("GRPO 单条消息模板返回了多个 token 序列")
        ids = ids[0]
    return [int(token) for token in ids]


def _append_environment_tokens(
    tokenizer: Any, messages: Sequence[Mapping[str, Any]], policy_text: str,
    observation: str, sampled_prefix: Sequence[int],
) -> tuple[List[int], List[Dict[str, Any]]]:
    """比较模板候选前缀，只返回可追加尾片段，绝不替换采样 token。"""
    appended = append_react_turn(messages, policy_text, observation)
    candidate = _chat_template_ids(tokenizer, appended)
    prefix = list(sampled_prefix)
    if candidate[:len(prefix)] != prefix:
        raise ValueError(
            "GRPO chat template 无法保留原始采样 token 前缀；"
            "不能通过 decode/re-encode 替换策略输出"
        )
    # 模板候选已包括策略实际采样的 EOS，前缀扣除后不会再次插入它。
    return candidate[len(prefix):], appended


def make_multiturn_rollout_func(
    environment_factory,
    max_turns: int = 6,
    *,
    tasks_by_id: Optional[Mapping[str, Mapping[str, Any]]] = None,
    prompt_task_ids: Optional[Mapping[str, str]] = None,
    training_metadata: Optional[Mapping[str, Any]] = None,
):
    """Create a TRL custom rollout function for textual ReAct models.

    Every assistant turn is sampled from the current policy. Tool observations
    are appended to the completion token stream with ``env_mask=0``; generated
    policy tokens use ``env_mask=1`` and therefore participate in GRPO loss.
    The full history is forwarded to the reward function as an extra field.
    """
    require_rollout_func_support()
    max_turns = max(1, int(max_turns))

    def run_batch(prompts, trainer, environments):
        import copy

        tokenizer = trainer.processing_class
        sampling_metadata = protocol_metadata(tokenizer)
        sampling_metadata.pop("training_protocol", None)
        sampling_metadata["sampling_protocol"] = sampling_metadata["context_version"]
        source_protocol = read_protocol_metadata(training_metadata)
        # 不要再按 num_generations 展开：TRL 交进来的 prompts 已经是重复过的
        # （num_generations=2 时收到的是 2 条一模一样的 prompt）。再展开一次会让
        # 返回条数变成 N*G*G，与 TRL 期望的 N*G 对不上，在 shuffle_sequence_dict
        # 处炸成 IndexError。
        received_prompts = [copy.deepcopy(p) for p in prompts]
        task_lookup = tasks_by_id or {}
        prompt_lookup = prompt_task_ids or {}
        rollout_task_ids = [
            _resolve_prompt_task_id(prompt, prompt_lookup)
            for prompt in received_prompts
        ]
        # 先恢复环境身份，再剔除隐藏字段。初始化必须先于policy输入编码，
        # 不能把 TaskSpec 的 setup 声明伪装成已成功执行的会话状态。
        expanded_prompts = [
            _strip_prompt_task_metadata(prompt) for prompt in received_prompts
        ]
        messages = [
            prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
            for prompt in expanded_prompts
        ]
        environments.extend(environment_factory() for _ in expanded_prompts)
        applied_setups = []
        for index, (task_id, environment) in enumerate(zip(rollout_task_ids, environments)):
            applied_setups.append(
                _prepare_rollout_environment(environment, task_id, task_lookup)
            )
            actual_setup = getattr(environment, "setup_context", None)
            if task_id in task_lookup and actual_setup is not None:
                from agentic_arxiv.runtime.prompt_templates import format_tool_description
                from agentic_arxiv.tools.tool_registry import registry

                messages[index] = build_react_messages(
                    task_lookup[task_id]["task"],
                    format_tool_description(registry.list_tools()),
                    visible_setup_context=actual_setup,
                )
        prompt_ids = [_chat_template_ids(tokenizer, prompt) for prompt in messages]

        trajectory_results = []
        completion_ids = [[] for _ in expanded_prompts]
        env_masks = [[] for _ in expanded_prompts]
        histories = [[] for _ in expanded_prompts]
        raw_assistant_turns = [[] for _ in expanded_prompts]
        termination_reasons = [None for _ in expanded_prompts]
        active = list(range(len(expanded_prompts)))
        full_ids = [list(ids) for ids in prompt_ids]

        for _turn in range(max_turns):
            if not active:
                break
            remaining = [trainer.max_completion_length - len(completion_ids[i]) for i in active]
            for index, budget in zip(active, remaining):
                if budget <= 0:
                    termination_reasons[index] = TerminationType.TOKEN_LIMIT.value
            keep = [(i, r) for i, r in zip(active, remaining) if r > 0]
            if not keep:
                break
            active = [i for i, _ in keep]

            turn_ids, _, _ = trainer._generate_single_turn(
                [full_ids[i] for i in active], None, {}
            )
            if len(turn_ids) != len(active):
                raise ValueError("GRPO 单轮采样条数与活动 prompt 条数不一致")
            next_active = []
            for batch_index, index in enumerate(active):
                budget = trainer.max_completion_length - len(completion_ids[index])
                # turn_ids 是 TRL 返回的完成 token；可能是 tensor，统一转成 int 列表，
                # 否则下一轮拼进 full_ids 后 torch.tensor() 无法处理
                generated = [int(t) for t in turn_ids[batch_index]]
                raw_text = tokenizer.decode(generated, skip_special_tokens=True)
                raw_assistant_turns[index].append(raw_text)
                # Match inference's Observation stop before applying the budget:
                # the budget itself may split the stop marker across tokens.
                policy_text = parse_react_response(raw_text).policy_text
                generated = generated[:budget]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                if policy_text != raw_text:
                    # Keep only an original sampled prefix, never re-tokenize it.
                    # A token overlapping the boundary must be dropped whole.
                    while generated and not policy_text.startswith(text):
                        generated.pop()
                        text = tokenizer.decode(generated, skip_special_tokens=True)
                if not text.strip() and not completion_ids[index]:
                    raise ValueError("GRPO 首轮没有合法策略 token，不能保留伪造 Observation 或补造 token")
                completion_ids[index].extend(generated)
                env_masks[index].extend([1] * len(generated))

                if not raw_text.strip():
                    histories[index].append(make_terminal_step(
                        TerminationType.MODEL_END.value, error="模型结束但未给出 FINISH",
                        raw_response=raw_text, policy_text=text,
                    ))
                    termination_reasons[index] = TerminationType.MODEL_END.value
                    continue

                parsed = replace(parse_react_response(text), raw_response=raw_text)
                if parsed.kind == ParseKind.FINISH:
                    histories[index].append(make_history_step(parsed))
                    termination_reasons[index] = "FINISH"
                    continue
                if parsed.kind == ParseKind.PARSE_ERROR:
                    histories[index].append(make_history_step(parsed))
                    termination_reasons[index] = (
                        TerminationType.TOKEN_LIMIT.value
                        if len(completion_ids[index]) >= trainer.max_completion_length
                        else "PARSE_ERROR"
                    )
                    continue

                action = parsed.action
                error = None
                result = None
                try:
                    result = _dispatch_environment_tool(
                        environments[index], action["name"], action["args"]
                    )
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)

                step_result = make_history_step(parsed, result=result, error=error)
                histories[index].append(step_result)
                observation = step_result["observation"]
                sampled_prefix = prompt_ids[index] + completion_ids[index]
                suffix_ids, messages[index] = _append_environment_tokens(
                    tokenizer, messages[index], text, observation, sampled_prefix,
                )
                suffix_ids = suffix_ids[: max(0, trainer.max_completion_length - len(completion_ids[index]))]
                completion_ids[index].extend(suffix_ids)
                env_masks[index].extend([0] * len(suffix_ids))
                full_ids[index] = prompt_ids[index] + completion_ids[index]
                if len(completion_ids[index]) < trainer.max_completion_length:
                    next_active.append(index)
                else:
                    termination_reasons[index] = TerminationType.TOKEN_LIMIT.value
            active = next_active

        for index, history in enumerate(histories):
            last_action = str(history[-1].get("action") or "") if history else ""
            clipped = (
                len(completion_ids[index]) >= trainer.max_completion_length
                and last_action != "FINISH"
            )
            reason = termination_reasons[index] or TerminationType.MAX_TURNS.value
            if reason in {TerminationType.MAX_TURNS.value, TerminationType.TOKEN_LIMIT.value}:
                history.append(make_terminal_step(reason, error=(
                    "达到最大迭代次数" if reason == TerminationType.MAX_TURNS.value
                    else "completion token 预算耗尽"
                )))
            trajectory_results.append({
                "history": history,
                "timing": {},
                "token_usage": {
                    "completion_tokens": len(completion_ids[index]),
                    "policy_tokens": sum(env_masks[index]),
                },
                "iteration_count": len(raw_assistant_turns[index]),
                "raw_assistant_turns": raw_assistant_turns[index],
                "task_id": rollout_task_ids[index],
                "setup_actions": applied_setups[index],
                "clipped": clipped,
                "reached_max_turns": (
                    len(raw_assistant_turns[index]) >= max_turns
                    and termination_reasons[index] is None
                ),
                "termination_reason": reason,
                **sampling_metadata,
                "training_protocol_metadata": source_protocol,
                "protocol_relation": evaluation_protocol_relation(source_protocol, sampling_metadata),
            })
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": None,
            "env_mask": env_masks,
            "trajectory_results": trajectory_results,
        }

    def rollout_func(prompts, trainer):
        environments = []
        try:
            return run_batch(prompts, trainer, environments)
        finally:
            for environment in environments:
                close = getattr(environment, "close", None)
                if callable(close):
                    close()

    return rollout_func


def build_prompt_dataset(tasks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """由任务集直接生成 GRPO 数据集。

    GRPO 是在线算法：只需要 prompt 与可验证奖励，**不需要预先生成轨迹数据**，
    因此这里不像 SFT 那样依赖离线数据生成入口。

    行中的消息使用共同任务/工具构造器，不把 setup 声明当成执行事实。
    rollout 在实际环境初始化后重建过滤后的初态，再进行首次编码。
    """
    from agentic_arxiv.runtime.prompt_templates import (
        format_tool_description,
    )
    from agentic_arxiv.tools.tool_registry import registry

    tools_description = format_tool_description(registry.list_tools())
    rows = []
    for task in tasks:
        prompt = build_react_messages(task["task"], tools_description)
        prompt[0]["_task_id"] = task["id"]
        rows.append({
            "prompt": prompt,
            "task_id": task["id"],
        })
    return rows


def load_mock_env(snapshot_path: Optional[Path] = None):
    """读取已有离线后端，缺快照明确失败，不降级为无工具评分。"""
    if snapshot_path is None or not Path(snapshot_path).exists():
        raise FileNotFoundError(f"GRPO 离线快照不存在: {snapshot_path}")
    from agentic_arxiv.runtime.env import MockArxivEnv
    return MockArxivEnv(snapshot_path=Path(snapshot_path), mode="replay")
