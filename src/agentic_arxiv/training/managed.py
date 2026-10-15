"""受管训练接线；裸入口不启用新恢复协议或停止控制。"""

from pathlib import Path

from agentic_arxiv.config import workspace_path


def prepare_managed_training(managed_context, *, stage, output_dir,
                             resume_from_checkpoint=None):
    """在模型加载前检查尝试目录和显式恢复资格，不改写历史目录。"""
    if not managed_context:
        if resume_from_checkpoint and stage in {"sft", "dpo"}:
            raise ValueError("SFT/DPO 完整恢复必须提供 --managed-context；仅重载模型不是续训")
        return None
    from agentic_arxiv.training.checkpoint import ManagedCheckpointContext

    context = ManagedCheckpointContext.from_file(
        workspace_path(managed_context), expected_stage=stage,
    )
    if not all(isinstance(getattr(context, field), str) and getattr(context, field)
               for field in ("output_dir", "control_dir", "log_dir")):
        raise ValueError("受管身份缺少当前尝试 output_dir/control_dir/log_dir")
    output = workspace_path(output_dir).resolve()
    if output != Path(context.output_dir).resolve():
        raise ValueError("--output-dir 必须是受管身份文件声明的当前尝试目录")
    operational_dirs = [output, Path(context.control_dir).resolve(), Path(context.log_dir).resolve()]
    if any(not path.is_relative_to(output.parent) or path == output.parent
           for path in operational_dirs):
        raise ValueError("受管输出、控制和日志目录必须隔离在当前尝试内")
    for index, path in enumerate(operational_dirs):
        if any(path.is_relative_to(other) or other.is_relative_to(path)
               for other in operational_dirs[index + 1:]):
            raise ValueError("受管输出、控制和日志目录不能互相重叠")
    from agentic_arxiv.training.distributed import TrainingContext
    if TrainingContext.from_environment().is_main and output.exists() and any(output.iterdir()):
        raise ValueError(f"受管训练要求全新的尝试输出目录，拒绝覆盖: {output}")
    if resume_from_checkpoint:
        source = workspace_path(resume_from_checkpoint).resolve()
        if source.is_relative_to(output) or output.is_relative_to(source.parent):
            raise ValueError("恢复必须写入新尝试，输出不能覆盖恢复来源或原尝试")
    _validate_input_fingerprints(context.identity["inputs"], output_dir=output)
    _validate_input_fingerprints(context.identity["models"], output_dir=output)
    return context


def _validate_input_fingerprints(identities, *, output_dir=None):
    """输入只读复核；上下文自洽不代表磁盘上的原始输入仍未改变。"""
    from agentic_arxiv.provenance import file_fingerprint, model_input_fingerprint

    checked = {}

    def visit(value):
        if not isinstance(value, dict):
            return
        if isinstance(value.get("path"), str) and "sha256" in value and "status" in value:
            if file_fingerprint(value["path"]) != value:
                raise ValueError(f"受管文件指纹已改变: {value['path']}")
        if "path" in value and "fingerprint" in value:
            path, expected = value["path"], value["fingerprint"]
            if not isinstance(path, str) or not isinstance(expected, dict):
                raise ValueError("受管输入身份缺少明确路径或结构化指纹")
            target = Path(path).resolve()
            if output_dir is not None and (target.is_relative_to(output_dir)
                                           or output_dir.is_relative_to(target)):
                raise ValueError(f"受管输出不能与只读输入重叠: {path}")
            kind = "model" if "source" in expected else "file"
            key = (kind, path)
            if key not in checked:
                checked[key] = (model_input_fingerprint(path) if kind == "model"
                                else file_fingerprint(path))
            if checked[key] != expected:
                raise ValueError(f"受管输入指纹已改变: {path}")
            # 模型目录指纹已覆盖 manifest；其中的历史来源不能当作当前文件。
            if value.get("base_model"):
                visit(value["base_model"])
            if "runtime_dependencies" in value:
                dependencies = value["runtime_dependencies"]
                if not isinstance(dependencies, dict):
                    raise ValueError("受管运行依赖必须是明确文件身份清单")
                for dependency in dependencies.values():
                    if not isinstance(dependency, dict) or not {"path", "fingerprint"} <= dependency.keys():
                        raise ValueError("受管运行依赖缺少文件身份")
                    visit(dependency)
            return
        if "inputs" in value and "models" not in value:
            visit(value["inputs"])
            visit(value.get("policies", {}))
            return
        for child in value.values():
            if isinstance(child, dict):
                visit(child)

    visit(identities)


def attach_managed_training(trainer, context, execution, *,
                            resume_from_checkpoint=None, state_provider=None,
                            state_restorer=None, protection_reason=None):
    """完整校验和附加状态恢复先于 train；标准状态继续由 Trainer 恢复。"""
    if context is None:
        return None, None
    from agentic_arxiv.training.checkpoint import attach_managed_checkpoint
    from agentic_arxiv.training.callbacks import ManagedStopCallback

    def checkpoint_state():
        # 保护器可能在 on_log 才触发，必须在提交清单前读取最终原因。
        reason = protection_reason() if protection_reason is not None else None
        trainer._managed_termination_reason = reason
        return state_provider() if state_provider is not None else {}

    adapter = attach_managed_checkpoint(
        trainer, context=context, training_context=execution,
        resume_from_checkpoint=resume_from_checkpoint,
        state_provider=checkpoint_state, state_restorer=state_restorer,
    )
    callback = ManagedStopCallback(
        context.control_dir, context.run_id, context.attempt_id,
        distributed=execution,
        checkpoint_committed=lambda step: adapter.last_checkpoint,
        protection_reason=protection_reason or (lambda: None),
    )
    trainer.add_callback(callback)
    return callback, adapter


def validate_managed_config(context, requested, configured):
    """核对真正执行的参数，不仅比较上下文文件与 checkpoint 自报的身份。"""
    if context is None:
        return
    expected = context.identity["training_config"]
    if (set(expected) != {"defaults", "trainer_parameters"}
            or not isinstance(expected["defaults"], dict)
            or not isinstance(expected["trainer_parameters"], dict)):
        raise ValueError("受管 training_config 必须明确 defaults 和 trainer_parameters")
    operational = {"output_dir", "run_name", "rollout_trace_path", "resume_from_checkpoint"}
    path_fields = {"model", "data", "data_manifest", "snapshot", "base_model", "split"}
    differences = []
    for key, value in expected["defaults"].items():
        if key in operational:
            continue
        actual = requested.get(key)
        if key in path_fields and actual:
            if key == "split" and ":" in str(actual):
                filename, split = str(actual).rsplit(":", 1)
                actual = f"{workspace_path(filename)}:{split}"
            elif key != "split":
                actual = str(workspace_path(actual))
        if actual != value:
            differences.append(f"defaults.{key}")
    if expected["trainer_parameters"] != configured["trainer_parameters"]:
        differences.append("trainer_parameters")
    epoch_budget = getattr(context, "epoch_budget", None)
    if epoch_budget is not None:
        if requested.get("max_steps") != -1:
            differences.append("epoch_budget.max_steps")
        if requested.get("epochs") != epoch_budget:
            differences.append("epoch_budget.epochs")
    elif requested.get("max_steps") != context.total_budget:
        differences.append("total_budget")
    if differences:
        raise ValueError(f"受管尝试执行配置与不可变身份不一致: {', '.join(differences)}")


def record_training_result(context, execution, *, training, validation, global_step,
                           reason=None, checkpoint=None, planned_steps=None,
                           best_effort=False, **metrics):
    """独立发布训练/验证事实；异常分支不启动任何额外 collective。"""
    if context is None or not execution.is_main:
        return
    from agentic_arxiv.training.control import atomic_control_json

    # Epoch 预算由 Trainer 按真实 dataloader/DDP/累积配置核定，不自行换算。
    planned_steps = planned_steps if type(planned_steps) is int and planned_steps > 0 else None
    total_budget = context.total_budget if context.total_budget is not None else planned_steps
    payload = {
        "schema_version": 1, "run_id": context.run_id, "attempt_id": context.attempt_id,
        "training": training, "validation": validation, "global_step": global_step,
        "reason": reason, "checkpoint": str(checkpoint) if checkpoint else None,
        "planned_steps": planned_steps, "total_budget": total_budget,
        "epoch_budget": getattr(context, "epoch_budget", None),
        **metrics,
    }
    try:
        atomic_control_json(Path(context.control_dir) / "training-result.json", payload)
    except Exception as exc:
        if not best_effort:
            raise
        print(f"无法保存受管结果诊断: {type(exc).__name__}")


def managed_cancellation_check(context):
    """验证按 episode 边界查询本尝试停止请求，不产生训练恢复点。"""
    if context is None:
        return None
    from agentic_arxiv.training.control import read_stop_request

    return lambda: read_stop_request(context.control_dir, context.run_id, context.attempt_id) is not None
