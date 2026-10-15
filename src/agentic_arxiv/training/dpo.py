"""可选离线 DPO：严格审计真实单步偏好，使用独立、冻结的完整 SFT 参考。"""

from __future__ import annotations

import gc
import inspect
import json
import math
import time
from functools import partial
from pathlib import Path

from agentic_arxiv.config import get_workspace, workspace_path
from agentic_arxiv.contracts import ParseKind
from agentic_arxiv.llm.artifacts import artifact_metadata
from agentic_arxiv.provenance import (
    build_run_manifest, file_fingerprint, load_dpo_package,
    model_input_fingerprint, require_training_protocol, source_fingerprint,
)
from agentic_arxiv.runtime.react_contract import (
    parse_react_response, protocol_metadata, validate_react_messages,
)
from agentic_arxiv.training.distributed import TrainingContext
from agentic_arxiv.training.dataloader_resume import preserve_managed_dataloader_epoch
from agentic_arxiv.training.managed import (
    prepare_managed_training, attach_managed_training, validate_managed_config,
    record_training_result,
    managed_cancellation_check,
)
from agentic_arxiv.training.mainline_config import (
    config_file_path, load_mainline_config, require_supported_config,
    resolve_model_source, stage_config, stage_config_source,
)
from agentic_arxiv.training.model_setup import (
    _assert_lora_only_trainable, _trainable_parameter_stats,
    load_reference_policy, load_training_policy,
)
from agentic_arxiv.training.observability import describe_logging, resolve_report_to
from agentic_arxiv.training.stage_verifier import StageVerifier

DPO_DEFAULTS = stage_config("dpo")["defaults"]
SFT_STAGES = {"sft_qlora", "sft_full"}


def _token_ids(value, *, label: str) -> list[int]:
    """不猜测 BatchEncoding 的长度，也不接受批处理或空 token 序列。"""
    if not isinstance(value, list) or not value or any(type(token) is not int for token in value):
        raise ValueError(f"{label} 必须为非空的单条 token ID 列表")
    return value


def _audit_token_boundaries(tokenizer, rows: list[dict], max_length: int) -> list[dict]:
    """逐项复现 TRL conversational 模板路径，将其前缀警告提升为拒绝条件。"""
    if type(max_length) is not int or max_length < 1:
        raise ValueError("DPO max_length 必须为正整数")
    if not rows:
        raise ValueError("DPO 数据不能为空")
    audited = []
    for number, row in enumerate(rows, 1):
        prompt = row["prompt"]
        validate_react_messages(prompt)
        prompt_ids = _token_ids(tokenizer.apply_chat_template(
            prompt, tools=None, add_generation_prompt=True,
            tokenize=True, return_dict=False,
        ), label=f"第 {number} 行 prompt")
        tokenized = {"prompt_ids": prompt_ids}
        for side in ("chosen", "rejected"):
            completion = row[side]
            if not isinstance(completion, list) or len(completion) != 1:
                raise ValueError(f"第 {number} 行 {side} 必须只有一条当前 assistant")
            validate_react_messages(prompt + completion, require_completion=True)
            response = completion[0]["content"]
            parsed = parse_react_response(response)
            if (not response.strip() or parsed.kind not in (ParseKind.CALL, ParseKind.FINISH)
                    or parsed.policy_text != response):
                raise ValueError(f"第 {number} 行 {side} 不是完整真实的单步 CALL/FINISH")
            processed = tokenizer.apply_chat_template(
                prompt + completion, tools=None, tokenize=True, return_dict=True,
            )
            full_ids = _token_ids(processed["input_ids"], label=f"第 {number} 行 {side}")
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError(f"第 {number} 行 {side} 的实际模板 token 改变了完整 prompt 前缀")
            if len(full_ids) <= len(prompt_ids):
                raise ValueError(f"第 {number} 行 {side} 没有当前目标 token")
            if len(full_ids) > max_length:
                raise ValueError(
                    f"第 {number} 行 {side} 长度 {len(full_ids)} 超过 max_length={max_length}；"
                    "严格 DPO 不截断上下文或目标，请显式提高长度预算"
                )
            tokenized[f"{side}_ids"] = full_ids[len(prompt_ids):]
        audited.append(tokenized)
    return audited


def _assert_prepared_dataset(dataset, audited: list[dict]) -> None:
    """训练器再次模板化的结果必须与前置审计逐 token 一致。"""
    if len(dataset) != len(audited):
        raise ValueError("DPOTrainer 预处理改变了已审计样本数")
    for number, (actual, expected) in enumerate(zip(dataset, audited), 1):
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(f"DPOTrainer 第 {number} 行 {key} 与实际模板审计不一致")


def _assert_independent_reference(policy, reference, optimizer=None) -> None:
    """同一来源不等于同一实例；共享 storage 的视图也必须拒绝。"""
    if policy is reference:
        raise ValueError("DPO policy/reference 不能是同一模型")
    reference_parameters = list(reference.parameters())
    if any(parameter.requires_grad for parameter in reference_parameters):
        raise ValueError("DPO reference 必须完全冻结")
    if reference.training:
        raise ValueError("DPO reference 必须处于 eval 模式")
    def storages(model):
        return {(str(parameter.device), parameter.untyped_storage().data_ptr())
                for parameter in model.parameters() if parameter.numel()}
    if storages(policy) & storages(reference):
        raise ValueError("DPO policy/reference 不能共享可变参数存储")
    if optimizer is not None:
        reference_ids = {id(parameter) for parameter in reference_parameters}
        if any(id(parameter) in reference_ids for group in optimizer.param_groups
               for parameter in group["params"]):
            raise ValueError("DPO reference 参数进入了策略优化器")


def _require_trainer_interface(trainer_cls) -> None:
    required = {"model", "ref_model", "args", "train_dataset", "processing_class"}
    missing = required - inspect.signature(trainer_cls.__init__).parameters.keys()
    if missing:
        raise ValueError(f"当前 DPOTrainer 缺少必要显式接口: {sorted(missing)}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(
    model: str = DPO_DEFAULTS["model"], data: str = DPO_DEFAULTS["data"],
    data_manifest: str = DPO_DEFAULTS["data_manifest"],
    output_dir: str = DPO_DEFAULTS["output_dir"], epochs: int = DPO_DEFAULTS["epochs"],
    max_steps: int = DPO_DEFAULTS["max_steps"], batch_size: int = DPO_DEFAULTS["batch_size"],
    grad_accum: int = DPO_DEFAULTS["grad_accum"], lr: float = DPO_DEFAULTS["lr"],
    beta: float = DPO_DEFAULTS["beta"], max_length: int = DPO_DEFAULTS["max_length"],
    qlora: bool = DPO_DEFAULTS["qlora"], lora_r: int = DPO_DEFAULTS["lora_r"],
    lora_alpha: int = DPO_DEFAULTS["lora_alpha"], lora_dropout: float = DPO_DEFAULTS["lora_dropout"],
    gradient_checkpointing: bool = DPO_DEFAULTS["gradient_checkpointing"],
    optim: str = DPO_DEFAULTS["optim"], inspect_only: bool = DPO_DEFAULTS["inspect_only"],
    seed: int = DPO_DEFAULTS["seed"], verify: bool = DPO_DEFAULTS["verify"],
    min_reward: float = DPO_DEFAULTS["min_reward"], report_to: str = DPO_DEFAULTS["report_to"],
    run_name: str = DPO_DEFAULTS["run_name"], base_model: str = DPO_DEFAULTS["base_model"],
    snapshot: str = DPO_DEFAULTS["snapshot"], config_path: str = None,
    managed_context: str = None, resume_from_checkpoint: str = None,
):
    requested_config = dict(locals())
    managed = prepare_managed_training(
        managed_context, stage="dpo", output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    configured = stage_config("dpo", config_path)
    validate_managed_config(managed, requested_config, configured)
    execution = TrainingContext.from_environment()
    effective_batch = execution.effective_batch_size(batch_size, grad_accum)
    for name, value in (("epochs", epochs), ("lr", lr), ("beta", beta)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"DPO {name} 必须为正有限数")
    if type(max_steps) is not int or max_steps == 0 or max_steps < -1:
        raise ValueError("DPO max_steps 必须为 -1 或正整数")
    if not math.isfinite(min_reward):
        raise ValueError("DPO 验证阈值必须为有限数")
    data_path = workspace_path(data)
    manifest_path = (workspace_path(data_manifest) if data_manifest
                     else data_path.with_suffix(data_path.suffix + ".manifest.json"))
    resolved = resolve_model_source(model, local_only=True)
    source_metadata = artifact_metadata(
        resolved, base_model=base_model, allowed_stages=SFT_STAGES,
        expected_protocol=protocol_metadata(), local_base_only=True,
    )
    source_model = model_input_fingerprint(resolved)
    if managed is not None:
        managed.validate_reference_identity(resolved)
    rows, _traces, manifest = load_dpo_package(
        data_path, manifest_path=manifest_path, expected_protocol=protocol_metadata(),
        expected_source_model=source_model, decision_parser=parse_react_response,
        messages_validator=validate_react_messages,
    )
    # inspect-only 只需要本地 tokenizer；此分支之前不导入 TRL/datasets 或加载策略权重。
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    current_protocol = protocol_metadata(tokenizer)
    for label, metadata in (("DPO 源 SFT", source_metadata), ("DPO 数据包", manifest)):
        require_training_protocol(metadata, current_protocol, source_label=label,
                                  require_chat_template_hash=True)
    audited = _audit_token_boundaries(tokenizer, rows, max_length)
    if inspect_only:
        print(f"DPO 审计通过：{len(rows)} 对，完整模板前缀/目标/长度已检查；未加载策略权重或训练")
        return

    import torch
    from agentic_arxiv.training import trl_compat  # noqa: F401
    from trl import DPOConfig, DPOTrainer
    from datasets import Dataset
    from agentic_arxiv.training.precision import pin_single_gpu, precision_flags, trainable_model_dtype

    execution.prepare_device(qlora=qlora)
    out_path = workspace_path(output_dir)
    final_path = out_path / "final"
    source_path = Path(resolved).resolve()
    if out_path.resolve().is_relative_to(source_path) or source_path.is_relative_to(out_path.resolve()):
        raise ValueError("DPO 输出目录不能与源 SFT 目录重叠")
    if final_path.exists() or (out_path / "run_manifest.json").exists():
        raise ValueError(f"拒绝覆盖已有 DPO 模型或运行记录，请选择新输出目录: {out_path}")
    snapshot_path = workspace_path(snapshot or "data/mock_arxiv_snapshot.json")
    if verify and not snapshot_path.is_file():
        raise ValueError(f"DPO 默认阶段验证所需快照不存在: {snapshot_path}")
    backends = resolve_report_to(report_to)
    actual_optim = "adamw_torch_fused" if not qlora and optim == "paged_adamw_8bit" else optim
    config_kwargs = {
        **configured["trainer_parameters"], "output_dir": str(out_path),
        "num_train_epochs": epochs, "max_steps": max_steps,
        "per_device_train_batch_size": batch_size, "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr, "beta": beta, "loss_type": ["sigmoid"],
        # 所有原始样本已严格拒绝超限；禁用 TRL 内部截断，不重写目标边界。
        "max_length": None, "sync_ref_model": False, "precompute_ref_log_probs": False,
        "label_smoothing": 0.0, "use_weighting": False,
        "f_divergence_type": "reverse_kl", "ld_alpha": None, "loss_weights": [1.0],
        "use_liger_kernel": False, "padding_free": False,
        "average_tokens_across_devices": False,
        "optim": actual_optim, "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if gradient_checkpointing else None,
        "seed": seed, "data_seed": seed, "report_to": backends,
        "run_name": run_name or out_path.name,
        "logging_dir": str(managed.log_dir if managed is not None else out_path / "logs"),
        **precision_flags(), **execution.trainer_kwargs(stage="dpo"),
    }
    require_supported_config(DPOConfig, config_kwargs)
    _require_trainer_interface(DPOTrainer)
    config = DPOConfig(**config_kwargs)
    pin_single_gpu(config)
    load_kwargs = dict(
        qlora=qlora, base_model=base_model, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, allowed_stages=SFT_STAGES,
        expected_protocol=current_protocol, local_base_only=True,
        model_dtype=trainable_model_dtype() if not qlora else None,
    )
    policy = load_training_policy(resolved, **load_kwargs)
    reference = load_reference_policy(resolved, **load_kwargs)
    _assert_independent_reference(policy, reference)
    policy.config.use_cache = False
    dataset = Dataset.from_list([{key: row[key] for key in ("prompt", "chosen", "rejected")} for row in rows])
    trainer = DPOTrainer(model=policy, ref_model=reference, args=config,
                         train_dataset=dataset, processing_class=tokenizer)
    execution.validate_trainer(trainer, stage="dpo")
    _assert_prepared_dataset(trainer.train_dataset, audited)
    _assert_independent_reference(trainer.model, trainer.ref_model, trainer.optimizer)
    stop_control, _checkpoint = attach_managed_training(
        trainer, managed, execution, resume_from_checkpoint=resume_from_checkpoint,
    )
    trainable, total, ratio = (_assert_lora_only_trainable(trainer.model) if qlora
                               else _trainable_parameter_stats(trainer.model))
    effective_adapters = {
        name: {"r": values.r, "lora_alpha": values.lora_alpha,
               "lora_dropout": values.lora_dropout,
               "target_modules": (values.target_modules if isinstance(values.target_modules, str)
                                  else sorted(values.target_modules or []))}
        for name, values in getattr(trainer.model, "peft_config", {}).items()
    }
    print(describe_logging(backends, out_path / "logs" if backends else None))
    print(f"DPO 计划：pairs={len(rows)}，global_pair_batch={effective_batch}，独立双模型，loss=sigmoid")
    memory = None
    sft_source = {"model": source_model, "manifest": file_fingerprint(Path(resolved) / "training_manifest.json"),
                  "stage": source_metadata["stage"]}
    verification = {"status": "pending" if verify else "skipped", "enabled": verify}
    trainer_state = trainer.state

    def save_run(reason):
        if not execution.is_main:
            return None
        record = build_run_manifest(
            requested_config=requested_config, resolved_config=config.to_dict(),
            derived_values={"effective_batch_size": effective_batch, "pair_count": len(rows),
                            "distributed_training": execution.metadata(), "memory": memory,
                            "trainable_parameters": trainable, "model_parameters": total,
                            "trainer_planned_steps": getattr(trainer_state, "max_steps", None),
                            "effective_adapters": effective_adapters,
                            **({"training_loss_semantics": "Trainer 原始返回值，不作为跨尝试累计均值或恢复等价证据",
                                "attempt_loss_sum": None, "attempt_mean_loss": None} if managed is not None else {}),
                            "audited_max_length": max_length, "configuration_source": stage_config_source("dpo", config_path)},
            source_baseline={"configuration_baseline": load_mainline_config(config_path)["source_baseline"],
                             "execution_source": source_fingerprint(
                                 [Path(__file__), Path(__file__).with_name("model_setup.py"),
                                  Path(__file__).with_name("distributed.py")], repo_root=get_workspace())},
            inputs={"sft_source": sft_source, "reference_source": sft_source,
                    "data": file_fingerprint(data_path), "data_manifest": file_fingerprint(manifest_path),
                    "snapshot": file_fingerprint(snapshot_path) if verify else None,
                    "mainline_configuration": file_fingerprint(config_file_path(config_path))},
            resource_overrides={**execution.resource_overrides(), **{
                key: {"baseline": DPO_DEFAULTS[key], "requested": requested_config[key]}
                for key in ("qlora", "batch_size", "grad_accum", "gradient_checkpointing", "optim", "max_length")
                if requested_config[key] != DPO_DEFAULTS[key]}},
            requested_steps=max_steps, actual_steps=getattr(trainer_state, "global_step", None),
            stop_reason=reason, protocol_metadata=current_protocol,
            protections={"complete_package_audit": True, "no_truncation": True,
                         "frozen_independent_sft_reference": True, "verification": verification},
        )
        _write_json(out_path / "run_manifest.json", record)
        return record

    training_started = time.monotonic()
    try:
        # Trainer 仅恢复可训练策略；reference 始终从最初的 SFT 独立重建。
        with preserve_managed_dataloader_epoch(
                trainer, enabled=managed is not None and bool(resume_from_checkpoint)):
            result = (trainer.train(resume_from_checkpoint=str(workspace_path(resume_from_checkpoint)))
                      if resume_from_checkpoint else trainer.train())
        _assert_independent_reference(trainer.model, trainer.ref_model, trainer.optimizer)
        train_loss = float(result.training_loss)
        if not math.isfinite(train_loss):
            raise ValueError(f"DPO training_loss 不是有限数: {train_loss}")
    except BaseException as exc:
        if stop_control is not None:
            stop_control.mark_failed(exc, getattr(trainer.state, "global_step", None))
        record_training_result(managed, execution, training="failed", validation="not_run",
                               global_step=getattr(trainer.state, "global_step", None),
                               planned_steps=getattr(trainer.state, "max_steps", None),
                               reason=f"training_exception:{type(exc).__name__}",
                               best_effort=True,
                               training_elapsed_seconds=time.monotonic() - training_started)
        # 训练器自己的集体通信错误不再启动额外 collective，避免掩盖原异常。
        trainer_state = trainer.state
        try:
            save_run(f"training_exception:{type(exc).__name__}")
        except Exception as record_error:
            print(f"DPO 训练异常时无法保存运行诊断: {type(record_error).__name__}")
        raise
    training_elapsed = time.monotonic() - training_started
    trainer_state = trainer.state
    actual_steps = int(getattr(trainer.state, "global_step", 0) or 0)
    planned = getattr(trainer.state, "max_steps", None)
    if managed is not None and (type(planned) is not int or planned < 1):
        record_training_result(managed, execution, training="failed", validation="not_run",
                               global_step=actual_steps, reason="missing_trainer_budget",
                               training_elapsed_seconds=training_elapsed)
        raise ValueError("受管 DPO 缺少 Trainer 实际核定的总优化步数，不能宣称预算完成")
    if managed is None and max_steps > 0:
        planned = max_steps
    reason = "trainer_stopped_early" if planned and actual_steps < planned else "max_steps_or_epochs_completed"
    if reason == "trainer_stopped_early" and stop_control is not None and stop_control.user_stopped:
        reason = "user_safe_stop"
    if reason != "max_steps_or_epochs_completed":
        record_training_result(
            managed, execution,
            training="user_safe_stop" if reason == "user_safe_stop" else "protection_stopped",
            validation="not_run", global_step=actual_steps, reason=reason,
            planned_steps=planned,
            checkpoint=stop_control.checkpoint if stop_control is not None else None,
            training_elapsed_seconds=training_elapsed,
        )
        verification.update(status="skipped", reason=reason)
    memory = execution.memory_summary()
    record = execution.run_on_main(lambda: save_run(reason))
    if reason != "max_steps_or_epochs_completed":
        print(f"DPO 训练已停止: {reason}；未导出 final 或运行正常完成验证")
        return
    # 所有 rank 参与 Trainer 保存；共享 tokenizer/manifest 只有全局主进程写入。
    trainer.save_model(str(final_path))

    def save_final(tokenizer_to_save):
        tokenizer_to_save.save_pretrained(str(final_path))
        if qlora:
            adapter_config = json.loads((final_path / "adapter_config.json").read_text(encoding="utf-8"))
            actual_base = adapter_config["base_model_name_or_path"]
            if not any((final_path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
                raise ValueError("DPO 策略 adapter 权重未保存")
        else:
            actual_base = source_metadata.get("base_model") or resolved
        exported = {
            **record, "stage": "dpo_qlora" if qlora else "dpo_full", "base_model": actual_base,
            "sft_source": sft_source, "reference_source": sft_source,
            "data": str(data_path), "data_manifest": file_fingerprint(manifest_path),
            "output": str(final_path), "train_loss": train_loss, "actual_steps": actual_steps,
            "batch_size": batch_size, "gradient_accumulation_steps": grad_accum,
            "effective_batch_size": effective_batch, "beta": beta, "loss_type": "sigmoid",
            "trainable_ratio_percent": ratio, "peak_vram_gib": memory["peak_allocated_gib"],
            "verification": verification,
        }
        _write_json(final_path / "training_manifest.json", exported)
        return exported

    exported = execution.run_on_main(partial(save_final, tokenizer))
    record_training_result(managed, execution, training="budget_completed",
                           validation="in_progress" if verify else "skipped",
                           global_step=actual_steps, reason=reason,
                           planned_steps=planned,
                           training_elapsed_seconds=training_elapsed)
    # 参考、优化器、策略可能被 Trainer/Accelerate 同时持有；先清空这些引用再加载验证模型。
    trainer.accelerator.free_memory()
    # 受管保存适配和停止回调也持有 Trainer，必须一起退出训练生命周期。
    del stop_control, _checkpoint, trainer, policy, reference, dataset, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def verify_final():
        from agentic_arxiv.runtime.env import MockArxivEnv
        verifier = StageVerifier(dpo_min_reward=min_reward,
                                 cancel_requested=managed_cancellation_check(managed))
        try:
            env = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")
            report = verifier.verify_dpo(model_path=str(final_path), env=env)
            verifier.save_report(report, final_path)
            status = ("interrupted" if getattr(report, "coverage", {}).get("execution") == "interrupted"
                      else "passed" if report.passed else "failed")
            verification.update(status=status,
                                report=str(final_path / "verification_report.json"))
            print(report.summary())
        except BaseException as exc:
            verification.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            _write_json(final_path / "verification_report.json", {
                "stage": "dpo", "model_path": str(final_path), "passed": False,
                "failures": [verification["error"]],
            })
            raise
        finally:
            exported["verification"] = dict(verification)
            record["protections"]["verification"] = dict(verification)
            _write_json(final_path / "training_manifest.json", exported)
            _write_json(out_path / "run_manifest.json", record)
        if not report.passed and verification["status"] != "interrupted":
            raise SystemExit("DPO 阶段验证失败；模型与报告已保留，不自动继续 GRPO")
        return dict(verification)

    if verify:
        verification_started = time.monotonic()
        try:
            verification.update(execution.run_on_main(verify_final))
        except BaseException as exc:
            record_training_result(managed, execution, training="budget_completed",
                                   validation="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                   global_step=actual_steps, reason=f"verification_exception:{type(exc).__name__}",
                                   planned_steps=planned,
                                   best_effort=True,
                                   training_elapsed_seconds=training_elapsed,
                                   stage_validation_elapsed_seconds=time.monotonic() - verification_started)
            raise
        record_training_result(managed, execution, training="budget_completed",
                               validation=verification["status"], global_step=actual_steps, reason=reason,
                               planned_steps=planned,
                               training_elapsed_seconds=training_elapsed,
                               stage_validation_elapsed_seconds=time.monotonic() - verification_started)
    print(f"DPO 训练产物已保存: {final_path}；验证状态={verification['status']}")
