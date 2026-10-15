"""固定版本 Trainer 的受管 checkpoint 事务和只读恢复校验。

控制端导入此模块不导入 torch、Transformers 或 TRL。文件完整性不构成
任意 pickle 的信任授权：调用端还必须限制来源为本系统登记的训练产物。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import wraps
from hashlib import sha256
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import shutil
from types import MethodType
from typing import Any, Callable


SCHEMA_VERSION = 1
MANIFEST_NAME = "managed_checkpoint.json"
COMPLETE_NAME = "managed_checkpoint.complete.json"
PINNED_VERSIONS = {"transformers": "4.57.6", "trl": "0.29.1"}
IDENTITY_KEYS = {"inputs", "training_config", "runtime", "models", "devices"}


class CheckpointError(ValueError):
    """不能安全保存或恢复受管训练状态。"""


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"无法读取 checkpoint 元数据 {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"checkpoint 元数据必须为对象: {path}")
    return value


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


@dataclass(frozen=True)
class ManagedCheckpointContext:
    stage: str
    run_id: str
    attempt_id: str
    total_budget: int | None
    identity: dict
    schema_version: int = SCHEMA_VERSION
    output_dir: str | None = None
    control_dir: str | None = None
    log_dir: str | None = None
    epoch_budget: int | None = None

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise CheckpointError("未知受管 checkpoint context 版本")
        if self.stage not in {"sft", "dpo", "grpo"}:
            raise CheckpointError("checkpoint stage 必须为 sft/dpo/grpo")
        if (not isinstance(self.run_id, str) or not isinstance(self.attempt_id, str)
                or not self.run_id or not self.attempt_id):
            raise CheckpointError("checkpoint 缺少 run_id/attempt_id")
        if self.epoch_budget is not None and (type(self.epoch_budget) is not int or self.epoch_budget < 1):
            raise CheckpointError("checkpoint 原 epoch 预算必须为正整数或 None")
        if self.total_budget is None:
            if self.epoch_budget is None:
                raise CheckpointError("checkpoint 未核定步数时必须保留正整数 epoch 预算")
        elif type(self.total_budget) is not int or self.total_budget < 1:
            raise CheckpointError("checkpoint 原总预算必须为正整数优化步数")
        if not isinstance(self.identity, dict) or IDENTITY_KEYS - self.identity.keys():
            raise CheckpointError(f"checkpoint identity 需要 {sorted(IDENTITY_KEYS)}")
        devices = self.identity["devices"]
        if (not isinstance(devices, list) or not devices
                or any(not isinstance(device, str) or not device for device in devices)
                or len(set(devices)) != len(devices)):
            raise CheckpointError("identity.devices 必须按 rank 顺序列出不同的 GPU UUID")
        for key in IDENTITY_KEYS - {"devices"}:
            if not isinstance(self.identity[key], dict) or not self.identity[key]:
                raise CheckpointError(f"identity.{key} 必须为非空身份/指纹对象")

    @classmethod
    def from_file(cls, path, *, expected_stage: str | None = None):
        result = cls.from_dict(_read_json(Path(path)))
        if expected_stage is not None and result.stage != expected_stage:
            raise CheckpointError(f"受管上下文阶段 {result.stage} != {expected_stage}")
        return result

    @classmethod
    def from_dict(cls, value: dict):
        allowed = {"schema_version", "stage", "run_id", "attempt_id", "total_budget", "identity",
                   "output_dir", "control_dir", "log_dir", "epoch_budget"}
        if set(value) - allowed:
            raise CheckpointError(f"未知受管上下文字段: {sorted(set(value) - allowed)}")
        try:
            return cls(**value)
        except TypeError as exc:
            raise CheckpointError(f"受管上下文字段缺失: {exc}") from exc

    def as_dict(self) -> dict:
        return {"schema_version": self.schema_version, "stage": self.stage,
                "run_id": self.run_id, "attempt_id": self.attempt_id,
                "total_budget": self.total_budget, "epoch_budget": self.epoch_budget,
                "identity": self.identity}

    def assert_identity(self, key: str, value: Any) -> None:
        if self.identity.get(key) != value:
            raise CheckpointError(f"checkpoint 身份不匹配: identity.{key}")

    def validate_reference_identity(self, source_model) -> None:
        from agentic_arxiv.provenance import model_input_fingerprint

        models = self.identity["models"]
        reference = models.get("reference")
        path = reference.get("path") if isinstance(reference, dict) else reference
        if not path or Path(path).resolve() != Path(source_model).resolve():
            raise CheckpointError("冻结参考不是本次运行原始绑定的模型来源")
        if (not isinstance(reference, dict) or not reference.get("fingerprint")
                or reference["fingerprint"] != model_input_fingerprint(source_model)):
            raise CheckpointError("冻结参考权重指纹不匹配或缺失")


def _files(directory: Path) -> dict:
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise CheckpointError(f"checkpoint 不接受符号链接: {path}")
        if path.is_file() and path not in {directory / MANIFEST_NAME, directory / COMPLETE_NAME}:
            result[path.relative_to(directory).as_posix()] = {
                "sha256": _digest(path), "size": path.stat().st_size,
            }
    return result


def _required_files(manifest: dict, directory: Path) -> None:
    files = manifest["files"]
    required = {"trainer_state.json", "training_args.bin", "optimizer.pt", "scheduler.pt"}
    size = manifest["world_size"]
    for rank in range(size):
        required.update({f"rank-{rank}.complete.json", f"managed_state_{rank}.pt"})
        required.add("rng_state.pth" if size == 1 else f"rng_state_{rank}.pth")
    precision = manifest.get("precision")
    if precision == {"scaler": "enabled"}:
        required.add("scaler.pt")
    elif precision != {"scaler": "not_applicable"}:
        raise CheckpointError("checkpoint 必须明确精度 scaler 状态或不适用")
    missing = required - files.keys()
    if missing:
        raise CheckpointError(f"checkpoint 缺少必要训练/rank 文件: {sorted(missing)}")
    if any(files[name]["size"] == 0 for name in required):
        raise CheckpointError("checkpoint 必要训练/rank 文件为空")
    weight_names = {"model.safetensors", "pytorch_model.bin", "adapter_model.safetensors", "adapter_model.bin"}
    index_names = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    if not weight_names.intersection(files):
        indexes = index_names.intersection(files)
        if not indexes:
            raise CheckpointError("checkpoint 缺少策略权重（仅元数据不能恢复）")
        for name in indexes:
            index = _read_json(directory / name)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise CheckpointError("策略分片索引为空")
            if set(weight_map.values()) - files.keys():
                raise CheckpointError("策略权重分片缺失")
    state = _read_json(directory / "trainer_state.json")
    if state.get("global_step") != manifest["global_step"] or state.get("max_steps") != manifest["total_budget"]:
        raise CheckpointError("Trainer 步数/原总预算与恢复清单不一致")
    if "epoch" not in state or state["epoch"] is None:
        raise CheckpointError("checkpoint 缺少 Trainer 数据消费 epoch 位置")
    for rank in range(size):
        receipt = _read_json(directory / f"rank-{rank}.complete.json")
        if receipt != {"rank": rank, "global_step": manifest["global_step"],
                       "attempt_id": manifest["attempt_id"], "schema_version": SCHEMA_VERSION}:
            raise CheckpointError(f"rank {rank} 状态不属于同一次 checkpoint 保存")


def _differences(left, right, prefix="") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        return [item for key in sorted(left.keys() | right.keys())
                for item in (_differences(left[key], right[key], f"{prefix}.{key}".strip("."))
                             if key in left and key in right else [f"{prefix}.{key}".strip(".")])]
    return [] if left == right else [prefix]


def validate_checkpoint(path, *, expected=None, require_remaining_budget: bool = True) -> dict:
    """只读核验完成标记、所有内容和身份；不导入/加载 pickle 或训练栈。"""
    directory = Path(path)
    if (directory.is_symlink() or not directory.is_dir()
            or any(".incomplete" in part for part in directory.parts)):
        raise CheckpointError("恢复来源不是已提交 checkpoint 目录")
    marker = _read_json(directory / COMPLETE_NAME)
    manifest = _read_json(directory / MANIFEST_NAME)
    if marker != {"schema_version": SCHEMA_VERSION, "manifest_sha256": _digest(directory / MANIFEST_NAME)}:
        raise CheckpointError("checkpoint 完成标记或清单哈希不匹配")
    context = ManagedCheckpointContext.from_dict({key: manifest.get(key) for key in (
        "schema_version", "stage", "run_id", "attempt_id", "total_budget", "epoch_budget", "identity")})
    if context.total_budget is None:
        raise CheckpointError("已提交 checkpoint 必须包含 Trainer 核定的正整数总步数")
    if (type(manifest.get("world_size")) is not int
            or manifest["world_size"] != len(context.identity["devices"])):
        raise CheckpointError("checkpoint world_size/rank 设备映射不匹配")
    if (type(manifest.get("global_step")) is not int
            or not 0 <= manifest["global_step"] <= context.total_budget):
        raise CheckpointError("checkpoint 逻辑步数非法")
    if type(manifest.get("additional_state_version")) is not int or manifest["additional_state_version"] != 1:
        raise CheckpointError("未知 checkpoint 附加状态版本")
    if manifest.get("data_position") != "trainer_global_step_epoch_skip":
        raise CheckpointError("不支持 checkpoint 数据消费位置协议")
    if manifest.get("files") != _files(directory):
        raise CheckpointError("checkpoint 文件清单/内容指纹不匹配（缺失、增加或损坏）")
    _required_files(manifest, directory)
    if require_remaining_budget:
        if manifest["global_step"] >= context.total_budget:
            raise CheckpointError("原定训练预算已耗尽，不能继续训练")
        if manifest.get("termination") not in {None, "user_stopped"}:
            raise CheckpointError("保护器或失败终止的 checkpoint 不允许普通恢复")
    if expected is not None:
        desired = expected.as_dict() if isinstance(expected, ManagedCheckpointContext) else dict(expected)
        for key in ("attempt_id", "output_dir", "control_dir", "log_dir"):
            desired.pop(key, None)
        allowed = {"schema_version", "stage", "run_id", "total_budget", "epoch_budget", "identity"}
        if set(desired) - allowed:
            raise CheckpointError("expected 包含不支持的身份字段")
        if "total_budget" in desired and desired["total_budget"] is None:
            if (type(desired.get("epoch_budget")) is not int or desired["epoch_budget"] < 1
                    or desired["epoch_budget"] != context.epoch_budget):
                raise CheckpointError("未核定步数的恢复检查必须保持原 epoch 预算")
            # 首次运行身份仅固定 epoch 数；实际总步数必须来自本清单，不能估算。
            desired.pop("total_budget")
        differences = _differences({key: manifest.get(key) for key in desired}, desired)
        if differences:
            raise CheckpointError("恢复配置/来源不兼容: " + ", ".join(differences))
    return manifest


load_checkpoint_manifest = validate_checkpoint


def check_trainer_interfaces(trainer, *, installed_versions=None) -> None:
    """在任何优化更新前拒绝未经适配的私有接口和分片路径。"""
    versions = installed_versions if installed_versions is not None else {
        name: version(name) for name in PINNED_VERSIONS
    }
    if any(versions.get(name) != required for name, required in PINNED_VERSIONS.items()):
        raise CheckpointError(f"受管恢复要求固定依赖 {PINNED_VERSIONS}，实际 {versions}")
    for name, arguments in {"_save_checkpoint": {"model", "trial"}, "_get_output_dir": {"trial"},
                            "_rotate_checkpoints": {"output_dir"}, "_load_rng_state": {"checkpoint"}}.items():
        method = getattr(trainer, name, None)
        if not callable(method) or not arguments <= inspect.signature(method).parameters.keys():
            raise CheckpointError(f"不支持的 Trainer checkpoint 接口: {name}")
    for name in ("_save_optimizer_and_scheduler", "_save_rng_state", "_save_scaler"):
        if not callable(getattr(trainer, name, None)):
            raise CheckpointError(f"缺少 Trainer checkpoint 接口: {name}")
    args = trainer.args
    forbidden = ("save_only_model", "push_to_hub", "save_on_each_node", "ignore_data_skip", "load_best_model_at_end")
    if any(getattr(args, name, False) for name in forbidden):
        raise CheckpointError("受管恢复不支持仅权重保存、Hub、每节点保存、跳过数据恢复或最佳模型回载")
    if any(getattr(trainer, name, False) for name in ("is_fsdp_enabled", "is_deepspeed_enabled")):
        raise CheckpointError("受管恢复首版仅支持单机普通 Trainer/DDP，不支持分片")
    strategy = str(getattr(args.save_strategy, "value", args.save_strategy))
    if strategy != "steps" or not getattr(args, "save_steps", 0):
        raise CheckpointError("受管恢复要求显式按优化步周期完整保存")
    if getattr(args, "dataloader_num_workers", 0) != 0:
        raise CheckpointError("首版精确恢复要求 dataloader_num_workers=0，避免不可恢复的预取状态")
    limit = getattr(args, "save_total_limit", None)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise CheckpointError("save_total_limit 必须为正整数或 None")


@contextmanager
def preserve_adapter_restore_contract(trainer, model):
    """仅受管恢复禁止 PEFT 自动升精度，并在加载优化器前核对参数契约。"""
    if not getattr(model, "peft_config", None):
        yield
        return
    original = getattr(model, "load_adapter", None)
    if not callable(original):
        raise CheckpointError("受管 adapter 恢复缺少 load_adapter 接口")
    signature = inspect.signature(original)
    if "autocast_adapter_dtype" not in signature.parameters:
        raise CheckpointError("受管 adapter 恢复要求可禁用自动精度转换的 PEFT 接口")
    before = {name: (parameter, parameter.dtype, parameter.requires_grad)
              for name, parameter in model.named_parameters()}
    active = tuple(model.active_adapters)
    optimizer = getattr(trainer, "optimizer", None)
    optimizer_parameters = tuple(tuple(id(parameter) for parameter in group["params"])
                                 for group in getattr(optimizer, "param_groups", []))
    absent = object()
    previous = vars(model).get("load_adapter", absent)

    @wraps(original)
    def load_adapter(*args, **kwargs):
        # Trainer 的标准调用（包括 GRPO ref/）没有传该参数，PEFT 默认会
        # 将 BF16/FP16 adapter 升至 FP32；不能只修补随后加载的根 default。
        bound = signature.bind(*args, **kwargs)
        bound.arguments["autocast_adapter_dtype"] = False
        return original(*bound.args, **bound.kwargs)

    model.load_adapter = load_adapter
    try:
        yield
    finally:
        if previous is absent:
            del model.load_adapter
        else:
            model.load_adapter = previous
    after = dict(model.named_parameters())
    if before.keys() != after.keys():
        raise CheckpointError("受管 adapter 恢复改变了模型参数集合")
    for name, (parameter, dtype, requires_grad) in before.items():
        restored = after[name]
        if restored is not parameter or restored.dtype != dtype or restored.requires_grad != requires_grad:
            raise CheckpointError(f"受管 adapter 恢复改变了参数对象、精度或冻结状态: {name}")
    if tuple(model.active_adapters) != active:
        raise CheckpointError("受管 adapter 恢复改变了活动策略")
    actual_optimizer = getattr(trainer, "optimizer", None)
    actual_parameters = tuple(tuple(id(parameter) for parameter in group["params"])
                              for group in getattr(actual_optimizer, "param_groups", []))
    if actual_optimizer is not optimizer or actual_parameters != optimizer_parameters:
        raise CheckpointError("受管 adapter 恢复改变了优化器参数引用")


def install_managed_adapter_restore(trainer):
    """保留标准 Trainer 加载顺序，仅保护显式受管 PEFT 模型的加载边界。"""
    if not getattr(getattr(trainer, "model", None), "peft_config", None):
        return
    original = getattr(trainer, "_load_from_checkpoint", None)
    if not callable(original):
        raise CheckpointError("受管 adapter 恢复缺少 Trainer checkpoint 加载接口")

    def load_checkpoint(instance, resume_from_checkpoint, model=None):
        target = instance.model if model is None else model
        with preserve_adapter_restore_contract(instance, target):
            return original(resume_from_checkpoint, model=model)

    trainer._load_from_checkpoint = MethodType(load_checkpoint, trainer)


class ManagedCheckpointAdapter:
    def __init__(self, trainer, *, context, training_context, resume_from_checkpoint=None,
                 state_provider=None, state_restorer=None):
        self.trainer = trainer
        self.context = context
        self.training_context = training_context
        self.resume_from_checkpoint = Path(resume_from_checkpoint).resolve() if resume_from_checkpoint else None
        self.state_provider = state_provider or (lambda: {})
        self.state_restorer = state_restorer or (lambda value: None)
        self.last_checkpoint = None
        self._restored = False
        self._budget_bound = False

    def bind_budget(self, *, args, state):
        """Trainer 计算完整原排程后、首次优化前核定 epoch 模式的总步数。"""
        actual = state.max_steps
        max_steps = getattr(args, "max_steps", None)
        epochs = getattr(args, "num_train_epochs", None)
        local = {"rank": self.training_context.rank, "actual": actual,
                 "max_steps": max_steps, "epochs": epochs}
        outcomes = self.training_context.gather(local)
        if (len(outcomes) != self.training_context.world_size
                or {entry["rank"] for entry in outcomes} != set(range(self.training_context.world_size))
                or any(type(entry["actual"]) is not int or entry["actual"] < 1
                       or entry["actual"] != actual for entry in outcomes)):
            raise CheckpointError(f"Trainer 各 rank 的核定总预算不一致或非法: {outcomes}")
        if self.context.epoch_budget is not None:
            if any(entry["max_steps"] != -1 or isinstance(entry["epochs"], bool)
                   or entry["epochs"] != self.context.epoch_budget for entry in outcomes):
                raise CheckpointError("epoch 恢复必须保留原 max_steps=-1 和 epoch 总预算，不得重写排程")
        elif any(entry["max_steps"] != self.context.total_budget for entry in outcomes):
            raise CheckpointError("Trainer 步数排程与原总预算不一致")
        if self.context.total_budget is not None and actual != self.context.total_budget:
            raise CheckpointError("Trainer 核定总步数与原预算不一致，不能重发或延长恢复预算")
        bound = replace(self.context, total_budget=actual)

        def publish():
            if bound.control_dir is None:
                if bound.epoch_budget is not None:
                    raise CheckpointError("epoch 预算核定需要当前尝试 control_dir 保存证据")
                return
            directory = Path(bound.control_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "resolved-budget.json"
            value = {"schema_version": SCHEMA_VERSION, "run_id": bound.run_id,
                     "attempt_id": bound.attempt_id, "total_budget": actual,
                     "epoch_budget": bound.epoch_budget}
            if path.exists():
                if _read_json(path) != value:
                    raise CheckpointError("当前尝试已有不同核定预算，不覆盖旧证据")
            else:
                _write_json(path, value)

        self.training_context.run_on_main(publish)
        self.context = bound
        self._budget_bound = True

    def validate_resume(self):
        if self.resume_from_checkpoint is None:
            return None
        manifest = validate_checkpoint(self.resume_from_checkpoint, expected=self.context)
        if self.context.total_budget is None:
            # 即使调用端只持有最初 epoch 身份，恢复也必须绑定原已提交预算。
            self.context = replace(self.context, total_budget=manifest["total_budget"])
        return manifest

    @property
    def latest_checkpoint(self):
        return self.last_checkpoint

    def restore_additional_state(self):
        if self.resume_from_checkpoint is None or self._restored:
            return
        self.validate_resume()
        import torch

        value = torch.load(self.resume_from_checkpoint / f"managed_state_{self.training_context.rank}.pt",
                           map_location="cpu", weights_only=False)
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise CheckpointError("未知的 rank 附加状态格式")
        self.state_restorer(value["state"])
        self._restored = True

    def _rotate(self, root: Path):
        limit = getattr(self.trainer.args, "save_total_limit", None)
        if limit is None:
            return
        if type(limit) is not int or limit < 1:
            raise CheckpointError("save_total_limit 必须为正整数或 None")
        eligible = []
        for path in root.glob("checkpoint-*"):
            if not path.is_dir() or path.is_symlink():
                continue
            try:
                manifest = validate_checkpoint(path, require_remaining_budget=False)
            except CheckpointError:
                continue
            if (manifest["run_id"], manifest["attempt_id"]) == (self.context.run_id, self.context.attempt_id):
                eligible.append((manifest["global_step"], path))
        eligible.sort()
        while len(eligible) > limit and len(eligible) > 1:
            index = next((i for i, (_, path) in enumerate(eligible)
                          if path.resolve() != self.resume_from_checkpoint and path != self.last_checkpoint), None)
            if index is None:
                break
            _, path = eligible.pop(index)
            shutil.rmtree(path)
        _fsync_dir(root)

    def _save(self, original, model, trial):
        trainer = self.trainer
        topology = self.training_context
        if trial is not None or getattr(trainer.state, "best_global_step", None):
            raise CheckpointError("受管 checkpoint 不支持超参搜索或 best-step 保存同步")
        step = trainer.state.global_step
        if self.context.total_budget is None:
            raise CheckpointError("epoch 总预算尚未在 Trainer.on_train_begin 核定，不能提交 checkpoint")
        if type(step) is not int or not 0 <= step <= self.context.total_budget:
            raise CheckpointError("Trainer 逻辑优化步数超出原预算或类型非法")
        if trainer.state.max_steps != self.context.total_budget:
            raise CheckpointError("Trainer 实际总预算与受管原预算不一致")
        root = Path(trainer.args.output_dir).resolve()
        name = f"checkpoint-{step}"
        transaction = root / f".{name}.incomplete"
        destination = root / name
        staging = transaction / name

        def prepare():
            root.mkdir(parents=True, exist_ok=True)
            if destination.exists() or transaction.exists():
                raise CheckpointError("checkpoint 目标或未完成事务已存在；保留证据，不覆盖")
            transaction.mkdir()
            staging.mkdir()
            _fsync_dir(root)

        topology.run_on_main(prepare)
        output_method = trainer._get_output_dir
        trainer._get_output_dir = lambda trial=None: str(transaction)
        error = None
        try:
            original(model, trial)
            import torch

            state_path = staging / f"managed_state_{topology.rank}.pt"
            torch.save({"schema_version": 1, "state": self.state_provider()}, state_path)
            _write_json(staging / f"rank-{topology.rank}.complete.json", {
                "schema_version": 1, "rank": topology.rank, "global_step": step,
                "attempt_id": self.context.attempt_id,
            })
        except Exception as exc:
            # 已损坏的通信/CUDA上下文不能为了报告保存错误再发起 collective。
            if (type(exc).__name__ in {"DistBackendError", "DistNetworkError", "DistStoreError"}
                    or any(token in str(exc).lower() for token in (
                        "nccl", "cuda", "connection closed", "process group", "collective"))):
                raise
            error = f"{type(exc).__name__}: {exc}"
        finally:
            trainer._get_output_dir = output_method
        outcomes = topology.gather({"rank": topology.rank, "error": error, "global_step": step})
        if (len(outcomes) != topology.world_size
                or {item["rank"] for item in outcomes} != set(range(topology.world_size))
                or any(item["error"] or item["global_step"] != step for item in outcomes)):
            raise CheckpointError(f"checkpoint rank 保存失败，保留未完成事务: {outcomes}")

        def commit():
            manifest = {**self.context.as_dict(), "world_size": topology.world_size,
                        "global_step": step, "additional_state_version": 1,
                        "data_position": "trainer_global_step_epoch_skip",
                        "precision": {"scaler": "enabled" if getattr(trainer.accelerator, "scaler", None)
                                      is not None else "not_applicable"},
                        "termination": getattr(trainer, "_managed_termination_reason", None),
                        "files": _files(staging)}
            _required_files(manifest, staging)
            for path in staging.rglob("*"):
                if path.is_file():
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
            _write_json(staging / MANIFEST_NAME, manifest)
            _write_json(staging / COMPLETE_NAME, {"schema_version": 1,
                        "manifest_sha256": _digest(staging / MANIFEST_NAME)})
            _fsync_dir(staging)
            os.replace(staging, destination)
            _fsync_dir(root)
            return str(destination)

        self.last_checkpoint = Path(topology.run_on_main(commit))
        topology.run_on_main(lambda: self._rotate(root))
        return None


def attach_managed_checkpoint(trainer, *, context: ManagedCheckpointContext, training_context,
                              resume_from_checkpoint=None, state_provider: Callable | None = None,
                              state_restorer: Callable | None = None):
    """只对显式受管 Trainer 接入，保留标准 Trainer.train 恢复语义。"""
    check_trainer_interfaces(trainer)
    if training_context.world_size != len(context.identity["devices"]):
        raise CheckpointError("当前训练拓扑与受管 GPU/rank 映射不一致")
    if (context.output_dir is not None
            and Path(trainer.args.output_dir).resolve() != Path(context.output_dir).resolve()):
        raise CheckpointError("Trainer 输出目录不是当前受管尝试目录")
    if hasattr(trainer, "_managed_checkpoint"):
        raise CheckpointError("Trainer 已接入受管 checkpoint")
    if context.stage == "grpo" and (state_provider is None or state_restorer is None):
        raise CheckpointError("GRPO 必须提供生成缓存与保护状态的保存/恢复适配")
    adapter = ManagedCheckpointAdapter(trainer, context=context, training_context=training_context,
                                       resume_from_checkpoint=resume_from_checkpoint,
                                       state_provider=state_provider, state_restorer=state_restorer)
    adapter.validate_resume()
    adapter.restore_additional_state()
    install_managed_adapter_restore(trainer)
    from transformers import TrainerCallback

    class BudgetBindingCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            adapter.bind_budget(args=args, state=state)
            return control

    adapter.budget_callback = BudgetBindingCallback()
    trainer.add_callback(adapter.budget_callback)
    original = trainer._save_checkpoint

    def save_checkpoint(self, model, trial):
        return adapter._save(original, model, trial)

    trainer._save_checkpoint = MethodType(save_checkpoint, trainer)
    # 标准实现会在写完公共文件后立即轮转；此处禁用，提交后由 adapter 处理。
    trainer._rotate_checkpoints = lambda *args, **kwargs: None
    trainer._managed_checkpoint = adapter
    return adapter
