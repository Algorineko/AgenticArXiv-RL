"""版本化严格 JSON 管理配置，不加载训练器、环境凭据或研究数据。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re


GROUP_STAGES = {"M0": "evaluate", "M1": "sft", "M2": "dpo", "M3": "grpo", "M4": "grpo", "C1": "grpo"}
PROFILE_REQUIRED = {"schema_version", "workspace", "run_root", "input_roots", "cache_root", "image", "uid", "gid", "gpus", "shm_size"}
RECIPE_REQUIRED = {"schema_version", "group", "stage", "training_config", "inputs", "training_overrides", "checkpoint", "evaluation", "budget"}
INPUT_KEYS = {"model", "base_model", "data", "data_manifest", "snapshot", "split", "source_sft", "source_dpo", "preference_cost"}
EVALUATION_TYPES = {
    "config": str, "kind": str, "repeat": int, "tasks": str, "task_ids": list,
    "backend": str, "local_device": str, "local_dtype": str, "seed": int,
    "prefix": str, "no_thinking": bool, "offline": bool, "snapshot": str,
    "save_traces": bool, "split": str, "task_set": str, "num_samples": int,
    "max_iterations": int, "sft_min_parse_rate": float, "dpo_min_reward": float,
    "grpo_min_reward": float,
}
NON_OVERRIDE_FIELDS = {"model", "base_model", "data", "data_manifest", "output_dir", "snapshot",
                       "resume_from_checkpoint", "rollout_trace_path", "inspect_only", "skip_data_manifest_check", "smoke"}


def _object(value, label, *, required=(), optional=()):
    if type(value) is not dict:
        raise ValueError(f"{label} 必须为对象")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f"{label} 缺失字段={sorted(missing)}，未知字段={sorted(unknown)}")
    return value


def _typed(value, expected, label, *, nullable=False):
    if value is None and nullable:
        return
    valid = type(value) is expected
    if expected is float:
        valid = type(value) in (int, float) and math.isfinite(value)
    if not valid:
        raise ValueError(f"{label} 必须为 {expected.__name__}" + (" 或 null" if nullable else ""))
    if expected is str and not value.strip():
        raise ValueError(f"{label} 不能是空字符串")


def _version(payload, label):
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError(f"{label}.schema_version 必须为整数 1")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 重复字段: {key}")
        result[key] = value
    return result


def read_json(path):
    """拒绝重复键与非有限数，避免解析器间的配置身份歧义。"""
    def reject(value):
        raise ValueError(f"JSON 不允许非有限数: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=reject)


def validate_profile(payload):
    _object(payload, "profile", required=PROFILE_REQUIRED, optional={"min_free_bytes"})
    _version(payload, "profile")
    for key in ("workspace", "run_root", "cache_root", "image", "shm_size"):
        _typed(payload[key], str, f"profile.{key}")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", payload["image"]):
        raise ValueError("profile.image 必须是经核验的固定 image@sha256:<64位摘要>，不能使用 latest")
    for key in ("uid", "gid", "min_free_bytes"):
        if key in payload:
            _typed(payload[key], int, f"profile.{key}")
            if payload[key] < (0 if key == "min_free_bytes" else 1):
                raise ValueError(f"profile.{key} 不允许该值；容器必须使用普通用户")
    for key in ("input_roots", "gpus"):
        _typed(payload[key], list, f"profile.{key}")
        if not payload[key]:
            raise ValueError(f"profile.{key} 必须显式列出允许范围")
        for item in payload[key]:
            _typed(item, str, f"profile.{key}[]")
        if len(set(payload[key])) != len(payload[key]):
            raise ValueError(f"profile.{key} 不允许重复项")
    if not re.fullmatch(r"[1-9][0-9]*[kKmMgG]?", payload["shm_size"]):
        raise ValueError("profile.shm_size 必须为正大小，例如 16g")
    return copy.deepcopy(payload)


def _reference(value, label):
    if value is None:
        return
    if type(value) is str:
        _typed(value, str, label)
        return
    _object(value, label, required={"kind"}, optional={"path", "registration_id", "run_id", "attempt_id", "artifact", "verification_ref"})
    kind = value["kind"]
    _typed(kind, str, f"{label}.kind")
    required = {"path": {"path"}, "registration": {"registration_id"}, "run": {"run_id", "attempt_id", "artifact"}}
    if kind not in required:
        raise ValueError(f"{label}.kind 必须为 path/registration/run")
    allowed = {"kind"} | required[kind]
    if set(value) - {"verification_ref"} != allowed:
        raise ValueError(f"{label} 的 {kind} 引用需要且仅接受 {sorted(allowed)}")
    for key in allowed - {"kind"}:
        _typed(value[key], str, f"{label}.{key}")
        if value[key].lower() == "latest":
            raise ValueError(f"{label} 不允许 latest 引用")
    if kind == "run" and value["artifact"] != "final":
        raise ValueError(f"{label} 只能明确引用 final，恢复点使用 resume --checkpoint")
    if "verification_ref" in value:
        verified = _object(value["verification_ref"], f"{label}.verification_ref", required={"run_id", "attempt_id"})
        for key, item in verified.items():
            _typed(item, str, f"{label}.verification_ref.{key}")
            if item.lower() == "latest":
                raise ValueError(f"{label}.verification_ref 不允许 latest 引用")


def validate_recipe(payload):
    _object(payload, "recipe", required=RECIPE_REQUIRED, optional={"task_pool", "filter_policy", "name"})
    _version(payload, "recipe")
    _typed(payload["group"], str, "recipe.group")
    _typed(payload["stage"], str, "recipe.stage")
    if payload["group"] not in GROUP_STAGES or payload["stage"] != GROUP_STAGES[payload["group"]]:
        raise ValueError("recipe.group 与 stage 不匹配")
    _typed(payload["training_config"], str, "recipe.training_config", nullable=True)
    _object(payload["inputs"], "recipe.inputs", optional=INPUT_KEYS)
    for key, value in payload["inputs"].items():
        _reference(value, f"recipe.inputs.{key}")
    overrides = _object(payload["training_overrides"], "recipe.training_overrides",
                        required={"defaults", "trainer_parameters"})
    for key, value in overrides.items():
        if type(value) is not dict:
            raise ValueError(f"recipe.training_overrides.{key} 必须为对象")
    checkpoint = payload["checkpoint"]
    if payload["stage"] == "evaluate":
        if checkpoint is not None or payload["training_config"] is not None or any(overrides.values()):
            raise ValueError("M0 只允许评测，不能包含训练配置、覆盖或 checkpoint 策略")
    else:
        _object(checkpoint, "recipe.checkpoint", required={"save_steps", "save_total_limit"})
        for key in checkpoint:
            _typed(checkpoint[key], int, f"recipe.checkpoint.{key}")
            if checkpoint[key] < 1:
                raise ValueError(f"recipe.checkpoint.{key} 必须为正整数，受管训练必须周期完整保存")
    evaluation = _object(payload["evaluation"], "recipe.evaluation", optional=EVALUATION_TYPES)
    for key, value in evaluation.items():
        _typed(value, EVALUATION_TYPES[key], f"recipe.evaluation.{key}", nullable=key in {"config", "tasks", "snapshot", "split", "prefix", "save_traces"})
    if evaluation.get("kind", "benchmark") not in ("benchmark", "stage"):
        raise ValueError("recipe.evaluation.kind 必须为 benchmark/stage")
    if evaluation.get("config") and set(evaluation) - {"config", "kind"}:
        raise ValueError("recipe.evaluation.config 与内联评测参数不能混用")
    if "task_ids" in evaluation and not all(type(item) is str and item for item in evaluation["task_ids"]):
        raise ValueError("recipe.evaluation.task_ids 必须为非空字符串列表")
    budget = _object(payload["budget"], "recipe.budget", required={"max_steps", "comparison_group", "costs", "accounting", "strict_match"}, optional={"comparison_reference", "epochs"})
    _typed(budget["max_steps"], int, "recipe.budget.max_steps", nullable=True)
    if budget["max_steps"] is not None and budget["max_steps"] < 1:
        raise ValueError("recipe.budget.max_steps 必须为正整数或 null")
    epochs = budget.get("epochs")
    _typed(epochs, int, "recipe.budget.epochs", nullable=True)
    if epochs is not None and epochs < 1:
        raise ValueError("recipe.budget.epochs 必须为正整数或 null")
    if budget["max_steps"] is not None and epochs is not None:
        raise ValueError("recipe.budget.max_steps 与 epochs 必须二选一，不能同时设置")
    if payload["stage"] == "evaluate" and (budget["max_steps"] is not None or epochs is not None):
        raise ValueError("M0 只执行评测，不能包含训练步数或轮数预算")
    if budget["comparison_group"] not in (None, "M4") or budget["accounting"] != "gpu_hours" or budget["strict_match"] is not False:
        raise ValueError("预算必须使用 gpu_hours，不能自动声明严格等预算；comparison_group 只能是 M4 或 null")
    comparison = budget.get("comparison_reference")
    if comparison is not None:
        _reference(comparison, "recipe.budget.comparison_reference")
        if type(comparison) is dict and comparison["kind"] not in {"path", "run"}:
            raise ValueError("预算对照必须引用明确 M4 配方/成本报告路径或运行尝试")
    costs = _object(budget["costs"], "recipe.budget.costs", required={"preference_generation_gpu_hours", "dpo_gpu_hours", "grpo_gpu_hours"})
    for key, value in costs.items():
        _typed(value, float, f"recipe.budget.costs.{key}", nullable=True)
        if value is not None and value < 0:
            raise ValueError(f"recipe.budget.costs.{key} 不能为负")
    for key in ("task_pool", "filter_policy"):
        if key in payload:
            _object(payload[key], f"recipe.{key}", required={"path", "fingerprint"})
            for name, value in payload[key].items():
                _typed(value, str, f"recipe.{key}.{name}", nullable=True)
    if "name" in payload:
        _typed(payload["name"], str, "recipe.name", nullable=True)
    return copy.deepcopy(payload)


def load_profile(path):
    return validate_profile(read_json(path))


def load_recipe(path):
    return validate_recipe(read_json(path))


def normalize_profile(profile, workspace=None):
    """先拒绝原始资源路径别名，再生成绝对路径，不能抹去符号链接证据。"""
    from .paths import canonical

    result = validate_profile(profile)
    root = Path(workspace or Path.cwd()).expanduser().resolve()
    location = Path(result["workspace"]).expanduser()
    root = canonical(location if location.is_absolute() else root / location, exists=False)
    result["workspace"] = str(root)

    def resource(value):
        location = Path(value).expanduser()
        return str(canonical(location if location.is_absolute() else root / location, exists=False))

    for key in ("run_root", "cache_root"):
        result[key] = resource(result[key])
    result["input_roots"] = [resource(value) for value in result["input_roots"]]
    return result


def parse_overrides(values=(), *, seed=None, gpus=None, name=None):
    result = {"defaults": {}, "trainer_parameters": {}}
    for value in values:
        key, separator, raw = value.partition("=")
        parts = key.split(".")
        if not separator or len(parts) != 2 or parts[0] not in result:
            raise ValueError("--set 必须为 defaults.KEY=JSON 或 trainer_parameters.KEY=JSON")
        group, field = parts
        if field in result[group]:
            raise ValueError(f"重复覆盖字段: {key}")
        result[group][field] = json.loads(raw, object_pairs_hook=_pairs)
    if seed is not None:
        if "seed" in result["defaults"]:
            raise ValueError("--seed 与 --set defaults.seed 冲突")
        result["defaults"]["seed"] = seed
    if gpus is not None:
        result["gpus"] = gpus.split(",") if isinstance(gpus, str) else gpus
    if name is not None:
        result["name"] = name
    return result


def resolve(profile, recipe, overrides=None, workspace=None):
    """解析资源、配方和显式覆盖，训练字段继续由唯一 stage_config 校验。"""
    from agentic_arxiv.training.mainline_config import _validate_section, stage_config, validate_grpo_options

    profile = normalize_profile(profile, workspace)
    recipe = validate_recipe(recipe)
    overrides = copy.deepcopy(overrides or {})
    _object(overrides, "overrides", optional={"defaults", "trainer_parameters", "gpus", "name"})
    root = Path(profile["workspace"])

    def path(value):
        item = Path(value).expanduser()
        return str((item if item.is_absolute() else root / item).resolve())

    training = None
    sources = {}
    if recipe["stage"] != "evaluate":
        config_path = path(recipe["training_config"]) if recipe["training_config"] else None
        training = copy.deepcopy(stage_config(recipe["stage"], config_path))
        sources["training_config"] = config_path or "packaged:resources/mainline.json"
        for group, values in training.items():
            for key in values:
                sources[f"training.{group}.{key}"] = sources["training_config"]
        for layer_name, layer in (("recipe", recipe["training_overrides"]), ("explicit", overrides)):
            for group in ("defaults", "trainer_parameters"):
                values = layer.get(group, {})
                if type(values) is not dict:
                    raise ValueError(f"{layer_name}.{group} 必须为对象")
                for key, value in values.items():
                    if key not in training[group] or key in NON_OVERRIDE_FIELDS:
                        raise ValueError(f"不允许覆盖 {group}.{key}；输入与输出必须使用明确管理字段")
                    training[group][key] = value
                    sources[f"training.{group}.{key}"] = layer_name
        _validate_section(recipe["stage"], training, stage_config(recipe["stage"]))
        if recipe["stage"] == "grpo":
            validate_grpo_options(**{key: training["defaults"][key] for key in ("loss_type", "importance_sampling_level", "scale_rewards")})
        checkpoint_group = "defaults" if recipe["stage"] == "grpo" else "trainer_parameters"
        for key, value in recipe["checkpoint"].items():
            supplied = recipe["training_overrides"].get(checkpoint_group, {}).get(key, value)
            if supplied != value:
                raise ValueError(f"recipe.checkpoint.{key} 与 training_overrides 保存策略冲突")
            explicit = overrides.get(checkpoint_group, {}).get(key, value)
            if type(explicit) is not int or explicit < 1:
                raise ValueError(f"受管保存策略 {key} 必须为正整数")
            training[checkpoint_group][key] = explicit
            sources[f"training.{checkpoint_group}.{key}"] = "explicit" if key in overrides.get(checkpoint_group, {}) else "recipe.checkpoint"
        budget = recipe["budget"]
        if budget["max_steps"] is not None:
            if training["defaults"]["max_steps"] != budget["max_steps"]:
                raise ValueError("budget.max_steps 必须与最终 training.defaults.max_steps 一致")
        elif budget.get("epochs") is not None:
            if training["defaults"]["max_steps"] != -1 or training["defaults"]["epochs"] != budget["epochs"]:
                raise ValueError("epoch 预算要求 training.defaults.max_steps=-1 且 epochs 与 budget.epochs 一致")
    elif any(overrides.get(key) for key in ("defaults", "trainer_parameters")):
        raise ValueError("M0 不能接收训练参数覆盖")
    if "gpus" in overrides:
        values = overrides["gpus"]
        if type(values) is not list or not values or any(type(item) is not str or not item for item in values) or len(set(values)) != len(values):
            raise ValueError("显式 GPU 必须为非空、无重复的字符串列表")
        if not set(values) <= set(profile["gpus"]):
            raise ValueError("显式 GPU 超出 profile 允许范围")
    if "name" in overrides:
        _typed(overrides["name"], str, "overrides.name")
    payload = {"schema_version": 1, "profile": profile, "recipe": recipe, "training": training,
               "overrides": overrides, "sources": sources}
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return payload
