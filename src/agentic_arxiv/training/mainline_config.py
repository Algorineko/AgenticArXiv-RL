"""SFT/可选 DPO/GRPO 配置与严格传参保护，不解析训练库默认值。"""

from __future__ import annotations

import dataclasses
import copy
import json
import math
from pathlib import Path
from agentic_arxiv.config import (
    default_mainline_config_resource,
    read_default_mainline_config,
    resolve_model_source,
    workspace_path,
)

DELEGATED_GRPO_FIELDS = (
    "loss_type", "epsilon_high", "mask_truncated_completions",
    "importance_sampling_level", "scale_rewards",
)


def config_file_path(path: str | Path | None = None):
    """实际配置来源：显式工作区文件或已安装的唯一默认资源。"""
    return workspace_path(path) if path else default_mainline_config_resource()


def load_mainline_config(path: str | Path | None = None) -> dict:
    config_path = config_file_path(path)
    payload = json.loads(
        config_path.read_text(encoding="utf-8") if path
        else read_default_mainline_config()
    )
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"主线配置 schema_version 必须为 1: {config_path}")
    required = {"schema_version", "source_baseline", "sft", "grpo"}
    if not required <= set(payload) or set(payload) - required - {"dpo"}:
        raise ValueError(f"主线配置包含缺失或未知顶层字段: {config_path}")
    if "dpo" not in payload:
        payload["dpo"] = copy.deepcopy(json.loads(read_default_mainline_config())["dpo"])
    for stage in ("sft", "dpo", "grpo"):
        section = payload[stage]
        groups = {"defaults", "trainer_parameters"} | ({"generation_defaults"} if stage == "dpo" else set())
        if not isinstance(section, dict) or set(section) != groups:
            raise ValueError(f"主线配置 {stage} 需要完整字段: {sorted(groups)}")
        if not all(isinstance(section[key], dict) for key in section):
            raise ValueError(f"主线配置 {stage} 的参数必须为对象")
    validate_grpo_options(**{
        key: payload["grpo"]["defaults"].get(key)
        for key in ("loss_type", "importance_sampling_level", "scale_rewards")
    })
    _validate_section("dpo", payload["dpo"], json.loads(read_default_mainline_config())["dpo"])
    return payload


def stage_config_source(stage: str, path: str | Path | None = None) -> dict:
    """记录实际默认来源；旧配置补值仅在内存进行，不写回用户文件。"""
    config_path = config_file_path(path)
    explicit = json.loads(config_path.read_text(encoding="utf-8")) if path else None
    fallback = stage == "dpo" and explicit is not None and "dpo" not in explicit
    source = {
        "stage": stage,
        "config_path": str(config_path),
        "defaults_source": str(default_mainline_config_resource()) if fallback else str(config_path),
        "used_packaged_dpo_defaults": fallback,
    }
    if stage == "dpo":
        source["baseline"] = {
            "kind": "design_defaults_not_validated_recipe",
            "change": "restore-optional-dpo-stage",
            "preserves_sft_grpo_baseline": True,
        }
    return source


def validate_grpo_options(*, loss_type=None, importance_sampling_level=None, scale_rewards=None):
    """保留未设置语义及主线数值参数，但不重新开放独立算法选择。"""
    if loss_type not in (None, "grpo"):
        raise ValueError("主线不支持算法变体 loss_type；未设置时继续委托 TRL")
    if importance_sampling_level not in (None, "token"):
        raise ValueError("主线不支持序列级算法变体 importance_sampling_level")
    if scale_rewards not in (None, "group", "none", "batch"):
        raise ValueError("scale_rewards 必须为 group/none/batch 或未设置")


def stage_config(stage: str, path: str | Path | None = None) -> dict:
    payload = load_mainline_config(path)
    baseline = load_mainline_config() if path else payload
    _validate_section(stage, payload[stage], baseline[stage])
    return payload[stage]


def _validate_section(stage: str, section: dict, baseline: dict) -> None:
    for group in baseline:
        references = baseline[group]
        values = section[group]
        expected = set(references)
        actual = set(values)
        if actual != expected:
            raise ValueError(
                f"{stage}.{group} 缺失字段={sorted(expected - actual)}，"
                f"未知字段={sorted(actual - expected)}"
            )
        for key, value in values.items():
            reference = references[key]
            if reference is None:
                expected_type = {"epsilon_high": float,
                                 "mask_truncated_completions": bool}.get(key, str)
                if value is None:
                    continue
            else:
                expected_type = type(reference)
            valid = (type(value) in (int, float) and math.isfinite(value)
                     if expected_type is float else type(value) is expected_type)
            if not valid:
                raise ValueError(f"{stage}.{group}.{key} 必须是 {expected_type.__name__}")


def require_supported_config(config_cls, kwargs: dict) -> dict:
    """显式参数不支持时失败，禁止训练库兼容层改变主线算法。"""
    supported = {field.name for field in dataclasses.fields(config_cls)}
    unsupported = sorted(set(kwargs) - supported)
    if unsupported:
        raise ValueError(
            f"当前 {config_cls.__name__} 不支持主线显式参数: {unsupported}；"
            "拒绝静默忽略，请使用支持这些字段的训练依赖"
        )
    return kwargs
