"""SFT/GRPO 主线的仓库配置与严格传参保护，不解析训练库默认值。"""

from __future__ import annotations

import dataclasses
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
    if set(payload) != {"schema_version", "source_baseline", "sft", "grpo"}:
        raise ValueError(f"主线配置包含缺失或未知顶层字段: {config_path}")
    for stage in ("sft", "grpo"):
        section = payload[stage]
        if not isinstance(section, dict) or set(section) != {"defaults", "trainer_parameters"}:
            raise ValueError(f"主线配置 {stage} 需要 defaults/trainer_parameters")
        if not all(isinstance(section[key], dict) for key in section):
            raise ValueError(f"主线配置 {stage} 的参数必须为对象")
    validate_grpo_options(**{
        key: payload["grpo"]["defaults"].get(key)
        for key in ("loss_type", "importance_sampling_level", "scale_rewards")
    })
    return payload


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
    for group in ("defaults", "trainer_parameters"):
        references = baseline[stage][group]
        values = payload[stage][group]
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
    return payload[stage]


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
