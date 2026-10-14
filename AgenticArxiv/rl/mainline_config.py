"""SFT/GRPO 主线的仓库配置与严格传参保护，不解析训练库默认值。"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "training" / "mainline.json"
DELEGATED_GRPO_FIELDS = (
    "loss_type", "epsilon_high", "mask_truncated_completions",
    "importance_sampling_level", "scale_rewards",
)


def config_file_path(path: str | Path | None = None) -> Path:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    return config_path


def load_mainline_config(path: str | Path | None = None) -> dict:
    config_path = config_file_path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
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


def parse_mainline_args(parser: argparse.ArgumentParser, stage: str, argv=None):
    """配置提供默认值；显式 CLI 覆盖，不默默忽略配置中的字段。"""
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--config", dest="config_path", default=None)
    selected, _ = selector.parse_known_args(argv)
    try:
        defaults = stage_config(stage, selected.config_path)["defaults"]
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    parser.add_argument("--config", dest="config_path", default=None,
                        help="小型主线配置 JSON；显式 CLI 参数优先")
    parser.set_defaults(**defaults)
    return parser.parse_args(argv)


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


def resolve_model_source(source: str, *, local_only: bool = False) -> str:
    """本地路径必须存在；显式 Hub 标识只允许用于 SFT 初始基座。"""
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.is_dir():
        return str(path.resolve())
    is_hub_id = (
        len(source.split("/")) == 2
        and not source.startswith(("outputs/", "models/", "./", "../", "/", "~"))
    )
    if not local_only and is_hub_id:
        return source
    raise ValueError(f"指定模型路径不存在或不是模型目录: {path}")
