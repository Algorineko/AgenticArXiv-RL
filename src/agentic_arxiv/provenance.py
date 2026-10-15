"""来源与文件记录工具；只消费显式输入，不查询运行时协议。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from loguru import logger as log

from agentic_arxiv.contracts import (
    OBSERVATION_DATA_PREFIX, Trajectory, business_action, copy_tool_result,
    trajectory_from_dict, trajectory_to_dict,
)


PROTOCOL_FIELDS = (
    "runtime_version", "context_version", "training_protocol",
    "prompt_template_hash", "chat_template_hash", "template_hash",
)
_REQUIRED_PROTOCOL_FIELDS = PROTOCOL_FIELDS[:4]
_PACKAGES = ("torch", "transformers", "trl", "accelerate", "datasets", "peft", "bitsandbytes")
DPO_DATA_KIND = "dpo-react-messages-v3"
DPO_DATA_VERSION = 1
DPO_PAIRING_VERSION = "strict-first-action-divergence-v1"
DPO_RESULT_CODES = (
    "selected", "not_selected", "duplicate_sample", "sampling_failure",
    "source_mismatch", "missing_record", "context_mismatch",
    "missing_response", "unusable_decision", "no_action_divergence",
    "nonfinite_score", "insufficient_reward_gap",
)
_SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:password|passwd|secret|credential|credentials|authorization|"
    r"api_key|access_key|private_key|secret_key|signing_key|encryption_key|"
    r"access_token|refresh_token|hub_token|hf_token|api_token)$"
)


def _sensitive_key(key: str) -> bool:
    key = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower().replace("-", "_")
    special_tokens = {"bos_token", "eos_token", "pad_token", "sep_token", "unk_token", "mask_token"}
    return (key in {"key", "token", "auth", "cookie", "cookies"}
            or (key.endswith("_token") and key not in special_tokens)
            or bool(_SENSITIVE_KEY.search(key)))


def _safe_string(value: str) -> str | None:
    """仅处理 URL 内的凭据；不读取环境变量或猜测任意字符串内容。"""
    if "://" not in value:
        return value
    try:
        parts = urlsplit(value)
        hostname = parts.netloc.rsplit("@", 1)[-1]
        query = [(key, item) for key, item in parse_qsl(parts.query, keep_blank_values=True)
                 if not _sensitive_key(key)]
        return urlunsplit((parts.scheme, hostname, parts.path, urlencode(query), parts.fragment))
    except ValueError:
        return None


def sanitize_metadata(value: Any) -> Any:
    """排除凭据字段；无法序列化的对象保持未知，而非写入可能泄密的 repr。"""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return sanitize_metadata(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return _safe_string(value)
    if isinstance(value, Mapping):
        return {str(key): sanitize_metadata(item) for key, item in value.items()
                if not _sensitive_key(str(key))}
    if isinstance(value, (list, tuple)):
        return [sanitize_metadata(item) for item in value]
    return None


def canonical_hash(value: Any) -> str:
    """对实际提供且已脱敏的内容求稳定 SHA256。"""
    payload = json.dumps(sanitize_metadata(value), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """对明确提供的文件进行流式哈希，不加载模型或递归搜索资产。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path | str | None) -> dict:
    """只读文件指纹；目录/缺失文件没有伪造的内容哈希。"""
    if path is None:
        return {"path": None, "sha256": None, "status": "unknown"}
    target = Path(path)
    result = {"path": str(target), "sha256": None, "status": "missing"}
    if target.is_dir():
        result["status"] = "directory"
    elif target.is_file():
        result.update(sha256=sha256_file(target), status="file", size_bytes=target.stat().st_size)
    return result


def model_input_fingerprint(
    source: Path | str | None, *, resolved_revision: str | None = None
) -> dict:
    """只哈希明确模型输入文件；Hub revision/缺失权重不猜填。

    不递归扫描输出目录。分片名称只来自标准权重 index，且不允许路径逃逸。
    """
    result = {
        "source": sanitize_metadata(source), "status": "unknown",
        "resolved_revision": sanitize_metadata(resolved_revision),
        "files": [], "weights_status": "unknown",
    }
    if source is None:
        return result
    root = Path(source)
    if not root.is_dir():
        result["status"] = "not_local_or_missing"
        return result
    result["status"] = "local"
    config_names = (
        "config.json", "adapter_config.json", "generation_config.json",
        "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json",
        "vocab.json", "merges.txt", "tokenizer.model", "added_tokens.json",
        "chat_template.jinja", "training_manifest.json",
    )
    weight_names = {
        "model.safetensors", "pytorch_model.bin",
        "adapter_model.safetensors", "adapter_model.bin",
    }
    index_names = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    selected = {name for name in config_names if (root / name).is_file()}
    selected.update(name for name in weight_names if (root / name).is_file())
    indexed_weights = set()
    for name in index_names:
        path = root / name
        if not path.is_file():
            continue
        selected.add(name)
        index = json.loads(path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
        if not isinstance(weight_map, Mapping):
            raise ValueError(f"模型权重 index 缺少 weight_map: {path}")
        for filename in weight_map.values():
            if not isinstance(filename, str):
                raise ValueError(f"模型权重 index 文件名必须为字符串: {path}")
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"模型权重 index 路径越界: {path}")
            indexed_weights.add(filename)
    selected.update(indexed_weights)
    result["files"] = [file_fingerprint(root / name) for name in sorted(selected)]
    weights = [entry for name, entry in zip(sorted(selected), result["files"])
               if name in weight_names or name in indexed_weights]
    if weights:
        result["weights_status"] = "known" if all(item["status"] == "file" for item in weights) else "missing"
    result["input_fingerprint"] = canonical_hash(result)
    return result


def dependency_versions(packages: Iterable[str] = _PACKAGES) -> dict:
    """当前实际安装版本，不作为历史依赖版本证据。"""
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def source_fingerprint(paths: Iterable[Path | str], *, repo_root: Path | str) -> dict:
    """保存本次执行源码的 revision 和文件哈希，不猜填历史基线。"""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"revision": revision, "files": [file_fingerprint(path) for path in paths]}


def read_protocol_metadata(source: Mapping[str, Any] | Path | str | None) -> dict:
    """读取版本声明；历史缺字段为未知，绝不填入当前活动版本。"""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            path = path / "training_manifest.json"
        source = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if source is not None and not isinstance(source, Mapping):
        raise ValueError("协议元数据必须为 JSON 对象")
    document = source or {}
    nested = document.get("protocol_metadata")
    if not isinstance(nested, Mapping):
        nested = document.get("protocol")
    nested = nested if isinstance(nested, Mapping) else {}
    result = {}
    for key in PROTOCOL_FIELDS:
        value = document.get(key)
        if value is None:
            value = nested.get(key)
        known = (isinstance(value, str) and value.strip()
                 and value.strip().lower() not in {"unknown", "none", "null", "n/a", "未知"})
        result[key] = value if known else None
    limit = document.get("observation_body_limit", nested.get("observation_body_limit"))
    if isinstance(limit, int) and not isinstance(limit, bool):
        result["observation_body_limit"] = limit
    known = sum(result[key] is not None for key in PROTOCOL_FIELDS[:-1])
    result["metadata_status"] = "unknown" if not known else ("known" if known == 5 else "partial")
    return result


def with_protocol_metadata(payload: Mapping[str, Any], protocol: Mapping[str, Any] | None) -> dict:
    """为新产物附加声明；没有实际事实时仍保存未知，不修改输入对象。"""
    return {**payload, "protocol_metadata": read_protocol_metadata(protocol)}


def require_training_protocol(
    source: Mapping[str, Any] | Path | str | None,
    expected_protocol: Mapping[str, Any] | str,
    *,
    source_label: str = "训练输入",
    require_chat_template_hash: bool = False,
) -> dict:
    """拒绝未知/不兼容训练协议；无 tokenizer 的数据不伪造 chat 哈希。"""
    actual = read_protocol_metadata(source)
    expected = (read_protocol_metadata(expected_protocol) if isinstance(expected_protocol, Mapping)
                else {"context_version": expected_protocol, "training_protocol": expected_protocol})
    if not expected.get("context_version") or not expected.get("training_protocol"):
        raise ValueError("当前训练协议未明确，不能检查输入兼容性")
    missing = [key for key in _REQUIRED_PROTOCOL_FIELDS if actual[key] is None]
    if require_chat_template_hash and actual["chat_template_hash"] is None:
        missing.append("chat_template_hash")
    if missing:
        raise ValueError(f"{source_label} 缺少必要协议元数据: {', '.join(missing)}")
    mismatches = [key for key in PROTOCOL_FIELDS[:-1]
                  if expected.get(key) is not None and actual[key] is not None
                  and actual[key] != expected[key]]
    if mismatches:
        details = ", ".join(f"{key}={actual[key]!r} (expected {expected[key]!r})"
                            for key in mismatches)
        raise ValueError(f"{source_label} 协议不兼容: {details}")
    return actual


def evaluation_protocol_relation(training: Any, evaluation: Any) -> dict:
    """跨协议评测明确标注；未知来源不能默认为同口径或兼容。"""
    train = read_protocol_metadata(training)
    evaluate = read_protocol_metadata(evaluation)
    # 执行记录的 context 是评测协议，不要求它宣称 checkpoint 的训练事实。
    compared_evaluation = {**evaluate, "training_protocol": evaluate["context_version"]}
    different = [key for key in PROTOCOL_FIELDS[:-1]
                 if train[key] is not None and compared_evaluation[key] is not None
                 and train[key] != compared_evaluation[key]]
    unknown = [key for key in PROTOCOL_FIELDS[:-1]
               if train[key] is None or compared_evaluation[key] is None]
    relation = "cross_protocol" if different else ("unknown" if unknown else "same_protocol")
    return {"relation": relation, "different_fields": different, "unknown_fields": unknown,
            "training": train, "evaluation": evaluate,
            "same_comparison_basis": relation == "same_protocol"}


def build_run_manifest(
    *,
    requested_config: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    derived_values: Mapping[str, Any],
    source_baseline: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
    resource_overrides: Mapping[str, Any] | None = None,
    protocol_metadata: Mapping[str, Any] | None = None,
    requested_steps: int | None = None,
    actual_steps: int | None = None,
    stop_reason: str | None = None,
    protections: Mapping[str, Any] | None = None,
    versions: Mapping[str, Any] | None = None,
) -> dict:
    """记录调用方可证明的事实，不执行训练、不推测库默认值或结束原因。"""
    requested = sanitize_metadata(requested_config)
    resolved = sanitize_metadata(resolved_config)
    result = {
        "run_manifest_version": 1,
        "source_baseline": sanitize_metadata(source_baseline),
        "requested_config": requested,
        "resolved_config": resolved,
        "derived_values": sanitize_metadata(derived_values),
        "requested_unset_fields": sorted(key for key, value in requested.items() if value is None),
        "resolved_unknown_fields": sorted(key for key, value in resolved.items() if value is None),
        "inputs": sanitize_metadata(inputs or {}),
        "resource_overrides": sanitize_metadata(resource_overrides or {}),
        "dependency_versions": sanitize_metadata(versions if versions is not None else dependency_versions()),
        "protocol_metadata": read_protocol_metadata(protocol_metadata),
        "requested_steps": requested_steps,
        "actual_steps": actual_steps,
        "stop_reason": stop_reason,
        "protections": sanitize_metadata(protections or {}),
    }
    # 输出位置和日志名不决定实验配置；资源覆盖、来源、版本和输入哈希决定。
    fingerprint = {key: value for key, value in result.items()
                   if key not in {"actual_steps", "stop_reason"}}
    for key in ("requested_config", "resolved_config"):
        fingerprint[key] = {name: value for name, value in fingerprint[key].items()
                            if name not in {"output_dir", "logging_dir", "run_name", "rollout_trace_path"}}
    result["run_fingerprint"] = canonical_hash(fingerprint)
    return result


def read_json(path: Path) -> Any:
    """读取显式 JSON 文件，不猜测输入所在位置。"""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, overwrite: bool = True) -> None:
    """写入 JSON；要求保全既有文件的调用方可以显式拒绝覆盖。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """逐行读取 JSONL 文件，跳过空行。"""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dpo_content_hash(value: Any) -> str:
    """Hash exact preference content, without metadata sanitization or normalization."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dpo_sample_fingerprint(row: Mapping[str, Any]) -> str:
    return dpo_content_hash({key: row[key] for key in ("prompt", "chosen", "rejected")})


def _dpo_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"DPO {label} 必须为有限数值")
    return float(value)


def dpo_trace_decisions(trace: Mapping[str, Any], *, decision_parser, messages_validator) -> list:
    """Validate actual request/response to history bindings using the caller's runtime API.

    Foundation code does not import the runtime. Callers must supply its real parser
    and message validator, not a compatibility parser for historical data.
    """
    requests = trace.get("requests")
    trajectory = trace.get("trajectory")
    history = trajectory.get("history") if isinstance(trajectory, Mapping) else None
    if not isinstance(requests, list) or not isinstance(history, list):
        raise ValueError("DPO trace 缺少实际 requests/history")
    if not requests:
        raise ValueError("DPO trace 缺少当前响应")
    identity = trace.get("identity")
    sampling = identity.get("sampling") if isinstance(identity, Mapping) else None
    protocol = identity.get("protocol_metadata") if isinstance(identity, Mapping) else None
    if (not isinstance(sampling, Mapping) or not isinstance(protocol, Mapping)
            or type(sampling.get("max_turns")) is not int
            or not 0 < len(requests) <= sampling["max_turns"]):
        raise ValueError("DPO 实际请求与采样预算不一致")
    decisions = []
    seen = set()
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            raise ValueError("DPO 请求记录必须为对象")
        messages_validator(request.get("messages"))
        if index:
            previous_request, previous_parsed, _, _ = decisions[-1]
            previous_step = history[index - 1]
            observation = previous_step.get("visible_observation")
            if (previous_parsed.kind != "CALL" or previous_step.get("framework_generated")
                    or not isinstance(observation, str)):
                raise ValueError("DPO 请求历史跨越真实终止或缺少环境反馈")
            expected_messages = [
                *previous_request["messages"],
                {"role": "assistant", "content": previous_parsed.policy_text},
                {"role": "user", "content": OBSERVATION_DATA_PREFIX + observation},
            ]
            if request["messages"] != expected_messages:
                raise ValueError("DPO 实际请求未保留此前真实 Thought/Action/Observation")
        options = request.get("request_options")
        if (not isinstance(options, Mapping)
                or options.get("temperature") != sampling.get("temperature")
                or options.get("max_tokens") != sampling.get("max_new_tokens")):
            raise ValueError("DPO 实际请求与采样设置不一致")
        step_index = request.get("step_index")
        if (isinstance(step_index, bool) or not isinstance(step_index, int)
                or step_index != index or step_index >= len(history) or step_index in seen):
            raise ValueError("DPO 请求缺少唯一真实 history 位置")
        seen.add(step_index)
        step = history[step_index]
        response = request.get("response")
        if not isinstance(step, Mapping) or not isinstance(response, Mapping):
            raise ValueError("DPO 缺少实际模型响应或历史记录")
        if (not protocol.get("chat_template_hash")
                or step.get("chat_template_hash") != protocol["chat_template_hash"]):
            raise ValueError("DPO 实际请求模板与协议声明不一致")
        choices = response.get("choices") or [{}]
        if not isinstance(choices, list) or not isinstance(choices[0], Mapping):
            raise ValueError("DPO 模型响应结构无效")
        raw = response.get("raw_model_content")
        if raw is None:
            raw = (choices[0].get("message") or {}).get("content") or ""
        if not isinstance(raw, str) or raw != step.get("raw_response"):
            raise ValueError("DPO 实际响应与 history 不一致")
        parsed = decision_parser(raw)
        if (request.get("policy_text") != parsed.policy_text
                or step.get("policy_text") != parsed.policy_text):
            raise ValueError("DPO policy_text 与真实策略边界不一致")
        if request.get("finish_reason") != choices[0].get("finish_reason"):
            raise ValueError("DPO 截断标识与实际响应不一致")
        usable = (bool(parsed.policy_text.strip())
                  and parsed.kind in {"CALL", "FINISH"}
                  and not step.get("framework_generated")
                  and request.get("finish_reason") == "stop"
                  and step.get("termination_reason") != "TOKEN_LIMIT")
        # Ignore only framework arguments already excluded at execution/scoring;
        # preserve message/target text verbatim, including their whitespace.
        action = business_action(parsed.action) if parsed.kind == "CALL" else str(parsed.kind.value)
        action = json.dumps(action, ensure_ascii=False, sort_keys=True, allow_nan=False)
        decisions.append((request, parsed, usable, action))
    # A framework budget marker may follow the last actual request; an unrecorded
    # model step may not. This also rejects partial request capture.
    if any(not step.get("framework_generated") for step in history[len(requests):]):
        raise ValueError("DPO history 含未绑定的模型响应")
    return decisions


def _validate_dpo_trace_identity(trace: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    identity = trace.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("DPO trace 缺少来源身份")
    for key in ("source_model", "source_split", "snapshot", "sampling", "scoring"):
        if not manifest.get(key) or identity.get(key) != manifest[key]:
            raise ValueError(f"DPO trace {key} 与数据包来源不一致")
    require_training_protocol(identity, manifest.get("protocol_metadata"),
                              source_label="DPO trace", require_chat_template_hash=True)
    task_id = trace.get("task_id")
    initialization = manifest.get("initialization")
    entry = initialization.get(task_id) if isinstance(initialization, Mapping) else None
    if (not isinstance(entry, Mapping) or not isinstance(entry.get("plan"), list)
            or canonical_hash(entry["plan"]) != entry.get("sha256")
            or identity.get("initialization") != entry["plan"]):
        raise ValueError("DPO trace 初始化与数据包不一致")
    task_hashes = manifest.get("task_definition_hashes")
    if (not isinstance(task_hashes, Mapping) or not task_hashes.get(task_id)
            or identity.get("task_definition_sha256") != task_hashes[task_id]):
        raise ValueError("DPO trace 任务定义与数据包不一致")


def validate_dpo_row(row: Mapping[str, Any], traces_by_id: Mapping[str, Any],
                     manifest: Mapping[str, Any], *, decision_parser,
                     messages_validator) -> None:
    """Audit a current-version single-step preference against both actual traces."""
    if row.get("kind") != DPO_DATA_KIND or row.get("version") != DPO_DATA_VERSION:
        raise ValueError("DPO 行格式不兼容；旧字符串偏好不自动迁移")
    task_id = row.get("task_id")
    if task_id not in manifest.get("task_ids", []):
        raise ValueError("DPO 行不属于声明的训练任务成员")
    if row.get("source_split") != manifest.get("source_split"):
        raise ValueError("DPO 行切分与数据声明不一致")
    require_training_protocol(row, manifest.get("protocol_metadata"),
                              source_label="DPO 行", require_chat_template_hash=True)
    messages_validator(row.get("prompt"))
    for side in ("chosen", "rejected"):
        target = row.get(side)
        if (not isinstance(target, list) or len(target) != 1
                or not isinstance(target[0], Mapping) or set(target[0]) != {"role", "content"}
                or target[0].get("role") != "assistant"
                or not isinstance(target[0].get("content"), str)):
            raise ValueError("DPO 目标必须为当前单条 assistant 消息")
        messages_validator([*row["prompt"], *target], require_completion=True)
    chosen_id, rejected_id = row.get("chosen_trace_id"), row.get("rejected_trace_id")
    if chosen_id == rejected_id or chosen_id not in traces_by_id or rejected_id not in traces_by_id:
        raise ValueError("DPO 行缺少两个不同的真实 trace 引用")
    traces = [traces_by_id[chosen_id], traces_by_id[rejected_id]]
    for trace in traces:
        if (trace.get("status") != "completed" or trace.get("task_id") != task_id
                or trace.get("source_split") != row.get("source_split")):
            raise ValueError("DPO trace 状态、任务或切分不一致")
        _validate_dpo_trace_identity(trace, manifest)
    if traces[0]["identity"] != traces[1]["identity"]:
        raise ValueError("DPO 配对 trace 来源身份不一致")
    decisions = [dpo_trace_decisions(trace, decision_parser=decision_parser,
                                   messages_validator=messages_validator) for trace in traces]
    step = row.get("decision_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("DPO decision_step 必须为非负整数")
    for index in range(step + 1):
        if any(index >= len(side) for side in decisions):
            raise ValueError("DPO 缺少当前真实响应；不得补造 FINISH")
        left, right = decisions[0][index], decisions[1][index]
        if left[0]["messages"] != right[0]["messages"]:
            raise ValueError("DPO 实际完整上下文不一致")
        if not left[2] or not right[2]:
            raise ValueError("DPO 当前决策不可用或已截断")
        divergent = left[3] != right[3]
        if divergent != (index == step):
            raise ValueError("DPO 目标不是首个真实动作分歧")
    for side, decision in zip(("chosen", "rejected"), (decisions[0][step], decisions[1][step])):
        if row["prompt"] != decision[0]["messages"] or row[side][0]["content"] != decision[1].policy_text:
            raise ValueError("DPO 目标或上下文不是所引用的实际请求/响应")
    high, low = [_dpo_number(trace.get("reward"), "trace reward") for trace in traces]
    gap = high - low
    threshold = _dpo_number(manifest.get("min_reward_gap"), "min_reward_gap")
    if threshold < 0 or gap <= threshold:
        raise ValueError("DPO 奖励差必须严格大于非负阈值")
    if (row.get("chosen_reward") != high or row.get("rejected_reward") != low
            or row.get("reward_gap") != gap):
        raise ValueError("DPO 行奖励与 trace 不一致")
    if (row.get("context_fingerprint") != dpo_content_hash(row["prompt"])
            or row.get("sample_fingerprint") != dpo_sample_fingerprint(row)):
        raise ValueError("DPO 行内容指纹不一致")


def load_dpo_package(data_path: Path | str, *, manifest_path: Path | str | None = None,
                     expected_protocol=None, expected_source_model=None,
                     decision_parser, messages_validator) -> tuple[list, list, dict]:
    """Read only a completed, hash-bound current DPO package; never repair input."""
    data_path = Path(data_path).resolve()
    manifest_path = Path(manifest_path) if manifest_path else Path(str(data_path) + ".manifest.json")
    manifest = read_json(manifest_path)
    if (not isinstance(manifest, Mapping) or manifest.get("kind") != DPO_DATA_KIND
            or manifest.get("version") != DPO_DATA_VERSION
            or manifest.get("status") != "complete"
            or manifest.get("pairing_version") != DPO_PAIRING_VERSION):
        raise ValueError("DPO 数据包未完成或格式/配对版本不兼容")
    require_training_protocol(manifest, expected_protocol or manifest,
                              source_label="DPO 数据包", require_chat_template_hash=True)
    source_model = manifest.get("source_model")
    if not isinstance(source_model, Mapping) or not source_model:
        raise ValueError("DPO 数据包缺少来源 SFT 指纹")
    if expected_source_model is not None and source_model != expected_source_model:
        raise ValueError("DPO 数据来源与当前 SFT 模型不一致")
    model_manifest = manifest.get("source_model_manifest")
    if not isinstance(model_manifest, Mapping) or not isinstance(model_manifest.get("path"), str):
        raise ValueError("DPO 数据包缺少来源 SFT manifest 指纹")
    model_manifest_path = Path(model_manifest["path"])
    model_manifest_path = (model_manifest_path if model_manifest_path.is_absolute()
                           else manifest_path.parent / model_manifest_path)
    if (not model_manifest_path.is_file()
            or sha256_file(model_manifest_path) != model_manifest.get("sha256")):
        raise ValueError("DPO 来源 SFT manifest 缺失或哈希不一致")
    manifest_entries = [entry for entry in source_model.get("files", [])
                        if Path(str(entry.get("path", ""))).name == "training_manifest.json"]
    if (len(manifest_entries) != 1
            or manifest_entries[0].get("sha256") != model_manifest.get("sha256")):
        raise ValueError("DPO 来源 SFT 指纹未绑定同一 training manifest")
    split = manifest.get("source_split")
    task_ids = manifest.get("task_ids")
    if (not isinstance(split, str) or not split.endswith(":train")
            or split == ":train" or not isinstance(task_ids, list) or not task_ids
            or any(not isinstance(item, str) or not item for item in task_ids)
            or len(set(task_ids)) != len(task_ids)):
        raise ValueError("DPO 数据包需要显式版本化 train 切分及唯一成员")
    split_file = manifest.get("split_file")
    if not isinstance(split_file, Mapping) or not isinstance(split_file.get("path"), str):
        raise ValueError("DPO 数据包缺少实际切分文件指纹")
    split_path = Path(split_file["path"])
    split_path = split_path if split_path.is_absolute() else manifest_path.parent / split_path
    if split != f"{split_path.name}:train":
        raise ValueError("DPO 来源切分名与实际文件名不一致")
    if not split_path.is_file() or sha256_file(split_path) != split_file.get("sha256"):
        raise ValueError("DPO 实际切分文件缺失或哈希不一致")
    split_payload = read_json(split_path)
    groups = split_payload.get("split") if isinstance(split_payload, Mapping) else None
    train_ids = groups.get("train") if isinstance(groups, Mapping) else None
    if (not isinstance(train_ids, list) or not train_ids
            or not set(task_ids).issubset(train_ids)):
        raise ValueError("DPO 声明任务不属于实际切分文件的 train 成员")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("DPO 数据包缺少文件清单")
    loaded = {}
    for name in ("data", "traces"):
        entry = files.get(name)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ValueError(f"DPO 数据包缺少 {name} 文件声明")
        path = Path(entry["path"])
        path = path if path.is_absolute() else manifest_path.parent / path
        if name == "data" and path.resolve() != data_path:
            raise ValueError("DPO 数据文件路径与 manifest 不一致")
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise ValueError(f"DPO {name} 文件缺失或哈希不一致: {path}")
        loaded[name] = read_jsonl(path)
        if entry.get("count") != len(loaded[name]):
            raise ValueError(f"DPO {name} 记录数不一致")
    rows, traces = loaded["data"], loaded["traces"]
    if not rows:
        raise ValueError("DPO 数据包不能为空")
    report_entry = manifest.get("report")
    if not isinstance(report_entry, Mapping) or not isinstance(report_entry.get("path"), str):
        raise ValueError("DPO 数据包缺少诊断报告指纹")
    report_path = Path(report_entry["path"])
    report_path = report_path if report_path.is_absolute() else manifest_path.parent / report_path
    if not report_path.is_file() or sha256_file(report_path) != report_entry.get("sha256"):
        raise ValueError("DPO 诊断报告缺失或哈希不一致")
    report = read_json(report_path)
    if (not isinstance(report, Mapping) or report.get("status") != "complete"
            or report.get("final_pairs") != len(rows)
            or report.get("requested_tasks") != len(task_ids)
            or report.get("min_reward_gap") != manifest.get("min_reward_gap")
            or report.get("sampling") != manifest.get("sampling")
            or report.get("scoring") != manifest.get("scoring")):
        raise ValueError("DPO 诊断报告状态、计数或配置与数据包不一致")
    by_id = {}
    trials = set()
    sampling = manifest.get("sampling")
    num_trials = sampling.get("num_trials") if isinstance(sampling, Mapping) else None
    if type(num_trials) is not int or num_trials < 2:
        raise ValueError("DPO 数据包缺少每任务请求采样数")
    for trace in traces:
        if not isinstance(trace, Mapping):
            raise ValueError("DPO trace 必须为对象")
        trace_id = trace.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id or trace_id in by_id:
            raise ValueError("DPO trace ID 缺失或重复")
        if trace.get("task_id") not in task_ids or trace.get("source_split") != split:
            raise ValueError("DPO trace 不属于声明的训练成员")
        trial_id = trace.get("trial_id")
        trial_key = (trace["task_id"], trial_id)
        if type(trial_id) is not int or not 0 <= trial_id < num_trials or trial_key in trials:
            raise ValueError("DPO trace 缺少唯一请求 trial 位置")
        if trace.get("status") not in {"completed", "invalid"}:
            raise ValueError("DPO trace 状态无效")
        expected_stream = f"dpo:{trace['task_id']}:{trial_id}"
        if (sampling.get("stream_key_format") != "dpo:{task_id}:{trial_id}"
                or trace_id != expected_stream or trace.get("stream_key") != expected_stream):
            raise ValueError("DPO trace 随机流与任务/trial 不一致")
        trials.add(trial_key)
        _validate_dpo_trace_identity(trace, manifest)
        by_id[trace_id] = trace
    expected_rollouts = len(task_ids) * num_trials
    expected_combinations = len(task_ids) * num_trials * (num_trials - 1) // 2
    counts = report.get("result_counts")
    combinations = report.get("combinations")
    if (len(traces) != expected_rollouts or report.get("requested_rollouts") != expected_rollouts
            or report.get("requested_combinations") != expected_combinations
            or not isinstance(counts, Mapping)
            or set(counts) != set(DPO_RESULT_CODES)
            or any(type(value) is not int or value < 0 for value in counts.values())
            or sum(counts.values()) != expected_combinations
            or not isinstance(combinations, list) or len(combinations) != expected_combinations):
        raise ValueError("DPO 请求分母、trace 数或结果分类总数不一致")
    actual_counts = dict.fromkeys(DPO_RESULT_CODES, 0)
    seen_combinations = set()
    selected_tasks = set()
    for combination in combinations:
        if not isinstance(combination, Mapping):
            raise ValueError("DPO 组合诊断必须为对象")
        tid, pair, code = (combination.get("task_id"), combination.get("trial_ids"),
                           combination.get("result"))
        if (tid not in task_ids or not isinstance(pair, list) or len(pair) != 2
                or any(type(trial) is not int for trial in pair)
                or not 0 <= pair[0] < pair[1] < num_trials or code not in actual_counts):
            raise ValueError("DPO 组合结果缺少唯一请求位置或结果码")
        key = (tid, *pair)
        if key in seen_combinations:
            raise ValueError("DPO 请求组合重复计数")
        seen_combinations.add(key)
        actual_counts[code] += 1
        if code == "selected":
            if tid in selected_tasks:
                raise ValueError("DPO 同一任务重复选择偏好对")
            selected_tasks.add(tid)
    completed_rollouts = sum(trace["status"] == "completed" for trace in traces)
    accepted = sum(actual_counts[code] for code in ("selected", "not_selected", "duplicate_sample"))
    if (counts != actual_counts or actual_counts["selected"] != len(rows)
            or selected_tasks != {row.get("task_id") for row in rows}
            or report.get("completed_rollouts") != completed_rollouts
            or report.get("invalid_rollouts") != expected_rollouts - completed_rollouts
            or report.get("accepted_combinations") != accepted
            or report.get("discarded_combinations") != expected_combinations - accepted
            or report.get("combination_acceptance_rate") != accepted / expected_combinations
            or report.get("task_yield_rate") != len(rows) / len(task_ids)):
        raise ValueError("DPO 诊断结果分类、产出率与实际请求不一致")
    seen_tasks = set()
    for row in rows:
        validate_dpo_row(row, by_id, manifest, decision_parser=decision_parser,
                         messages_validator=messages_validator)
        if row["task_id"] in seen_tasks:
            raise ValueError("DPO 每任务最多一个偏好对")
        seen_tasks.add(row["task_id"])
    return rows, traces, dict(manifest)


def write_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool = True,
) -> None:
    """覆盖写 JSONL；UTF-8、ensure_ascii=False，每行一条且行尾换行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def save_trajectory(traj: Trajectory, filepath: Path) -> None:
    """保存单条轨迹到 JSONL，保留原追加写语义。"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trajectory_to_dict(traj), ensure_ascii=False) + "\n")


def load_trajectories(filepath: Path) -> List[Trajectory]:
    """读取历史轨迹；缺失文件为空，缺失协议字段保持未知。"""
    if not filepath.exists():
        return []
    trajectories = []
    with filepath.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                trajectories.append(trajectory_from_dict(json.loads(line)))
    return trajectories


def save_papers_to_file(papers: List[Dict], output_path: str) -> None:
    """将论文列表保存为既有文本格式；文件失败不改变工具主流程。"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("=" * 80 + "\n")
            handle.write("ArXiv 计算机科学论文列表\n")
            handle.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            handle.write(f"论文数量: {len(papers)}\n")
            handle.write("=" * 80 + "\n\n")
            for i, paper in enumerate(papers, 1):
                handle.write(f"论文 {i}: {paper.get('title', '无标题')}\n")
                handle.write("-" * 60 + "\n")
                handle.write(f"ID: {paper.get('id', 'N/A')}\n")
                handle.write(f"作者: {', '.join(paper.get('authors', ['未知']))}\n")
                handle.write(f"发表时间: {paper.get('published', 'N/A')}\n")
                handle.write(f"更新时间: {paper.get('updated', 'N/A')}\n")
                handle.write(f"主要分类: {paper.get('primary_category', 'N/A')}\n")
                handle.write(f"所有分类: {', '.join(paper.get('categories', []))}\n")
                handle.write(f"PDF链接: {paper.get('pdf_url', 'N/A')}\n")
                if paper.get("comment"):
                    handle.write(f"备注: {paper.get('comment', '')}\n")
                handle.write(f"摘要: {paper.get('summary', '无摘要')}\n")
                if paper.get("links"):
                    handle.write("相关链接:\n")
                    for link in paper.get("links", []):
                        handle.write(f"  - {link}\n")
                handle.write("\n" + "=" * 80 + "\n\n")
        log.info(f"论文已保存到: {output_path}")
    except Exception as error:
        log.error(f"保存论文到文件时出错: {str(error)}")


def export_episode_artifacts(
    artifact_paths: Mapping[str, Path], destination: Path,
) -> Dict[str, str]:
    """仅交付 episode 的普通自有文件，不跟随指向共享输入的链接。"""
    exported = {}
    for name, root in artifact_paths.items():
        root = Path(root)
        for source in sorted(root.rglob("*")):
            if source.is_symlink() or not source.is_file():
                continue
            target = destination / "artifacts" / name / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            exported[str(source)] = str(target)
    return exported


def write_result(
    result: Mapping[str, Any], path: Path, *,
    artifact_exports: Mapping[str, str] | None = None,
) -> None:
    """独占交付完整结果；日期转 ISO，但不把模型观测换成评分投影。"""
    payload = copy_tool_result({**result, "artifact_exports": dict(artifact_exports or {})})
    write_json(path, payload, overwrite=False)
