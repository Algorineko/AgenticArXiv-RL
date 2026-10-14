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
    Trajectory, copy_tool_result, trajectory_from_dict, trajectory_to_dict,
)


PROTOCOL_FIELDS = (
    "runtime_version", "context_version", "training_protocol",
    "prompt_template_hash", "chat_template_hash", "template_hash",
)
_REQUIRED_PROTOCOL_FIELDS = PROTOCOL_FIELDS[:4]
_PACKAGES = ("torch", "transformers", "trl", "accelerate", "datasets", "peft", "bitsandbytes")
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
