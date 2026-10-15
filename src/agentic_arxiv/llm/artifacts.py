"""本地训练产物的纯元数据合同，不依赖训练入口或模型运行时。"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping

from agentic_arxiv.config import resolve_model_source
from agentic_arxiv.provenance import (
    file_fingerprint, read_protocol_metadata, require_training_protocol,
    model_input_fingerprint,
)


TRAINING_STAGES = frozenset({
    "sft_qlora", "sft_full", "dpo_qlora", "dpo_full", "grpo_qlora", "grpo_full",
})
SFT_STAGES = frozenset({"sft_qlora", "sft_full"})


def _read_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{label} 无法读取: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} 必须是 JSON 对象: {path}")
    return value


def _require_full_weights(local: Path) -> None:
    if not (local / "config.json").is_file():
        raise SystemExit(f"全参模型 config.json 缺失: {local}")
    if any((local / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        return
    indexes = [local / name for name in (
        "model.safetensors.index.json", "pytorch_model.bin.index.json",
    ) if (local / name).is_file()]
    if not indexes:
        raise SystemExit(f"全参模型权重或分片索引缺失: {local}")
    for index in indexes:
        mapping = _read_object(index, "模型权重 index").get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise SystemExit(f"模型权重 index 缺少 weight_map: {index}")
        for filename in mapping.values():
            if not isinstance(filename, str) or not filename:
                raise SystemExit(f"模型权重 index 文件名无效: {index}")
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise SystemExit(f"模型权重 index 路径越界: {index}")
            if not (local / relative).is_file():
                raise SystemExit(f"模型权重分片缺失: {local / relative}")


def adapter_base_model(
    adapter_path: Path, override: str | None = None, *, local_only: bool = False,
) -> str:
    """读取明确基座；离线模式只接受已有本地目录或 Hub 缓存。"""
    config_path = adapter_path / "adapter_config.json"
    config = _read_object(config_path, "adapter_config")
    value = override or config.get("base_model_name_or_path")
    if not isinstance(value, str) or not value:
        raise SystemExit(f"adapter_config 未声明 base_model_name_or_path: {config_path}")
    try:
        source = resolve_model_source(value)
    except ValueError as exc:
        raise SystemExit(
            f"adapter 引用的基座模型不存在: {value}；"
            "换机器后可显式传 --base-model 指定同一基座的有效位置"
        ) from exc
    if local_only and not Path(source).is_dir():
        try:
            from huggingface_hub import snapshot_download

            # local_files_only 保证缓存不存在时失败，不能联网补齐权重。
            source = snapshot_download(repo_id=source, local_files_only=True)
        except (ImportError, OSError, ValueError) as exc:
            raise SystemExit(f"基座模型没有可用的本地缓存: {source}: {exc}") from exc
    if local_only and override:
        declared = adapter_base_model(adapter_path, local_only=True)
        if Path(source).resolve() != Path(declared).resolve():
            def identity(path):
                _require_full_weights(Path(path))
                fingerprint = model_input_fingerprint(path)
                return {
                    str(Path(item["path"]).relative_to(path)): item["sha256"]
                    for item in fingerprint["files"]
                    if Path(item["path"]).name == "config.json"
                    or item["path"].endswith((".safetensors", ".bin", ".index.json"))
                }
            if identity(source) != identity(declared):
                raise SystemExit("显式基座与 adapter 声明基座的本地内容不一致")
    return source


def artifact_metadata(
    model_path: str,
    *,
    base_model: str | None = None,
    allowed_stages: Iterable[str] | None = None,
    expected_protocol: Mapping | str | None = None,
    local_base_only: bool = False,
) -> dict:
    """校验明确全参/adapter 产物；协议由调用方提供，绝不补写。"""
    local = Path(resolve_model_source(model_path, local_only=True))
    manifest_path = local / "training_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"训练模型缺少必要 training_manifest: {manifest_path}")
    metadata = _read_object(manifest_path, "training_manifest")
    stage = metadata.get("stage")
    stages = TRAINING_STAGES if allowed_stages is None else frozenset(allowed_stages)
    if not isinstance(stage, str) or stage not in TRAINING_STAGES or stage not in stages:
        raise SystemExit(f"不支持的训练产物 stage={stage!r}: {manifest_path}")
    if stage.startswith("dpo_"):
        sft = metadata.get("sft_source")
        reference = metadata.get("reference_source")
        if not isinstance(sft, dict) or sft.get("stage") not in SFT_STAGES:
            raise SystemExit("DPO training_manifest 缺少明确的 SFT 来源")
        if not isinstance(reference, dict) or reference != sft:
            raise SystemExit("DPO training_manifest 的冻结参考来源与 SFT 起点不一致")
    is_adapter = (local / "adapter_config.json").is_file()
    if is_adapter != stage.endswith("_qlora"):
        raise SystemExit("训练 manifest 的全参/适配器类型与产物文件不一致")
    if is_adapter:
        if not metadata.get("base_model"):
            raise SystemExit("适配器 training_manifest 缺少 base_model")
        if not any((local / name).is_file() for name in (
            "adapter_model.safetensors", "adapter_model.bin",
        )):
            raise SystemExit(f"适配器权重缺失: {local}")
        config = _read_object(local / "adapter_config.json", "adapter_config")
        if config.get("base_model_name_or_path") != metadata["base_model"]:
            raise SystemExit("适配器基座声明与 training_manifest 的 base_model 不一致")
        source = adapter_base_model(local, base_model, local_only=local_base_only)
        if local_base_only:
            _require_full_weights(Path(source))
    else:
        _require_full_weights(local)
    if expected_protocol is not None:
        require_training_protocol(
            metadata, expected_protocol, source_label="训练输入模型",
            require_chat_template_hash=True,
        )
    return metadata


def artifact_lineage(model_path: str, metadata: Mapping) -> dict:
    """保留实际输入阶段和其声明的上游链，不要求上游目录仍在本机。"""
    return {
        "source_stage": metadata["stage"],
        "source_manifest": file_fingerprint(Path(model_path) / "training_manifest.json"),
        "source_model": str(model_path),
        "sft_source": metadata.get("sft_source"),
        "reference_source": metadata.get("reference_source"),
        "upstream_lineage": metadata.get("source_model_lineage"),
    }


def require_artifact_tokenizer(metadata: Mapping, tokenizer) -> None:
    """只核对实际模板；协议版本由上层传入，不在本地层推断。"""
    getter = getattr(tokenizer, "get_chat_template", None)
    try:
        template = getter() if callable(getter) else getattr(tokenizer, "chat_template", None)
    except (TypeError, ValueError) as exc:
        raise ValueError("产物 tokenizer 无法确定实际 chat template") from exc
    if not isinstance(template, str) or not template:
        raise ValueError("产物 tokenizer 缺少实际 chat template")
    actual = sha256(template.encode("utf-8")).hexdigest()
    declared = read_protocol_metadata(metadata)["chat_template_hash"]
    if actual != declared:
        raise ValueError("产物 tokenizer chat template 与 training_manifest 不一致")
