"""纯文件元数据/源码检查，不加载模型、不新增响应或轨迹。"""

import ast
import json
from pathlib import Path

import pytest

from agentic_arxiv.llm.artifacts import (
    SFT_STAGES, adapter_base_model, artifact_lineage, artifact_metadata,
)
from agentic_arxiv.runtime.react_contract import protocol_metadata


@pytest.fixture
def artifact(tmp_path):
    # 仅复用既有权重存在性检查的临时文件输入，不含模型参数。
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").touch()
    protocol = {**protocol_metadata(), "chat_template_hash": "declared-template"}
    metadata = {"stage": "sft_full", "protocol_metadata": protocol}
    (model / "training_manifest.json").write_text(json.dumps(metadata))
    return model, metadata


@pytest.mark.parametrize("stage", ["sft_full", "grpo_full", "dpo_full"])
def test_declared_stages_and_source_lineage(artifact, stage):
    model, metadata = artifact
    metadata["stage"] = stage
    if stage == "dpo_full":
        metadata["sft_source"] = {"stage": "sft_full", "model": {"source": "SFT"}}
        metadata["reference_source"] = dict(metadata["sft_source"])
    (model / "training_manifest.json").write_text(json.dumps(metadata))
    result = artifact_metadata(str(model), expected_protocol=protocol_metadata())
    assert result == metadata
    lineage = artifact_lineage(str(model), result)
    assert lineage["source_stage"] == stage
    assert lineage["source_manifest"]["status"] == "file"
    assert lineage["sft_source"] == metadata.get("sft_source")


def test_missing_manifest_weights_and_unknown_stage_rejected(artifact):
    model, metadata = artifact
    manifest = model / "training_manifest.json"
    manifest.unlink()
    with pytest.raises(SystemExit, match="training_manifest"):
        artifact_metadata(str(model))
    metadata["stage"] = "legacy_dpo"
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(SystemExit, match="stage"):
        artifact_metadata(str(model))
    metadata["stage"] = "sft_full"
    manifest.write_text(json.dumps(metadata))
    (model / "model.safetensors").unlink()
    with pytest.raises(SystemExit, match="权重"):
        artifact_metadata(str(model))


def test_unknown_protocol_and_non_sft_start_rejected(artifact):
    model, metadata = artifact
    metadata.pop("protocol_metadata")
    (model / "training_manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="协议元数据"):
        artifact_metadata(str(model), expected_protocol=protocol_metadata())
    metadata["stage"] = "grpo_full"
    (model / "training_manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(SystemExit, match="stage"):
        artifact_metadata(str(model), allowed_stages=SFT_STAGES)


@pytest.mark.parametrize("filename", ["missing.safetensors", "../outside.safetensors"])
def test_index_does_not_hide_missing_or_unsafe_shards(artifact, filename):
    model, _ = artifact
    (model / "model.safetensors").unlink()
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"weight": filename},
    }))
    with pytest.raises(SystemExit, match="分片缺失|路径越界"):
        artifact_metadata(str(model))


def test_adapter_base_declaration_and_strict_override(artifact, tmp_path):
    base, metadata = artifact
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").touch()
    config = {"base_model_name_or_path": str(base)}
    (adapter / "adapter_config.json").write_text(json.dumps(config))
    metadata.update(stage="sft_qlora", base_model=str(base))
    (adapter / "training_manifest.json").write_text(json.dumps(metadata))
    assert artifact_metadata(str(adapter), local_base_only=True) == metadata
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text('{"model_type":"different"}')
    (other / "model.safetensors").touch()
    with pytest.raises(SystemExit, match="本地内容不一致"):
        adapter_base_model(adapter, str(other), local_only=True)
    # GRPO 既有显式 relocation 接口没有被 DPO 的严格策略改写。
    assert adapter_base_model(adapter, str(other)) == str(other)
    metadata["base_model"] = str(other)
    (adapter / "training_manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(SystemExit, match="基座声明"):
        artifact_metadata(str(adapter))


def test_reference_loader_is_explicit_and_frozen_in_source():
    path = Path(__file__).parents[1] / "src/agentic_arxiv/training/model_setup.py"
    tree = ast.parse(path.read_text())
    reference = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "load_reference_policy")
    source = ast.unparse(reference)
    assert "AutoModelForCausalLM.from_pretrained" in source
    assert "PeftModel.from_pretrained" in source
    assert "merge_and_unload" in source
    assert "parameter.requires_grad = False" in source
    assert "reference.eval()" in source
    assert "get_peft_model" not in source
    assert "disable_adapter" not in source


def test_strict_local_tokenization_does_not_duplicate_template_special_tokens():
    path = Path(__file__).parents[1] / "src/agentic_arxiv/llm/local.py"
    tree = ast.parse(path.read_text())
    client = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "TransformersLLMClient")
    methods = {node.name: node for node in client.body if isinstance(node, ast.FunctionDef)}
    assert "self._artifact_mode = artifact" in ast.unparse(methods["__init__"])
    assert "client._artifact_mode = artifact" in ast.unparse(methods["from_loaded"])
    source = ast.unparse(methods["chat_completions"])
    assert "{'add_special_tokens': False} if getattr(self, '_artifact_mode', False) else {}" in source
    assert "self.tokenizer(prompt, return_tensors='pt', **tokenization_kwargs)" in source


def test_loaded_client_keeps_explicit_artifact_mode_without_loading_or_generating():
    from agentic_arxiv.llm.local import TransformersLLMClient

    model, tokenizer = object(), object()
    default = TransformersLLMClient.from_loaded(model, tokenizer)
    strict = TransformersLLMClient.from_loaded(model, tokenizer, artifact=True)
    assert default._artifact_mode is False
    assert strict._artifact_mode is True
    assert strict.model is model and strict.tokenizer is tokenizer
