"""实验输入只读解析、来源链和登记；不加载模型或执行前置阶段。"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from agentic_arxiv.llm.artifacts import _require_full_weights, artifact_lineage, artifact_metadata
from agentic_arxiv.provenance import file_fingerprint, model_input_fingerprint, read_protocol_metadata
from agentic_arxiv.runtime.react_contract import protocol_metadata
from .paths import canonical
from .store import FileStore, identifier, read_json


GROUP_STAGES = {"M1": "sft", "M2": "dpo", "M3": "grpo", "M4": "grpo", "C1": "grpo"}


def _validate_artifact_input(validator, *args, **kwargs):
    """适配旧产物校验的文本退出错误，不截获普通进程退出。"""
    try:
        return validator(*args, **kwargs)
    except SystemExit as exc:
        if not isinstance(exc.code, str):
            raise
        raise ValueError(exc.code) from exc


def _path(value, workspace=None):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("输入必须填写真实路径")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(workspace or Path.cwd()) / path
    return path.resolve(strict=True)


def _reference(reference):
    if reference is None:
        raise ValueError("输入必须填写真实路径或明确登记引用")
    if isinstance(reference, str):
        return {"kind": "path", "path": reference}
    if not isinstance(reference, Mapping):
        raise ValueError("输入引用必须是路径或明确引用对象")
    kind = reference.get("kind")
    fields = {"path": {"kind", "path"}, "registration": {"kind", "registration_id"},
              "run": {"kind", "run_id", "attempt_id", "artifact"}}
    if kind not in fields or set(reference) - {"verification_ref"} != fields[kind]:
        raise ValueError("输入引用字段不完整或含未知字段；run 必须明确 attempt_id 和 artifact")
    if "verification_ref" in reference:
        evidence = reference["verification_ref"]
        if not isinstance(evidence, Mapping) or set(evidence) != {"run_id", "attempt_id"}:
            raise ValueError("verification_ref 必须明确 run_id 和 attempt_id")
        for key in evidence:
            identifier(evidence[key])
            if evidence[key].lower() == "latest":
                raise ValueError("verification_ref 不允许 latest")
    return dict(reference)


def resolve_file_reference(reference, *, workspace=None):
    """非模型输入仅接受明确路径，禁止搜索最新数据或隐式生成。"""
    ref = _reference(reference)
    if ref["kind"] != "path":
        raise ValueError("数据、快照及策略输入必须使用明确路径")
    path = _path(ref["path"], workspace)
    if not path.is_file():
        raise ValueError(f"输入不是文件: {path}")
    return {"path": str(path), "fingerprint": file_fingerprint(path)}


def _dpo_package_dependencies(data, manifest_record):
    """只解析 load_dpo_package 会读取的文件，不递归历史来源或挂载包父目录。"""
    manifest_path = Path(manifest_record["path"])
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("files"), Mapping):
        raise ValueError("DPO 数据包缺少文件清单")
    entries = {"source_model_manifest": manifest.get("source_model_manifest"),
               "split_file": manifest.get("split_file"),
               "data": manifest["files"].get("data"),
               "traces": manifest["files"].get("traces"), "report": manifest.get("report")}
    dependencies = {}
    for name, entry in entries.items():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str) or not entry["path"]:
            raise ValueError(f"DPO 数据包缺少 {name} 文件声明")
        path = Path(entry["path"])
        path = canonical(path if path.is_absolute() else manifest_path.parent / path)
        actual = file_fingerprint(path)
        if actual["status"] != "file" or actual["sha256"] != entry.get("sha256"):
            raise ValueError(f"DPO {name} 文件缺失或哈希不一致: {path}")
        if name == "data":
            if str(path) != data["path"] or actual != data["fingerprint"]:
                raise ValueError("DPO 数据文件路径或指纹与 manifest 不一致")
        else:
            dependencies[name] = {"path": str(path), "fingerprint": actual}
    return dependencies


def _report_bound(report, path, metadata):
    if not isinstance(report, Mapping):
        return False
    relation = report.get("protocol", {})
    return (report.get("stage") == metadata["stage"].split("_", 1)[0]
            and isinstance(report.get("model_path"), str)
            and Path(report["model_path"]).expanduser().resolve() == path
            and isinstance(relation, Mapping)
            and relation.get("training") == read_protocol_metadata(metadata))


def _verification(path, metadata):
    report_path = path / "verification_report.json"
    report = read_json(report_path) if report_path.is_file() else {}
    declared = metadata.get("verification", {})
    bound = _report_bound(report, path, metadata)
    passed = bound and report.get("passed") is True and declared.get("status") not in {
        "failed", "skipped", "pending", "interrupted",
    }
    return {"status": "passed" if passed else "unknown" if not report else "not_passed",
            "report": file_fingerprint(report_path)}


def _explicit_verification(store, reference, result):
    run_id, attempt_id = reference["run_id"], reference["attempt_id"]
    run = store.read_run(run_id)
    attempt = store.read_attempt(run_id, attempt_id)
    record_path = store.attempt_dir(run_id, attempt_id) / "output" / "evaluation-result.json"
    evidence = read_json(record_path)
    report = evidence.get("report", {})
    planned = run.get("evaluation", {})
    valid = (run.get("stage") == "evaluate"
             and attempt.get("run_id") == run_id and attempt.get("attempt_id") == attempt_id
             and evidence.get("schema_version") == 1 and evidence.get("kind") == "stage"
             and evidence.get("run_id") == run_id and evidence.get("attempt_id") == attempt_id
             and evidence.get("execution") == "complete" and evidence.get("passed") is True
             and evidence.get("diagnostic", False) is False
             and evidence.get("qualification_eligible") is True
             and planned.get("diagnostic", False) is False
             and evidence.get("training_executed") is False
             and evidence.get("source", {}).get("fingerprint") == result["fingerprint"]
             and evidence.get("source", {}).get("base_model") == result["base_model"]
             and planned.get("kind") == "stage"
             and evidence.get("configuration") == planned.get("config")
             and planned.get("source", {}).get("fingerprint") == result["fingerprint"]
             and report.get("passed") is True
             and _report_bound(report, Path(result["path"]), result["metadata"]))
    if not valid:
        raise ValueError("显式阶段重验证据未完整通过或与当前模型/协议/来源不一致")
    return {"status": "passed", "report": file_fingerprint(record_path),
            "verification_ref": dict(reference), "kind": "stage"}


def resolve_model_reference(reference, *, store_root, expected_stage=None,
                            require_verified=True, workspace=None):
    """解析准确路径、独立登记或明确尝试 final；复查登记指纹，不更新旧身份。"""
    ref = _reference(reference)
    store = FileStore(store_root)
    saved = None
    if ref["kind"] == "registration":
        saved = read_json(store.root / "registrations" / identifier(ref["registration_id"]) / "reference.json")
        if saved.get("schema_version") != 1:
            raise ValueError("不支持的登记版本")
        path = _path(saved["path"])
    elif ref["kind"] == "run":
        if ref["artifact"] != "final":
            raise ValueError("模型引用仅允许明确 final；checkpoint 不自动取得最终产物资格")
        attempt = store.read_attempt(ref["run_id"], ref["attempt_id"])
        if attempt.get("training") != "budget_completed":
            raise ValueError("受管尝试尚未预算完成")
        path = _path(str(store.attempt_dir(ref["run_id"], ref["attempt_id"]) / "output" / "final"))
        saved = attempt.get("final_artifact")
        if not isinstance(saved, dict):
            raise ValueError("受管 final 缺少提交时输入身份，不能重新登记掩盖变化")
    else:
        path = _path(ref["path"], workspace)
    stages = None if expected_stage is None else {f"{expected_stage}_full", f"{expected_stage}_qlora"}
    metadata = _validate_artifact_input(
        artifact_metadata, str(path), allowed_stages=stages,
        expected_protocol=protocol_metadata(), local_base_only=True,
    )
    base = metadata.get("base_model")
    if not isinstance(base, str) or not base:
        raise ValueError("训练产物缺少明确 base_model 声明")
    base_path = _path(base, workspace)
    _validate_artifact_input(_require_full_weights, base_path)
    base_identity = model_input_fingerprint(base_path)
    if base_identity["weights_status"] != "known":
        raise ValueError("训练产物基座缺少可核对的本地权重")
    result = {"path": str(path), "metadata": metadata, "reference": ref,
              "fingerprint": model_input_fingerprint(path),
              "lineage": artifact_lineage(str(path), metadata),
              "base_model": {"path": str(base_path), "fingerprint": base_identity},
              "model_verification": _verification(path, metadata)}
    if saved:
        for key in ("fingerprint", "base_model"):
            if saved.get(key) != result[key]:
                raise ValueError(f"已登记输入发生变化: {key}；不会更新旧指纹")
        original = saved.get("model_verification", saved.get("verification"))
        if original != result["model_verification"]:
            raise ValueError("已登记输入发生变化: verification；不会更新旧指纹")
    result["verification"] = result["model_verification"]
    evidence_ref = ref.get("verification_ref") or (saved or {}).get("verification_ref")
    if evidence_ref:
        result["verification_ref"] = evidence_ref
        result["verification"] = _explicit_verification(store, evidence_ref, result)
        if saved and saved.get("verification_ref") == evidence_ref and saved.get("verification") != result["verification"]:
            raise ValueError("已登记阶段重验证据发生变化；不会更新旧指纹")
    if require_verified and result["verification"]["status"] != "passed":
        raise ValueError("模型阶段验证未通过或未知，不能自动提升为后续实验输入")
    return result


def register_artifact(store_root, group, model_path, *, artifact_id=None, metadata=None):
    """仅在管理根创建引用，未知历史成本/验证保持未知，不取得训练恢复资格。"""
    if group not in GROUP_STAGES:
        raise ValueError("登记仅支持 M1、M2、M3、M4、C1 训练产物")
    ref = {"kind": "path", "path": str(model_path)}
    if metadata and metadata.get("verification_ref"):
        ref["verification_ref"] = metadata["verification_ref"]
    resolved = resolve_model_reference(ref, store_root=store_root,
                                       expected_stage=GROUP_STAGES[group], require_verified=False)
    return FileStore(store_root).create_registration({
        **resolved, "group": group, "registration_only": True, "resume_eligible": False,
        "historical_execution": "unknown", "costs": cost_summary(metadata or {}),
    }, artifact_id=artifact_id)


def _sft_identity(resolved):
    metadata = resolved["metadata"]
    if metadata["stage"].startswith("sft_"):
        return {"model": resolved["fingerprint"],
                "manifest": file_fingerprint(Path(resolved["path"]) / "training_manifest.json"),
                "stage": metadata["stage"]}
    source = metadata.get("sft_source")
    lineage = metadata.get("source_model_lineage")
    while source is None and isinstance(lineage, Mapping):
        source = lineage.get("sft_source")
        lineage = lineage.get("upstream_lineage")
    if not isinstance(source, Mapping):
        raise ValueError("产物无法核对共同 SFT 来源链")
    return dict(source)


def cost_summary(budget):
    """未知成本不是零；偏好生成与 DPO 不因复用消失，不自动宣称严格匹配。"""
    costs = budget.get("costs", {})
    names = ("preference_generation_gpu_hours", "dpo_gpu_hours", "grpo_gpu_hours")
    result = {}
    for name in names:
        value = costs.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError(f"成本必须非负有限数或未知: {name}")
        result[name] = value
    known = [value for value in result.values() if value is not None]
    return {"accounting": "allocated_gpu_hours", "components": result,
            "known_total_gpu_hours": sum(known) if known else None,
            "total_gpu_hours": sum(result.values()) if all(v is not None for v in result.values()) else None,
            "unknown_components": [name for name, value in result.items() if value is None],
            "strict_budget_match": False, "comparison_group": budget.get("comparison_group")}


def _comparison(reference, store_root, workspace):
    ref = _reference(reference)
    if ref["kind"] == "path":
        resolved = resolve_file_reference(ref, workspace=workspace)
        document = read_json(resolved["path"])
        if not isinstance(document, Mapping) or document.get("group") != "M4":
            raise ValueError("C1 对照文件必须明确声明 group=M4 的配方或成本报告")
        return {"reference": ref, **resolved, "configuration": document}
    if ref["kind"] != "run" or ref["artifact"] != "final":
        raise ValueError("C1 对照必须是明确 M4 配方/报告路径或准确运行/尝试引用")
    store = FileStore(store_root)
    run = store.read_run(ref["run_id"])
    store.read_attempt(ref["run_id"], ref["attempt_id"])
    if run.get("group") != "M4":
        raise ValueError("C1 的运行对照不是 M4")
    config_path = store.run_dir(ref["run_id"]) / "resolved-config.json"
    return {"reference": ref, "path": str(config_path), "fingerprint": file_fingerprint(config_path),
            "configuration": read_json(config_path).get("recipe", {})}


def audit_recipe_inputs(recipe, *, store_root, workspace=None):
    """解析单项阶段输入并审计分支身份；不执行数据准备或训练。"""
    group = recipe["group"]
    inputs = recipe["inputs"]
    resolved = {}
    for name, reference in inputs.items():
        if name in {"model", "source_sft", "source_dpo", "base_model"}:
            if group in {"M0", "M1"} or name == "base_model":
                ref = _reference(reference)
                if ref["kind"] != "path":
                    raise ValueError("Base 必须使用明确本地路径")
                path = _path(ref["path"], workspace)
                _validate_artifact_input(_require_full_weights, path)
                identity = model_input_fingerprint(path)
                if identity["weights_status"] != "known":
                    raise ValueError("Base 缺少完整本地权重")
                resolved[name] = {"path": str(path), "fingerprint": identity}
            else:
                stage = "sft" if name == "source_sft" or group in {"M2", "M3", "C1"} else "dpo"
                resolved[name] = resolve_model_reference(reference, store_root=store_root,
                    expected_stage=stage, workspace=workspace)
        else:
            resolved[name] = resolve_file_reference(reference, workspace=workspace)
    # 两个别名只能选定同一个实际产物，避免预检与训练采用不同来源。
    if group == "M4" and {"model", "source_dpo"} <= resolved.keys():
        if resolved["model"]["path"] != resolved["source_dpo"]["path"]:
            raise ValueError("M4 的 model 与 source_dpo 必须指向同一 DPO 产物；请只保留选定的引用")
    if group == "M2":
        if not {"data", "data_manifest"} <= resolved.keys():
            raise ValueError("DPO 需要明确 data 和 data_manifest 输入")
        resolved["data_manifest"]["runtime_dependencies"] = _dpo_package_dependencies(
            resolved["data"], resolved["data_manifest"],
        )
    if group in {"M2", "M3", "M4", "C1"}:
        if "source_sft" not in resolved:
            raise ValueError("必须显式填写共同 source_sft 引用")
        primary = resolved.get("source_dpo") if group == "M4" else resolved.get("model")
        primary = primary or resolved.get("model")
        if primary is None:
            raise ValueError("缺少明确模型输入；M4 必须人工准备 DPO")
        if _sft_identity(primary) != _sft_identity(resolved["source_sft"]):
            raise ValueError("分支模型与共同 SFT 来源不一致")
    policies = {}
    if group in {"M3", "M4", "C1"}:
        for name in ("task_pool", "filter_policy"):
            policy = recipe.get(name) or {}
            actual = resolve_file_reference(policy.get("path"), workspace=workspace)
            if not policy.get("fingerprint") or policy["fingerprint"] != actual["fingerprint"]["sha256"]:
                raise ValueError(f"{name} 必须记录与实际文件一致的指纹；禁止隐式重筛选")
            policies[name] = actual
    comparison = None
    if group == "C1":
        budget = recipe.get("budget", {})
        if budget.get("comparison_group") != "M4" or not budget.get("comparison_reference"):
            raise ValueError("C1 必须预先明确 comparison_group=M4 和精确 comparison_reference")
        comparison = _comparison(budget["comparison_reference"], store_root, workspace)
        other = comparison["configuration"]
        other_sft = other.get("inputs", {}).get("source_sft")
        if other_sft:
            sft = resolve_model_reference(other_sft, store_root=store_root, expected_stage="sft", workspace=workspace)
            if _sft_identity(sft) != _sft_identity(resolved["source_sft"]):
                raise ValueError("C1 与明确 M4 对照的共同 SFT 来源不一致")
        for name in ("task_pool", "filter_policy"):
            declared = other.get(name, {}).get("fingerprint")
            if declared and declared != policies[name]["fingerprint"]["sha256"]:
                raise ValueError(f"C1 与明确 M4 对照的 {name} 不一致")
    return {"inputs": resolved, "policies": policies, "costs": cost_summary(recipe.get("budget", {})),
            "comparison": comparison,
            "common_sft": _sft_identity(resolved["source_sft"]) if "source_sft" in resolved else None}
