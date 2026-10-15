#!/usr/bin/env python3
"""只读核对源码配置基线与主线配置，不导入训练器或解析库默认值。

本工具的输出只支持源码/配置对照；不能证明训练、历史实验等价或真实
tokenizer 集成。配置构造中不能静态确认的表达式会列为人工审查项。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from importlib import resources
from pathlib import Path


if __package__:
    from .source_mapping import (
        DEFAULT_SOURCE_MAP, load_source_map, mapped_paths, source_index, workspace_path,
    )
else:
    from source_mapping import (
        DEFAULT_SOURCE_MAP, load_source_map, mapped_paths, source_index, workspace_path,
    )


OMITTED = (
    "loss_type", "epsilon_high", "mask_truncated_completions",
    "importance_sampling_level", "scale_rewards",
)
RETIRED = {
    "dapo", "dynamic_sampling", "dynamic_sampling_std_threshold",
    "dynamic_sampling_max_resamples",
}
ENTRY_CORRECTIONS = {
    "model": "outputs/sft_qlora/final",
    "max_turns": 6,
    "task_set": None,
}


def _literal(node: ast.AST, defaults: dict, constants: dict):
    """仅接受字面值或已知配置引用，绝不执行源文件表达式。"""
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        if node.value.id in ("SFT_DEFAULTS", "GRPO_DEFAULTS"):
            return defaults[ast.literal_eval(node.slice)]
    return ast.literal_eval(node)


def audit(
    baseline_path: Path,
    config_path: Path | None,
    *,
    workspace: Path,
    source_map: str | Path = DEFAULT_SOURCE_MAP,
) -> dict:
    workspace = workspace.resolve()
    baseline_path = workspace_path(workspace, baseline_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if config_path is None:
        resource = resources.files("agentic_arxiv").joinpath("resources/mainline.json")
        config_raw = resource.read_bytes()
        config_label = "agentic_arxiv:resources/mainline.json"
    else:
        config_path = workspace_path(workspace, config_path)
        config_raw = config_path.read_bytes()
        config_label = str(config_path)
    config = json.loads(config_raw)
    source_mapping = load_source_map(workspace, source_map)
    sources = tuple(baseline["stages"][stage]["source"] for stage in ("sft", "grpo"))
    source_before = source_index(workspace, sources, source_mapping)
    failures = []
    checks = []
    manual = []

    def check(name, passed, detail):
        record = {"check": name, "passed": passed, "detail": detail}
        checks.append(record)
        if not passed:
            failures.append(record)

    source_ref = config.get("source_baseline", {})
    check("源码版本绑定", source_ref.get("source_revision") == baseline["source_revision"],
          "比较记录的起点版本，不要求当前 HEAD 仍等于起点。")
    for stage in ("sft", "grpo"):
        stage_baseline = baseline["stages"][stage]
        defaults = config[stage]["defaults"]
        original = {row["field"]: row["default"]
                    for row in stage_baseline["function_defaults"]}
        for key, value in original.items():
            if stage == "grpo" and key in RETIRED:
                check(f"{stage}.{key} 退出", key not in defaults,
                      "只删除退出分支，不将其数值复制到保留路径。")
                continue
            expected = ENTRY_CORRECTIONS.get(key, value) if stage == "grpo" else value
            check(f"{stage}.{key} 默认值", key in defaults and defaults[key] == expected,
                  {"baseline": value, "expected": expected,
                   "actual": defaults.get(key),
                   "entry_correction": stage == "grpo" and key in ENTRY_CORRECTIONS})
        path = stage_baseline["source"]
        check(f"{stage} 起点文件哈希绑定",
              source_ref.get("source_sha256", {}).get(path)
              == baseline["sources"][path]["sha256"],
              "绑定不可变起点哈希，不将重构后的源文件要求为旧哈希。")
        candidates = []
        for current_path in mapped_paths(workspace, path, source_mapping):
            raw = current_path.read_bytes()
            tree = ast.parse(raw.decode("utf-8"))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "main":
                    candidates.append((current_path, raw, tree, node))
        check(f"{stage} 迁移后 main 唯一定位", len(candidates) == 1,
              {"original_source": path, "matches": [str(row[0]) for row in candidates]})
        if len(candidates) != 1:
            continue
        current_path, raw, tree, main = candidates[0]
        current_name = current_path.relative_to(workspace).as_posix()
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        try:
                            constants[target.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass
        main_defaults = {}
        for arg, node in zip(main.args.args[-len(main.args.defaults):], main.args.defaults):
            try:
                main_defaults[arg.arg] = _literal(node, defaults, constants)
            except (ValueError, TypeError, KeyError):
                manual.append({"field": f"{stage}.{arg.arg}", "path": current_name,
                               "line": node.lineno, "expression": ast.unparse(node),
                               "reason": "默认表达式不是工具允许的静态配置引用。"})
        for key, value in defaults.items():
            check(f"{stage}.{key} main 默认绑定",
                  key in main_defaults and main_defaults[key] == value,
                  {"configured": value, "function": main_defaults.get(key)})

        dictionary = None
        for node in ast.walk(main):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                if any(isinstance(target, ast.Name)
                       and target.id in ("config_kwargs", "cfg_kwargs")
                       for target in node.targets):
                    dictionary = node.value
                    break
        configured_values = {}
        if dictionary is not None:
            for key, value in zip(dictionary.keys, dictionary.values):
                if key is None:
                    expression = ast.unparse(value)
                    if expression == "configured['trainer_parameters']":
                        configured_values.update(config[stage]["trainer_parameters"])
                    elif expression != "_precision_flags()":
                        manual.append({"path": current_name, "line": value.lineno,
                                       "expression": expression,
                                       "reason": "需人工核对配置字典展开来源。"})
                    continue
                configured_values[ast.literal_eval(key)] = value
        check(f"{stage} 配置字典可定位", dictionary is not None,
              "只解析 AST，不构造真实训练配置。")
        for mapping in stage_baseline["config_and_manifest_mappings"]:
            if mapping["target_container"] not in ("config_kwargs", "cfg_kwargs"):
                continue
            try:
                expected = ast.literal_eval(mapping["expression"])
            except (ValueError, TypeError):
                continue
            value = configured_values.get(mapping["field"])
            try:
                actual = ast.literal_eval(value) if isinstance(value, ast.AST) else value
            except (ValueError, TypeError):
                manual.append({"field": f"{stage}.{mapping['field']}", "path": current_name,
                               "reason": "固定训练参数改为动态表达式，需人工确认来源。"})
                check(f"{stage}.{mapping['field']} 固定训练参数", False,
                      "原可静态核对字段不可因迁移降为未知；须修复静态来源定位。")
                continue
            check(f"{stage}.{mapping['field']} 固定训练参数",
                  mapping["field"] in configured_values and actual == expected,
                  {"baseline": expected, "actual": actual})
        constructor = "SFTConfig" if stage == "sft" else "GRPOConfig"
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        guard_calls = [node for node in calls if isinstance(node.func, ast.Name)
                       and node.func.id == "require_supported_config"]
        constructor_calls = [node for node in calls if isinstance(node.func, ast.Name)
                             and node.func.id == constructor]
        check(f"{stage} 显式配置接口保护",
              bool(guard_calls) and bool(constructor_calls)
              and min(node.lineno for node in guard_calls)
              < min(node.lineno for node in constructor_calls),
              "要求在配置构造前调用保护；保护函数语义还需源码审查。")
        if stage == "grpo":
            for field in OMITTED:
                expected_assignment = f"cfg_kwargs['{field}'] = {field}"
                found = any(isinstance(node, ast.If)
                            and ast.unparse(node.test) == f"{field} is not None"
                            and any(ast.unparse(statement) == expected_assignment
                                    for statement in node.body)
                            for node in ast.walk(main))
                check(f"grpo.{field} None 省略条件", found and defaults.get(field) is None,
                      "is not None 保留显式 False；None 不传给训练库，不填有效值。")
        checks.append({"check": f"{stage} 审查源码身份", "passed": True,
                       "detail": {"original_path": path, "current_path": current_name,
                                  "sha256": hashlib.sha256(raw).hexdigest(),
                                  "note": "当前实现哈希，不替换起点哈希。"}})

    check("源码在检查中未改变", source_before == source_index(
        workspace, sources, load_source_map(workspace, source_map)),
          "显式映射的当前实现与旧身份分别比较，不修改任一源码或历史指纹。")

    return {
        "schema_version": 1,
        "status": "failed" if failures else "static_checks_passed",
        "evidence_boundary": "仅源码/配置静态对照，不导入训练器、不运行快照/训练，不证明历史等价或库解析有效值。",
        "baseline": str(baseline_path),
        "configuration": config_label,
        "configuration_sha256": hashlib.sha256(config_raw).hexdigest(),
        "source_map": str(workspace_path(workspace, source_map)),
        "source_index": source_before,
        "checks": checks,
        "failures": failures,
        "manual_review": manual,
        "not_verified": ["训练库有效值", "真实 tokenizer 集成", "训练行为", "完整环境执行"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd(),
                        help="显式工作区；相对路径均相对此目录")
    parser.add_argument("--source-map", type=Path, default=DEFAULT_SOURCE_MAP,
                        help="独立的新旧源码映射，不修改旧来源指纹")
    parser.add_argument("--baseline", type=Path,
                        default=Path("artifacts/mainline_baseline/source_configuration.json"))
    parser.add_argument("--configuration", type=Path,
                        help="显式配置；省略时读取安装包内默认资源")
    args = parser.parse_args()
    result = audit(args.baseline, args.configuration, workspace=args.workspace,
                   source_map=args.source_map)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
