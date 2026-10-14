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
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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


def audit(baseline_path: Path, config_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
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
        raw = (ROOT / path).read_bytes()
        tree = ast.parse(raw.decode("utf-8"))
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
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
                manual.append({"field": f"{stage}.{arg.arg}", "path": path,
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
                        manual.append({"path": path, "line": value.lineno,
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
                manual.append({"field": f"{stage}.{mapping['field']}", "path": path,
                               "reason": "固定训练参数改为动态表达式，需人工确认来源。"})
                continue
            check(f"{stage}.{mapping['field']} 固定训练参数", actual == expected,
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
                       "detail": {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}})

    return {
        "schema_version": 1,
        "status": "failed" if failures else "static_checks_passed",
        "evidence_boundary": "仅源码/配置静态对照，不导入训练器、不运行快照/训练，不证明历史等价或库解析有效值。",
        "baseline": str(baseline_path.relative_to(ROOT)),
        "configuration": str(config_path.relative_to(ROOT)),
        "checks": checks,
        "failures": failures,
        "manual_review": manual,
        "not_verified": ["训练库有效值", "真实 tokenizer 集成", "训练行为", "完整环境执行"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=ROOT / "artifacts/mainline_baseline/source_configuration.json")
    parser.add_argument("--configuration", type=Path,
                        default=ROOT / "configs/training/mainline.json")
    args = parser.parse_args()
    result = audit(args.baseline.resolve(), args.configuration.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
