"""对实际源码检查职责边界，不导入业务模块或执行研究流程。"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "agentic_arxiv"
NAMESPACE = "agentic_arxiv"
FOUNDATIONS = {"contracts", "config", "provenance"}
SHARED = FOUNDATIONS | {"state", "tools", "llm", "runtime", "tasks", "scoring"}
ALLOWED = {
    "contracts": set(),
    "config": {"contracts"},
    "provenance": {"contracts"},
    "state": FOUNDATIONS,
    "tools": FOUNDATIONS | {"state"},
    "llm": FOUNDATIONS | {"state"},
    "runtime": FOUNDATIONS | {"state", "tools", "llm"},
    "tasks": FOUNDATIONS,
    "scoring": {"contracts", "provenance"},
    "evaluation": SHARED,
    "data_pipeline": SHARED,
    "training": SHARED | {"evaluation"},
    "cli": SHARED | {"training", "evaluation", "data_pipeline"},
}
OLD_NAMESPACES = {"AgenticArxiv", "agents", "benchmark", "rl", "models", "utils"}
# 两个训练入口不再作为另一阶段的共享实现。
TRAINING_ENTRYPOINTS = {"agentic_arxiv.training.sft", "agentic_arxiv.training.grpo"}


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(PACKAGE).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([NAMESPACE, *parts])


def _owner(module: str) -> str:
    parts = module.split(".")
    if module == "agentic_arxiv.__main__":
        return "__root__"
    return parts[1] if len(parts) > 1 else "__root__"


def _is_reference(module: str) -> bool:
    return module == "agentic_arxiv.tasks.reference" or module.startswith(
        "agentic_arxiv.tasks.reference."
    )


def _relative_import(node: ast.ImportFrom, module: str, path: Path) -> str:
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    if node.level:
        return importlib.util.resolve_name("." * node.level + (node.module or ""), package)
    return node.module or ""


def _static_imports(tree: ast.AST, module: str, path: Path):
    # ast.walk 同时覆盖函数内导入、条件导入和 __init__ 重导出。
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            base = _relative_import(node, module, path)
            yield node.lineno, base
            if base == NAMESPACE or base.startswith(NAMESPACE + ".") or node.module is None:
                for alias in node.names:
                    if alias.name != "*":
                        yield node.lineno, f"{base}.{alias.name}"


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _aliases(tree: ast.AST) -> dict[str, str]:
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _resolved_name(node: ast.AST, aliases: dict[str, str]) -> str:
    name = _dotted(node)
    head, _, tail = name.partition(".")
    return aliases.get(head, head) + (f".{tail}" if tail else "")


def _tool_module_inventory(tree: ast.AST) -> tuple[str, ...]:
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "TOOL_MODULES" for target in targets):
                inventory = ast.literal_eval(node.value)
                assert isinstance(inventory, dict), "工具动态注册清单必须是静态字典"
                assert all(isinstance(key, str) for key in inventory)
                return tuple(inventory)
    raise AssertionError("工具 bootstrap 必须声明完整 TOOL_MODULES 清单")


def _dynamic_imports(tree: ast.AST, module: str):
    aliases = _aliases(tree)
    constants = {}
    for declaration in ast.iter_child_nodes(tree):
        if isinstance(declaration, (ast.Assign, ast.AnnAssign)):
            targets = declaration.targets if isinstance(declaration, ast.Assign) else [declaration.target]
            if isinstance(declaration.value, ast.Constant) and isinstance(declaration.value.value, str):
                for target in targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = declaration.value.value
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = _resolved_name(node.func, aliases)
        if function not in {"importlib.import_module", "__import__"}:
            continue
        assert node.args, f"{module}:{node.lineno}: 无法检查空动态导入"
        target = node.args[0]
        if isinstance(target, ast.Name) and target.id in constants:
            target = ast.Constant(value=constants[target.id])
        if isinstance(target, ast.Constant) and isinstance(target.value, str):
            name = target.value
            if name.startswith("."):
                # 包内动态相对导入也必须给出明确的字面量 package。
                package = node.args[1] if len(node.args) > 1 else next(
                    (item.value for item in node.keywords if item.arg == "package"), None
                )
                assert isinstance(package, ast.Constant) and isinstance(package.value, str), (
                    f"{module}:{node.lineno}: 动态相对导入缺少静态 package"
                )
                name = importlib.util.resolve_name(name, package.value)
            yield node.lineno, name
        elif module == "agentic_arxiv.tools.bootstrap" and isinstance(target, ast.Name) and target.id == "module":
            for name in _tool_module_inventory(tree):
                yield node.lineno, name
        else:
            pytest.fail(f"{module}:{node.lineno}: 未声明的动态导入 {ast.unparse(target)}")


def _check_dependency(source: str, target: str, location: str):
    head = target.split(".")[0]
    assert head not in OLD_NAMESPACES, f"{location}: 活动旧命名空间 {target}"
    # 旧裸 tools/config/state 不能作为已安装包的隐式别名。
    assert head not in ALLOWED, f"{location}: 裸项目命名空间 {target}"
    if head != NAMESPACE or target == NAMESPACE:
        return
    source_owner, target_owner = _owner(source), _owner(target)
    if source_owner == "__root__":
        assert target_owner == "cli" and source == "agentic_arxiv.__main__", (
            f"{location}: 根包必须轻量，不能重导出 {target}"
        )
        return
    assert source_owner in ALLOWED, f"{location}: 未声明的模块职责 {source_owner}"
    if source_owner == target_owner:
        if source_owner == "tasks" and _is_reference(target):
            assert _is_reference(source), f"{location}: 纯任务层不能重导出或加载 reference 执行层"
        if source_owner == "training" and any(target == entry or target.startswith(entry + ".") for entry in TRAINING_ENTRYPOINTS):
            assert any(source == entry and target.startswith(entry) for entry in TRAINING_ENTRYPOINTS), (
                f"{location}: 训练入口不能成为共享辅助模块 {target}"
            )
        return
    allowed = ALLOWED[source_owner]
    if source_owner == "tasks" and _is_reference(source):
        allowed = allowed | {"runtime"}
    assert target_owner in allowed, f"{location}: {source_owner} 不允许依赖 {target}"
    if source_owner == "training" and target_owner == "evaluation":
        assert any(target == service or target.startswith(service + ".") for service in (
            "agentic_arxiv.evaluation.episode", "agentic_arxiv.evaluation.service",
        )), (
            f"{location}: training 只能使用模型无关公共 episode 评测服务，不能导入 {target}"
        )


def test_real_source_dependency_allowlist():
    paths = sorted(PACKAGE.rglob("*.py"))
    assert paths, "缺少 src/agentic_arxiv 实际源码"
    for path in paths:
        module = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, target in [*_static_imports(tree, module, path), *_dynamic_imports(tree, module)]:
            _check_dependency(module, target, f"{path.relative_to(ROOT)}:{line}")


def test_no_source_path_injection_or_namespace_aliases():
    paths = [
        *PACKAGE.rglob("*.py"),
        *ROOT.joinpath("scripts").rglob("*.py"),
        *ROOT.joinpath("tests").rglob("*.py"),
    ]
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases = _aliases(tree)
        for node in ast.walk(tree):
            name = _resolved_name(node, aliases)
            assert not (name == "sys.path" or name.startswith("sys.path.")), (
                f"{path.relative_to(ROOT)}:{node.lineno}: 禁止源码搜索路径注入"
            )
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Subscript) and _resolved_name(target.value, aliases) == "sys.modules":
                        assert not (
                            isinstance(target.slice, ast.Constant)
                            and isinstance(target.slice.value, str)
                            and target.slice.value.split(".")[0] in OLD_NAMESPACES | set(ALLOWED)
                        ), f"{path.relative_to(ROOT)}:{node.lineno}: 禁止旧命名空间别名"


def test_scoring_is_read_only_computation():
    # 不允许通过标准库绕过评分层对文件、进程、网络和工具执行的限制。
    forbidden_imports = {"os", "io", "pathlib", "shutil", "subprocess", "socket", "requests", "importlib"}
    forbidden_calls = {
        "open", "eval", "exec", "__import__", "read_text", "read_bytes", "write_text", "write_bytes",
        "load_trajectory", "save_trajectory", "save_jsonl", "read_jsonl", "execute", "execute_action", "step", "reset",
    }
    paths = sorted(PACKAGE.joinpath("scoring").rglob("*.py"))
    pure_provenance = {"sanitize_metadata", "canonical_hash"}
    assert paths, "缺少 scoring 实际源码"
    for path in paths:
        module = _module_name(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases = _aliases(tree)
        for line, target in _static_imports(tree, module, path):
            assert target.split(".")[0] not in forbidden_imports, f"{path.relative_to(ROOT)}:{line}: 评分导入 I/O 层 {target}"
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "agentic_arxiv.provenance":
                assert all(alias.name in pure_provenance for alias in node.names), (
                    f"{path.relative_to(ROOT)}:{node.lineno}: 评分只能引用已提供数据的纯来源函数"
                )
            if isinstance(node, ast.Call):
                name = _resolved_name(node.func, aliases)
                if name.startswith("agentic_arxiv.provenance."):
                    assert name.rpartition(".")[2] in pure_provenance, (
                        f"{path.relative_to(ROOT)}:{node.lineno}: 评分不能调用来源文件函数 {name}"
                    )
                assert name not in {"compile", "builtins.compile"}, (
                    f"{path.relative_to(ROOT)}:{node.lineno}: 评分不能动态编译执行代码"
                )
                assert name.rpartition(".")[2] not in forbidden_calls, (
                    f"{path.relative_to(ROOT)}:{node.lineno}: 评分不能调用 I/O 或执行函数 {name}"
                )


def test_tool_dynamic_inventory_uses_only_owned_modules():
    path = PACKAGE / "tools" / "bootstrap.py"
    inventory = _tool_module_inventory(ast.parse(path.read_text(encoding="utf-8")))
    assert inventory
    for module in inventory:
        assert module.startswith("agentic_arxiv.tools."), module
        assert (PACKAGE / Path(*module.split(".")[1:])).with_suffix(".py").is_file(), module


def test_root_pytest_configuration_excludes_archive(pytestconfig, request):
    assert pytestconfig.rootpath == ROOT
    assert pytestconfig.getini("testpaths") == ["tests"]
    assert not pytestconfig.getini("pythonpath"), "测试也必须使用实际安装，不能补源码搜索路径"
    assert "docs" in pytestconfig.getini("norecursedirs")
    assert "artifacts" in pytestconfig.getini("norecursedirs")
    for item in request.session.items:
        path = Path(str(item.path)).resolve()
        assert path.is_relative_to(ROOT / "tests"), f"活动发现越界: {path}"
        assert path.name != "test_mainline_behavior_baseline.py", "非活动基线被收集"
