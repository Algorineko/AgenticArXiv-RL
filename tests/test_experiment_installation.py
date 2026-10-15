"""安装资源和轻量管理导入集成检查；只构造控制元数据，不加载研究输入。"""

from importlib import resources
import json
import os
from pathlib import Path
import subprocess
import sys


def test_installed_experiment_resources_are_complete_and_unconfigured():
    root = resources.files("agentic_arxiv").joinpath("resources", "experiments")
    groups = {"M0", "M1", "M2", "M3", "M4", "C1"}
    recipes = {name: json.loads(root.joinpath("recipes", name + ".json").read_text())
               for name in groups}
    assert {recipe["group"] for recipe in recipes.values()} == groups
    assert recipes["M0"]["stage"] == "evaluate"
    assert recipes["M0"]["checkpoint"] is None
    assert all(recipe["budget"]["max_steps"] is None for recipe in recipes.values())
    assert json.loads(root.joinpath("profiles", "server.json").read_text())["image"] is None
    compose = json.loads(root.joinpath("compose.json").read_text())
    assert len(compose["services"]) == 1


def test_management_module_inventory_is_importable_without_training_or_env(tmp_path):
    # 在仓库外的真实安装环境导入全部管理模块，而不是仅测试根包。
    (tmp_path / ".env").write_text("AGENTIC_EXPERIMENT_SENTINEL=must-not-load\n")
    code = """
import importlib, pkgutil, sys, os
import agentic_arxiv.experiments as package
for module in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
    importlib.import_module(module.name)
heavy = {'torch','transformers','trl','accelerate','datasets','peft','bitsandbytes'}
assert not heavy.intersection(sys.modules), heavy.intersection(sys.modules)
assert os.environ.get('AGENTIC_EXPERIMENT_SENTINEL') is None
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("AGENTIC_EXPERIMENT_SENTINEL", None)
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert {path.name for path in tmp_path.iterdir()} == {".env"}


def test_unconfigured_installed_profile_fails_before_runtime_or_creation(tmp_path):
    root = resources.files("agentic_arxiv").joinpath("resources", "experiments")
    with resources.as_file(root.joinpath("profiles", "server.json")) as profile:
        command = [sys.executable, "-m", "agentic_arxiv", "experiment", "list",
                   "--profile", str(profile), "--json"]
        result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["completion_scope"] == "management_operation_only"
    assert not list(tmp_path.iterdir())
