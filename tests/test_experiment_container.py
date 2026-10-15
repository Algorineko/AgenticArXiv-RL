"""容器渲染/构建上下文离线工程检查，不构建镜像或启动任务。"""
import importlib.util
import hashlib
import json
from pathlib import Path
import re
import subprocess

import pytest

from agentic_arxiv.experiments.container import render_compose


ROOT = Path(__file__).resolve().parents[1]


def profile_and_paths(tmp_path):
    inputs = tmp_path / "inputs"
    model = inputs / "model"
    model.mkdir(parents=True)
    run_root = tmp_path / "runs"
    attempt = run_root / "runs/r1/attempts/a1"
    paths = [attempt / child for child in ("output", "control", "logs")]
    cache = tmp_path / "cache"
    for path in [*paths, cache]:
        path.mkdir(parents=True)
    profile = {"schema_version": 1, "workspace": str(tmp_path), "run_root": str(run_root),
               "input_roots": [str(inputs)], "cache_root": str(cache),
               "image": "example/training@sha256:" + "a" * 64,
               "uid": 1000, "gid": 1000, "gpus": ["GPU-a", "GPU-b"], "shm_size": "8gb"}
    return profile, model, paths


def render(tmp_path, **overrides):
    profile, model, paths = profile_and_paths(tmp_path)
    params = {"name": "test-r1-a1", "command": ["agentic-arxiv", "train", "sft"],
              "inputs": [model], "output_dir": paths[0], "control_dir": paths[1],
              "logs_dir": paths[2], "labels": {"run_id": "r1"}}
    params.update(overrides)
    return render_compose(profile, **params)


def test_compose_limits_to_explicit_resources(tmp_path):
    service = render(tmp_path)["services"]["experiment"]
    assert service["user"] == "1000:1000"
    assert service["pull_policy"] == "never"
    assert service["restart"] == "no"
    assert service["read_only"] is True
    assert service["network_mode"] == "none"
    assert service["cap_drop"] == ["ALL"]
    assert service["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == ["GPU-a", "GPU-b"]
    assert not set(service) & {"build", "ports", "privileged", "ipc", "pid"}
    assert service["volumes"][0]["read_only"] is True
    assert len(service["volumes"]) == 5
    assert all("docker.sock" not in item["source"] for item in service["volumes"])
    assert service["command"] == ["agentic-arxiv", "train", "sft"]


def test_cpu_only_render_has_no_gpu_reservation(tmp_path):
    assert "deploy" not in render(tmp_path, gpu_uuids=[])["services"]["experiment"]


def test_readonly_config_inside_attempt_is_allowed(tmp_path):
    profile, model, paths = profile_and_paths(tmp_path)
    config = paths[0].parent / "resolved-config.json"
    config.write_text("{}")
    result = render_compose(profile, name="test", command=["true"], inputs=[model],
                            output_dir=paths[0], control_dir=paths[1], labels={},
                            extra_readonly=[config])
    assert any(item["source"] == str(config) and item["read_only"]
               for item in result["services"]["experiment"]["volumes"])


def test_managed_history_final_is_mounted_readonly(tmp_path):
    profile, _, paths = profile_and_paths(tmp_path)
    previous_final = Path(profile["run_root"]) / "runs/previous/attempts/a1/output/final"
    previous_final.mkdir(parents=True)
    result = render_compose(profile, name="test", command=["python", "-m", "agentic_arxiv"],
                            inputs=[previous_final], output_dir=paths[0], control_dir=paths[1],
                            logs_dir=paths[2], labels={})
    service = result["services"]["experiment"]
    assert service["entrypoint"] == []
    assert service["command"] == ["python", "-m", "agentic_arxiv"]
    mount = next(item for item in service["volumes"] if item["source"] == str(previous_final))
    assert mount["read_only"] is True


@pytest.mark.parametrize("relative", ["output", "control", "output/checkpoint-50"])
def test_other_managed_history_paths_are_not_inputs(tmp_path, relative):
    profile, _, paths = profile_and_paths(tmp_path)
    previous = Path(profile["run_root"]) / "runs/previous/attempts/a1" / relative
    previous.mkdir(parents=True)
    with pytest.raises(ValueError, match="输入必须位于允许根目录"):
        render_compose(profile, name="test", command=["true"], inputs=[previous],
                       output_dir=paths[0], control_dir=paths[1], labels={})


def test_managed_final_cannot_overlap_current_attempt_writable(tmp_path):
    profile, _, paths = profile_and_paths(tmp_path)
    current_final = paths[0] / "final"
    current_final.mkdir()
    with pytest.raises(ValueError, match="可写目录不能覆盖"):
        render_compose(profile, name="test", command=["true"], inputs=[current_final],
                       output_dir=paths[0], control_dir=paths[1], labels={})


@pytest.mark.parametrize("change", ["root", "tag", "shell", "overlap", "unknown_gpu"])
def test_renderer_rejects_unsafe_values(tmp_path, change):
    profile, model, paths = profile_and_paths(tmp_path)
    if change == "root":
        profile["uid"] = 0
    elif change == "tag":
        profile["image"] = "example/training:latest"
    params = dict(name="test", command=["true"], inputs=[model], output_dir=paths[0],
                  control_dir=paths[1], labels={})
    if change == "shell":
        params["command"] = "true; echo unsafe"
    elif change == "overlap":
        params["control_dir"] = paths[0]
    elif change == "unknown_gpu":
        params["gpu_uuids"] = ["GPU-a", "GPU-a"]
    with pytest.raises(ValueError):
        render_compose(profile, **params)


def test_build_context_excludes_assets_and_identity_is_truthful(tmp_path):
    spec = importlib.util.spec_from_file_location("prepare_build", ROOT / "scripts/prepare_container_build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "context"
    manifest = module.prepare(ROOT, target)
    assert len(manifest["tree_sha256"]) == 64
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True)
    assert manifest["dirty"] is bool(status.strip())
    names = set(manifest["files"])
    assert not any(name.startswith(("data/", "artifacts/", "output/", ".git/", ".venv/")) for name in names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
    assert json.loads((target / "build-identity.json").read_text()) == manifest
    with pytest.raises(ValueError):
        module.prepare(ROOT, target)


def test_dependency_lock_has_complete_hashes_sources_and_baseline():
    value = (ROOT / "containers/requirements-linux-py310-cu126.lock").read_text()
    evidence = json.loads((ROOT / "containers/lock-resolution.json").read_text())
    assert evidence["lock_sha256"] == hashlib.sha256(value.encode()).hexdigest()
    packages = re.findall(r"^([\w.-]+)==([^\s\\]+)", value, re.MULTILINE)
    assert len(packages) == evidence["package_count"]
    assert dict(packages)["torch"] == "2.10.0+cu126"
    assert dict(packages)["trl"] == "0.29.1"
    assert "nvidia-cuda-runtime-cu12" in dict(packages)
    for block in re.split(r"\n(?=[\w.-]+==)", value)[1:]:
        assert "--hash=sha256:" in block
        assert "# from https://" in block


def test_dockerfile_fixed_base_and_normal_wheel_install():
    value = (ROOT / "containers/Dockerfile").read_text()
    base = json.loads((ROOT / "containers/base-image.json").read_text())
    assert base["reference"].removeprefix("docker.io/library/") in value
    assert "--require-hashes" in value
    assert "python -m build --no-isolation --wheel" in value
    assert "--no-deps /opt/agentic-build/wheels/*.whl" in value
    assert "pip check" in value
    assert "site-packages" in value
    assert "latest" not in value and "-e ." not in value
