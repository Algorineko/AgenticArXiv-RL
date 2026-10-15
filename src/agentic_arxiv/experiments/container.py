"""只渲染单任务 Compose 资源，不拉取、构建或启动容器。"""
from dataclasses import asdict, is_dataclass
from importlib.resources import files
import json
from pathlib import Path
import re

from .paths import canonical, input_path, minimal_inputs, overlaps, validate_layout


def render_compose(profile, *, name, command, inputs, output_dir, control_dir,
                   labels, gpu_uuids=None, logs_dir=None, extra_readonly=()):
    """返回可直接序列化成 JSON 的 Compose 文档，输入保留原绝对路径。"""
    if is_dataclass(profile):
        profile = asdict(profile)
    elif hasattr(profile, "to_dict"):
        profile = profile.to_dict()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
        raise ValueError("非法容器名称")
    if not re.fullmatch(r"[^\s@]+@sha256:[a-f0-9]{64}", profile["image"]):
        raise ValueError("镜像必须固定到 sha256 摘要")
    if any(type(profile[key]) is not int or profile[key] <= 0 for key in ("uid", "gid")):
        raise ValueError("容器必须使用普通 UID/GID")
    if not isinstance(command, (tuple, list)) or not command or any(
            not isinstance(part, str) or "\x00" in part for part in command):
        raise ValueError("命令必须为非空参数数组")
    gpus = list(profile["gpus"] if gpu_uuids is None else gpu_uuids)
    if len(set(gpus)) != len(gpus) or any(not isinstance(gpu, str) or not gpu for gpu in gpus):
        raise ValueError("GPU 必须为明确且不重复的设备身份")
    readonly = minimal_inputs([
        input_path(path, profile["input_roots"], managed_root=profile["run_root"])
        for path in inputs
    ])
    validate_layout(profile["run_root"], readonly, cache_root=profile.get("cache_root"))
    readonly += [canonical(path) for path in extra_readonly]
    root = canonical(profile["run_root"], exists=False)
    writable = [canonical(path) for path in (output_dir, control_dir, logs_dir) if path]
    if any(root not in path.parents for path in writable):
        raise ValueError("输出/控制/日志必须位于管理根目录内")
    if any(overlaps(left, right) for index, left in enumerate(writable)
           for right in writable[index + 1:]):
        raise ValueError("尝试可写目录不能重叠")
    if profile.get("cache_root"):
        writable.append(canonical(profile["cache_root"]))
    if any(overlaps(left, right) for left in readonly for right in writable):
        raise ValueError("可写目录不能覆盖只读输入或配置")
    resource = files("agentic_arxiv").joinpath("resources/experiments/compose.json")
    result = json.loads(resource.read_text(encoding="utf-8"))
    service = result["services"]["experiment"]
    service.update(image=profile["image"], container_name=name,
                   user=f'{profile["uid"]}:{profile["gid"]}', command=list(command),
                   labels=dict(labels), shm_size=profile["shm_size"])
    # 镜像 ENTRYPOINT 是 agentic-arxiv；参数数组由调用方提供，不经过 shell。
    service["entrypoint"] = []
    service["volumes"] = [
        {"type": "bind", "source": str(path), "target": str(path), "read_only": ro,
         "bind": {"create_host_path": False}}
        for ro, paths in ((True, readonly), (False, writable)) for path in paths
    ]
    if profile.get("cache_root"):
        service["environment"].update(HF_HOME=str(Path(profile["cache_root"]) / "huggingface"),
                                     HOME=str(profile["cache_root"]))
    if gpus:
        service["deploy"] = {"resources": {"reservations": {"devices": [
            {"driver": "nvidia", "device_ids": gpus, "capabilities": ["gpu"]}
        ]}}}
    return result
