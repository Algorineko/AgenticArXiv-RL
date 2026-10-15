"""Docker/NVIDIA 参数数组边界，可注入执行器进行无容器工程测试。"""
from __future__ import annotations

import csv
import io
import json
import re
import subprocess


class RuntimeUnavailable(RuntimeError):
    exit_code = 4

    def __init__(self, message, *, uncertain=True):
        super().__init__(message)
        self.uncertain = uncertain


class IdentityMismatch(RuntimeError):
    exit_code = 2


class DockerRuntime:
    def __init__(self, runner=None):
        self.runner = runner or subprocess.run

    def call(self, arguments, *, allow_missing=False, input_text=None):
        try:
            kwargs = dict(capture_output=True, text=True, timeout=30, check=False)
            if input_text is not None:
                kwargs["input"] = input_text
                kwargs["timeout"] = 300
            result = self.runner(list(arguments), **kwargs)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeUnavailable(str(exc)) from exc
        if result.returncode:
            if allow_missing and ("No such object:" in result.stderr or "No such container:" in result.stderr):
                return None
            raise RuntimeUnavailable(result.stderr.strip() or f"命令失败: {result.returncode}", uncertain=False)
        return result.stdout

    def inspect(self, name):
        raw = self.call(["docker", "container", "inspect", name], allow_missing=True)
        if raw is None:
            return None
        try:
            records = json.loads(raw)
            if not isinstance(records, list) or len(records) != 1:
                raise ValueError("容器查询必须返回唯一对象")
            if not isinstance(records[0], dict) or any(not isinstance(records[0].get(key), dict) for key in ("Config", "HostConfig", "State")):
                raise ValueError("容器查询字段不完整")
            return records[0]
        except (TypeError, ValueError) as exc:
            raise RuntimeUnavailable("容器查询响应无效") from exc

    def verify(self, container, attempt, profile):
        if container is None:
            raise IdentityMismatch("目标容器不存在")
        labels = container.get("Config", {}).get("Labels") or {}
        expected = {"org.agentic-arxiv.managed": "1", "org.agentic-arxiv.run": attempt["run_id"],
                    "org.agentic-arxiv.attempt": attempt["attempt_id"],
                    "org.agentic-arxiv.owner": f"{profile['uid']}:{profile['gid']}"}
        if any(labels.get(key) != value for key, value in expected.items()):
            raise IdentityMismatch("容器标签归属不匹配")
        if container.get("Name", "").lstrip("/") != attempt["container_name"]:
            raise IdentityMismatch("容器确定性名称不匹配")
        if container.get("Config", {}).get("User") != expected["org.agentic-arxiv.owner"]:
            raise IdentityMismatch("容器执行用户不匹配")
        if container.get("Config", {}).get("Image") != profile["image"]:
            raise IdentityMismatch("容器镜像身份不匹配")
        if attempt.get("container_id") and container.get("Id") != attempt["container_id"]:
            raise IdentityMismatch("容器ID不匹配")
        requests = container.get("HostConfig", {}).get("DeviceRequests") or []
        devices = [device for request in requests for device in (request.get("DeviceIDs") or [])]
        if devices != attempt.get("gpu_uuids", []):
            raise IdentityMismatch("容器GPU映射不匹配")
        visible = [item.removeprefix("CUDA_VISIBLE_DEVICES=") for item in (container.get("Config", {}).get("Env") or []) if item.startswith("CUDA_VISIBLE_DEVICES=")]
        if attempt.get("gpu_uuids") and visible != [",".join(attempt["gpu_uuids"])]:
            raise IdentityMismatch("容器CUDA UUID/rank顺序不匹配")
        return container

    def image(self, reference):
        raw = self.call(["docker", "image", "inspect", reference])
        try:
            image = json.loads(raw)[0]
            if not isinstance(image, dict):
                raise ValueError("镜像查询必须返回对象")
            if reference not in image.get("RepoDigests", []):
                raise ValueError("本地镜像摘要与配置不匹配")
            labels = image.get("Config", {}).get("Labels") or {}
            if not re.fullmatch(r"[0-9a-f]{64}", labels.get("io.agentic-arxiv.code-identity", "")) or labels.get("io.agentic-arxiv.recovery-protocol") != "1":
                raise ValueError("本地镜像缺少固定代码身份或恢复协议标签")
            return image
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeUnavailable("固定镜像不可核验") from exc

    def devices(self):
        raw = self.call(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
        rows = list(csv.reader(io.StringIO(raw)))
        devices = {}
        for row in rows:
            if len(row) != 2 or not row[1].strip().startswith("GPU-"):
                raise RuntimeUnavailable("GPU身份查询响应无效")
            index, uuid = (item.strip() for item in row)
            devices[index] = uuid
            devices[uuid] = uuid
        if not devices:
            raise RuntimeUnavailable("没有可核验GPU")
        return devices

    def resolve_gpus(self, selected, allowed):
        devices = self.devices()
        if isinstance(selected, str):
            selected = selected.split(",")
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("必须显式选择不重复的GPU")
        try:
            result = [devices[str(item)] for item in selected]
            permitted = {devices[str(item)] for item in allowed}
        except KeyError as exc:
            raise ValueError(f"未知GPU: {exc.args[0]}") from exc
        if len(set(result)) != len(result) or not set(result) <= permitted:
            raise ValueError("GPU别名重复或不在允许范围")
        return result

    def check_idle(self, uuids):
        raw = self.call(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
        for row in csv.reader(io.StringIO(raw)):
            if not row:
                continue
            if len(row) != 2 or not row[0].strip().startswith("GPU-") or not row[1].strip().isdigit():
                raise RuntimeUnavailable("GPU进程状态未知")
            if row[0].strip() in uuids:
                raise ValueError(f"GPU存在计算进程: {row[0].strip()} pid={row[1].strip()}")

    def create_arguments(self, profile, attempt, command, mounts):
        labels = {"org.agentic-arxiv.managed": "1", "org.agentic-arxiv.run": attempt["run_id"],
                  "org.agentic-arxiv.attempt": attempt["attempt_id"],
                  "org.agentic-arxiv.owner": f"{profile['uid']}:{profile['gid']}"}
        result = ["docker", "create", "--name", attempt["container_name"], "--pull", "never",
                  "--user", labels["org.agentic-arxiv.owner"], "--restart", "no",
                  "--read-only", "--network", "none", "--entrypoint", "python", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                  "--shm-size", str(profile["shm_size"]), "--tmpfs", "/tmp:rw,nosuid,nodev"]
        if attempt.get("gpu_uuids"):
            result += ["--gpus", '"device=' + ",".join(attempt["gpu_uuids"]) + '"']
            result += ["--env", "CUDA_VISIBLE_DEVICES=" + ",".join(attempt["gpu_uuids"])]
        for key, value in labels.items():
            result += ["--label", f"{key}={value}"]
        for mount in mounts:
            value = f"type=bind,src={mount['source']},dst={mount['target']}"
            if mount["readonly"]:
                value += ",readonly"
            result += ["--mount", value]
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            result += ["--env", f"{key}=1"]
        if profile.get("cache_root"):
            result += ["--env", f"HF_HOME={profile['cache_root']}/huggingface", "--env", f"HOME={profile['cache_root']}"]
        if command[:1] != ["python"]:
            raise ValueError("受管容器命令必须使用固定Python入口")
        return result + [profile["image"], *command[1:]]


Docker = DockerRuntime
