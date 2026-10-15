"""显式单作业生命周期，未知状态保守保留资源，不自动重试或排队。"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil

from .docker import DockerRuntime, RuntimeUnavailable
from .paths import canonical, input_path, validate_layout, mounts
from .store import FileStore, atomic_json, now, parse_record_time, read_json


class ResourceConflict(ValueError):
    exit_code = 3


def _input_paths(audit):
    # 历史来源与对照配置仅供审计，不是容器运行依赖。
    paths = set()
    for section in ("inputs", "policies"):
        for record in audit.get(section, {}).values():
            paths.add(record["path"])
            if record.get("base_model"):
                paths.add(record["base_model"]["path"])
            for dependency in record.get("runtime_dependencies", {}).values():
                paths.add(dependency["path"])
    return sorted(paths)


def _confirmed_start_failure(attempt, container):
    failure = attempt.get("start_failure")
    state = container.get("State", {})
    return (isinstance(failure, dict) and failure.get("container_id") == container.get("Id")
            and state.get("Status") == "created" and state.get("Running") is False)


class ExperimentService:
    def __init__(self, profile, workspace=None, runner=None):
        self.profile = profile
        self.workspace = Path(workspace or profile["workspace"])
        self.store = FileStore(profile["run_root"])
        self.runtime = DockerRuntime(runner)

    def handle(self, operation, args, resolved=None):
        operation = operation.rsplit(".", 1)[-1]
        if operation in {"check", "start"}:
            return getattr(self, operation)(resolved, gpus=getattr(args, "gpus", None), name=getattr(args, "name", None))
        if operation == "list":
            return {"runs": self.list_runs()}
        if operation == "status":
            return self.status(args.run_id)
        if operation == "logs":
            return {"logs": self.logs(args.run_id, rank=getattr(args, "rank", None), tail=getattr(args, "tail", 100))}
        if operation == "stop":
            return self.stop(args.run_id, force=getattr(args, "force", False))
        if operation == "resume":
            return self.resume(args.run_id, checkpoint=args.checkpoint, gpus=getattr(args, "gpus", None))
        if operation == "register":
            from .artifacts import register_artifact
            metadata = read_json(args.metadata) if getattr(args, "metadata", None) else None
            return register_artifact(self.store.root, args.group, args.model, metadata=metadata,
                                     artifact_id=getattr(args, "artifact_id", None))
        if operation == "evaluate":
            return self.evaluate(args)
        raise ValueError(f"未知管理操作: {operation}")

    def _validate_host_paths(self, paths):
        if self.profile["uid"] != os.getuid() or self.profile["gid"] not in {os.getgid(), *os.getgroups()}:
            raise ValueError("管理命令必须由profile指定的普通用户及其所属组执行；不自动改所有权")
        root = validate_layout(self.profile["run_root"], paths, cache_root=self.profile.get("cache_root"))
        parent = root
        while not parent.exists():
            parent = parent.parent
        if not os.access(parent, os.W_OK | os.X_OK):
            raise ValueError("管理输出根目录不可写")
        minimum = self.profile.get("min_free_bytes", 0)
        if shutil.disk_usage(parent).free < minimum:
            raise ValueError("输出目录剩余空间不足")
        if self.profile.get("cache_root"):
            cache = canonical(self.profile["cache_root"])
            if not cache.is_dir() or not os.access(cache, os.W_OK | os.X_OK):
                raise ValueError("显式缓存目录不存在或不可写")

    def check(self, resolved, *, gpus=None, name=None):
        from .artifacts import audit_recipe_inputs
        if not resolved:
            raise ValueError("必须先解析服务器 profile 和配方")
        recipe = resolved["recipe"]
        for reference in recipe["inputs"].values():
            raw = reference if isinstance(reference, str) else reference.get("path") if isinstance(reference, dict) else None
            if raw:
                raw_path = Path(raw).expanduser()
                canonical(raw_path if raw_path.is_absolute() else self.workspace / raw_path)
        audit = audit_recipe_inputs(recipe, store_root=self.store.root, workspace=self.workspace)
        paths = [input_path(path, self.profile["input_roots"], managed_root=self.store.root) for path in _input_paths(audit)]
        self._validate_host_paths(paths)
        image = self.runtime.image(self.profile["image"])
        devices = []
        if recipe["stage"] != "evaluate":
            training = resolved["training"]
            total = recipe.get("budget", {}).get("max_steps")
            epochs = recipe.get("budget", {}).get("epochs")
            steps_match = type(total) is int and total > 0 and epochs is None and training["defaults"].get("max_steps") == total
            epochs_match = total is None and type(epochs) is int and epochs > 0 and training["defaults"].get("max_steps") == -1 and training["defaults"].get("epochs") == epochs
            if not steps_match and not epochs_match:
                raise ValueError("受管预算必须明确选择max_steps或epochs，并与原训练参数一致")
            checkpoint = recipe.get("checkpoint", {})
            if type(checkpoint.get("save_steps")) is not int or checkpoint["save_steps"] < 1:
                raise ValueError("受管训练必须显式启用周期完整 checkpoint")
        devices = self.runtime.resolve_gpus(gpus, self.profile["gpus"])
        self.runtime.check_idle(devices)
        if recipe["stage"] != "evaluate":
            training = self._render_training(resolved, audit)
            # 执行基座已按产物合同归一化；不要让预检重新应用独立输入覆盖。
            # 完整输入审计和只读挂载仍保留该基座记录。
            preflight_inputs = {key: value for key, value in audit["inputs"].items() if key != "base_model"}
            preflight = ["docker", "run", "--rm", "--interactive", "--pull", "never", "--network", "none",
                         "--user", f"{self.profile['uid']}:{self.profile['gid']}", "--read-only", "--cap-drop", "ALL",
                         "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,nosuid,nodev",
                         "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1", "--env", "HF_DATASETS_OFFLINE=1",
                         "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "CUDA_VISIBLE_DEVICES="]
            for path in paths:
                preflight += ["--mount", f"type=bind,src={path},dst={path},readonly"]
            preflight += ["--entrypoint", "python", self.profile["image"], "-m", "agentic_arxiv.experiments.preflight"]
            report = json.loads(self.runtime.call(preflight, input_text=json.dumps({"stage": recipe["stage"], "training": training,
                "inputs": preflight_inputs, "workspace": "/tmp"})))
            if not isinstance(report, dict) or report.get("schema_version") != 1 or report.get("status") != "passed" or report.get("stage") != recipe["stage"]:
                raise ValueError("只读容器预检未通过")
        else:
            planned = self._prepare_m0(resolved, {"audit": audit}, persist=False)
            extra_paths = [input_path(path, self.profile["input_roots"], managed_root=self.store.root) for path in planned["inputs"]]
            self._validate_host_paths(extra_paths)
        return {"checked": True, "resolved": resolved, "audit": audit,
                "input_paths": [str(path) for path in paths], "gpu_uuids": devices,
                "requested_gpus": gpus,
                "image_id": image.get("Id"), "image_metadata": image, "training_success": False,
                "resource_guarantee": "仅本管理根内防争用，不是跨用户独占保证"}

    def _claims(self):
        path = self.store.root / "gpu-claims.json"
        return read_json(path) if path.exists() else {}

    def _claim(self, attempt, *, locked=False):
        if not locked:
            with self.store.lock():
                return self._claim(attempt, locked=True)
        claims = self._claims()
        for gpu in attempt["gpu_uuids"]:
            if gpu in claims:
                owner = claims[gpu]
                raise ResourceConflict(f"GPU已有持久活动声明: {gpu} {owner['run_id']}/{owner['attempt_id']}；请先核对原尝试")
        for gpu in attempt["gpu_uuids"]:
            claims[gpu] = {"run_id": attempt["run_id"], "attempt_id": attempt["attempt_id"]}
        atomic_json(self.store.root / "gpu-claims.json", claims)

    def _release(self, attempt):
        with self.store.lock():
            claims = self._claims()
            for gpu in attempt["gpu_uuids"]:
                if claims.get(gpu) == {"run_id": attempt["run_id"], "attempt_id": attempt["attempt_id"]}:
                    del claims[gpu]
            atomic_json(self.store.root / "gpu-claims.json", claims)

    def start(self, resolved, *, gpus=None, name=None):
        checked = self.check(resolved, gpus=gpus, name=name)
        if resolved["recipe"]["stage"] == "evaluate":
            return self._start_m0(resolved, checked)
        run = self.store.create_run({"group": resolved["recipe"]["group"], "stage": resolved["recipe"]["stage"],
                                     "name": name, "image": self.profile["image"],
                                     "total_budget": resolved["recipe"]["budget"]["max_steps"],
                                     "epoch_budget": resolved["recipe"]["budget"].get("epochs")},
                                    resolved, checked["audit"])
        return self._launch_training(run, checked)

    def _render_training(self, resolved, audit):
        training = copy.deepcopy(resolved["training"])
        defaults = training["defaults"]
        inputs = audit["inputs"]
        for key in ("model", "base_model", "data", "data_manifest", "snapshot"):
            if key in inputs and key in defaults:
                defaults[key] = inputs[key]["path"]
        source = "source_dpo" if resolved["recipe"]["group"] == "M4" else "source_sft"
        if ("model" not in inputs or source == "source_dpo") and source in inputs:
            defaults["model"] = inputs[source]["path"]
        policy = next((value for value in inputs.values() if value.get("path") == defaults.get("model")), None)
        if policy is None:
            raise ValueError("训练 model 必须明确绑定已审计输入")
        if "base_model" in defaults:
            # 已审计的产物阶段决定加载合同；本次 qlora 开关只决定资源路径。
            # 全参产物的基座仍保留在审计身份/挂载中，但不能作为 adapter 覆盖传入。
            source_stage = policy.get("metadata", {}).get("stage", "")
            if source_stage.endswith("_qlora") and defaults.get("base_model"):
                from agentic_arxiv.llm.artifacts import adapter_base_model
                from .artifacts import _validate_artifact_input

                candidate = Path(defaults["base_model"]).expanduser()
                candidate = candidate if candidate.is_absolute() else self.workspace / candidate
                _validate_artifact_input(adapter_base_model, Path(policy["path"]),
                                         override=str(candidate), local_only=True)
            defaults["base_model"] = (policy["base_model"]["path"]
                                      if source_stage.endswith("_qlora") else None)
        if "split" in inputs and resolved["recipe"]["stage"] == "grpo":
            split = defaults.get("split")
            if not isinstance(split, str) or ":" not in split:
                raise ValueError("GRPO split 必须在 defaults.split 显式填写 FILE:train 或 FILE:rl_train")
            filename, name = split.rsplit(":", 1)
            candidate = Path(filename)
            candidate = candidate if candidate.is_absolute() else self.workspace / candidate
            if candidate.resolve() != Path(inputs["split"]["path"]) or name not in {"train", "rl_train"}:
                raise ValueError("GRPO切分必须与明确输入文件一致并选train/rl_train")
            defaults["split"] = f"{inputs['split']['path']}:{name}"
        elif resolved["recipe"]["stage"] == "grpo":
            raise ValueError("GRPO必须显式登记训练split文件，不使用内置任务池")
        from agentic_arxiv.training.mainline_config import _validate_section, stage_config
        _validate_section(resolved["recipe"]["stage"], training, stage_config(resolved["recipe"]["stage"]))
        return training

    def _launch_training(self, run, checked, source_checkpoint=None):
        resolved = copy.deepcopy(checked["resolved"])
        inputs = checked["audit"]["inputs"]
        training = self._render_training(resolved, checked["audit"])
        defaults = training["defaults"]
        policy = next((value for value in inputs.values() if value.get("path") == defaults.get("model")), None)
        if not policy:
            raise ValueError("未能将训练模型参数绑定到已审计输入")
        base = policy.get("base_model")
        research = {key: copy.deepcopy(training[key]) for key in ("defaults", "trainer_parameters")}
        for key in ("output_dir", "run_name", "rollout_trace_path", "resume_from_checkpoint"):
            research["defaults"].pop(key, None)
        identity = {"inputs": checked["audit"], "training_config": research,
                    "runtime": {"image": self.profile["image"], "image_id": checked["image_id"]},
                    "models": {"policy": policy, "reference": policy if run["stage"] in {"dpo", "grpo"} else None,
                               "base": base}, "devices": checked["gpu_uuids"]}
        source_step = 0
        resolved_total = run["total_budget"]
        if source_checkpoint:
            from agentic_arxiv.training.checkpoint import validate_checkpoint
            expected = {"stage": run["stage"], "run_id": run["run_id"], "total_budget": run["total_budget"], "identity": identity}
            if run.get("epoch_budget") is not None:
                expected["epoch_budget"] = run["epoch_budget"]
            manifest = validate_checkpoint(source_checkpoint, expected=expected)
            source_step = manifest["global_step"]
            resolved_total = manifest["total_budget"]
            if manifest["attempt_id"] != Path(source_checkpoint).parent.parent.name:
                raise ValueError("恢复清单attempt身份与历史目录不一致")
        with self.store.lock():
            # 声明与尝试创建在同一短锁内；竞争者不会留下第二个活动尝试。
            for gpu in checked["gpu_uuids"]:
                if gpu in self._claims():
                    raise ResourceConflict(f"GPU已有活动声明: {gpu}")
            attempt = self.store.create_attempt(run["run_id"], {"gpu_uuids": checked["gpu_uuids"],
                "requested_gpus": checked.get("requested_gpus", checked["gpu_uuids"]), "source_checkpoint_step": source_step,
                "source_checkpoint": str(source_checkpoint) if source_checkpoint else None}, locked=True)
            attempt["container_name"] = f"arxiv-{run['run_id']}-{attempt['attempt_id']}"
            self.store.update_attempt(run["run_id"], attempt["attempt_id"], {"container_name": attempt["container_name"]}, locked=True)
            self._claim(attempt, locked=True)
        try:
            command, extra = self._prepare_training_job(run, checked, attempt, training, identity, resolved_total, source_checkpoint)
        except (OSError, ValueError, RuntimeError) as exc:
            return self._planning_failed(run, attempt, exc)
        return self.launch_job(run, attempt, command, checked["input_paths"], extra_readonly=extra)

    def _planning_failed(self, run, attempt, error):
        try:
            self.store.update_attempt(run["run_id"], attempt["attempt_id"], {"execution": "create_failed", "training": "failed", "error": str(error)})
            self.store.event(run["run_id"], attempt["attempt_id"], "planning_failed", error=str(error))
        finally:
            self._release(attempt)
        return {"started": False, "management_exit_code": 2, "execution": "create_failed", "run_id": run["run_id"], "attempt_id": attempt["attempt_id"], "error": str(error)}

    def _prepare_training_job(self, run, checked, attempt, training, identity, resolved_total, source_checkpoint):
        defaults = training["defaults"]
        directory = self.store.attempt_dir(run["run_id"], attempt["attempt_id"])
        validate_layout(self.store.root, checked["input_paths"], cache_root=self.profile.get("cache_root"), attempt_dir=directory)
        atomic_json(directory / "image-metadata.json", checked.get("image_metadata", {}))
        context = {"schema_version": 1, "stage": run["stage"], "run_id": run["run_id"],
                   "attempt_id": attempt["attempt_id"], "total_budget": resolved_total, "identity": identity,
                   "output_dir": str(directory / "output"), "control_dir": str(directory / "control"), "log_dir": str(directory / "logs")}
        if run.get("epoch_budget") is not None:
            context["epoch_budget"] = run["epoch_budget"]
        atomic_json(directory / "control" / "managed-context.json", context)
        defaults["output_dir"] = str(directory / "output")
        if run["stage"] == "grpo":
            defaults["rollout_trace_path"] = str(directory / "output" / "rollouts.jsonl") if defaults.get("save_rollout_traces") else None
        from agentic_arxiv.config import read_default_mainline_config
        config = json.loads(read_default_mainline_config())
        config[run["stage"]] = training
        atomic_json(directory / "training-config.json", config)
        args = ["--config", str(directory / "training-config.json"), "--output-dir", str(directory / "output"),
                "--managed-context", str(directory / "control" / "managed-context.json")]
        if run["stage"] == "sft" and checked["audit"]["inputs"].get("snapshot"):
            args += ["--snapshot", checked["audit"]["inputs"]["snapshot"]["path"]]
        if source_checkpoint:
            args += ["--resume-from-checkpoint", str(source_checkpoint)]
        count = len(attempt["gpu_uuids"])
        if count == 1:
            command = ["python", "-m", "agentic_arxiv", "train", run["stage"], *args]
        else:
            command = ["python", "-m", "agentic_arxiv.training.launch", "--num-processes", str(count),
                       "--gpus", ",".join(attempt["gpu_uuids"]), "--log-dir", str(directory / "logs"), run["stage"], *args]
        extra = [directory / "training-config.json"]
        if source_checkpoint:
            extra.append(Path(source_checkpoint))
        return command, extra

    def launch_job(self, run, attempt, command, inputs, *, extra_readonly=()):
        directory = self.store.attempt_dir(run["run_id"], attempt["attempt_id"])
        try:
            from .container import render_compose
            labels = {"org.agentic-arxiv.managed": "1", "org.agentic-arxiv.run": attempt["run_id"],
                      "org.agentic-arxiv.attempt": attempt["attempt_id"],
                      "org.agentic-arxiv.owner": f"{self.profile['uid']}:{self.profile['gid']}"}
            rendered = render_compose(self.profile, name=attempt["container_name"], command=command,
                inputs=inputs, output_dir=directory / "output", control_dir=directory / "control",
                logs_dir=directory / "logs", labels=labels, gpu_uuids=attempt["gpu_uuids"], extra_readonly=extra_readonly)
            atomic_json(directory / "compose.json", rendered)
            bindings = mounts(inputs, directory, self.profile.get("cache_root"), extra_readonly)
            arguments = self.runtime.create_arguments(self.profile, attempt, command, bindings)
            atomic_json(directory / "docker-command.json", arguments)
        except (OSError, ValueError, RuntimeError) as exc:
            return self._planning_failed(run, attempt, exc)
        self.store.update_attempt(run["run_id"], attempt["attempt_id"], {"execution": "starting"})
        self.store.event(run["run_id"], attempt["attempt_id"], "create_intent", command=arguments)
        phase = "create"
        try:
            # 第二次复查防止预检后已出现的外部占用；不自动换卡。
            self.runtime.check_idle(attempt["gpu_uuids"]) if attempt["gpu_uuids"] else None
            result = self.runtime.call(arguments)
            attempt = self.store.update_attempt(run["run_id"], attempt["attempt_id"], {"container_id": result.strip()})
            container = self.runtime.verify(self.runtime.inspect(attempt["container_name"]), attempt, self.profile)
            phase = "start"
            self.runtime.call(["docker", "start", container["Id"]])
            phase = "inspect_started"
            container = self.runtime.verify(self.runtime.inspect(attempt["container_name"]), attempt, self.profile)
        except (RuntimeError, ValueError) as exc:
            if phase == "start" and isinstance(exc, RuntimeUnavailable) and not exc.uncertain:
                # 明确的 start 拒绝与超时分开存证；仍须核验容器未启动才释放。
                attempt = self.store.update_attempt(run["run_id"], attempt["attempt_id"], {
                    "start_failure": {"container_id": attempt["container_id"], "at": now(), "error": str(exc)}})
            self.store.event(run["run_id"], attempt["attempt_id"], "launch_uncertain", error=str(exc))
            try:
                container = self.runtime.inspect(attempt["container_name"])
                if container is None:
                    if getattr(exc, "uncertain", False):
                        raise RuntimeUnavailable("客户端结果丢失，暂时不存在不能证明创建未发生")
                    self.store.update_attempt(run["run_id"], attempt["attempt_id"], {"execution": "create_failed", "training": "failed", "error": str(exc)})
                    self._release(attempt)
                    return {"started": False, "management_exit_code": 4, "run_id": run["run_id"], "attempt_id": attempt["attempt_id"], "execution": "create_failed", "error": str(exc)}
                self.runtime.verify(container, attempt, self.profile)
                self.store.update_attempt(run["run_id"], attempt["attempt_id"], {"container_id": container["Id"]})
            except (RuntimeUnavailable, ValueError, RuntimeError):
                container = None
            if container is not None and _confirmed_start_failure(attempt, container):
                self.store.update_attempt(run["run_id"], attempt["attempt_id"], {
                    "execution": "create_failed", "training": "failed", "error": str(exc)})
                self.store.event(run["run_id"], attempt["attempt_id"], "start_failed", error=str(exc))
                self._release(attempt)
                return {"started": False, "management_exit_code": 4, "execution": "create_failed",
                        "run_id": run["run_id"], "attempt_id": attempt["attempt_id"], "error": str(exc)}
            if not container or not container.get("State", {}).get("Running"):
                self.store.update_attempt(run["run_id"], attempt["attempt_id"], {"execution": "unknown", "error": str(exc)})
                return {"started": False, "management_exit_code": 5, "execution": "unknown", "run_id": run["run_id"], "attempt_id": attempt["attempt_id"], "claims_retained": True}
        started_at = container.get("State", {}).get("StartedAt")
        self.store.update_attempt(run["run_id"], attempt["attempt_id"], {"execution": "running", "training": "not_started" if attempt.get("evaluation_only") else "in_progress", "started_at": started_at if parse_record_time(started_at) is not None else None})
        return {"started": True, "run_id": run["run_id"], "attempt_id": attempt["attempt_id"], "training_success": False}

    def list_runs(self):
        return [{**run, "status": self.status(run["run_id"])} for run in self.store.list_runs()]

    def status(self, run_id):
        run = self.store.read_run(run_id)
        if not run.get("current_attempt"):
            return {"run_id": run_id, "execution": "preparing", "training": "not_started", "validation": "not_run"}
        attempt = self.store.read_attempt(run_id)
        directory = self.store.attempt_dir(run_id, attempt["attempt_id"])
        result = dict(attempt)
        for key in ("started_at", "finished_at"):
            if key in result and parse_record_time(result[key]) is None:
                result[key] = None
        if attempt.get("execution") == "create_failed" and not attempt.get("container_id"):
            return result
        try:
            container = self.runtime.verify(self.runtime.inspect(attempt["container_name"]), attempt, self.profile)
            state = container["State"]
            if parse_record_time(state.get("StartedAt")) is not None:
                result["started_at"] = state["StartedAt"]
            for key in ("uncertainty", "claims_retained", "management_exit_code"):
                result.pop(key, None)
            if state.get("Running"):
                result["execution"] = "stopping" if (directory / "control" / "stop-request.json").exists() else "running"
            elif state.get("Status") in {"exited", "dead"}:
                result.update(execution="exited", exit_code=state.get("ExitCode"))
                if parse_record_time(state.get("FinishedAt")) is not None:
                    result["finished_at"] = state["FinishedAt"]
                self._release(attempt)
            elif _confirmed_start_failure(attempt, container):
                result.update(execution="create_failed", training="failed", error=attempt["start_failure"]["error"])
                self._release(attempt)
            else:
                result["execution"] = "starting" if state.get("Status") == "created" else "unknown"
        except (RuntimeUnavailable, RuntimeError, KeyError, TypeError) as exc:
            result.update(execution="unknown", uncertainty=str(exc), claims_retained=True, management_exit_code=5)
        evidence = directory / "control" / "training-result.json"
        if evidence.is_file():
            record = read_json(evidence)
            if record.get("run_id") == run_id and record.get("attempt_id") == attempt["attempt_id"]:
                if record.get("training") in {"budget_completed", "protection_stopped", "failed", "user_safe_stop"}:
                    result["training"] = record["training"]
                if record.get("validation") in {"not_run", "in_progress", "passed", "failed", "skipped", "interrupted"}:
                    result["validation"] = record["validation"]
                for key in ("global_step", "reason", "checkpoint", "planned_steps", "total_budget", "training_elapsed_seconds", "stage_validation_elapsed_seconds", "checkpoint_step", "generated_tokens", "peak_memory_bytes"):
                    if key in record:
                        result[key] = record[key]
        if attempt.get("evaluation_only"):
            result["training"] = "not_started"
            evaluation_path = directory / "output" / "evaluation-result.json"
            if evaluation_path.is_file():
                evaluation = read_json(evaluation_path)
                if evaluation.get("run_id") == run_id and evaluation.get("attempt_id") == attempt["attempt_id"]:
                    result["evaluation"] = evaluation
                    if run.get("evaluation", {}).get("kind") == "stage":
                        result["validation"] = ("passed" if evaluation.get("passed") is True else "failed") if evaluation.get("execution") == "complete" else evaluation.get("execution", "interrupted")
                    else:
                        result["validation"] = "not_run"
                    evaluation["gpu_uuids"] = attempt["gpu_uuids"]
                    evaluation_record = run.get("evaluation", {})
                    evaluation_id = evaluation_record.get("evaluation_id") or attempt.get("evaluation_id")
                    if evaluation_id:
                        atomic_json(self.store.run_dir(run_id) / "evaluations" / evaluation_id / "evaluation.json", evaluation)
        stop_status = directory / "control" / "stop-status.json"
        if stop_status.is_file():
            stopped = read_json(stop_status)
            if stopped.get("run_id") == run_id and stopped.get("attempt_id") == attempt["attempt_id"]:
                if stopped.get("status") == "failed":
                    result["training"] = "failed"
                elif stopped.get("status") == "user_safe_stop" and result["execution"] == "exited" and result["training"] not in {"failed", "protection_stopped", "budget_completed"}:
                    from agentic_arxiv.training.checkpoint import validate_checkpoint
                    try:
                        from agentic_arxiv.training.checkpoint import ManagedCheckpointContext
                        validate_checkpoint(stopped["checkpoint"], expected=ManagedCheckpointContext.from_file(directory / "control" / "managed-context.json"))
                        result["training"] = "user_safe_stop"
                    except (ValueError, OSError, KeyError) as exc:
                        result.update(training="failed", reason=f"停止断点核验失败: {exc}")
        force_requested = attempt.get("force_requested_at") or attempt.get("forced_at")
        if force_requested and not attempt.get("evaluation_only") and result["execution"] == "exited" and result["training"] not in {"budget_completed", "protection_stopped", "failed"}:
            result["training"] = "forced_interrupted"
        elif not attempt.get("evaluation_only") and result.get("execution") == "exited" and result.get("exit_code", 0) != 0 and result["training"] not in {"budget_completed", "protection_stopped", "failed"}:
            result["training"] = "failed"
        if result.get("execution") == "exited" and result.get("exit_code", 0) != 0 and result["validation"] == "in_progress":
            result["validation"] = "interrupted"
        if force_requested and result["execution"] == "exited":
            result["force_uncertain"] = False
        if attempt.get("evaluation_only") and result.get("evaluation", {}).get("execution") != "complete" and result.get("execution") == "exited" and (force_requested or result.get("exit_code", 0) != 0):
            result["training"] = "not_started"
            result["evaluation"] = dict(result.get("evaluation", {}), execution="interrupted" if force_requested else "failed", passed=False)
            if run.get("evaluation", {}).get("kind") == "stage":
                result["validation"] = "interrupted" if force_requested else "failed"
        if result["execution"] != "exited" and result["training"] == "user_safe_stop":
            result["training"] = "in_progress"
        total = run.get("total_budget")
        resolved_budget_path = directory / "control" / "resolved-budget.json"
        if resolved_budget_path.is_file():
            resolved_budget = read_json(resolved_budget_path)
            if resolved_budget.get("run_id") == run_id and resolved_budget.get("attempt_id") == attempt["attempt_id"] and resolved_budget.get("epoch_budget") == run.get("epoch_budget"):
                actual_total = resolved_budget.get("total_budget")
                if type(actual_total) is int and actual_total > 0:
                    total = actual_total
                    result["resolved_total_budget"] = actual_total
        if total is None and type(result.get("planned_steps")) is int and result["planned_steps"] > 0:
            total = result["planned_steps"]
        if result["training"] == "budget_completed" and (type(total) is not int or result.get("global_step", -1) < total):
            result.update(training="failed", reason="完成证据缺少核定总预算或未达到原定优化步预算")
        # 进程仍可写验证报告/manifest；只有退出后才冻结可复用身份。
        if result["training"] == "budget_completed" and result["execution"] == "exited" and result["validation"] != "in_progress" and not result.get("final_artifact"):
            from .artifacts import resolve_model_reference
            try:
                result["final_artifact"] = resolve_model_reference(str(directory / "output" / "final"), store_root=self.store.root, expected_stage=run["stage"], require_verified=False)
            except (ValueError, OSError) as exc:
                result.update(validation="failed", reason=str(exc))
        changes = {key: value for key, value in result.items() if key not in {"updated_at"}}
        self.store.update_attempt(run_id, attempt["attempt_id"], changes)
        from .evaluation import attempt_observability, summarize_costs
        result["observability"] = attempt_observability(directory)
        attempts = [read_json(path) for path in (self.store.run_dir(run_id) / "attempts").glob("*/attempt.json")]
        evaluations = [read_json(path) for path in (self.store.run_dir(run_id) / "evaluations").glob("*/evaluation.json")]
        inputs = read_json(self.store.run_dir(run_id) / "inputs.json")
        result["costs"] = summarize_costs(attempts, evaluations, inputs.get("costs"))
        atomic_json(directory / "status.json", result)
        return result

    def logs(self, run_id, *, rank=None, tail=100):
        if type(tail) is not int or tail < 1:
            raise ValueError("日志 tail 必须为正整数")
        attempt = self.store.read_attempt(run_id)
        if rank is not None:
            if type(rank) is not int or not 0 <= rank < len(attempt["gpu_uuids"]):
                raise ValueError("rank不在当前尝试范围")
            directory = self.store.attempt_dir(run_id, attempt["attempt_id"]) / "logs"
            files = sorted(directory.glob(f"*/attempt_*/{rank}/stdout.log")) + sorted(directory.glob(f"rank-{rank}.log"))
            if not files:
                return "指定rank日志尚不可用"
            return "\n".join("\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:]) for path in files)
        container = self.runtime.verify(self.runtime.inspect(attempt["container_name"]), attempt, self.profile)
        return self.runtime.call(["docker", "logs", "--tail", str(tail), container["Id"]])

    def stop(self, run_id, *, force=False):
        attempt = self.store.read_attempt(run_id)
        container = self.runtime.verify(self.runtime.inspect(attempt["container_name"]), attempt, self.profile)
        if not container.get("State", {}).get("Running"):
            raise ValueError("目标尝试未运行，不提交停止操作")
        if force:
            self.store.event(run_id, attempt["attempt_id"], "force_intent", container_id=container["Id"])
            self.store.update_attempt(run_id, attempt["attempt_id"], {"force_requested_at": now(), "force_target_id": container["Id"], "execution": "stopping"})
            try:
                self.runtime.call(["docker", "kill", "--signal", "KILL", container["Id"]])
            except RuntimeUnavailable as exc:
                self.store.update_attempt(run_id, attempt["attempt_id"], {"force_uncertain": True, "force_error": str(exc)})
                self.store.event(run_id, attempt["attempt_id"], "force_uncertain", error=str(exc))
                return {"force_requested": True, "management_exit_code": 5, "force_uncertain": True,
                        "run_id": run_id, "attempt_id": attempt["attempt_id"], "claims_retained": True}
            self.store.update_attempt(run_id, attempt["attempt_id"], {"forced_at": now(), "force_uncertain": False})
            return {"force_requested": True, "checkpoint_guarantee": "仅此前完整提交的断点可恢复", "status": self.status(run_id)}
        from agentic_arxiv.training.control import request_stop
        request = request_stop(self.store.attempt_dir(run_id, attempt["attempt_id"]) / "control", run_id, attempt["attempt_id"])
        self.store.update_attempt(run_id, attempt["attempt_id"], {"execution": "stopping"})
        return {"stop_requested": True, "request": request, "execution": "stopping", "safe_stop_confirmed": False}

    def resume(self, run_id, *, checkpoint, gpus=None):
        status = self.status(run_id)
        if status["execution"] not in {"exited", "create_failed"} or status["training"] in {"budget_completed", "protection_stopped"}:
            raise ValueError("仅明确已退出、未耗尽预算且未保护终止的运行允许恢复")
        run = self.store.read_run(run_id)
        source = canonical(checkpoint)
        old_attempts = self.store.run_dir(run_id) / "attempts"
        try:
            relative = source.relative_to(old_attempts)
        except ValueError as exc:
            raise ValueError("恢复点必须属于本运行历史尝试，不能引用外部pickle") from exc
        if len(relative.parts) != 3 or relative.parts[1] != "output" or not relative.parts[2].startswith("checkpoint-"):
            raise ValueError("恢复点必须是本运行历史尝试的明确checkpoint目录")
        source_attempt = self.store.read_attempt(run_id, relative.parts[0])
        if source_attempt.get("training") == "protection_stopped":
            raise ValueError("不能绕过历史尝试的保护终止")
        checked = self.check(read_json(self.store.run_dir(run_id) / "resolved-config.json"), gpus=gpus or status["gpu_uuids"])
        if checked["gpu_uuids"] != status["gpu_uuids"]:
            raise ValueError("恢复必须保持相同GPU UUID和rank顺序")
        return self._launch_training(run, checked, source)

    def _start_m0(self, resolved, checked):
        prepared = self._prepare_m0(resolved, checked)
        return self._launch_evaluation(prepared, gpus=checked["gpu_uuids"])

    def _prepare_m0(self, resolved, checked, *, persist=True):
        from .evaluation import prepare_evaluation
        config = dict(resolved["recipe"]["evaluation"])
        kind = config.pop("kind", "benchmark")
        if config.get("config"):
            location = Path(config["config"])
            config = read_json(location if location.is_absolute() else self.workspace / location)
        inputs = checked["audit"]["inputs"]
        if "snapshot" in inputs:
            if config.get("offline") is not True:
                raise ValueError("M0显式快照必须配合offline=true固定评测配置")
            config["snapshot"] = inputs["snapshot"]["path"]
        if "split" in inputs:
            value = config.get("split")
            if not isinstance(value, str) or ":" not in value:
                raise ValueError("M0评测split必须显式FILE:NAME，不猜测留出集")
            filename, subset = value.rsplit(":", 1)
            location = Path(filename)
            if (location if location.is_absolute() else self.workspace / location).resolve() != Path(inputs["split"]["path"]):
                raise ValueError("M0评测切分与登记输入文件不一致")
            config["split"] = f"{inputs['split']['path']}:{subset}"
        return prepare_evaluation(self.store, inputs["model"]["path"], kind, config, workspace=self.workspace, persist=persist)

    def evaluate(self, args):
        from .evaluation import prepare_evaluation
        config = read_json(args.evaluation_config)
        prepared = prepare_evaluation(self.store, args.reference, args.kind, config, workspace=self.workspace,
                                      diagnostic=getattr(args, "diagnostic", False))
        return self._launch_evaluation(prepared, gpus=args.gpus)

    def _launch_evaluation(self, prepared, gpus=None):
        from .evaluation import build_evaluation_command
        paths = [input_path(path, self.profile["input_roots"], managed_root=self.store.root) for path in prepared["inputs"]]
        self._validate_host_paths(paths)
        self.runtime.image(self.profile["image"])
        devices = self.runtime.resolve_gpus(gpus, self.profile["gpus"])
        self.runtime.check_idle(devices)
        run = prepared["run"]
        with self.store.lock():
            for gpu in devices:
                if gpu in self._claims():
                    raise ResourceConflict(f"GPU已有活动声明: {gpu}")
            attempt = self.store.create_attempt(run["run_id"], {"gpu_uuids": devices, "evaluation_only": True,
                "evaluation_id": prepared["context"]["evaluation"]["evaluation_id"]}, locked=True)
            name = f"arxiv-{run['run_id']}-{attempt['attempt_id']}"
            attempt = self.store.update_attempt(run["run_id"], attempt["attempt_id"], {"container_name": name}, locked=True)
            self._claim(attempt, locked=True)
        directory = self.store.attempt_dir(run["run_id"], attempt["attempt_id"])
        evaluation = prepared["context"]["evaluation"]
        try:
            command = build_evaluation_command(**{key: evaluation[key] for key in
                ("model_path", "kind", "config", "stage", "source", "workspace", "diagnostic")},
                output_dir=directory / "output", control_dir=directory / "control",
                run_id=run["run_id"], attempt_id=attempt["attempt_id"])
        except (OSError, ValueError, RuntimeError) as exc:
            return self._planning_failed(run, attempt, exc)
        return self.launch_job(run, attempt, command, paths)
