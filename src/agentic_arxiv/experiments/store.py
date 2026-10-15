"""单机管理记录：短期锁、同文件系统原子提交，不修改研究输入。"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
import uuid


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_record_time(value):
    """缺失、无时区或 Docker 的零时间均不构成运行计时证据。"""
    if not isinstance(value, str):
        return None
    # Docker RFC3339Nano 的九位小数需归一化到 Python 3.10 支持的微秒。
    value = re.sub(r"(\d{2}:\d{2}:\d{2})\.(\d+)",
                   lambda match: match[1] + "." + match[2][:6].ljust(6, "0"), value)
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.year > 1 and parsed.utcoffset() is not None else None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("非法记录标识")
    return value


class FileStore:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()

    @contextlib.contextmanager
    def lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".management.lock").open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def run_dir(self, run_id):
        return self.root / "runs" / identifier(run_id)

    def attempt_dir(self, run_id, attempt_id):
        return self.run_dir(run_id) / "attempts" / identifier(attempt_id)

    def read_run(self, run_id):
        return read_json(self.run_dir(run_id) / "run.json")

    def read_attempt(self, run_id, attempt_id=None):
        if attempt_id is None:
            attempt_id = self.read_run(run_id)["current_attempt"]
        return read_json(self.attempt_dir(run_id, attempt_id) / "attempt.json")

    def _new(self, parent, prefix):
        parent.mkdir(parents=True, exist_ok=True)
        while True:
            record_id = prefix + uuid.uuid4().hex
            directory = parent / record_id
            try:
                directory.mkdir()
                return record_id, directory
            except FileExistsError:
                continue

    def create_run(self, identity, resolved_config, inputs):
        with self.lock():
            run_id, directory = self._new(self.root / "runs", "run-")
            record = dict(identity, schema_version=1, run_id=run_id, created_at=now(), current_attempt=None)
            atomic_json(directory / "resolved-config.json", resolved_config)
            atomic_json(directory / "inputs.json", inputs)
            atomic_json(directory / "run.json", record)
        return record

    def create_attempt(self, run_id, identity, *, locked=False):
        if not locked:
            with self.lock():
                return self.create_attempt(run_id, identity, locked=True)
        run = self.read_run(run_id)
        if run.get("current_attempt"):
            previous = self.read_attempt(run_id)
            if previous.get("execution") not in {"exited", "create_failed"}:
                raise ValueError("运行已有活动或状态未知的尝试")
        attempt_id, directory = self._new(self.run_dir(run_id) / "attempts", "attempt-")
        record = dict(identity, schema_version=1, run_id=run_id, attempt_id=attempt_id,
                      created_at=now(), execution="preparing", training="not_started", validation="not_run")
        for child in ("control", "logs", "output"):
            (directory / child).mkdir()
        atomic_json(directory / "attempt.json", record)
        run["current_attempt"] = attempt_id
        atomic_json(self.run_dir(run_id) / "run.json", run)
        return record

    def update_attempt(self, run_id, attempt_id, changes, *, locked=False):
        if not locked:
            with self.lock():
                return self.update_attempt(run_id, attempt_id, changes, locked=True)
        record = self.read_attempt(run_id, attempt_id)
        immutable = {"run_id", "attempt_id", "created_at", "gpu_uuids", "source_checkpoint"}
        if any(key in record and record[key] != value for key, value in changes.items() if key in immutable):
            raise ValueError("尝试身份不可修改")
        record.update(changes, updated_at=now())
        atomic_json(self.attempt_dir(run_id, attempt_id) / "attempt.json", record)
        return record

    def event(self, run_id, attempt_id, kind, **details):
        with self.lock():
            path = self.attempt_dir(run_id, attempt_id) / "events.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(time=now(), kind=kind, **details), ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def list_runs(self):
        directory = self.root / "runs"
        return [read_json(path) for path in sorted(directory.glob("*/run.json"))] if directory.exists() else []

    def create_registration(self, reference, artifact_id=None):
        with self.lock():
            if artifact_id is None:
                record_id, directory = self._new(self.root / "registrations", "artifact-")
            else:
                record_id = identifier(artifact_id)
                directory = self.root / "registrations" / record_id
                directory.mkdir(parents=True, exist_ok=False)
            record = dict(reference, schema_version=1, artifact_id=record_id, created_at=now())
            atomic_json(directory / "reference.json", record)
            return record

    def create_evaluation(self, run_id, record):
        with self.lock():
            self.read_run(run_id)
            record_id, directory = self._new(self.run_dir(run_id) / "evaluations", "evaluation-")
            result = dict(record, schema_version=1, evaluation_id=record_id, created_at=now())
            atomic_json(directory / "evaluation.json", result)
            return result


RunStore = FileStore
