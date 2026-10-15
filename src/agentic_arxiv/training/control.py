"""Attempt-scoped file control protocol; safe to import without a training stack."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid


CONTROL_VERSION = 1
REQUEST_FILE = "stop-request.json"
STATUS_FILE = "stop-status.json"


def atomic_control_json(path: Path, value: dict[str, Any]) -> None:
    """Replace a snapshot durably without exposing partially written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
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


def read_stop_request(control_dir: str | Path, run_id: str, attempt_id: str) -> dict | None:
    """Only the exact current attempt may consume a request; old requests are inert."""
    path = Path(control_dir) / REQUEST_FILE
    if not path.exists():
        return None
    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("停止请求必须为 JSON 对象")
    if request.get("run_id") != run_id or request.get("attempt_id") != attempt_id:
        return None
    if set(request) != {"schema_version", "run_id", "attempt_id", "request_id"} or (
        type(request["schema_version"]) is not int
        or request["schema_version"] != CONTROL_VERSION
        or not isinstance(request["request_id"], str) or not request["request_id"]
    ):
        raise ValueError("不支持或损坏的停止请求")
    return request


def request_stop(control_dir: str | Path, run_id: str, attempt_id: str) -> dict:
    """Submit once under a short lock; retries preserve the original request ID."""
    if not isinstance(run_id, str) or not run_id or not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("停止请求需要明确的 run_id 和 attempt_id")
    directory = Path(control_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".stop.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = read_stop_request(directory, run_id, attempt_id)
        if existing is not None:
            return existing
        request = {"schema_version": CONTROL_VERSION, "run_id": run_id,
                   "attempt_id": attempt_id, "request_id": uuid.uuid4().hex}
        atomic_control_json(directory / REQUEST_FILE, request)
        return request
