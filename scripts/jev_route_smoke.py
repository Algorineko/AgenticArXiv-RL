"""Run a small Jev tool-routing experiment on the built-in benchmark tasks.

This script intentionally evaluates only the *next tool choice*.  It does not
load the local Qwen policy, execute tools, or change the GRPO training path.

Usage (from the repository root):

    python scripts/jev_route_smoke.py
    python scripts/jev_route_smoke.py --split dev
    python scripts/jev_route_smoke.py --limit 3 --fresh
    python scripts/jev_route_smoke.py --task-set legacy
    python scripts/jev_route_smoke.py --dry-run

The API key is read from ``TYPESAFE_API_KEY``.  When the variable is absent,
the script asks for it interactively without echoing it.  Successful rows are
checkpointed after every request; rerunning the same command resumes them.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "AgenticArxiv"
sys.path.insert(0, str(PACKAGE_ROOT))

from agents.prompt_templates import build_visible_setup_context  # noqa: E402
from benchmark.tasks import (  # noqa: E402
    get_all_tasks,
    get_dependency_chain,
    get_task_by_id,
)
from benchmark.tasks_expanded import get_expanded_tasks  # noqa: E402
from tools.bootstrap import require_all_tools  # noqa: E402
from tools.tool_registry import registry  # noqa: E402


API_BASE = "https://api.typesafe.ai"
MODEL = "jev-latest"
INPUT_PRICE_PER_MILLION = 0.042
TRANSIENT_ATTEMPTS = 5
SPLIT_PATH = REPO_ROOT / "data" / "splits" / "v3_81.json"
EXPERIMENT_VERSION = "v2"
SNAPSHOT_CANDIDATES = (
    REPO_ROOT / "data" / "mock_arxiv_snapshot.json",
    REPO_ROOT
    / "artifacts"
    / "remote_gpu_evidence_20260927"
    / "data"
    / "smoke_arxiv_snapshot.json",
    REPO_ROOT
    / "repo"
    / "evidence"
    / "remote_gpu_evidence_20260927"
    / "data"
    / "smoke_arxiv_snapshot.json",
)


class _TransientRequestError(RuntimeError):
    """All retries for one HTTP transport were exhausted."""


def _tool_criteria() -> Dict[str, str]:
    require_all_tools("Jev 路由实验")
    criteria = {
        tool["name"]: str(tool.get("description") or tool["name"])
        for tool in registry.list_tools()
    }
    criteria["FINISH"] = (
        "Do not call a tool. Select this when the task is already complete OR "
        "cannot be executed safely: required session/paper context is missing, "
        "a paper reference is invalid or outside the available candidate list, "
        "the requested action is unsupported, or no available tool can satisfy it."
    )
    return criteria


def _expected_first_tool(task: Mapping[str, Any]) -> str:
    expected = list(task.get("expected_tools") or [])
    if not expected:
        return "FINISH"
    return str(expected[0])


def _effective_setup(task: Mapping[str, Any]) -> List[Dict[str, Any]]:
    setup = [dict(action) for action in (task.get("setup") or [])]
    task_id = str(task.get("id") or "")
    if task_id:
        for dependency_id in get_dependency_chain(task_id)[:-1]:
            dependency = get_task_by_id(dependency_id)
            if dependency is None:
                continue
            tools = list(dependency.get("expected_tools") or [])
            arguments = list(dependency.get("expected_tool_args") or [])
            setup.extend(
                {
                    "name": tool,
                    "args": dict(arguments[index] or {})
                    if index < len(arguments)
                    else {},
                }
                for index, tool in enumerate(tools)
            )
    return setup


def _visible_context_for(
    task: Mapping[str, Any], papers: Sequence[Mapping[str, Any]] = ()
) -> str:
    """Render setup plus completed dependency actions as pre-existing state.

    The normal BenchmarkRunner executes ``depends_on`` tasks in the same
    session before evaluating the current task.  A router-only experiment does
    not execute those agents, so recreate only the state their expected tool
    calls would establish.  These are predecessor actions, never the current
    task's expected answer.
    """
    setup = _effective_setup(task)
    context = build_visible_setup_context({"setup": setup})
    if not papers:
        return context
    lines = [context, "[Current candidate papers in session]"]
    for index, paper in enumerate(papers, 1):
        paper_id = str(paper.get("id") or paper.get("paper_id") or "unknown")
        title = str(paper.get("title") or "untitled")
        lines.append(f"{index}. {paper_id} — {title}")
    return "\n".join(lines)


def _state_for(
    task: Mapping[str, Any], papers: Sequence[Mapping[str, Any]] = ()
) -> Dict[str, Any]:
    visible_context = _visible_context_for(task, papers)
    return {
        "agent": "AgenticArXiv ReAct research assistant",
        "task": task["task"],
        "existing_session_state": visible_context,
        "decision": (
            "Select only the immediate next action. Select FINISH instead of a "
            "tool when required context is absent, a reference is invalid or "
            "out of range, or the requested operation is unsupported."
        ),
    }


def _payload(
    task: Mapping[str, Any],
    criteria: Mapping[str, str],
    papers: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    return {
        "state": _state_for(task, papers),
        "model": MODEL,
        "questions": {
            "next_tool": {
                "type": "choice",
                "instructions": (
                    "Which single tool should the agent call next? Select the "
                    "immediate next tool. Choose FINISH when no tool call is needed "
                    "or when the request cannot be executed with the visible state "
                    "and available tools."
                ),
                "criteria": dict(criteria),
            }
        },
    }


def _resolve_snapshot(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"snapshot does not exist: {path}")
        return path
    return next((path for path in SNAPSHOT_CANDIDATES if path.exists()), None)


def _paper_lists_from_snapshot(
    tasks: Sequence[Mapping[str, Any]], snapshot_path: Path | None
) -> Dict[str, List[Dict[str, Any]]]:
    """Replay setup searches so the router sees the same candidate titles/IDs."""
    if snapshot_path is None:
        return {}
    from rl.env import MockArxivEnv

    require_all_tools("Jev routing snapshot context")
    environment = MockArxivEnv(snapshot_path=snapshot_path, mode="replay")
    paper_lists: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        current: List[Dict[str, Any]] = []
        for action in _effective_setup(task):
            name = str(action.get("name") or "")
            if name not in {
                "get_recently_submitted_cs_papers",
                "search_arxiv_papers",
            }:
                continue
            try:
                result = environment.execute_tool(name, dict(action.get("args") or {}))
            except (KeyError, TypeError, ValueError):
                continue
            if isinstance(result, list):
                current = [dict(paper) for paper in result if isinstance(paper, dict)]
        if current:
            paper_lists[str(task["id"])] = current
    return paper_lists


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


class _ResilientClient:
    """Small HTTP client that discards a broken TLS connection before retrying."""

    def __init__(
        self,
        *,
        attempts: int = TRANSIENT_ATTEMPTS,
        transport: str = "auto",
        session_factory: Callable[[], requests.Session] = requests.Session,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.attempts = max(1, attempts)
        self.transport = transport
        self._active_transport = "curl" if transport == "curl" else "requests"
        self._session_factory = session_factory
        self._sleep = sleep
        self._session = self._session_factory()
        self._curl = shutil.which("curl.exe") or shutil.which("curl")

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "_ResilientClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _reset_session(self) -> None:
        self._session.close()
        self._session = self._session_factory()

    def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], float]:
        if self._active_transport == "curl":
            return self._request_json_curl(
                method, url, headers=headers, payload=payload
            )
        try:
            return self._request_json_requests(
                method, url, headers=headers, payload=payload
            )
        except _TransientRequestError as requests_error:
            if self.transport != "auto" or self._curl is None:
                raise
            print(
                "[FALLBACK] requests/OpenSSL failed; switching to "
                "curl/Windows Schannel",
                file=sys.stderr,
            )
            self._active_transport = "curl"
            try:
                return self._request_json_curl(
                    method, url, headers=headers, payload=payload
                )
            except _TransientRequestError as curl_error:
                raise RuntimeError(
                    "both HTTP transports failed; "
                    f"requests={requests_error}; curl={curl_error}"
                ) from curl_error

    def _request_json_requests(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], float]:
        started = time.perf_counter()
        transient_errors = (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        )
        last_error: BaseException | None = None
        for attempt in range(1, self.attempts + 1):
            retry_reason: str | None = None
            try:
                response = self._session.request(
                    method,
                    url,
                    headers=dict(headers),
                    json=dict(payload) if payload is not None else None,
                    timeout=(10, 45),
                )
            except transient_errors as exc:
                last_error = exc
                retry_reason = f"{type(exc).__name__}: {exc}"
                # An SSLEOF can leave urllib3's pooled connection unusable.
                self._reset_session()
            else:
                if response.status_code not in {429, 500, 502, 503, 504}:
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    if not response.ok:
                        raise RuntimeError(
                            f"TypeSafe API returned HTTP {response.status_code}: "
                            f"{response.text[:1000]}"
                        )
                    try:
                        return response.json(), elapsed_ms
                    except ValueError as exc:
                        raise RuntimeError(
                            "TypeSafe API returned a non-JSON response: "
                            f"{response.text[:1000]}"
                        ) from exc
                retry_reason = f"HTTP {response.status_code}"
                last_error = RuntimeError(
                    f"{retry_reason}: {response.text[:1000]}"
                )
                if response.status_code >= 500:
                    self._reset_session()

            if attempt == self.attempts:
                break
            # Jitter avoids retrying in lockstep with a briefly unhealthy edge.
            delay = 0.75 * (2 ** (attempt - 1)) + random.uniform(0.0, 0.25)
            print(
                f"[RETRY] {retry_reason}; attempt {attempt}/{self.attempts}, "
                f"wait {delay:.2f}s",
                file=sys.stderr,
            )
            self._sleep(delay)

        raise _TransientRequestError(
            f"TypeSafe request failed after {self.attempts} attempts: {last_error}"
        ) from last_error

    def _request_json_curl(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], float]:
        if self._curl is None:
            raise _TransientRequestError("curl executable was not found")
        started = time.perf_counter()
        body = json.dumps(payload, ensure_ascii=False) if payload is not None else None
        last_error = "unknown curl error"
        for attempt in range(1, self.attempts + 1):
            command = [
                self._curl,
                "--silent",
                "--show-error",
                "--http1.1",
                "--ipv4",
                "--connect-timeout",
                "10",
                "--max-time",
                "45",
                "--request",
                method,
                url,
            ]
            for name, value in headers.items():
                command.extend(["--header", f"{name}: {value}"])
            if body is not None:
                command.extend(["--data-binary", "@-"])
            command.extend(["--write-out", "\n__HTTP_STATUS__:%{http_code}"])
            completed = subprocess.run(
                command,
                input=body,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            response_body, marker, status_text = completed.stdout.rpartition(
                "\n__HTTP_STATUS__:"
            )
            status = int(status_text.strip()) if marker and status_text.strip().isdigit() else 0
            if completed.returncode == 0 and 200 <= status < 300:
                try:
                    decoded = json.loads(response_body)
                except ValueError as exc:
                    raise RuntimeError(
                        "TypeSafe API returned non-JSON through curl: "
                        f"{response_body[:1000]}"
                    ) from exc
                elapsed_ms = (time.perf_counter() - started) * 1000
                return decoded, elapsed_ms
            if completed.returncode == 0 and status not in {429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"TypeSafe API returned HTTP {status} through curl: "
                    f"{response_body[:1000]}"
                )
            last_error = (
                completed.stderr.strip()
                or f"curl exit={completed.returncode}, HTTP {status}"
            )
            if attempt == self.attempts:
                break
            delay = 0.75 * (2 ** (attempt - 1)) + random.uniform(0.0, 0.25)
            print(
                f"[RETRY curl] {last_error}; attempt {attempt}/{self.attempts}, "
                f"wait {delay:.2f}s",
                file=sys.stderr,
            )
            self._sleep(delay)
        raise _TransientRequestError(
            f"curl transport failed after {self.attempts} attempts: {last_error}"
        )


def _summarize(
    rows: Sequence[Mapping[str, Any]],
    *,
    planned_tasks: int,
    models: Mapping[str, Any],
    models_ms: float | None,
) -> Dict[str, Any]:
    total_input = sum(int(row["input_tokens"]) for row in rows)
    latencies = [float(row["elapsed_ms"]) for row in rows]
    confidences = [float(row["confidence"]) for row in rows]
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "model": MODEL,
        "model_lookup_ms": round(models_ms, 2) if models_ms is not None else None,
        "available_models": dict(models),
        "planned_tasks": planned_tasks,
        "tasks": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "mean_confidence": statistics.fmean(confidences) if confidences else 0.0,
        "mean_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "total_input_tokens": total_input,
        "total_output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "estimated_cost_usd": total_input / 1_000_000 * INPUT_PRICE_PER_MILLION,
    }


def _report(
    rows: Sequence[Mapping[str, Any]],
    *,
    planned_tasks: int,
    models: Mapping[str, Any],
    models_ms: float | None,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "config": dict(config),
        "summary": _summarize(
            rows,
            planned_tasks=planned_tasks,
            models=models,
            models_ms=models_ms,
        ),
        "results": list(rows),
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _resume_rows(
    output: Path,
    tasks: Sequence[Mapping[str, Any]],
    enabled: bool,
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    if not enabled or not output.exists():
        return []
    try:
        existing = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[WARN] cannot read checkpoint {output}: {exc}", file=sys.stderr)
        return []
    existing_config = existing.get("config") or {}
    if existing_config.get("experiment_version") != config.get("experiment_version"):
        print(
            "[WARN] checkpoint belongs to a different prompt version; "
            "starting a new evaluation",
            file=sys.stderr,
        )
        return []
    expected_by_id = {str(t["id"]): _expected_first_tool(t) for t in tasks}
    valid: Dict[str, Dict[str, Any]] = {}
    for raw in existing.get("results") or []:
        task_id = str(raw.get("task_id") or "")
        if task_id in expected_by_id and raw.get("expected") == expected_by_id[task_id]:
            valid[task_id] = dict(raw)
    return [valid[str(t["id"])] for t in tasks if str(t["id"]) in valid]


def _evaluate(
    tasks: Sequence[Mapping[str, Any]],
    api_key: str,
    criteria: Mapping[str, str],
    *,
    output: Path,
    resume: bool,
    lookup_models: bool,
    attempts: int,
    transport: str,
    paper_lists: Mapping[str, Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    rows = _resume_rows(output, tasks, resume, config)
    completed = {str(row["task_id"]) for row in rows}
    for row in rows:
        print(
            f"[RESUME] {row['task_id']}: selected={row['selected']} "
            f"correct={row['correct']}"
        )

    headers = _headers(api_key)
    models: Dict[str, Any] = {"status": "skipped"}
    models_ms: float | None = None
    _write_report(
        output,
        _report(
            rows,
            planned_tasks=len(tasks),
            models=models,
            models_ms=models_ms,
            config=config,
        ),
    )
    with _ResilientClient(attempts=attempts, transport=transport) as client:
        if lookup_models:
            try:
                models, models_ms = client.request_json(
                    "GET", f"{API_BASE}/v1/models", headers=headers
                )
            except RuntimeError as exc:
                # Model discovery is informational; routing can still work.
                models = {"status": "unavailable", "error": str(exc)}
                print(f"[WARN] model lookup failed; continuing: {exc}", file=sys.stderr)

        for task in tasks:
            task_id = str(task["id"])
            if task_id in completed:
                continue
            response, elapsed_ms = client.request_json(
                "POST",
                f"{API_BASE}/v1/systemone",
                headers=headers,
                payload=_payload(task, criteria, paper_lists.get(task_id, ())),
            )
            answer = response["answers"]["next_tool"]
            usage = response.get("usage") or {}
            selected = str(answer["choice"])
            expected = _expected_first_tool(task)
            row = {
                "task_id": task_id,
                "task": task["task"],
                "expected": expected,
                "selected": selected,
                "correct": selected == expected,
                "confidence": float(answer.get("confidence", 0.0)),
                "probabilities": answer.get("probabilities") or {},
                "elapsed_ms": round(elapsed_ms, 2),
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
            }
            rows.append(row)
            completed.add(task_id)
            marker = "PASS" if row["correct"] else "FAIL"
            print(
                f"[{marker}] {row['task_id']}: expected={expected} "
                f"selected={selected} confidence={row['confidence']:.3f} "
                f"latency={row['elapsed_ms']:.0f}ms"
            )
            _write_report(
                output,
                _report(
                    rows,
                    planned_tasks=len(tasks),
                    models=models,
                    models_ms=models_ms,
                    config=config,
                ),
            )

    return _report(
        rows,
        planned_tasks=len(tasks),
        models=models,
        models_ms=models_ms,
        config=config,
    )


def _select_tasks(
    task_set: str,
    split: str,
    limit: int,
    *,
    split_path: Path | None = None,
) -> List[Dict[str, Any]]:
    """Select expanded tasks from one frozen manifest, including for ``all``."""
    if task_set == "legacy":
        if split != "all":
            raise ValueError("--split is only supported with --task-set expanded")
        tasks = get_all_tasks()
    else:
        tasks = get_expanded_tasks()
        payload = json.loads((split_path or SPLIT_PATH).read_text(encoding="utf-8"))
        groups = payload.get("split") or {}
        if split != "all" and split not in groups:
            raise ValueError(
                f"unknown split {split!r}; choose from all, "
                f"{', '.join(sorted(groups))}"
            )
        by_id = {str(task["id"]): task for task in tasks}
        selected_ids = (
            {task_id for ids in groups.values() for task_id in ids}
            if split == "all" else groups[split]
        )
        missing = sorted(task_id for task_id in selected_ids if task_id not in by_id)
        if missing:
            raise ValueError(f"split contains unknown task ids: {missing}")
        tasks = (
            [task for task in tasks if str(task["id"]) in selected_ids]
            if split == "all" else [by_id[task_id] for task_id in selected_ids]
        )
    return tasks[:limit] if limit > 0 else tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Small Jev tool-routing experiment")
    parser.add_argument(
        "--task-set", choices=("legacy", "expanded"), default="expanded"
    )
    parser.add_argument(
        "--split",
        default="all",
        help="expanded split: all, train, dev, iid_test, or ood_test",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=SPLIT_PATH,
        help="frozen expanded-task manifest (default: data/splits/v3_81.json)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="0 evaluates the complete selection"
    )
    parser.add_argument(
        "--task-ids", nargs="+", default=None, help="evaluate only these task IDs"
    )
    parser.add_argument(
        "--only-failures-from",
        default=None,
        help="evaluate task IDs marked incorrect in an existing Jev report",
    )
    parser.add_argument(
        "--snapshot",
        default=None,
        help="offline snapshot used to expose actual candidate paper titles/IDs",
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument("--fresh", action="store_true", help="ignore saved results")
    parser.add_argument(
        "--lookup-models",
        action="store_true",
        help="also query /v1/models (not required for routing)",
    )
    parser.add_argument("--attempts", type=int, default=TRANSIENT_ATTEMPTS)
    parser.add_argument(
        "--transport",
        choices=("auto", "requests", "curl"),
        default="auto",
        help="auto falls back from requests/OpenSSL to curl/Schannel",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        tasks = _select_tasks(
            args.task_set, args.split, args.limit, split_path=args.split_file,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    requested_ids = set(args.task_ids or [])
    if args.only_failures_from:
        try:
            prior = json.loads(Path(args.only_failures_from).read_text(encoding="utf-8"))
            requested_ids.update(
                str(row["task_id"])
                for row in prior.get("results") or []
                if not row.get("correct")
            )
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"cannot load --only-failures-from: {exc}")
    if requested_ids:
        available_ids = {str(task["id"]) for task in tasks}
        unknown = sorted(requested_ids - available_ids)
        if unknown:
            parser.error(f"unknown or out-of-split task IDs: {unknown}")
        tasks = [task for task in tasks if str(task["id"]) in requested_ids]
    if not tasks:
        parser.error("the selected task set is empty")
    criteria = _tool_criteria()
    try:
        snapshot_path = _resolve_snapshot(args.snapshot)
        paper_lists = _paper_lists_from_snapshot(tasks, snapshot_path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        preview = {
            "task_set": args.task_set,
            "split": args.split,
            "split_file": str(args.split_file.resolve()) if args.task_set == "expanded" else None,
            "task_count": len(tasks),
            "tool_count": len(criteria),
            "snapshot": str(snapshot_path) if snapshot_path else None,
            "tasks_with_candidate_papers": len(paper_lists),
            "first_payload": _payload(
                tasks[0], criteria, paper_lists.get(str(tasks[0]["id"]), ())
            ),
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    load_dotenv(REPO_ROOT / ".env.local", override=False)
    api_key = os.getenv("TYPESAFE_API_KEY") or getpass.getpass(
        "TypeSafe API key: "
    ).strip()
    if not api_key:
        raise SystemExit("TYPESAFE_API_KEY is required")

    if args.output:
        output = Path(args.output)
    else:
        suffix = "_targeted" if requested_ids else ""
        if args.task_set == "expanded" and args.split_file.resolve() != SPLIT_PATH.resolve():
            suffix += f"_{args.split_file.stem}"
        output = (
            REPO_ROOT
            / "artifacts"
            / f"jev_route_{args.task_set}_{args.split}_{EXPERIMENT_VERSION}{suffix}.json"
        )
    config = {
        "experiment_version": EXPERIMENT_VERSION,
        "task_set": args.task_set,
        "split": args.split,
        "split_file": str(args.split_file.resolve()) if args.task_set == "expanded" else None,
        "limit": args.limit,
        "model": MODEL,
        "router_scope": "next_tool_only",
        "transport": args.transport,
        "snapshot": str(snapshot_path) if snapshot_path else None,
        "candidate_context_tasks": len(paper_lists),
    }
    try:
        report = _evaluate(
            tasks,
            api_key,
            criteria,
            output=output,
            resume=not args.fresh,
            lookup_models=args.lookup_models,
            attempts=args.attempts,
            transport=args.transport,
            paper_lists=paper_lists,
            config=config,
        )
    except RuntimeError as exc:
        raise SystemExit(
            f"Experiment paused: {exc}\n"
            f"Completed rows remain in {output}. Run the same command to resume."
        ) from exc
    _write_report(output, report)
    print("\nSummary")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
