"""Deterministic end-to-end proof for the optional Jev routing interface.

This smoke does not call a model provider or TypeSafe.  It replays the exact
System One response shape through the real ``JevToolRouter`` adapter, then runs
the ordinary ReActAgent, MockArxivEnv, benchmark metric extractor and report.
The existing ``jev_route_expanded_all.json`` remains the live-API evidence.
Together they separate two questions cleanly:

1. can live Jev classify the project's tools? (81-task pilot)
2. can that decision replace the policy routing interface end to end? (this)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "AgenticArxiv"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.agent_engine import ReActAgent  # noqa: E402
from agents.side_effects import LocalSideEffectManager  # noqa: E402
from benchmark.metrics import extract_metrics, is_strict_success  # noqa: E402
from benchmark.report import BenchmarkReport  # noqa: E402
from benchmark.tasks_expanded import get_expanded_tasks  # noqa: E402
from rl.env import MockArxivEnv  # noqa: E402
from routing.jev import JevToolRouter  # noqa: E402
from tools.bootstrap import require_all_tools  # noqa: E402


TASK_ID = "search_AI_1d_3"
SEARCH_TOOL = "get_recently_submitted_cs_papers"
CONTROL_TOOL = "download_arxiv_pdf"
ACTION = {
    "name": SEARCH_TOOL,
    "args": {"aspect": "AI", "days": 1, "max_results": 3},
}


class ScriptedQwenContractClient:
    """A deterministic stand-in for Qwen's tool-argument generation contract."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def chat_completions(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            content = (
                "Thought: 调用检索工具并生成正确参数\n"
                f"Action: {json.dumps(ACTION, ensure_ascii=False)}"
            )
        else:
            content = "Thought: 任务已经完成\nAction: FINISH"
        return {
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }


class ReplayResponse:
    status_code = 200

    def __init__(self, choice: str, confidence: float) -> None:
        self.choice = choice
        self.confidence = confidence

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return {
            "answers": {
                "next_tool": {
                    "choice": self.choice,
                    "confidence": self.confidence,
                    "probabilities": {self.choice: self.confidence},
                }
            },
            "usage": {"input_tokens": 20, "output_tokens": 2},
        }


class SystemOneContractReplaySession:
    """Replay the provider's typed response while retaining the real adapter."""

    def __init__(self) -> None:
        self.responses = [
            ReplayResponse(SEARCH_TOOL, 0.95),
            ReplayResponse("FINISH", 0.98),
        ]
        self.requests: List[Dict[str, Any]] = []

    def post(self, _url, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected extra router request")
        return self.responses.pop(0)


def _task() -> Dict[str, Any]:
    return next(task for task in get_expanded_tasks() if task["id"] == TASK_ID)


def _run_once(task: Dict[str, Any], snapshot: Path, router=None):
    client = ScriptedQwenContractClient()
    agent = ReActAgent(
        client,
        side_effect_mgr=LocalSideEffectManager(),
        env=MockArxivEnv(snapshot_path=snapshot, mode="replay"),
        max_iterations=3,
        tool_router=router,
    )
    result = agent.run(
        task=task["task"],
        agent_model="scripted-qwen-contract",
        session_id=f"jev_integration_{'jev' if router else 'policy'}",
    )
    return result, client


def _prompt_visibility(client: ScriptedQwenContractClient) -> Dict[str, bool]:
    if not client.calls:
        return {
            "selected_tool_visible": False,
            "unselected_control_tool_visible": False,
        }
    prompt = client.calls[0]["messages"][0]["content"]
    # The policy prompt and guided fixed-tool prompt use different headings.
    # Inspect only the injected schema block, not static examples elsewhere.
    if "唯一允许的工具：" in prompt:
        tool_block = prompt.split("唯一允许的工具：", 1)[1].split(
            "当前任务：", 1
        )[0]
    else:
        tool_block = prompt.split("你有以下工具可以使用：", 1)[1].split(
            "当前任务：", 1
        )[0]
    return {
        "selected_tool_visible": SEARCH_TOOL in tool_block,
        "unselected_control_tool_visible": CONTROL_TOOL in tool_block,
    }


def _load_live_pilot(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"status": "not_found", "path": _display_path(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    return {
        "status": "completed",
        "source": _display_path(path),
        "model": summary.get("model"),
        "tasks": summary.get("tasks"),
        "correct": summary.get("correct"),
        "accuracy": summary.get("accuracy"),
        "mean_confidence": summary.get("mean_confidence"),
        "mean_latency_ms": summary.get("mean_latency_ms"),
        "estimated_cost_usd": summary.get("estimated_cost_usd"),
    }


def run(snapshot: Path, live_pilot: Path) -> Dict[str, Any]:
    require_all_tools("Jev project integration smoke")
    task = _task()

    policy_result, policy_client = _run_once(task, snapshot, router=None)

    replay_session = SystemOneContractReplaySession()
    jev_router = JevToolRouter(
        api_key="contract-replay-placeholder",
        min_confidence=0.80,
        max_retries=0,
        session=replay_session,
        sleep=lambda _seconds: None,
    )
    jev_result, jev_client = _run_once(task, snapshot, router=jev_router)

    policy_metrics = extract_metrics(task, policy_result, "policy", 0)
    jev_metrics = extract_metrics(task, jev_result, "jev", 0)
    report = BenchmarkReport([policy_metrics, jev_metrics], model="scripted-qwen-contract")

    policy_visibility = _prompt_visibility(policy_client)
    jev_visibility = _prompt_visibility(jev_client)
    second_state = replay_session.requests[1]["json"]["state"][
        "history_and_environment"
    ]
    checks = {
        "policy_strict_success": is_strict_success(policy_metrics),
        "jev_strict_success": is_strict_success(jev_metrics),
        "same_exact_tool_sequence": (
            policy_metrics.tool_call_sequence == jev_metrics.tool_call_sequence
        ),
        "policy_sees_all_tools": (
            policy_visibility["selected_tool_visible"]
            and policy_visibility["unselected_control_tool_visible"]
        ),
        "jev_restricts_qwen_to_selected_schema": (
            not jev_client.calls
            or (
                jev_visibility["selected_tool_visible"]
                and not jev_visibility["unselected_control_tool_visible"]
            )
        ),
        "jev_uses_validated_deterministic_arguments": (
            jev_result["routing"]["decisions"][0].get("argument_source")
            == "deterministic"
        ),
        "jev_reconsiders_after_observation": "Observation:" in second_state,
        "jev_decisions_were_used": all(
            decision.get("used")
            for decision in jev_result["routing"]["decisions"]
        ),
        "policy_has_no_external_decisions": not policy_result["routing"]["decisions"],
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"integration smoke failed: {failed}")

    return {
        "experiment": "jev-project-interface-smoke-v1",
        "scope": (
            "Deterministic end-to-end interface proof. Provider responses are "
            "contract replays; live provider quality is reported separately."
        ),
        "snapshot": _display_path(snapshot),
        "task_id": TASK_ID,
        "checks": checks,
        "policy": {
            "qwen_calls": len(policy_client.calls),
            "tool_sequence": policy_metrics.tool_call_sequence,
            "strict_success": is_strict_success(policy_metrics),
            "routing": policy_result["routing"],
        },
        "jev": {
            "qwen_calls": len(jev_client.calls),
            "tool_sequence": jev_metrics.tool_call_sequence,
            "strict_success": is_strict_success(jev_metrics),
            "routing": jev_result["routing"],
        },
        "benchmark_summary": report.summary_by_agent(),
        "timing_scope": (
            "Timing in this deterministic contract replay is not performance "
            "evidence; use the live pilot for provider latency."
        ),
        "live_routing_pilot": _load_live_pilot(live_pilot),
        "conclusion": (
            "The policy and Jev paths are switch-compatible at the project "
            "boundary. Both complete the same task with the same exact tool "
            "sequence; the Jev path constrains Qwen to the selected schema and "
            "passes decisions through the normal environment and benchmark."
        ),
    }


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=(
            REPO_ROOT
            / "AgenticArxiv"
            / "tests"
            / "fixtures"
            / "jev_integration_snapshot.json"
        ),
    )
    parser.add_argument(
        "--live-pilot",
        type=Path,
        default=REPO_ROOT / "artifacts" / "jev_route_expanded_all.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "jev_project_integration_smoke.json",
    )
    args = parser.parse_args()
    for path in (args.snapshot, args.live_pilot):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    result = run(args.snapshot, args.live_pilot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["checks"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
