"""复用已有历史轨迹，验证冻结参考与只读重评分边界。"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agentic_arxiv.cli import parse_args
from agentic_arxiv.evaluation.rescore_traces import _task_with_references, execute
from agentic_arxiv.provenance import read_jsonl
from agentic_arxiv.scoring.metrics import extract_metrics, is_strict_success
from agentic_arxiv.tasks.tasks_expanded import get_expanded_tasks


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "artifacts/eval_grpo_v381_iid_r3_s45"


@pytest.fixture
def recorded_case():
    task = next(task for task in get_expanded_tasks() if task["id"] == "ref_ctrl_id_cache")
    trace = next(row for row in read_jsonl(HISTORY / "traces.jsonl") if row["task_id"] == task["id"])
    # 参考只取任务中明确声明的论文 ID，不从实际策略结果推断标准答案。
    references = [task["expected_tool_args"][0]["ref"]]
    return task, trace, references


def test_frozen_reference_keeps_equivalent_ref_and_strict_success(recorded_case):
    task, trace, references = recorded_case
    literal = extract_metrics(task, trace, trace["agent_type"], trace["trial"])
    prepared = _task_with_references(task, trace, {task["id"]: references}, location="已有轨迹")
    semantic = extract_metrics(prepared, trace, trace["agent_type"], trace["trial"])
    assert literal.arg_score == 0.0
    assert semantic.arg_score == semantic.ref_score == 1.0
    assert is_strict_success(semantic)
    assert "expected_paper_ids" not in task
    assert "expected_paper_ids" not in trace


def test_inline_reference_is_read_without_mutating_the_input(recorded_case):
    task, trace, references = recorded_case
    frozen = {**trace, "expected_paper_ids": references}
    prepared = _task_with_references(task, frozen, {}, location="已有轨迹")
    assert prepared["expected_paper_ids"] == references
    assert prepared["expected_paper_ids"] is not references


@pytest.mark.parametrize("references", [[], [None], [123], [""], ["a", "b"]])
def test_invalid_reference_is_not_a_literal_fallback(recorded_case, references):
    task, trace, _ = recorded_case
    with pytest.raises(ValueError, match="必须与 expected_tools 对齐"):
        _task_with_references(task, trace, {task["id"]: references}, location="已有轨迹")


def test_conflicting_frozen_references_are_rejected(recorded_case):
    task, trace, references = recorded_case
    conflicting = {**trace, "expected_paper_ids": ["2608.14528v1"]}
    with pytest.raises(ValueError, match="冻结参考冲突"):
        _task_with_references(task, conflicting, {task["id"]: references}, location="已有轨迹")


def test_search_only_task_does_not_require_paper_references():
    task = next(task for task in get_expanded_tasks() if task["category"] == "search")
    assert _task_with_references(task, {}, {}, location="已有任务") is task


def _arguments(output, references=None):
    return SimpleNamespace(
        workspace=ROOT, traces=HISTORY / "traces.jsonl", summary=HISTORY / "summary.json",
        output=output, snapshot=None, references=references, task_set="expanded", split=None,
    )


def test_missing_reference_fails_before_any_report_is_written(recorded_case, tmp_path):
    task, _, _ = recorded_case
    with patch("agentic_arxiv.evaluation.rescore_traces.get_expanded_tasks", return_value=[task]):
        with pytest.raises(ValueError, match="无法重算论文身份、参数及严格成功指标"):
            execute(_arguments(tmp_path / "output"))
    assert not (tmp_path / "output").exists()


def test_reference_file_restores_old_scores_without_running_an_environment(recorded_case, tmp_path):
    task, _, references = recorded_case
    reference_path = tmp_path / "references.json"
    reference_path.write_text(json.dumps({task["id"]: references}), encoding="utf-8")
    inputs = (HISTORY / "traces.jsonl", HISTORY / "summary.json", reference_path)
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs]
    with (
        patch("agentic_arxiv.evaluation.rescore_traces.get_expanded_tasks", return_value=[task]),
        patch("agentic_arxiv.tasks.reference.attach_expected_paper_ids", side_effect=AssertionError("不得准备参考")),
        patch("agentic_arxiv.runtime.multiturn_env.AgenticArxivMultiTurnEnv", side_effect=AssertionError("不得创建环境")),
    ):
        execute(_arguments(tmp_path / "output", reference_path))
    output = json.loads((tmp_path / "output/summary.json").read_text(encoding="utf-8"))
    old = json.loads((HISTORY / "summary.json").read_text(encoding="utf-8"))
    expected = {row["trial"]: row for row in old["details"] if row["task_id"] == task["id"]}
    assert len(output["details"]) == len(expected)
    for row in output["details"]:
        assert row["arg_score"] == expected[row["trial"]]["arg_score"] == 1.0
        assert row["ref_score"] == expected[row["trial"]]["ref_score"] == 1.0
    assert before == [hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs]


def test_runner_freezes_only_supplied_references_outside_policy_history(recorded_case):
    from agentic_arxiv.evaluation.run_benchmark import _trace_record
    from agentic_arxiv.evaluation.runner import BenchmarkRunner

    task, trace, references = recorded_case
    prepared = {**task, "expected_paper_ids": references}
    runner = BenchmarkRunner(offline=True, repeat=1)
    runner._llm_client = object()
    with (
        patch("agentic_arxiv.tasks.reference.attach_expected_paper_ids", return_value=[prepared]),
        patch("agentic_arxiv.evaluation.runner.execute_episode", return_value=trace),
    ):
        result = runner.run_all([task])[0]
    record = _trace_record(result)
    assert record["expected_paper_ids"] == references
    assert record["expected_paper_ids"] is not references
    assert "expected_paper_ids" not in result.raw_result
    assert "expected_paper_ids" not in task
    assert is_strict_success(result.metrics)


def test_reference_input_resolves_against_explicit_workspace(tmp_path):
    _, args = parse_args([
        "--workspace", str(tmp_path), "evaluate", "rescore", "--traces", "traces.jsonl",
        "--summary", "summary.json", "--output", "out", "--references", "references.json",
    ])
    assert args.references == str(tmp_path / "references.json")
    assert args.snapshot is None
