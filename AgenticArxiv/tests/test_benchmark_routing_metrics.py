"""Router decisions must survive benchmark extraction, reporting, and traces."""

import json
from types import SimpleNamespace

import pytest

from benchmark.metrics import (
    JEV_INPUT_PRICE_PER_MILLION_USD,
    TaskMetrics,
    extract_metrics,
)
from benchmark.report import BenchmarkReport
from benchmark.run_benchmark import _trace_record


def _result():
    return {
        "history": [
            {
                "thought": "download",
                "action": json.dumps({
                    "name": "download_arxiv_pdf",
                    "args": {"ref": 1},
                }),
                "observation": "ok",
            },
            {"thought": "done", "action": "FINISH", "observation": "done"},
        ],
        "total_time_ms": 400,
        "iteration_count": 2,
        "timing": {
            "total_router_ms": 150,
            "total_llm_ms": 100,
            "total_tool_ms": 50,
            "framework_overhead_ms": 100,
        },
        "token_usage": {
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "total_tokens": 25,
        },
        "routing": {
            "mode": "jev",
            "decisions": [
                {
                    "selected_tool": "download_arxiv_pdf",
                    "confidence": 0.95,
                    "accepted": True,
                    "used": True,
                    "usage": {"input_tokens": 100, "output_tokens": 5},
                },
                {
                    "selected_tool": "FINISH",
                    "confidence": 0.40,
                    "accepted": False,
                    "used": False,
                    "reason": "low_confidence",
                    "usage": {"input_tokens": 20, "output_tokens": 2},
                },
            ],
        },
    }


def test_extracts_router_usage_fallback_latency_cost_and_first_choice():
    metrics = extract_metrics(
        {"id": "download", "expected_tools": ["download_arxiv_pdf"]},
        _result(),
        "regex",
        0,
    )

    assert metrics.total_router_ms == 150
    assert metrics.router_mode == "jev"
    assert metrics.router_decision_count == 2
    assert metrics.router_accepted_count == 1
    assert metrics.router_used_count == 1
    assert metrics.router_fallback_count == 1
    assert metrics.router_first_selected_tool == "download_arxiv_pdf"
    assert metrics.router_first_accurate is True
    assert metrics.router_first_applicable is True
    assert metrics.router_confidence_count == 2
    assert metrics.router_mean_confidence == pytest.approx(0.675)
    assert metrics.router_input_tokens == 120
    assert metrics.router_output_tokens == 7
    assert metrics.router_estimated_cost_usd == pytest.approx(
        120 / 1_000_000 * JEV_INPUT_PRICE_PER_MILLION_USD
    )
    assert metrics.router_fallback_reasons == {"low_confidence": 1}


def test_report_exposes_router_coverage_accuracy_latency_cost_and_reasons():
    metrics = extract_metrics(
        {"id": "download", "expected_tools": ["download_arxiv_pdf"]},
        _result(),
        "regex",
        0,
    )
    report = BenchmarkReport([metrics], model="qwen")
    summary = report.summary_by_agent()["regex"]

    assert summary["router_acceptance_rate"] == 0.5
    assert summary["router_use_rate"] == 0.5
    assert summary["router_fallback_rate"] == 0.5
    assert summary["router_first_accuracy"] == 1.0
    assert summary["router_mean_confidence"] == pytest.approx(0.675)
    assert summary["router_ms_per_decision"] == 75.0
    assert summary["router_fallback_reasons"] == {"low_confidence": 1}
    assert "### 外部工具路由" in report.comparison_table_md()
    assert "首步路由准确率" in report.comparison_table_md()


def test_policy_only_report_marks_router_rates_not_applicable():
    summary = BenchmarkReport(
        [TaskMetrics(task_id="t", agent_type="regex", trial=0)],
        model="qwen",
    ).summary_by_agent()["regex"]

    assert summary["router_decisions"] == 0
    assert summary["router_acceptance_rate"] is None
    assert summary["router_use_rate"] is None
    assert summary["router_first_accuracy"] is None


def test_trace_record_keeps_raw_routing_and_timing_for_rescoring():
    raw = _result()
    benchmark_result = SimpleNamespace(
        task_id="download",
        agent_type="regex",
        trial=2,
        session_id="session",
        raw_result=raw,
    )

    trace = _trace_record(benchmark_result)

    assert trace["routing"] == raw["routing"]
    assert trace["timing"] == raw["timing"]
    assert trace["token_usage"] == raw["token_usage"]
    assert trace["total_time_ms"] == 400
