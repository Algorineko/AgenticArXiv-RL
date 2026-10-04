"""Tests for schema-safe deterministic arguments after an external route."""

from routing.arguments import resolve_routed_arguments


def test_extracts_explicit_keyword_query_window_and_limit():
    result = resolve_routed_arguments(
        tool_name="search_arxiv_papers",
        task=(
            "按关键词检索 arXiv：all:agentic reinforcement learning，"
            "最近 30 天，最多返回 5 篇论文"
        ),
        history="",
    )
    assert result.status == "resolved"
    assert result.args == {
        "query": "all:agentic reinforcement learning",
        "days": 30,
        "max_results": 5,
    }


def test_extracts_direct_arxiv_id_without_requiring_candidate_membership():
    result = resolve_routed_arguments(
        tool_name="download_arxiv_pdf",
        task="下载 2608.14528v1 这篇论文",
        history="",
    )
    assert result.status == "resolved"
    assert result.args == {"ref": "2608.14528v1"}


def test_extracts_translation_reference_and_service():
    result = resolve_routed_arguments(
        tool_name="translate_arxiv_pdf",
        task="用 google 服务翻译第1篇论文",
        history="",
    )
    assert result.args == {"ref": 1, "service": "google"}


def test_active_reference_is_explicit_null():
    result = resolve_routed_arguments(
        tool_name="get_paper_cache_status",
        task="查一下我刚下载的那篇论文的缓存状态",
        history="",
    )
    assert result.args == {"ref": None}


def test_non_positive_reference_is_blocked_before_execution():
    result = resolve_routed_arguments(
        tool_name="download_arxiv_pdf",
        task="下载第0篇论文",
        history="",
    )
    assert result.status == "blocked"
    assert result.args == {}
    assert result.reason == "non_positive_reference"


def test_multi_reference_advances_then_reports_completion():
    task = "第1篇和第2篇，分别查一下缓存状态"
    first = resolve_routed_arguments(
        tool_name="get_paper_cache_status", task=task, history=""
    )
    assert first.args == {"ref": 1}
    history_one = (
        'Thought: x\nAction: {"name":"get_paper_cache_status",'
        '"args":{"ref":1}}\nObservation: ok'
    )
    second = resolve_routed_arguments(
        tool_name="get_paper_cache_status", task=task, history=history_one
    )
    assert second.args == {"ref": 2}
    history_two = history_one + (
        '\n\nThought: x\nAction: {"name":"get_paper_cache_status",'
        '"args":{"ref":2}}\nObservation: ok'
    )
    complete = resolve_routed_arguments(
        tool_name="get_paper_cache_status", task=task, history=history_two
    )
    assert complete.status == "complete"
