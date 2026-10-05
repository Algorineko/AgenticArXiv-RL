"""Offline tests for the optional Jev routing experiment driver."""

import importlib.util
import json
from pathlib import Path
from unittest import mock

import requests
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "jev_route_smoke.py"
SPEC = importlib.util.spec_from_file_location("jev_route_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
jev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(jev)


class _Response:
    status_code = 200
    ok = True
    text = ""

    def json(self):
        return {"ok": True}


class _Session:
    def __init__(self, error=None):
        self.error = error
        self.closed = False

    def request(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        return _Response()

    def close(self):
        self.closed = True


def test_ssl_failure_discards_session_and_retries():
    sessions = [
        _Session(requests.exceptions.SSLError("temporary EOF")),
        _Session(),
    ]
    delays = []
    client = jev._ResilientClient(
        attempts=2,
        session_factory=lambda: sessions.pop(0),
        sleep=delays.append,
    )
    first = client._session
    with mock.patch.object(jev.random, "uniform", return_value=0.0):
        payload, elapsed = client.request_json(
            "GET", "https://example.invalid", headers={}
        )
    assert payload == {"ok": True}
    assert elapsed >= 0
    assert first.closed is True
    assert delays == [0.75]
    client.close()


def test_checkpoint_resume_keeps_only_matching_tasks(tmp_path):
    tasks = [
        {"id": "a", "expected_tools": ["search_arxiv_papers"]},
        {"id": "b", "expected_tools": ["download_arxiv_pdf"]},
    ]
    output = tmp_path / "checkpoint.json"
    output.write_text(
        json.dumps(
            {
                "results": [
                    {"task_id": "a", "expected": "search_arxiv_papers"},
                    {"task_id": "b", "expected": "wrong_old_answer"},
                    {"task_id": "removed", "expected": "FINISH"},
                ]
            }
        ),
        encoding="utf-8",
    )
    existing = json.loads(output.read_text(encoding="utf-8"))
    existing["config"] = {"experiment_version": "v2"}
    output.write_text(json.dumps(existing), encoding="utf-8")
    rows = jev._resume_rows(
        output,
        tasks,
        enabled=True,
        config={"experiment_version": "v2"},
    )
    assert [row["task_id"] for row in rows] == ["a"]


def test_expanded_split_sizes_are_pinned():
    assert len(jev._select_tasks("expanded", "all", 0)) == 81
    assert len(jev._select_tasks("expanded", "dev", 0)) == 8
    assert len(jev._select_tasks("expanded", "iid_test", 0)) == 18
    assert len(jev._select_tasks("expanded", "ood_test", 0)) == 4


def test_all_selection_uses_the_same_frozen_manifest_as_named_splits() -> None:
    """New task families must not silently enter the frozen all-task probe."""
    payload = json.loads(jev.SPLIT_PATH.read_text(encoding="utf-8"))
    expected = {task_id for ids in payload["split"].values() for task_id in ids}
    selected = jev._select_tasks("expanded", "all", 0)
    assert {task["id"] for task in selected} == expected


def test_explicit_v7_manifest_selects_its_all_and_dev_tasks() -> None:
    """A newer benchmark can be selected explicitly without changing defaults."""
    manifest = SCRIPT.parents[1] / "data" / "splits" / "v7_86.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    selected = jev._select_tasks("expanded", "all", 0, split_path=manifest)
    assert len(selected) == payload["task_count"] == 86
    expected = {task_id for ids in payload["split"].values() for task_id in ids}
    assert {task["id"] for task in selected} == expected
    selected_dev = jev._select_tasks("expanded", "dev", 0, split_path=manifest)
    assert [task["id"] for task in selected_dev] == payload["split"]["dev"]
    assert len(jev._select_tasks("expanded", "all", 0)) == 81


def test_all_limit_applies_after_frozen_membership_filter(tmp_path: Path) -> None:
    """The limit must not admit a new task placed before frozen members."""
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps({"split": {"train": ["first"], "dev": ["second"]}}),
        encoding="utf-8",
    )
    with mock.patch.object(
        jev, "get_expanded_tasks",
        return_value=[{"id": "new"}, {"id": "second"}, {"id": "first"}],
    ):
        selected = jev._select_tasks("expanded", "all", 1, split_path=manifest)
    assert [task["id"] for task in selected] == ["second"]


def test_all_selection_rejects_unknown_manifest_task_ids(tmp_path: Path) -> None:
    """A broken frozen manifest must fail rather than quietly shrink the probe."""
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps({"split": {"train": ["missing-task-id"]}}), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing-task-id"):
        jev._select_tasks("expanded", "all", 0, split_path=manifest)


def test_cli_dry_run_records_explicit_manifest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI previews expose the chosen manifest and its task count without an API."""
    manifest = SCRIPT.parents[1] / "data" / "splits" / "v7_86.json"
    with mock.patch.object(
        jev.sys, "argv",
        [str(SCRIPT), "--dry-run", "--split-file", str(manifest)],
    ), mock.patch.object(jev, "_resolve_snapshot", return_value=None):
        jev.main()
    preview = json.loads(capsys.readouterr().out)
    assert preview["task_count"] == 86
    assert Path(preview["split_file"]) == manifest.resolve()


def test_visible_context_includes_runtime_candidate_titles_without_gold():
    task = {
        "id": "custom",
        "task": "download the matching paper",
        "setup": [
            {
                "name": "get_recently_submitted_cs_papers",
                "args": {"aspect": "AI", "days": 7, "max_results": 5},
            }
        ],
        "expected_tools": ["download_arxiv_pdf"],
    }
    context = jev._visible_context_for(
        task,
        [{"id": "2608.14528v1", "title": "Learning State Across Sessions"}],
    )
    assert "2608.14528v1" in context
    assert "Learning State Across Sessions" in context
    assert "download_arxiv_pdf" not in context


def test_finish_criteria_explicitly_covers_invalid_requests():
    finish = jev._tool_criteria()["FINISH"]
    assert "invalid" in finish
    assert "outside the available candidate list" in finish
    assert "unsupported" in finish
