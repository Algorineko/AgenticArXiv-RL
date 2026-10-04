"""Real-PDF coverage for translated text, continuation, and readiness."""

import hashlib
import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.schemas import Paper, PdfAsset, TranslateAsset
from models.store_memory import MemoryStore
from tools import paper_content_tool as reader


def write_pdf(path, text):
    with pymupdf.open() as doc:
        page = doc.new_page()
        assert page.insert_textbox(
            pymupdf.Rect(40, 40, 550, 780), text,
            fontname="china-s", fontsize=11,
        ) >= 0
        doc.save(path)


@pytest.fixture
def translated_store(tmp_path, monkeypatch):
    backend = MemoryStore(make_dirs=False)
    monkeypatch.setattr("models.store._backend", backend)
    papers = [Paper(id="2601.00001v1", title="First paper"),
              Paper(id="2601.00002v1", title="Second paper")]
    backend.set_last_papers("s", papers)
    backend.set_last_active_paper_id("s", papers[1].id)
    paths = []
    for index, paper in enumerate(papers, 1):
        raw = tmp_path / f"raw{index}.pdf"
        mono = tmp_path / f"mono{index}.pdf"
        write_pdf(raw, "Abstract\nSOURCE_ONLY original abstract.")
        write_pdf(mono, f"TRANSLATED_{index}\n中文译文第{index}篇。\n" +
                  "\n".join(f"第{line}行：正文内容用于检查连续读取。" for line in range(25)))
        backend.upsert_pdf_asset(PdfAsset(
            paper_id=paper.id, local_path=str(raw), status="READY"))
        backend.upsert_translate_asset(TranslateAsset(
            paper_id=paper.id, output_mono_path=str(mono), status="READY"))
        paths.append(mono)
    return backend, papers, paths


def test_reads_translation_and_continues_without_gaps(translated_store):
    assert hasattr(reader, "get_translated_paper_content"), "translated reader is missing"
    backend, papers, paths = translated_store
    chunks, offset = [], 0
    while True:
        result = reader.get_translated_paper_content("s", 1, offset, 37)
        assert result["paper_id"] == papers[0].id
        assert result["source"] == "translated"
        assert result["source_sha256"] == hashlib.sha256(paths[0].read_bytes()).hexdigest()
        assert result["offset"] == offset
        assert 0 < len(result["content"]) <= 37
        assert result["next_offset"] == offset + len(result["content"])
        chunks.append(result["content"])
        offset = result["next_offset"]
        if not result["has_more"]:
            break
    text = "".join(chunks)
    assert text == reader._pdf_to_text(str(paths[0]))
    assert "TRANSLATED_1" in text and "中文译文" in text
    assert "SOURCE_ONLY" not in text and "TRANSLATED_2" not in text
    assert backend.get_last_active_paper_id("s") == papers[0].id


@pytest.mark.parametrize("ref", [2, "第2篇", "2601.00002v1", "Second", None])
def test_reference_forms_select_the_same_translation(translated_store, ref):
    _, papers, _ = translated_store
    result = reader.get_translated_paper_content("s", ref)
    assert result["paper_id"] == papers[1].id
    assert "TRANSLATED_2" in result["content"]


@pytest.mark.parametrize("state", ["missing", "TRANSLATING", "FAILED", "missing_file", "image_only"])
def test_unavailable_translation_fails_without_changing_reference(translated_store, state):
    backend, papers, paths = translated_store
    if state == "missing":
        backend.delete_translate_asset(papers[0].id)
    elif state == "missing_file":
        paths[0].unlink()
    elif state == "image_only":
        paths[0].unlink()
        with pymupdf.open() as doc:
            doc.new_page()
            doc.save(paths[0])
    else:
        backend.update_translate_asset(papers[0].id, status=state)
    with pytest.raises(ValueError):
        reader.get_translated_paper_content("s", 1)
    assert backend.get_last_active_paper_id("s") == papers[1].id


@pytest.mark.parametrize("args", [dict(ref=" "), dict(offset=-1), dict(offset=10_000),
                                  dict(max_chars=0), dict(max_chars=4001)])
def test_invalid_read_parameters_fail(translated_store, args):
    with pytest.raises(ValueError):
        reader.get_translated_paper_content(session_id="s", **args)


def test_record_once_replays_any_chunk_without_pdf(translated_store, tmp_path):
    from tools.bootstrap import register_all_tools
    from rl.env import MockArxivEnv

    register_all_tools()
    backend, papers, paths = translated_store
    snapshot = tmp_path / "snapshot.json"
    recorder = MockArxivEnv(snapshot, mode="record")
    expected = recorder.execute_tool("get_translated_paper_content", {"session_id": "s", "ref": 1})
    recorder.save_snapshot()
    assert len(recorder.snapshot["get_translated_paper_content"]) == 1
    paths[0].unlink()
    backend.reset()
    backend.set_last_papers("s", papers)
    replay = MockArxivEnv(snapshot, mode="replay")
    with pytest.raises(ValueError, match="not ready"):
        replay.execute_tool("get_translated_paper_content", {"session_id": "s", "ref": 1})
    replay.execute_tool("translate_arxiv_pdf", {"session_id": "s", "ref": 1})
    result = replay.execute_tool("get_translated_paper_content", {"session_id": "s", "ref": 1})
    assert result == expected
    continued = replay.execute_tool("get_translated_paper_content", {
        "session_id": "s", "ref": 1, "offset": 20, "max_chars": 13})
    assert continued["content"] == expected["content"][20:33]
    assert replay.stats["real_calls"] == 0


def test_multiturn_readiness_is_local_and_resettable(translated_store, tmp_path):
    import json
    from tools.bootstrap import register_all_tools
    from rl.env import MockArxivEnv
    from rl.multiturn_env import AgenticArxivMultiTurnEnv

    register_all_tools()
    _, papers, _ = translated_store
    snapshot = tmp_path / "snapshot.json"
    recorder = MockArxivEnv(snapshot, mode="record")
    expected = recorder.execute_tool("get_translated_paper_content", {"session_id": "s", "ref": 1})
    recorder.save_snapshot()
    data = json.loads(snapshot.read_text(encoding="utf-8"))
    data["get_recently_submitted_cs_papers"] = {
        MockArxivEnv._make_key({"aspect": "AI", "max_results": 2}): {
            "args": {"aspect": "AI"}, "result": [paper.model_dump() for paper in papers]}}
    snapshot.write_text(json.dumps(data), encoding="utf-8")
    first, second = AgenticArxivMultiTurnEnv(snapshot), AgenticArxivMultiTurnEnv(snapshot)
    for environment in (first, second):
        environment.reset()
        environment.get_recently_submitted_cs_papers("AI", max_results=2)
    with pytest.raises(ValueError, match="not ready"):
        first.get_translated_paper_content(1)
    first.translate_arxiv_pdf(1)
    assert first.get_translated_paper_content(None) == expected
    with pytest.raises(ValueError, match="not ready"):
        second.get_translated_paper_content(1)
    first.reset()
    first.get_recently_submitted_cs_papers("AI", max_results=2)
    with pytest.raises(ValueError, match="not ready"):
        first.get_translated_paper_content(1)


def test_backfill_preserves_old_entries_and_session_state(translated_store, tmp_path):
    import json
    from scripts.backfill_translated_content import backfill_translated_content

    backend, papers, paths = translated_store
    old = {"get_recently_submitted_cs_papers": {"existing": {"result": []}}}
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(old), encoding="utf-8")
    assert backfill_translated_content(snapshot, {papers[0].id: paths[0]}) == 1
    data = json.loads(snapshot.read_text(encoding="utf-8"))
    assert data["get_recently_submitted_cs_papers"] == old["get_recently_submitted_cs_papers"]
    assert len(data["get_translated_paper_content"]) == 1
    assert backend.get_last_active_paper_id("s") == papers[1].id


def test_budgeted_observation_retains_structure_and_visible_cursor(translated_store):
    import json

    assert hasattr(reader, "format_translated_observation"), "budgeted serializer is missing"
    result = reader.get_translated_paper_content("s", 1, max_chars=4000)
    text = reader.format_translated_observation(result, max_chars=330)
    visible = json.loads(text)
    assert len(text) <= 330
    assert 0 < len(visible["content"]) < len(result["content"])
    assert visible["next_offset"] == len(visible["content"])
    assert visible["has_more"]
    following = reader.get_translated_paper_content("s", 1, visible["next_offset"])
    assert visible["content"] + following["content"] == result["content"]
    with pytest.raises(ValueError, match="budget"):
        reader.format_translated_observation(result, max_chars=10)


@pytest.mark.parametrize("mutation", [{"content": ""}, {"source": "original"},
                                     {"source_sha256": ""}, {"next_offset": -1}])
def test_invalid_translated_result_fails_metrics_and_reward(translated_store, mutation):
    import json
    from benchmark.metrics import extract_metrics, is_strict_success
    from rl.reward import RewardCalculator

    payload = reader.get_translated_paper_content("s", 1)
    task = {"id": "translated", "expected_tools": ["translate_arxiv_pdf", "get_translated_paper_content"],
            "expected_tool_args": [{"ref": 1}, {"ref": 1}],
            "expected_paper_ids": [payload["paper_id"], payload["paper_id"]]}
    payload.update(mutation)
    result = {"history": [
        {"thought": "translate", "action": json.dumps({"name": "translate_arxiv_pdf", "args": {"ref": 1}}),
         "observation": json.dumps({"paper_id": payload["paper_id"], "status": "READY"})},
        {"thought": "read", "action": json.dumps({"name": "get_translated_paper_content", "args": {"ref": 1}}),
         "observation": json.dumps(payload, ensure_ascii=False)},
        {"thought": "done", "action": "FINISH", "observation": "任务完成"},
    ]}
    assert not is_strict_success(extract_metrics(task, result, "regex", 0))
    reward, _ = RewardCalculator().compute_reward(task, result)
    assert reward <= -0.75


def test_grpo_delivers_complete_translation_and_masks_it(translated_store):
    import json
    from rl.grpo_reward import make_multiturn_rollout_func

    payload = reader.get_translated_paper_content("s", 1, max_chars=4000)
    action = 'Thought: read\nAction: {"name":"get_translated_paper_content","args":{"ref":1}}'

    class Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(character) for character in text]}

        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(token) for token in ids)

    class Environment:
        def reset(self):
            pass

        def get_translated_paper_content(self, ref):
            return payload

    class Trainer:
        processing_class = Tokenizer()
        max_completion_length = 650

        def __init__(self):
            self.inputs = []

        def _generate_single_turn(self, prompts, images, fields):
            self.inputs.append(prompts)
            text = action if len(self.inputs) == 1 else "Thought: done\nAction: FINISH"
            return [[ord(character) for character in text]], None, {}

    trainer = Trainer()
    output = make_multiturn_rollout_func(Environment, max_turns=2)(["task"], trainer)
    observation = output["trajectory_results"][0]["history"][0]["observation"]
    visible = json.loads(observation)
    assert visible["next_offset"] == len(visible["content"])
    assert "TRANSLATED_1" in visible["content"]
    assert observation in trainer.processing_class.decode(trainer.inputs[1][0])
    delivered = "".join(chr(token) for token, mask in zip(output["completion_ids"][0], output["env_mask"][0]) if mask == 0)
    assert observation in delivered


def test_opt_in_tasks_generate_grounded_expert_data(translated_store, tmp_path):
    import json

    module_path = Path(__file__).resolve().parents[1] / "benchmark" / "translated_tasks.py"
    assert module_path.exists(), "opt-in translated task set is missing"
    from benchmark.translated_tasks import get_translated_specs
    from rl.env import MockArxivEnv
    from scripts.generate_sft_data import generate_deterministic_trajectories
    from tools.bootstrap import register_all_tools
    from agents.prompt_templates import format_tool_description
    from tools.tool_registry import registry

    register_all_tools()
    _, papers, _ = translated_store
    snapshot = tmp_path / "snapshot.json"
    recorder = MockArxivEnv(snapshot, mode="record")
    recorder.execute_tool("get_translated_paper_content", {"session_id": "s", "ref": 1})
    recorder.snapshot["get_recently_submitted_cs_papers"] = {
        MockArxivEnv._make_key({"aspect": "AI", "max_results": 2}): {
            "args": {"aspect": "AI", "max_results": 2},
            "result": [paper.model_dump() for paper in papers]}}
    recorder.save_snapshot()
    specs = get_translated_specs(snapshot)
    assert len(specs) == 6 and len({spec.id for spec in specs}) == 6
    assert all(spec.requires_offline for spec in specs)
    rows = generate_deterministic_trajectories(
        specs, MockArxivEnv(snapshot, mode="replay"),
        format_tool_description(registry.list_tools()), source_split="translated-test",
        replay_translation=True,
    )
    assert rows
    assert "TRANSLATED_1" in json.dumps(rows, ensure_ascii=False)
    assert all(row["source_split"] == "translated-test" for row in rows)
