"""Small local-checkpoint smoke test; preserves full actions and delivered contexts."""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "AgenticArxiv"))
    os.environ["STORE_BACKEND"] = "memory"
    for name in ("PDF_RAW_PATH", "PDF_TRANSLATED_PATH", "PDF_FIGURES_PATH"):
        os.environ[name] = str(args.output / name.lower())

    import pymupdf
    import torch
    from agents.agent_engine import ReActAgent
    from agents.prompt_templates import build_visible_setup_context
    from agents.side_effects import LocalSideEffectManager
    from benchmark.metrics import extract_metrics, is_strict_success
    from models.schemas import Paper, TranslateAsset
    from models.store import store, use_memory_store
    from rl.env import MockArxivEnv
    from tools.bootstrap import register_all_tools
    from utils.llm_client import TransformersLLMClient

    register_all_tools()
    paper = Paper(id="2601.00001v1", title="Deterministic Agent Replay")
    mono = args.output / "fixture-mono.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_textbox(
            pymupdf.Rect(40, 40, 550, 780),
            "译文识别码 ORCHID-731。\n方法采用确定性快照回放。\n实验结果显示相同输入产生相同输出。",
            fontname="china-s", fontsize=11,
        )
        doc.save(mono)
    use_memory_store()
    store.set_last_papers("record", [paper])
    store.upsert_translate_asset(TranslateAsset(
        paper_id=paper.id, status="READY", output_mono_path=str(mono)))
    snapshot = args.output / "snapshot.json"
    recorder = MockArxivEnv(snapshot, mode="record")
    recorder.execute_tool("get_translated_paper_content", {"session_id": "record", "ref": 1})
    recorder.snapshot["get_recently_submitted_cs_papers"] = {
        MockArxivEnv._make_key({"aspect": "AI", "max_results": 1}): {
            "args": {"aspect": "AI", "max_results": 1}, "result": [paper.model_dump()]}}
    recorder.save_snapshot()

    started = time.monotonic()
    client = TransformersLLMClient(str(args.model), device="auto", dtype="bfloat16", seed=42)
    load_seconds = time.monotonic() - started
    torch.cuda.reset_peak_memory_stats()

    class RecordingClient:
        model_name = str(args.model)

        def __init__(self):
            self.messages = []

        def chat_completions(self, **kwargs):
            self.messages.append(kwargs["messages"])
            return client.chat_completions(**kwargs)

    search = {"name": "get_recently_submitted_cs_papers", "args": {"aspect": "AI", "max_results": 1}}
    translate = {"name": "translate_arxiv_pdf", "args": {"ref": 1}}
    results = []
    for cached in (True, False):
        use_memory_store(reset=True)
        environment = MockArxivEnv(snapshot, mode="replay")
        session_id = "model_cached" if cached else "model_chain"
        setup = [search, translate] if cached else [search]
        for action in setup:
            environment.execute_tool(action["name"], {**action["args"], "session_id": session_id})
        mode = "cached" if cached else "chain"
        task = {
            "id": mode,
            "task": ("读取已完成翻译的第1篇论文的中文译文首段，最多128个字符。" if cached else
                     "先翻译第1篇论文，再读取该论文的中文译文首段，最多128个字符。"),
            "setup": setup,
            "expected_tools": ["get_translated_paper_content"] if cached else
                              ["translate_arxiv_pdf", "get_translated_paper_content"],
            "expected_tool_args": [{"ref": 1, "offset": 0, "max_chars": 128}] if cached else
                                  [{"ref": 1}, {"ref": 1, "offset": 0, "max_chars": 128}],
            "expected_paper_ids": [paper.id] * (1 if cached else 2),
        }
        recording = RecordingClient()
        agent = ReActAgent(recording, side_effect_mgr=LocalSideEffectManager(), env=environment,
                           max_iterations=4, llm_extra={"temperature": 0.0})
        start = time.monotonic()
        raw = agent.run(task["task"], session_id=session_id, agent_model=str(args.model),
                        initial_history=build_visible_setup_context(task))
        metrics = extract_metrics(task, raw, "regex", 0, session_id)
        results.append({
            "case": mode, "strict_success": is_strict_success(metrics),
            "seconds": time.monotonic() - start, "metrics": metrics.to_dict(),
            "history": raw["history"], "reply": raw.get("reply"),
            "delivered_contexts": recording.messages,
        })
        print(mode, "strict_success=", is_strict_success(metrics), "tools=", metrics.tool_call_sequence, flush=True)
    report = {
        "model": str(args.model), "dtype": str(client.model.dtype),
        "device": str(next(client.model.parameters()).device), "load_seconds": load_seconds,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30, "cases": results,
    }
    destination = args.output / "model-smoke.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Report:", destination, flush=True)


if __name__ == "__main__":
    main()
