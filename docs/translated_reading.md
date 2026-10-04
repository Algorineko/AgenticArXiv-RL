# Reading completed translations

`get_translated_paper_content(ref=1, offset=0, max_chars=1000)` reads a completed mono PDF registered as a READY `TranslateAsset`. It reuses the existing PyMuPDF text extraction and normalization, returns at most 4000 characters per call, and never starts a translator. Pending/failed translations, missing PDFs, PDFs without text, blank references, and invalid offsets are explicit tool errors.

The observation contains `paper_id`, `status`, `source="translated"`, `source_sha256`, `content`, `offset`, `next_offset`, and `has_more`. Continue with `next_offset` while `has_more` is true. An offset is measured in Unicode characters of the normalized extracted text, not PDF bytes or model tokens. The SHA256 identifies the mono PDF; it records provenance rather than measuring translation quality.

Live Agent translation remains asynchronous. A request to submit a translation can finish after enqueueing; a request to read it requires READY content. A pending task must be described as pending. The first release does not wait for asynchronous jobs or claim that the policy has learned the new action.

## Offline snapshot backfill

Create a JSON manifest mapping actual paper IDs to already completed mono PDFs; relative paths are resolved against the manifest's directory:

```json
{"2601.00001v1": "pdf_translated/2601.00001v1-mono.pdf"}
```

Run from the repository root:

```bash
python scripts/backfill_translated_content.py --snapshot data/mock_arxiv_snapshot.json --translations translated-pdfs.json
```

This operation does not download papers, invoke translation, or change existing search entries. One complete normalized document is stored per paper, so arbitrary legal offsets/budgets replay without precomputing every chunk. Reads in ordinary offline Agents and multi-turn environments require translation state in the current rollout. A stored document alone does not satisfy that precondition.

## Opt-in evaluation and expert data

```bash
python -m AgenticArxiv.benchmark.run_benchmark --task-set translated --snapshot data/mock_arxiv_snapshot.json --offline --backend transformers --model /path/to/owner-grpo-checkpoint --local-device cuda --local-dtype bfloat16 --agents regex --repeat 1
```

The task set derives cached-read and translate-then-read tasks, with 64/128/256-character budgets, only for translated papers also present in a recorded search pool. It evaluates bounded first-fragment retrieval, not complete-paper understanding. Historical task sets and frozen splits are unchanged.

For expert data, freeze a separate versioned split JSON using the existing `{"version": 1, "split": {"train": [], "dev": [], "iid_test": [], "ood_test": []}}` format, filling each list with chosen generated task IDs. Keep every variant of one paper on the same side of the train/evaluation boundary. Use separate paper manifests for model adaptation and evaluation.

```bash
python scripts/generate_sft_data.py --task_set translated --snapshot data/mock_arxiv_snapshot.json --split data/splits/translated_v1.json:train --output data/sft/translated_train.jsonl
```

Only an explicit `PATH:train` split is accepted. Expert actions are executed against the snapshot; invalid or ungrounded results cannot produce accepted SFT rows. GRPO/OPD budget the new result before serialization, preserve a complete observation and the visible continuation, and keep environment tokens out of policy loss. Metrics and reward share one result validator for paper identity, translated provenance, nonempty text, and cursor consistency.

## Verification on 2026-10-04

- `python -X utf8 -m pytest AgenticArxiv/tests/ -q -ra`: **753 passed, 4 skipped, 567 subtests passed**. The skips are the three pre-existing manual network/API smoke cases and one audit case needing the default offline snapshot. TRL emits its existing experimental GKD warning. On Windows, use UTF-8 mode: the unmodified suite's default-encoding file reads otherwise fail in three cases under GBK.
- CI fatal-error flake8 checks (`E9,F63,F7,F82`) passed for the application and changed scripts.
- The translated expert-data CLI executed a frozen synthetic fixture task and produced two validated SFT rows. This verifies data plumbing; no model training was performed.
- Owner checkpoint `Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO` loaded locally as BF16 on an RTX 4060 Laptop 8 GB. Load time was 13.30 seconds; peak CUDA allocated memory during inference was 3.56 GiB.
- Two zero-shot cases were tried once each: cached translated first-fragment reading, and translate-then-read. **Strict success: 0/2.** The model selected old tools and supplied `max_results` to translation instead of calling the new reader. These are recorded model failures, not evidence that the policy learned the new action. Fine-tuning and translation-quality evaluation remain future work.

Reproduce the small model check with `python -X utf8 scripts/eval_translated_reading.py --model /path/to/owner-grpo-checkpoint --output /path/to/smoke-output`. The script creates its own Chinese PDF fixture, snapshot, and artifact paths; `model-smoke.json` preserves actions, tool results, every delivered model context, timings, and memory statistics. A completed evaluation can report failed cases; inspect `strict_success` rather than treating exit status as policy success.
