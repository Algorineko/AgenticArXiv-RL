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
