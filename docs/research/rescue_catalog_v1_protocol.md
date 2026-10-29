# Frozen protocol: real-metadata development screen v1

Status: fixed before any model run on this catalog. The scripted reference passed 72/72; 44 targeted software checks passed. This is a small development screen to choose the next experiment, not a confirmatory result or an RSI method.

## Catalog

`artifacts/rescue_catalog_v1/manifest.json` SHA256:
`aeaf9f1aaf14262ea575a5ba043d50d9fad7ad8e19a8f8af48dfe9ec2735ea73`.

Record four arXiv Atom feeds (AI, LG, CL, CV), each requesting 24 metadata records in the preceding seven-day window. Keep the raw responses, queries, timestamps, and hashes. Deterministically remove paper-ID overlaps (ignoring version suffixes) in AI/LG/CL/CV priority order. The frozen replay includes real titles, IDs, abstracts, and authors; PDF downloads and translation remain local stubs. No PDF understanding, translation quality, live-network performance or scientific reasoning claims follow.

The 72 tasks use four paper-disjoint splits:

| Split | Category | Tasks | Use |
|---|---|---:|---|
| train | AI | 12 | Reserved; no training this phase |
| search_dev | LG | 24 | This four-arm development comparison |
| promotion_val | CL | 12 | Reserved; no model scores this phase |
| final_test | CV | 24 | Reserved new compositions; no model scores this phase |

Train/dev/promotion share six primitive families. Final holds six new compositions built from the same primitive tools. Paper identity and composition are controlled; category and split are confounded, parameter variants share a family, and all tasks have been constructed with known tools. This is not a broad out-of-domain or capability benchmark. No split can be presented as fully independent simply by counting 24 parameter instances. The final model-evaluation path is disabled in the development entry point.

## Arms and order

1. Qwen/Qwen2.5-1.5B-Instruct, original prompt.
2. Released AgenticArXiv GRPO, original prompt.
3. Same instruction-tuned base, fixed completion-check prompt.
4. Same GRPO, same fixed completion-check prompt.

The instruction-tuned base is a healthier initialization comparator, not a raw pretrained language model. The released GRPO checkpoint also includes prior SFT; cross-model differences do not isolate a causal RL effect. The completion check is the exact manual string already frozen during the infrastructure smoke phase. It is not tuned on this new catalog, a search method, or self-improvement.

All arms: one physical GPU (default 1), bfloat16, greedy, seed 45, one trial per task, max 384 new tokens per action, at most six ReAct iterations per task, four CPU calculation threads, 35% per-process GPU allocator bound, minimum 16000MiB free at launch, 900 seconds per arm. Record model weight hashes, code hashes, original histories, and full task coverage. Execute sequentially. If an arm fails due to an execution issue, stop the batch and keep its artifacts; any engineering rerun uses a new output root with an explicit failure note.

## Preflight and readouts

Run the scripted gold policy through the actual parser/tool/metric path: must pass all 72. Run an always-FINISH negative control on search_dev: must pass 0/24. No test labels or expected actions are available to the model client; only the scripted reference uses them.

Primary descriptive readouts are task-level strict counts, paired transitions for the prompt change, false-finish counts, and per-family success. Also report prompt/completion tokens and total rollout time. Inference cost includes added prompt tokens. All artifacts including failure traces are retained.

There is no preselected claim that a method must win. A positive change on this tiny developer-built split is only a reason to design an independent larger test. Do not pool greedy repeats as independent evidence, optimize prompts after seeing the results and call them held-out, infer causal RL gains from the base-vs-pipeline difference, or unlock the final split to chase a favorable headline.

## Resource decision

If static prompting repairs the released model only partially, prioritize healthy-start/data/Harness diagnosis before weight training. If the base or all arms hit the ceiling, do not start RL on this suite: increase difficulty in a new version while preserving v1. If all arms hit the floor, inspect oracle validity, representation and budgets before adding seeds. Start the RL-by-Harness factorial only after a separate benchmark and matched training setup can answer an actual hypothesis.
