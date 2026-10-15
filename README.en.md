<p align="center">
  <a href="README.md">🇨🇳 中文</a> | <a href="README.en.md">🇬🇧 English</a> | <a href="README.es-ES.md">🇪🇸 Español</a>
</p>

# AgenticArXiv-RL — Agentic RL Training Environment

> **An Agentic RL training environment built on a ReAct agent + arXiv tools**
> The default training mainline is SFT → GRPO, with an explicit optional SFT → DPO → GRPO path; inference and evaluation use text ReAct with verifiable rewards (RLVR).
> The ultimate goal: **a lightweight multimodal paper assistant for on-device deployment** — an end-to-end paper-reading model spanning retrieval, download, translation, summarization, and figure understanding.

<p align="center">
  <img src="imgs/AgenticArXiv-RL.jpg" alt="AgenticArXiv-RL project overview" width="800"/>
</p>

---

## 🎯 Project Positioning

This project turns arXiv paper retrieval/download/translation/interpretation tasks into a **trainable reinforcement learning environment**, focused on:

1. **Verifiable Reward**: rule-based rewards (tool-call accuracy, task completion, parsing errors, etc.), no human annotation required
2. **Direct mainline and optional preference stage**: default SFT → GRPO or explicit SFT → DPO → GRPO, with QLoRA by default and explicit full-parameter training
3. **Lightweight engineering**: pure Python + JSONL storage, no database, no frontend — focused on offline training

**Non-goals**: a production-grade arXiv app, a Web UI, a real-time translation service (the original web app has been removed from this repo; see the original [AgenticArXiv](https://github.com/Algorineko/AgenticArXiv)).

---

## Quick Start

Run installation commands from the repository root, using `.venv`. Installation has no frontend or database steps. Once installed, `agentic-arxiv` works from any directory without `PYTHONPATH`.

```bash
git clone https://github.com/Algorineko/AgenticArXiv-RL.git
cd AgenticArXiv-RL
python3 -m venv .venv
source .venv/bin/activate
pip install .

agentic-arxiv evaluate rollout --help
agentic-arxiv evaluate run --help
agentic-arxiv run --help
```

The base install supports external APIs and historical-record reading without local-model or training dependencies. Install `pip install '.[local]'` for local inference/evaluation and DPO data generation, or `pip install '.[train]'` for SFT/DPO/GRPO training. For API inference, explicitly set `LLM_BASE_URL`, `LLM_API_KEY`, and `MODEL`; never commit credentials or a real `.env`. Local training/evaluation does not require an API key. Install `pip install '.[translate]'` for real PDF translation; no asynchronous application service is required.

Global `--workspace` defaults to the current directory. Relative configuration, model, data, snapshot, and output paths are workspace-relative; absolute paths are unchanged. To use existing research inputs from elsewhere, place `agentic-arxiv --workspace /actual/workspace` before the subcommand. Missing inputs do not fall back to the source tree. `.env` is read from the workspace only for an actual operation; help does not read credentials or create logs. See the [architecture migration guide](docs/architecture_migration.md) for old-to-new entry mappings.

```bash
# A separate future run with an existing local model and snapshot; not executed here
agentic-arxiv evaluate rollout \
  --task-id search_01 --output-dir traces/react_messages \
  --model outputs/sft_qlora/final --offline \
  --snapshot data/mock_arxiv_snapshot.json --max-iterations 6
```

Ordinary `run` defaults to API/live. `evaluate rollout` defaults to offline; explicit `--no-offline` enables live tools. Tool mode and model backend are separate choices; see the [runtime contract](docs/react_runtime.md).

Task execution needs an existing model or available API. Offline tools require an existing snapshot and do not fetch missing inputs. Installation/help checks are not task execution; this engineering change did not call external APIs, train models, or evaluate model performance.

---

## 📚 Core Concepts

### MDP Design

| Dimension | Definition |
|------|------|
| **State** | Task description + conversation history + tool results |
| **Action** | 10 tools (see below) + FINISH |
| **Reward** | Five-component multi-granular verifiable reward (format / tool / argument / process / outcome) |
| **Transition** | `execute_tool(action) → observation` (`MockArxivEnv` replays offline snapshots — deterministic and reproducible) |

### Action Space (10 Tools)

1. `get_recently_submitted_cs_papers(aspect, days, max_results)` — browse by subfield + time window
2. `search_arxiv_papers(query, max_results, days=None)` — keyword/title/author search
3. `download_arxiv_pdf(ref)` — download a PDF
4. `translate_arxiv_pdf(ref)` — translate a PDF (pdf2zh)
5. `get_paper_cache_status(ref)` — query cache status
6. `get_paper_content(ref, section=None)` — read the abstract or a chosen section (deterministic extraction)
7. `summarize_paper(ref, style, max_words)` — env-side summary (tldr / structured / bullet)
8. `extract_paper_figures(ref)` — extract figure files and captions
9. `analyze_figure(ref, figure_no, question=None)` — figure analysis: the env calls a local VLM to read the figure (multimodal environment enabled)
10. `get_translated_content(ref, page=1)` — read the translation one page at a time (the translated Chinese PDF, deterministic extraction; page 1 usually holds the title and abstract)

> These are common business parameters, not the full registry schema. `session_id` is injected at execution, never written into policy Actions or SFT targets.

> **Historical ability under the original protocol, not rerun under the new protocol.** The full "search → download → read → summarize → extract figures → analyze figures" interpretation loop is now wired end to end: figure analysis is performed env-side by the project's post-trained [FigureQA VLM](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA), which also records the snapshots, and on the policy side the [SFT-T5](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5) checkpoint has learned that four-step chain.
> **Admission rule**: a bigger action space is not automatically a better one — a new tool is admitted only if it "unlocks a new class of tasks", not merely because it "might be useful". See [Toolset Evolution Design](docs/toolset_evolution.md) for the full design decision chain.

### Verifiable Reward (Five Components)

**Multi-granular verifiable reward** (`src/agentic_arxiv/scoring/reward.py`). Five components normalized to `[-1, 1]`, plus diagnostic components and non-compensable failure caps:

| Component | Default weight | Signal |
|------|:---:|------|
| `format` | 1 | Whether each step's action is a valid JSON tool call or a terminator |
| `tool` | 3 | **Order-aware LCS-F1** between the predicted and expected tool sequences |
| `argument` | 2 | Argument key recall × exact-value accuracy |
| `process` | 1 | Bonus for valid steps − penalties for parse failures / execution failures / redundant calls |
| `outcome` | 3 | Correct completion +1, wrong-path completion +0.25, forced stop −0.5, error −1 |
| `result_quality` / `efficiency` | Diagnostic gate | Whether the observation genuinely succeeded; severe redundancy caps the reward |

- **Curriculum learning**: the source default still scales `tool`/`argument`/`outcome` by 1/3 for the first 30 steps (`RewardCalculator.schedule`); cleanup did not change it to 0. Historical findings report format 0.983 at the SFT start and no distinguishable difference between curriculum arms. A future explicit `--reward-curriculum-steps 0` override must be recorded in the manifest, not described as preserving the default.
- **Non-compensable failures**: parse failures, execution failures, and fake FINISH can only earn negative rewards and cannot be offset by format points; when `analyze_figure` lacks a valid answer, the total reward is capped at −0.75.
- All rewards are **rule-based and verifiable** (RLVR); every trajectory logs `reward_components` in detail for auditing. See the [multi-granular reward doc](docs/multigranular_rl.md) for the exact scoring rules.

### Episode Isolation

Each episode owns its mutable session and artifact directory. Snapshots and prefetched assets are shared read-only; reset/close only clean episode-owned paths, never other episodes or user files. The default full-task budget is 6 turns, including FINISH. A request to finish is not proof of task success.

---

## Shared Message Protocol

The only active protocol is `react-runtime-v2` / `react-messages-v3`: an initial user message establishes task/tools/actual setup, followed by assistant turns and user environment Observations explicitly marked as data. SFT supervises only the current completion. GRPO preserves sampled IDs, policy mask=1 and environment/inserted-boundary mask=0, checks the token prefix strictly, preserves existing EOS, and does not expand by G again in rollout.

Old/unknown data or trained artifacts cannot be silently labeled compatible, and skipping a manifest check does not bypass protocol validation. Reports separate the actual execution protocol from the source model's real training metadata, with same_protocol/cross_protocol/unknown relations; unknown is not a comparable reproduction. Real tokenizer prefixes, gradients, and model ability were not verified here. See the [runtime contract](docs/react_runtime.md).

## Training Mainline: Default SFT → GRPO, Optional DPO

These are commands for a separate future run, not engineering acceptance steps. They need new-protocol SFT data and its manifest, base-model/training resources, and an existing offline snapshot. None of these inputs were generated, and no training was run in this change.

First install `pip install '.[train]'`. The package resource `agentic_arxiv/resources/mainline.json` adds a separate DPO section while preserving the SFT/GRPO defaults and original source-baseline subtrees. Older configurations missing the whole DPO section receive recorded defaults in memory, without file changes. An explicit `--config` overrides the package default, and explicit CLI arguments take precedence.

```bash
# Default QLoRA: GRPO directly loads the SFT adapter and base-model metadata
agentic-arxiv train sft \
  --data data/sft/sft_react_messages_v3_train_mix.jsonl \
  --output-dir outputs/sft_qlora
agentic-arxiv train grpo \
  --model outputs/sft_qlora/final \
  --task-set expanded --split data/splits/v7_86.json:train \
  --snapshot data/mock_arxiv_snapshot.json --max-turns 6

# Explicit full-parameter path, not an adapter directory
agentic-arxiv train sft \
  --data data/sft/sft_react_messages_v3_train_mix.jsonl \
  --no-qlora --output-dir outputs/sft_full
agentic-arxiv train grpo \
  --model outputs/sft_full/final \
  --no-qlora --task-set expanded --split data/splits/v7_86.json:train \
  --snapshot data/mock_arxiv_snapshot.json --max-turns 6
```

Formal GRPO explicitly binds a task set and a `FILE:NAME` training split. `--smoke` selects the 8 smoke tasks; the 86-task set is never implicitly all used for training. The historical 33 tasks in `v6_grpo_train.json:rl_train` are not the 55 tasks in `v7_86.json:train`.

The small mainline configuration preserves explicit source values, defaults, and conditional argument forwarding. Unset loss, importance-sampling, or reward-scaling fields remain delegated to the training library, not guessed defaults or claims of historical recipe equivalence. Missing inputs/metadata fail explicitly, without switching models or APIs. See the [mainline guide](docs/rl_building.md) for future data generation, augmentation, mixing, resource overrides, messages/masks, and compatibility.

Guards retain length checks, variance diagnostics, Canary, SFT parse-rate threshold ≥ 0.3, GRPO mean-reward threshold ≥ −0.2, precision handling, and logging checks. Validation uses complete episodes, not first-action correctness; real-model threshold success was not verified here.

### Explicit Optional DPO (Examples Not Executed)

Without this stage, GRPO still defaults to `outputs/sft_qlora/final`; it never scans DPO directories or starts intermediate stages. Generation requires a current-protocol local SFT artifact, an existing snapshot, and an explicit versioned expanded `:train` split, never held-out data, bare train, or historical `rl_train` subsets:

```bash
agentic-arxiv data dpo generate \
  --model outputs/sft_qlora/final \
  --task-set expanded --split data/splits/v7_86.json:train \
  --snapshot data/mock_arxiv_snapshot.json \
  --output data/dpo/dpo_react_messages_v3_train.jsonl
agentic-arxiv train dpo --model outputs/sft_qlora/final \
  --data data/dpo/dpo_react_messages_v3_train.jsonl --inspect-only
agentic-arxiv train dpo --model outputs/sft_qlora/final \
  --data data/dpo/dpo_react_messages_v3_train.jsonl \
  --snapshot data/mock_arxiv_snapshot.json --output-dir outputs/dpo
agentic-arxiv train grpo --model outputs/dpo/final \
  --task-set expanded --split data/splits/v7_86.json:train \
  --snapshot data/mock_arxiv_snapshot.json --max-turns 6
```

Pairs require exactly equal complete actual messages, including Thought/setup/Observation text, and two real, untruncated current CALL/FINISH responses. Missing steps never become fabricated FINISH. Only current assistant targets enter the preference objective. `--inspect-only` audits hashes, train membership, provenance, protocol, real tokenizer prefixes/targets and full lengths without loading policy weights or training; a local tokenizer is still required. Overlength or incompatible templates fail rather than truncate. At most one pair per task is selected; trajectory rewards rank preferences, not establish single-action causality. Reports use requested tasks and requested unordered combinations as separate denominators. Low yield only warns; zero pairs retain traces/failure reports and exit nonzero, without a successful training package.

Independent policy/reference models start from the same complete SFT, including its adapter; the reference stays frozen and shares no mutable parameters. Two models cost extra memory; `--no-qlora` selects full-parameter training explicitly, never a different reference. Epochs 3, lr `5e-6`, beta 0.1 and QLoRA batch 1/accumulation 8 are design defaults, not verified historical recipes. Default full-episode Canary validation reports reward, strict success, parsing, false finishes, termination, coverage and protocol relation. Failure retains model/report and exits nonzero; `--no-verify` records skipped, not PASS. Training completion, smoke validation and improvement over SFT are separate claims; GRPO never starts automatically. See the [mainline guide](docs/rl_building.md). No generation, training or model evaluation ran here; real tokenizer integration, reference invariance after optimization, GPU/DDP and benefits remain unverified.

OPD, PPO, GRPO variant selection, self-evolution orchestration, independent VLM post-training, Jev, MCP/Skill CLI, Web/HTTP, MySQL, and SSE entries remain retired. DPO restoration does not restore old scripts/imports or upgrade old data. Existing figure-tool inference, snapshot construction, T5 backfill, bad-case scoring, and external LLM APIs remain.

### Historical Release Results (original protocol, not rerun)

The release information and numbers below retain their original experimental scope. They are not evidence of current model ability under the new message protocol. New-protocol training rejects incompatible inputs; cross-protocol evaluation must be labeled explicitly.

> Released: [AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT) (full fine-tune of Qwen2.5-1.5B, 2 epochs / 2628 expert trajectories, loss 0.079; covers the first 8 tools only).

> Released: [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO) (33 signal tasks, fully offline snapshot replay). Offline evaluation (seed 45 / 3 repeats, strict success rate pass³), SFT → GRPO: **rl_train 0.081→0.636, dev 0.000→0.375, iid_test 0.056→0.444, ood_test 0.000→0.500**.

> Released: [AgenticArXiv-RL-Qwen3-VL-4B-FigureQA](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA) (451 figures / 73 papers, split by paper to prevent leakage; on the 94 held-out figures ROUGE-1 0.105→0.126 and ROUGE-L 0.081→0.102).

---

## 📦 Released Models

> **This is a historical release list.** Retired variants are not trainable entries, and old checkpoint behavior under the new context is not guaranteed.

| Model | Base | Notes |
|------|------|------|
| [AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT) | Qwen2.5-1.5B | Stage 1 full-parameter SFT (first 8 tools) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO) | Qwen2.5-1.5B | Stage 3 GRPO (four-split evaluation above) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-GSPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-GSPO) | Qwen2.5-1.5B | GRPO + sequence-level importance sampling (tied with the baseline on all four splits, see the paradigm comparison doc) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DrGRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DrGRPO) | Qwen2.5-1.5B | Unbiased GRPO variant (rl_train 0.657 / dev 0.375 / iid 0.444 / ood 0.500, pass³) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DAPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO-DAPO) | Qwen2.5-1.5B | DAPO objective without dynamic sampling (rl_train 0.667 / dev 0.500 / iid 0.481 / ood 0.500, pass³; best of the three variants) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5) | Qwen2.5-1.5B | SFT + the `analyze_figure` four-step chain (rl_train 0.485 / dev 0.250 / iid 0.278 / ood 0.250, pass³) |
| [AgenticArXiv-RL-Qwen3-VL-4B-FigureQA](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA) | Qwen3-VL-4B | env-side figure-analysis VLM |

All models are hosted on ModelScope (`Algorineko/AgenticArXiv-RL-*`); each model card includes the offline evaluation table and an honest statement of limitations.

---

## 🧪 Task Suite and Evaluation

- **Smoke set** (`src/agentic_arxiv/tasks/tasks.py`): 8 tasks (search / download / translate / cache / composite)
- **Expanded set** (`src/agentic_arxiv/tasks/tasks_expanded.py`): 86 tasks across fourteen template categories (search / keyword_search / ref_form / composite / state / optional / constraint / long_chain / infeasible / paper_reading / paper_summary / figure_extraction / figure_analysis / **translation_reading**), enabled with `agentic-arxiv evaluate run --task-set expanded`. Both task sets go through the same `TaskSpec`: `expected_tools` / `expected_tool_args` are derived from the same `steps`, so two hand-written lists can never drift apart.
- **Splits**: `data/splits/v7_86.json` has train 55 / dev 8 / iid_test 19 / ood_test 4. The historical `v6_grpo_train.json:rl_train` has 33 tasks and is a different source. Future signal selection writes a new file using declared train members only; dev/iid/ood never enter training, and no rates were remeasured here.
- **Evaluation metrics (historical numbers)**: `pass^k` reliability (tau-bench convention), `false_finish` (the degenerate `always_finish` policy scores 91.5% vs `reference` 0%), `ref_score` (compares the `paper_id` rather than the `ref` spelling), and cost normalized by successful runs.
- **Discrimination gate (historical statistics)** (current entry `agentic-arxiv evaluate baselines`): deterministic degenerate policies must clear per-category thresholds — "always search cs.AI" scores 0.833→0.446 on retrieval categories, and "calling tools when none are needed" moves +0.165→−0.235.
- **Bad-case replay** (`agentic-arxiv evaluate badcases replay` + `eval/eval_cases.jsonl`, 17 cases): failed trajectories are frozen into permanent regression cases; replay runs only the scorers, and `pytest` is the gate; `hack/*` records reward-hacking patterns with threshold assertions, doubling as a reward-hacking case library.

```bash
agentic-arxiv evaluate run \
  --backend transformers --model outputs/grpo/final \
  --task-set expanded --split data/splits/v7_86.json:iid_test \
  --offline --snapshot data/mock_arxiv_snapshot.json \
  --repeat 3 --output artifacts/eval_react_messages_iid
```

---

## Dependencies

The sole dependency declaration is [pyproject.toml](pyproject.toml); see the [architecture migration guide](docs/architecture_migration.md) for installation and retired entry mappings. Base `pip install .` includes `arxiv`, `requests`, `python-dotenv`, `loguru`, `pydantic`, PyMuPDF, and Pillow, without the training stack.

- `pip install '.[local]'`: torch, transformers, accelerate, and peft for local inference, default LoRA adapters, and optional model tools, without trainers.
- `pip install '.[train]'`: the same local constraints, plus trl, datasets, bitsandbytes, and packaging.
- `pip install '.[translate]'`: pdf2zh for real PDF translation.
- `pip install '.[logging]'`: tensorboard for training curves; wandb remains separately installed and explicitly enabled.
- `pip install '.[dev]'`: pytest, coverage, flake8, build, and metadata checks, without implicitly installing train.

Library constraints and interface guards remain unchanged; Fire is retired. Figure/summary backends need explicit local models. Missing optional dependencies fail when used, without shrinking the complete 10-tool registry. Help does not load real tokenizers, models, or trainers, and does not prove their integration behavior. Web/database/alternate-agent dependencies are not installation options.

---

## 🔗 Related Resources

- **Docs**: [Toolset Evolution Design](docs/toolset_evolution.md) · [Multi-granular Reward](docs/multigranular_rl.md) · [Metric Statistics](docs/metric_stats.md) · [Roadmap Research Notes](docs/roadmap_notes.md) · [TRL documentation](https://huggingface.co/docs/trl/)
- **Method papers**: RLVR; DPO (Stanford, 2023); [On-Policy Distillation (Thinking Machines, 2025)](https://thinkingmachines.ai/blog/on-policy-distillation/); [GKD (arXiv:2306.13649)](https://arxiv.org/abs/2306.13649)
- **Original project**: [AgenticArXiv](https://github.com/Algorineko/AgenticArXiv) (web app edition, FastAPI + Vue3 + MySQL)

---

## 🤝 Contributing

Issues and PRs are welcome! Workflow: Fork → feature branch → `pip install -e '.[dev]'` → `pytest tests/` → `feat:`-prefixed commits → PR. Model-free training regressions also need train dependencies; see [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

MIT License

---

## 🙋 FAQ

**Q: How does this differ from the original AgenticArXiv?** The original is a production-grade web app (FastAPI + Vue3 + MySQL, three agent architectures); this project is a pure Python + JSONL RL training environment with single-policy text ReAct, default SFT → GRPO, optional DPO and verifiable rewards.

**Q: Why GRPO instead of PPO?** GRPO needs no value model (lower memory footprint), suits small models at the 1.5B scale, and is simple to implement and debug; PPO is a better fit for production-grade training of large models.

**Q: Why don't summary/figure-analysis quality enter the reward?** Quality scoring would require an LLM-as-judge, which brings non-determinism and a wider hacking surface. By design, interpretation is reduced to a **tool-call decision problem** (when to call, what to call, and with which arguments) — fully rule-judgeable and reproducible, which is a prerequisite for RLVR.

---

## Historical Roadmap (not current work items)

> **The following is the original direction record, not a current execution or implementation plan.** Numbers, old flags, and paths retain their historical context. Retired entries cannot be run; old pending experiments are not acceptance tasks for this change. Current future commands and unexecuted scope are in the [mainline guide](docs/rl_building.md).


> Full rationale and introduction paths in [docs/roadmap_notes.md](docs/roadmap_notes.md). Feel free to pick up an item (see 🤝 Contributing).

### P0 — On-Device Multimodal Paper Assistant (ultimate goal)

- [ ] **Multimodality on the policy side**: move the VLM from the env side into the policy (observations carry figures), reusing the TRL vision-language path already validated in `train_vlm_figure_qa.py`; candidate bases Qwen3-VL-4B / Qwen2.5-VL-2B (on-device tier)
- [ ] **End-to-end reading chain**: retrieval → download → translation → summarization → figure analysis closed-loop within a single model, optimized for on-device inference (quantization + 2-4B scale)
- [x] **Translated text into the context**: added `get_translated_content`, which reads the pdf2zh translation page by page, deterministically; it replays from offline snapshots (recorded with `build_snapshot --translate-max-ref N`) and is graded by the reading-tool result-quality rule

### P1 — Next-Generation Agentic RL Algorithms

- [x] **GSPO / Dr.GRPO wiring**: `--importance_sampling_level sequence` (sequence-level importance sampling, GSPO) and `--loss_type dr_grpo` (unbiased length normalization) are wired into `train_grpo.py`, orthogonal to and composable with the `--dapo` preset
- [x] **Train and compare the new methods**: GSPO / Dr.GRPO / DAPO variants trained on the frozen splits and released (shared recipe aligned with the baseline, seed 42 / 60 steps; offline pass³, GRPO baseline: rl_train 0.636 / dev 0.375 / iid 0.444 / ood 0.500):
  - **GSPO** (sequence-level IS): 0.636 / 0.375 / 0.444 / 0.500 — identical to the baseline on every split (different weights, diverging reward curves, same converged behavior)
  - **Dr.GRPO** (unbiased objective): 0.657 / 0.375 / 0.444 / 0.500 — slightly ahead on rl_train; best ood tool accuracy 0.67 / false-finish 0.33
  - **DAPO** (clip-higher 0.28 + truncation masking + β=0; dynamic sampling infeasible on the 33-task pool, therefore off): **0.667 / 0.500 / 0.481 / 0.500** — best on rl_train / dev / iid
  - Details and implementation notes in `docs/rl_paradigm_comparison.md`
- [ ] **SAO-style asynchronous training**: port verl `fully_async_policy` / AReaL, starting with skip-observation masking and DIS two-sided clipping ([arXiv:2607.07508](https://arxiv.org/abs/2607.07508), official code not released)
- [ ] **Cross-step credit assignment**: GiGPO / ARPO-style group-relative cross-step advantages, to ease the trajectory-level sparse signal on long-horizon chains (in-house research item)

### P2 — Jev-Style Discriminative Decision Component

- [ ] **Jev-style discriminative head**: non-autoregressive structured decisions (choice/score, at ~1/400 the cost of an LLM); candidate landing spots: ① a `result_quality` discriminative head replacing the rules ② a reward-component router (replacing the fixed curriculum schedule) ③ structured decoding for tool selection (a closed enumeration is naturally suited to a discriminative model). Constraint: deterministic inference, or diagnostics-only, to keep RLVR reproducible

### P3 — Bounded RSI (restricted self-evolution)

- [ ] **Self-evolving data loop**: held-out evaluation exposes weaknesses → the bad-case library grows automatically → parametric derivation of targeted data → retrain → freeze a new held-out set (reuses the existing evaluation/data/training pipelines; all prerequisites are in place). Acceptance: the four-split pass³ must not regress across rounds, and the bad-case library only grows (the four-step loop of [arXiv:2609.11873](https://arxiv.org/abs/2609.11873)). **Orchestrator landed** (`AgenticArxiv/rl/self_evolve.py`: diagnose → mine → select → synthesize → freeze → gate, pure JSON, no model needed; `generate_parametric_sft_data.py --only-parents` for targeted data); a real training round with the four-split comparison is still pending (see [docs/roadmap_notes.md](docs/roadmap_notes.md))

### ⛔ Environment Blockers

- [ ] **vLLM-accelerated sampling**: TRL 0.29 requires vLLM 0.10.2–0.12.0, and the platform-customized 0.6.2 installed on this machine makes GRPOTrainer fail at import. This is an environment dependency rather than a code change — waiting for the platform to ship a matching build.

---

**Start your Agentic RL training journey!** 🚀
