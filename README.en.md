<p align="center">
  <a href="README.md">🇨🇳 中文</a> | <a href="README.en.md">🇬🇧 English</a> | <a href="README.es-ES.md">🇪🇸 Español</a>
</p>

# AgenticArXiv-RL — Agentic RL Training Environment

> **An Agentic RL training environment built on ReAct Agent + arXiv tools**  
> Supports a progressive SFT/DPO/GRPO/PPO training pipeline, plus an optional OPD (on-policy distillation) route, for research on LLM Agent reinforcement learning

<p align="center">
  <img src="imgs/AgenticArXiv-RL.jpg" alt="AgenticArXiv-RL Project Overview" width="800"/>
</p>

---

## 🎯 Project Overview

This project transforms arXiv paper retrieval/download/translation tasks into a **trainable reinforcement learning environment**, with a focus on:

1. **Verifiable Reward**: Rule-based rewards (tool call accuracy, task completion, parsing errors, etc.) — no human annotation required
2. **Progressive Training**: SFT (Supervised Fine-Tuning) → DPO (Direct Preference Optimization) → GRPO (Group Relative Policy Optimization) → PPO (Proximal Policy Optimization)
3. **Lightweight Engineering**: Pure Python + JSONL storage, no MySQL/FastAPI/frontend — focused on offline training

**Non-goals**: Production-grade arXiv application, Web UI, real-time translation service (these features have been archived to `archive/`)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- LLM API (OpenAI API format compatible, e.g., Claude, Gemini, Qwen, etc.)
- `.venv` virtual environment

### 1️⃣ Clone the Project

```bash
git clone https://github.com/Algorineko/AgenticArXiv-RL.git
cd AgenticArXiv-RL
```

All following commands are intended to be run from the repository root
`AgenticArXiv-RL/`.

### 2️⃣ Environment Setup

**Create a virtual environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

**Install dependencies**:
```bash
pip install -r AgenticArxiv/requirements.txt
```

**Configure LLM API**:
```bash
cat > AgenticArxiv/.env << 'EOF'
# LLM API Configuration
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
MODEL=gpt-4-turbo

# Optional: PDF path configuration
PDF_RAW_PATH=./output/pdf_raw
PDF_TRANSLATED_PATH=./output/pdf_translated
EOF
```

### 3️⃣ Test Rollout

```bash
python -m AgenticArxiv.rl.rollout search_01 traces/train/
```

**Example output (reward varies with the actual trajectory and is in `[-1, 1]`)**:
```
✅ Task search_01 rollout completed
   Reward: 1.00
   Metrics: task_completed=True, tool_call_accurate=True
   Trajectory saved to: traces/train/rollout_20260621_150000.jsonl
```

---

## 📚 Core Concepts

### MDP Design

| Dimension | Definition |
|-----------|------------|
| **State** | Task description + conversation history + tool results |
| **Action** | 8 tools (arXiv browsing/keyword search/download/translate/cache query/paper reading/summarization/figure extraction) + FINISH |
| **Reward** | Five-component multi-granular verifiable reward (format / tool / argument / process / outcome, see below) |
| **Transition** | `execute_tool(action) → observation` (`MockArxivEnv` offline snapshot replay, deterministic & reproducible) |

### Action Space (8 Tools)

1. `get_recently_submitted_cs_papers(aspect, days, max_results)` — Search arXiv papers
2. `download_arxiv_pdf(ref, session_id)` — Download PDF
3. `translate_arxiv_pdf(ref, session_id)` — Translate PDF
4. `get_paper_cache_status(ref, session_id)` — Query cache status
5. `search_arxiv_papers(query, max_results, days=None)` — Search by keyword, title, or author
6. `get_paper_content(ref, session_id, section=None)` — Read an abstract or a selected section from a downloaded paper
7. `summarize_paper(ref, style, max_words)` — Env-side summary of a downloaded paper (tldr / structured / bullet)
8. `extract_paper_figures(ref)` — Extract a downloaded paper's figure files and captions

> The "search → download → read → summarize → extract figures" interpretation loop is now complete. Figure *analysis* (T5, which needs an env-side VLM) is still a design proposal; see "🧰 Toolset Evolution Design" below.

### Verifiable Reward Components

**Multi-granular verifiable reward** (`rl/reward.py`, inspired by LLM-TIR's hierarchical reward). The original five components keep their curriculum weights; two additional diagnostics (`result_quality` and `efficiency`) provide non-compensable safety gates:

| Component | Default weight | Signal |
|-----------|:---:|--------|
| `format` | 1 | Fraction of steps whose action is a valid JSON tool call or a terminal token |
| `tool` (tool sequence) | 3 | Order-aware **LCS-F1** between predicted and expected tool sequences (`benchmark/metrics.py` strict matching) |
| `argument` (parameters) | 2 | Parameter-key recall × exact value accuracy; automatically skipped when a task has no `expected_tool_args` |
| `process` | 1 | Valid-step credit minus parse/execution-failure and unnecessary-call penalties |
| `outcome` | 3 | Correct completion +1, completion with a wrong tool path +0.25, forced stop −0.5, error −1 |

| `result_quality` | diagnostic / gate | Checks whether tool observations represent a grounded success; empty, fallback, and failed observations are penalized |
| `efficiency` | diagnostic / gate | Normalizes redundant calls and failed retries against the task's expected steps |

**Curriculum learning**: for the first 30 training steps the `tool` / `argument` / `outcome` weights are scaled by 1/3 (learn the ReAct protocol first, semantics later); full weights apply from step 30 onward (`RewardCalculator.schedule`).

**Non-compensable failures**: parse errors, tool execution failures, and false FINISH actions cannot be rescued by format points. Single-step GRPO rollouts mark framework-generated terminal steps as `forced_finish`, so they do not receive the full terminal bonus.

**Key point**: All rewards are **verifiable** (rule-based), requiring no human annotation — corresponding to the RLVR (Reinforcement Learning with Verifiable Reward) framework. Every trajectory records a `reward_components` breakdown for auditing and reward-hacking analysis.

### Rollout isolation

`rl/sandbox.py` provides `RolloutSandbox`, which records the initial environment, in-memory store, and explicitly configured artifact roots, then restores that baseline before the next trajectory. Multi-turn GRPO, regular rollouts, and offline benchmarks use the same reset contract so search results, download/translation state, files, and counters cannot leak across trajectories.

---

## 🛠️ Training Pipeline (SFT → DPO → GRPO / OPD)

### Stage 1: SFT (Supervised Fine-Tuning)

**Goal**: Teach the model basic tool calling formats.

**Steps**:
1. Generate expert demonstrations:
   ```bash
   python scripts/generate_sft_data.py
   ```
2. Train:
   ```bash
   python -m AgenticArxiv.rl.train_sft
   ```
3. Output: `./outputs/sft/final` model

> **Released weights** (Qwen2.5-1.5B full-parameter SFT, 2 epochs on 2628 parameterised expert trajectories, loss 0.079):
> [🤗 ModelScope · AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT)
> This checkpoint covers only the first 8 tools. `analyze_figure` (T5) was added later; T5 expert data can now be generated optionally, but this released model was not trained on it.

**Data format** (`data/sft/sft_train.jsonl`):
```json
{
  "messages": [
    {"role": "system", "content": "You are an arXiv paper retrieval Agent..."},
    {"role": "user", "content": "Search for AI papers from the last 7 days"},
    {"role": "assistant", "content": "{\"name\":\"get_recently_submitted_cs_papers\",\"arguments\":{...}}"}
  ]
}
```

---

### Stage 2: DPO (Direct Preference Optimization)

**Goal**: Train the model to prefer correct tool selections and reject incorrect routing.

**Steps**:
1. Use the SFT model for rollout, collecting chosen/rejected pairs:
   ```bash
   python scripts/generate_dpo_data.py
   ```

This command samples the local Hugging Face model at `outputs/sft/final` and
does not require `LLM_API_KEY`. Common optional arguments:

```bash
python scripts/generate_dpo_data.py \
  --model outputs/sft/final \
  --num_rollouts_per_task 8 \
  --temperature 0.8 \
  --seed 42
```

When `data/mock_arxiv_snapshot.json` exists, tool calls are replayed offline for
reproducibility; otherwise generation falls back to the live tools. Only
trajectories whose reward gap exceeds `--min_reward_gap` (default 0.05) **and**
whose first tool action differs are paired. A run that yields very few pairs
warns loudly rather than quietly training DPO on a handful of samples — a
well-fitted SFT model is deterministic per task, so even 6 samples per task can
contain no within-group disagreement.
2. Train:
   ```bash
   python -m AgenticArxiv.rl.train_dpo
   ```
3. Output: `./outputs/dpo/final` model

**Data format** (`data/dpo/dpo_train.jsonl`):
```json
{
  "prompt": "Search for AI papers from the last 7 days",
  "chosen": "{\"name\":\"get_recently_submitted_cs_papers\",...}",
  "rejected": "{\"name\":\"download_arxiv_pdf\",...}"
}
```

---

### Stage 3: GRPO (Group Relative Policy Optimization)

**Goal**: Online training with verifiable rewards, no value model needed.

**Steps**:
```bash
python -m AgenticArxiv.rl.train_grpo
```

**Output**: `./outputs/grpo/final` model

**Advantages**:
- No reward model required (DPO's drawback: unable to learn online)
- No value model required (PPO's drawback: high VRAM overhead)
- Suitable for small models (e.g., Qwen2.5-1.5B)

**Multi-turn rollout and reward scoring** (`rl/grpo_reward.py`): the current policy generates a ReAct action at every turn, an independent `MockArxivEnv` executes it, and the observation is appended back to context until completion or `--max_turns`. Assistant tokens participate in GRPO loss while environment tokens are excluded with `env_mask=0`; the complete trajectory is scored by the shared five-component `RewardCalculator`.

```bash
python -m AgenticArxiv.rl.build_snapshot
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --max_turns 4

# Record training curves (same flag across all stages)
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --report_to tensorboard
tensorboard --logdir outputs/grpo/logs

# DAPO-family options (defaults change nothing, so historical runs stay comparable)
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --dapo
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --epsilon_high 0.2   # clip-higher only

# Multi-GPU (DDP; launch accelerate with the interpreter that has the training deps)
accelerate launch --config_file configs/accelerate/ddp_2gpu.yaml \
  -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --no-qlora
```

**Multi-GPU**: `configs/accelerate/` ships both a DDP and an FSDP config. Training scripts detect the `LOCAL_RANK` that `accelerate launch` exports and stop pinning a single GPU (pinning would hide the other ranks' devices); single-process runs are still pinned, which keeps the Trainer from falling back to `DataParallel` and segfaulting. QLoRA's `device_map={"": 0}` is incompatible with multi-process launches, so that combination fails loudly instead of silently training on one card. FSDP requires `torch>=2.6`.

**Training curves** (`rl/observability.py`): `--report_to` accepts `none` / `auto` / `tensorboard` / `wandb` (comma-separated), shared by all five stages (SFT / DPO / GRPO / OPD / PPO). On top of TRL's built-in metrics it logs:

| Metric group | Contents | Why log it separately |
|---|---|---|
| `reward_components/*` | format / tool / argument / process / outcome | each bounded to `[-1,1]` and independent of the weights |
| `reward_weights/*` | current curriculum weights | the curriculum suppresses tool/argument/outcome weights for the first 30 steps, so total reward alone reads "weights opened up" as "policy regressed" |
| `rollout/*` | turns / finished / parse_error_rate / tool_error_rate | when reward drops, separates "policy regressed" from "never learned to stop, burns every turn up to max_turns" |

### Training Quality Guards (auto-verification)

The training pipeline bakes in several layers of automatic validation that turn "silent training failures" into loud errors:

- **Generation-length check**: before training, verifies that `max_completion_length` fits the canonical action, preventing zero-gradient idle runs where the model can never emit a complete action
- **Zero-variance guard**: aborts training (with fix suggestions) when the in-group reward variance stays at 0, meaning all advantages are zero (`RewardVarianceGuard`)
- **Canary evaluation**: samples on fixed tasks every N steps and early-stops if performance degrades below the threshold repeatedly (`CanaryCallback`)
- **Stage verification**: each stage's output model must pass a minimum quality threshold — SFT parse rate ≥ 0.3, DPO mean reward ≥ −0.3, GRPO mean reward ≥ −0.2 (`StageVerifier`, skip with `--no-verify`)
- **Adaptive mixed precision**: bf16 first on CUDA, falling back to fp16, disabled on CPU / MPS (`rl/precision.py`)
- **Logging-backend validation**: a backend named in `--report_to` that is not installed fails before the model is loaded, so you never finish a run only to find no curves (`rl/observability.py`)

### Stage 3': OPD (On-Policy Distillation, optional, swappable with GRPO)

**Goal**: distill the student's ReAct action ability from a strong teacher's per-token signal, with no reward involved during training.

**Steps**:
```bash
python -m AgenticArxiv.rl.train_opd --model outputs/sft/final --teacher Qwen/Qwen2.5-7B-Instruct

# Multi-turn Agentic OPD: execute Action → Observation → Action
python -m AgenticArxiv.rl.train_opd --model outputs/sft/final \
  --teacher Qwen/Qwen2.5-7B-Instruct --max_turns 4 \
  --snapshot data/mock_arxiv_snapshot.json
```

**Output**: `./outputs/opd/final` model

**Positioning**: OPD is a **full training paradigm**, not a single trick — the student samples on-policy from task prompts, the teacher scores every token with its logprobs, and the loss is the reverse KL `D_KL(π_student ‖ π_teacher)` (mode-seeking). Relative to GRPO it is the "teacher route" vs the "reward route":

| Dimension | GRPO | OPD |
|---|---|---|
| Learning signal | five-component verifiable reward (sparse, trajectory-level) | teacher per-token logprobs (dense) |
| Extra model | none | teacher model (needs local weights for logprobs; external APIs don't expose per-token distributions) |
| Performance ceiling | can explore beyond the teacher | converges to teacher behavior |
| Best when | verifiable reward exists, no teacher | strong teacher available, want to save RL exploration cost |

OPD can also be used as a trick: warm start before RL, teacher-KL regularization inside RL, or verl's PG-OPD (reverse KL treated as a reward for policy gradient).

**Single-turn and multi-turn modes**: `--max_turns 1` (the default) preserves TRL GKDTrainer's original single prompt/completion behavior. `--max_turns > 1` enables Agentic OPD: every trajectory receives an independent `MockArxivEnv`, tool calls are executed, and the real Observation is inserted into the next turn. Prompt and Observation labels are `-100`; reverse-KL is computed only on student-generated assistant tokens. The first version requires identical teacher/student token-to-id vocabularies and fails before training if they differ. Student and teacher share GPU memory (a 1.5B student plus 7B teacher use roughly 17GB for bf16 weights alone; gradients, optimizer states, and logits need additional memory, so a smaller teacher or student is recommended on a 32GB GPU). Verified on trl 1.5.1 (`trl.experimental.gkd`); `beta=1.0` selects reverse-KL (locked by a numeric unit test in `tests/test_opd.py`). Canary and stage verification remain shared with GRPO: OPD itself uses no reward, but its output model must still pass environment checks.

**Offline comparison**: run SFT, `--max_turns 1`, `--max_turns > 1`, and GRPO with the same snapshot and task set, then compare the common `StageVerifier` results in each output directory's `final/verification_report.json`. Those rewards are evaluation-only and never enter the OPD loss.

---

### Stage 4: PPO (Proximal Policy Optimization) — ⚠️ unavailable with the current dependency set

**Goal**: online fine-tuning of policy and value network with a standard Actor-Critic architecture.

**Status**: TRL deprecated and then **removed** the classic PPO trainer — `PPOTrainer` / `PPOConfig` / `AutoModelForCausalLMWithValueHead` do not exist in the `trl>=0.28.0` that `requirements.txt` allows, so `train_ppo.py` cannot even be imported. It now says so explicitly at import time instead of raising an ImportError with no visible cause.

Making PPO run means rewriting the script against the `trl.experimental.ppo` API (different config fields and dataset format) or pinning trl below 0.9. In this project PPO is a learning-oriented comparison: with a verifiable reward, GRPO is the route; with a strong teacher, OPD is — both cheaper in VRAM than PPO.

```bash
# exits immediately with the explanation above
python -m AgenticArxiv.rl.train_ppo --model outputs/grpo/final
```

**Output** (once rewritten): `./outputs/ppo/final` model

---

## 📂 Directory Structure

```
AgenticArXiv-RL/
├─ AgenticArxiv/                     # ⭐ Python package (RL training environment)
│  ├─ agents/                        # Agent core
│  │  ├─ base_agent.py              # Generic ReAct loop
│  │  ├─ agent_engine.py            # ReActAgent (RL policy)
│  │  ├─ context_manager.py
│  │  ├─ prompt_templates.py
│  │  └─ side_effects.py           # Decoupled side effects interface
│  ├─ tools/                         # Tool layer (action space)
│  │  ├─ tool_registry.py          # Tool registry
│  │  ├─ arxiv_tool.py             # arXiv category/keyword search
│  │  ├─ pdf_download_tool.py      # PDF download
│  │  ├─ pdf_translate_tool.py     # PDF translation
│  │  ├─ cache_status_tool.py      # Cache query
│  │  ├─ paper_content_tool.py     # Deterministic paper-content reading (T2)
│  │  ├─ paper_summary_tool.py     # Env-side paper summarization (T3)
│  │  └─ paper_figures_tool.py     # Deterministic figure extraction (T4)
│  ├─ benchmark/                     # ⭐ Verifiable Reward source
│  │  ├─ metrics.py               # TaskMetrics, strict tool-sequence & argument matching
│  │  ├─ tasks.py                 # BENCHMARK_TASKS (8 smoke tasks)
│  │  ├─ tasks_expanded.py        # Expanded task set (81 tasks, 13 template families)
│  │  ├─ task_spec.py             # TaskSpec: expected_tools / expected_tool_args derived from steps
│  │  ├─ badcases.py              # Bad-case verdicts & replay
│  │  ├─ splits.py                # Template-level train/iid/ood split
│  │  ├─ baselines.py · run_baselines.py  # Deterministic degenerate-policy baselines (per-category gates)
│  │  ├─ runner.py · run_benchmark.py     # Benchmark executor & CLI entry
│  │  └─ report.py                # Metrics report
│  ├─ rl/                            # ⭐ RL core
│  │  ├─ train_sft.py              # ⭐ SFT training
│  │  ├─ train_dpo.py              # ⭐ DPO training
│  │  ├─ train_grpo.py             # ⭐ GRPO training (with training guards)
│  │  ├─ train_ppo.py              # ⭐ PPO training (Actor-Critic)
│  │  ├─ train_opd.py              # ⭐ OPD training (on-policy distillation, swappable with GRPO)
│  │  ├─ opd_multiturn.py          # Multi-turn rollout, Observation masks, and custom GKD loss
│  │  ├─ env.py                    # RLEnv + MockArxivEnv (offline snapshot env)
│  │  ├─ multiturn_env.py          # TRL multi-turn env adapter (environment_factory, one instance per generation)
│  │  ├─ reward.py                 # RewardCalculator (5-component reward + curriculum)
│  │  ├─ grpo_reward.py            # GRPO reward adapter (single-step completion → trajectory)
│  │  ├─ rollout.py                # Offline rollout data collection
│  │  ├─ trajectory.py             # Trajectory + JSONL read/write
│  │  ├─ build_snapshot.py         # Build arXiv offline snapshot (only network step)
│  │  ├─ canary.py                 # Periodic in-training evaluation (early stop)
│  │  ├─ stage_verifier.py         # Per-stage model quality threshold verification
│  │  ├─ precision.py              # Mixed-precision strategy (bf16/fp16/CPU)
│  │  └─ observability.py          # Logging backends + reward-component curves
│  ├─ models/                        # Storage layer (store_memory for RL, store_mysql for Web)
│  ├─ services/                      # Side-effect services (event_bus / log / runtime)
│  ├─ api/ · mcp_protocol/ · skill_cli/   # Archived Web / MCP / Skill compatibility layers
│  ├─ utils/                         # llm_client, logger, PDF utilities
│  ├─ tests/                         # Unit tests (unittest / pytest)
│  └─ requirements.txt
├─ configs/accelerate/                # Multi-GPU launch configs (ddp_2gpu / fsdp_2gpu)
├─ scripts/                          # Data generation
│  ├─ generate_sft_data.py          # Expert trajectories via LLM API
│  ├─ generate_parametric_sft_data.py  # Parameterized expert trajectories from task templates (no API)
│  ├─ augment_sft_data.py           # Semantics-preserving phrasing augmentation
│  ├─ build_sft_train_mix.py        # Auditable training mix + manifest
│  └─ generate_dpo_data.py          # Preference pairs from local SFT model sampling
├─ docs/
│  ├─ rl_building.md               # Full transformation plan
│  ├─ multigranular_rl.md         # Multi-granular reward design (5 components + curriculum)
│  └─ metric_stats.md            # Metrics/statistics plan
├─ data/                             # Datasets (sft/ & dpo/ are gitignored — generate first)
│  ├─ sft/                           # SFT dataset (JSONL)
│  ├─ dpo/                           # DPO pairs (JSONL)
│  ├─ splits/                        # Template-level train/iid/ood splits (v1 → v3_81)
│  └─ mock_arxiv_snapshot.json       # MockEnv snapshot
├─ eval/                             # Bad-case replay loop
│  ├─ badcase_replay.py             # Replay / capture CLI (no LLM needed)
│  ├─ eval_cases.jsonl              # Case library, doubles as the reward-hacking case library
│  └─ readme.md
├─ traces/                           # Trajectory storage (JSONL, gitignored)
├─ archive/                          # Archived (original Web app: PDFMathTranslate / arxiv-api / weather-agent)
├─ AgenticArxivWeb/                  # Original Vue3 frontend (archived)
├─ bin/ · Makefile · Overview.md     # Legacy Web startup scripts & docs (to be modernized)
└─ README.md / README.en.md / README.es-ES.md   # 🇨🇳 🇬🇧 🇪🇸
```

---

## 🔬 Usage Examples

### 1. Rollout (Collect Trajectories)

```bash
# Single task
python -m AgenticArxiv.rl.rollout search_01 traces/train/

# Batch rollout
python -m AgenticArxiv.rl.rollout --all --output_dir traces/train/
```

### 2. Training Pipeline (SFT → DPO → GRPO)

```bash
# Step 1: Generate SFT data
python scripts/generate_sft_data.py

# Step 2: SFT training
python -m AgenticArxiv.rl.train_sft

# Step 3: Generate DPO data (requires SFT model)
python scripts/generate_dpo_data.py

# Step 4: DPO training
python -m AgenticArxiv.rl.train_dpo

# Step 5: GRPO training
python -m AgenticArxiv.rl.train_grpo
```

### 3. Reward Computation Test

```python
from rl.reward import RewardCalculator
from benchmark.tasks import get_task_by_id

task_def = get_task_by_id('search_01')
# Construct a mock result
result = {
    'history': [
        {'thought': '...', 'action': '...', 'observation': '...'},
        {'thought': '...', 'action': 'FINISH', 'observation': '...'},
    ],
    'timing': {...},
    'token_usage': {...},
    'iteration_count': 2,
}

reward_calc = RewardCalculator()
reward, metrics = reward_calc.compute_reward(task_def, result)
print(f'Reward: {reward:.2f}')  # Reward range: [-1, 1]; the value depends on the trajectory
```

---

## 🧪 Task Set & Evaluation

### Smoke set (`benchmark/tasks.py`, 8 tasks)

| ID | Task | Type | Expected Tool |
|----|------|------|---------------|
| `search_01` | Search for AI papers from the last 7 days | Search | `get_recently_submitted_cs_papers` |
| `search_02` | Get ML papers from the last 3 days | Search | `get_recently_submitted_cs_papers` |
| `search_03` | Search for NLP papers from the last 7 days | Search | `get_recently_submitted_cs_papers` |
| `search_04` | Search all computer science categories from the last 7 days | Search | `get_recently_submitted_cs_papers` |
| `download_01` | Download the 1st paper as PDF | Download | `download_arxiv_pdf` |
| `translate_01` | Translate the 1st paper | Translation | `translate_arxiv_pdf` |
| `cache_01` | Check cache status of the 1st paper | Cache | `get_paper_cache_status` |
| `composite_01` | Search + Download | Composite | `get_recently_submitted_cs_papers`, `download_arxiv_pdf` |

### Expanded set (`benchmark/tasks_expanded.py`, 81 tasks)

Enabled via `run_benchmark.py --task-set expanded`, covering twelve template families: search / keyword_search / ref_form / composite / state / optional / constraint / long_chain / infeasible / paper_reading / paper_summary / figure_extraction. Both sets go through the `TaskSpec` in `benchmark/task_spec.py`: `expected_tools` and `expected_tool_args` are derived from the same `steps`, so two hand-maintained lists can never drift apart.

The interpretation families (`paper_reading` / `paper_summary`) pin "this key must be absent" with an explicit `{key: None}` in the argument oracle: `argument_match_score` only counts matches over expected keys and does not penalize extra ones, so without declaring the `section` key, a task asking for title+abstract would accept `section="method"` at full argument score.

### Evaluation metrics & splits

Beyond "success rate + mean tokens + mean iterations", the report includes:

- **`pass^k` reliability** (tau-bench convention, estimated per task then averaged; tasks with too few samples are skipped, not counted as 0)
- **`false_finish`**: ends with FINISH but the expected tools were not all called — degenerate policies measure `always_finish` 91.5% vs `reference` 0%
- **`ref_score`**: compares the resolved `paper_id` rather than the literal `ref` string, removing both false positives and false negatives
- **Cost normalized by successes** (`skill_cli` corrected from 43% to 99% more expensive) and failure-mode breakdown

Tasks are split by **template** into train/iid_test/ood_test (`benchmark/splits.py`, pinned in `data/splits/v1.json`); `--split` is wired into both `run_benchmark.py` and `train_grpo.py`; `rl_train` keeps only the middle success-rate band — tasks at both ends have zero in-group variance and produce no gradient.

### Discrimination gates & badcase replay

- **Per-category discrimination gate** (`benchmark/run_baselines.py`): quantifies reward discrimination with deterministic degenerate policies and enforces per-category thresholds (`tests/test_reward_discrimination.py`) — after fixing four argument-matching score leaks, "always search cs.AI regardless of the task" dropped from 0.833 to 0.446 on search tasks, and "call a tool when doing nothing is correct" went from +0.165 to −0.235.
- **Badcase replay** (`eval/badcase_replay.py` + `eval/eval_cases.jsonl`): freezes a failing trajectory together with its original verdict into a permanent regression case; replay only re-runs the scorer, no LLM / network / tools needed, so `pytest` itself is the gate (`tests/test_badcases.py::ShippedCasesTest`). Cases are `open` (bug still present) or `fixed` (already fixed — reproducing it again is a regression, exit code 1); `hack/*` records degenerate reward-hacking trajectories with threshold assertions ("this behavior must not score X", guarding the two holes fixed in #40 and #47), doubling as the reward-hacking case library — complementary to the `run_baselines.py` gates: **gates watch means, cases pin single failures**. `--save-traces` + `capture` pick bad cases from real trajectories by `false_finish` / `ref_score` / tool sequence, not just crashes.

---

## 📊 Metrics Monitoring

### Reward Curves

Monitor using TensorBoard or wandb:
```bash
tensorboard --logdir ./outputs/grpo/logs
```

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `reward` | Average reward | ↑ Increase |
| `kl_div` | KL divergence (vs reference model) | ↔ Stable (not too large) |
| `task_completed_rate` | Task success rate | ↑ Increase |
| `tool_call_accurate_rate` | Tool call accuracy | ↑ Increase |
| `parse_failures` | Parse failure count | ↓ Decrease |
| `tool_exec_failures` | Tool execution failure count | ↓ Decrease |

---

## 🛡️ Dependencies

**Core dependencies** (`requirements.txt`, covers rollout / benchmark / all five training stages):
```txt
torch>=2.0.0
transformers>=4.45.0
trl>=0.28.0               # floor set by multi-turn GRPO: only 0.28.0+ calls rollout_func on the
                          # non-vLLM path — older versions silently degrade multi-turn rollouts;
                          # verified on 0.29.1 (OPD verified on 1.5.1)
datasets>=2.14.0
accelerate>=0.25.0
arxiv
requests
python-dotenv
loguru
pydantic>=2.0
fire
```

**Optional dependencies** (`requirements-extra.txt`, install as needed; the core training pipeline does not require them):
- `pdf2zh` — real PDF translation (training/benchmarks use the mock; only needed for translation eval/demos)
- `fastapi` / `uvicorn` / `sqlalchemy` / `pymysql` — only for running the archived Web version
- `tensorboard` (recommended, zero-config and offline) / `wandb` — training-curve backends for `--report_to`

---

## 🔗 Related Resources

### Model weights
- [AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT) — stage-1 SFT checkpoint, Qwen2.5-1.5B full-parameter fine-tune (ModelScope)

### Official Documentation
- [TRL Documentation](https://huggingface.co/docs/trl/)
- [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [DPOTrainer](https://huggingface.co/docs/trl/en/dpo_trainer)
- [GRPOTrainer](https://huggingface.co/docs/trl/en/grpo_trainer)

### Papers
- **InstructGPT** (OpenAI, 2022): RLHF three-stage pipeline (SFT → RM → PPO)
- **DPO** (Stanford, 2023): Direct Preference Optimization
- **RLVR**: Reinforcement Learning with Verifiable Reward
- **On-Policy Distillation** ([Thinking Machines Lab, 2025](https://thinkingmachines.ai/blog/on-policy-distillation/)): source of the OPD recipe (student on-policy sampling + teacher per-token reverse KL)
- **GKD / On-Policy Distillation of Language Models** (Agarwal et al., ICLR 2024, [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)): generalized JSD distillation, the basis of TRL's GKDTrainer
- **Rethinking On-Policy Distillation of Large Language Models** ([arXiv:2604.13016](https://arxiv.org/abs/2604.13016)): independent replication and analysis of the OPD recipe

### Original AgenticArXiv (Web Application)
This project is derived from [AgenticArXiv](https://github.com/Algorineko/AgenticArXiv). The original version includes:
- FastAPI backend + Vue3 frontend
- Three Agent architectures (ReAct/MCP/Skill)
- Real-time SSE push, MySQL storage, PDF translation service

These features have been archived to `archive/`.

---

## 🤝 Contributing

Issues and Pull Requests are welcome!

### Development Workflow
1. Fork this repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

MIT License

---

## 🙋 FAQ

### Q: What's the difference from the original AgenticArXiv?

| Dimension | Original AgenticArXiv | This Project (AgenticArXiv-RL) |
|-----------|----------------------|-------------------------------|
| **Focus** | Production-grade arXiv app | RL training research environment |
| **Architecture** | FastAPI + Vue3 + MySQL | Pure Python + JSONL |
| **Agent Modes** | 3 types (ReAct/MCP/Skill) | ReAct only (streamlined) |
| **Core Features** | Real-time translation, SSE, Web UI | SFT/DPO/GRPO training |
| **Dependencies** | Heavy (14+ packages) | Lightweight (11 core packages + optional extras) |

### Q: Why keep only ReAct and archive MCP/Skill?

RL training focuses on a single policy (ReAct with regular parsing). MCP/Skill add complexity without changing the core logic.

### Q: Why switch to JSONL instead of MySQL?

- **Portability**: JSONL requires no database dependency
- **Lightweight**: Better suited for offline RL training scenarios
- **TRL Compatible**: TRL datasets natively support JSONL

### Q: Why GRPO instead of PPO?

GRPO is more suitable for lightweight learning projects:
- ✅ No additional value model required (lower VRAM/training overhead)
- ✅ Suitable for small models (e.g., Qwen2.5-1.5B)
- ✅ Simple implementation, easier to debug

PPO is better suited for production-grade large model training (7B+), which is beyond the scope of this learning demo.

### Q: How to choose between OPD, SFT, DPO and GRPO?

| Scenario | Recommendation |
|------|------|
| Expert demonstrations available, learn the action format first | SFT |
| Preference pairs (good/bad trajectories), no online reward | DPO |
| Verifiable reward, want to improve beyond the baseline online | GRPO |
| Strong teacher model, want to save RL exploration cost | OPD |

They complement each other: SFT is the starting point of every route; OPD and GRPO both sit on top of SFT — the former distills from a teacher (ceiling = teacher), the latter optimizes against verifiable reward (can explore beyond), and either output can initialize or benchmark the other.

---

## 🧰 Toolset Evolution Design (incremental roadmap)

> The end goal of this project is a **locally deployed lightweight LLM that independently handles arXiv paper retrieval, download, and interpretation**. T1–T4 are implemented; T5 remains a design proposal.

### Current state and gaps

| Tool | Capability | Boundary |
|------|------------|----------|
| `get_recently_submitted_cs_papers` | Category + time-window search | Only understands `cat:cs.*` + submission date; no pagination; abstracts truncated to 200 chars |
| `search_arxiv_papers` | Keyword / title / author search | No pagination; offline queries absent from the snapshot return an explicitly flagged deterministic fallback |
| `download_arxiv_pdf` | Download PDF | — |
| `translate_arxiv_pdf` | Full-paper translation via pdf2zh | Produces a translated PDF file; the translated body does not enter the model's context; depends on the optional extra |
| `get_paper_cache_status` | Cache query | — |
| `get_paper_content` | Read the abstract or a method/result/conclusion section | Requires a downloaded PDF; deterministic extraction with no LLM call |
| `summarize_paper` | Produce a summary for a given style/budget | Requires a downloaded PDF; deterministic extractive backend by default, optional `local_model` backend |
| `extract_paper_figures` | Extract embedded figure images + captions, return file paths | Requires a downloaded PDF; vector-only papers (no embedded raster) report `count: 0` |

Three conclusions:

1. **The retrieval half-loop is connected**: both time-window browsing and keyword/title/author lookup are available; pagination remains unimplemented.
2. **The interpretation loop is connected**: reading, summarizing, and figure extraction are all deterministic tools, so the model can carry a paper through the whole chain on its own. Figure *semantic* analysis (T5) remains unimplemented — it needs a VLM resident on the env side.
3. **A bigger action space is not automatically better**: the policy is a ~1.5B model, and every added tool enlarges the tool-selection and JSON-format learning burden. The admission bar for a new tool is "it enables a new task category", not "it might be useful" — of the 5 candidates below, T1/T2 are the critical path, T3 is the main increment, T4 has landed, T5 is optional.

### Proposed new tools (in dependency order)

| Priority | Tool | Design | Why it stays RLVR-friendly |
|--------|------|----------|----------------------|
| **T1** ✅ | `search_arxiv_papers(query, max_results, days=None)` | Keyword search mapped to the arXiv API's `all:` / `ti:` / `au:` fields; coexists with the existing tool (time-window browsing vs precise lookup are different task types) | Expected tools/args still derive from `task_spec.steps`; `MockArxivEnv` replays offline keyed by a hash of the query string, and unseen queries degrade **deterministically** (return a fixed subset, explicitly flagged in the observation) — reproducible, and it prevents the model from mistaking an empty result for a successful search |
| **T2** ✅ | `get_paper_content(ref, section=None)` | PDF → plain text (PyMuPDF); returns title/abstract by default, per section (method / result / conclusion) on demand | Deterministic text extraction, no LLM involved; extraction results pre-stored in the snapshot. **It is the prerequisite of every interpretation task** |
| **T3** ✅ | `summarize_paper(ref, style, max_words)` | Summarize a paper: an **env-side** summary (input from T2's text) is returned | What is trainable is "when to call it, on which ref, whether style/length args are right" — all rule-checkable; summary quality itself is **not rewarded** (see below) |
| **T4** ✅ | `extract_paper_figures(ref)` | Figure/table preparation: extract figure images + captions, return file paths | Deterministic; verify "correct ref + count ≥ 1" |
| **T5** (optional, multimodal) | `analyze_figure(ref, figure_no, question=None)` | Figure analysis: an env-side local VLM (e.g. Qwen2.5-VL) reads the figure and answers | Rules only judge "was it called correctly, are the args right"; VLM answer quality does not enter the reward, keeping third-party model noise out of the policy gradient |

**T3's summarization backend**: the original design called for "an env-side local summarizer model". The implementation defaults to a **deterministic extractive backend** instead (whole sentences selected per section, trimmed to the word budget, no sampling, no model) for three reasons: the same trajectory replays byte-identically at any later time; it does not add a second weights-dependent prerequisite on top of `build_snapshot`; and because the reward grades only the tool-call decision and never the summary text, a model backend contributes nothing to the training signal. When more natural-language output is wanted, `SUMMARY_BACKEND=local_model SUMMARY_MODEL_PATH=<local model dir>` switches to the model backend (greedy decoding, still deterministic) at the cost of loading weights during snapshot construction.

Companion task templates (following the `tasks_expanded.py` families, all declaratively derived from `task_spec.steps`):

- `search_kw_*`: keyword-search tasks (T1)
- `paper_reading`: search → download → read content (T2), 5 tasks
- `paper_summary`: search → download → summarize (T3), 5 tasks
- `figure_extraction`: search → download → extract figures (T4), 4 tasks
- `long_chain`'s `chain_ai5_read_then_summary`: read → summarize as a 4-step chain (T2+T3)
- `analyze_figure(ref, figure_no)` (T5, optional): search → download → extract figures → figure analysis, enabled only in multimodal environments

### Downstream design for training & evaluation

1. **Snapshot extension**: `build_snapshot.py` does everything in one pass — besides search results, **pre-extract full text and figure files** for snapshot papers; every new tool replays offline, preserving the "build_snapshot is the only network step" contract.
2. **Zero reward changes**: the five-component scheme is reused as is; `expected_tools` / `expected_tool_args` derive from `steps`, so `reward.py` and the curriculum stay untouched.
3. **Reward-hacking prevention**: interpretation tools open a new surface for "call tools randomly to farm process points" — reuse the `run_baselines.py` per-category gates + pin single cases in `eval/eval_cases.jsonl` (e.g. calling `summarize_paper` with a ref that points to a non-existent paper must lose points).
4. **The reward problem of summary quality (deliberately not done)**: turning "is the summary good" into a reward needs LLM-as-judge or rubric scoring, which introduces non-deterministic rewards and a new hacking surface. The design first reduces summarization to a **tool-invocation decision problem** (when to call, on whom); quality evaluation is a separate long-term project.
5. **The multimodal boundary (deliberately isolated)**: T5's VLM lives only on the env side; the policy remains a text-only small model — the action space only contains "call it or not, how to ask", outsourcing figure comprehension to the environment. Only when the policy itself becomes multimodal would putting images into the observation be considered.
6. **Hardware bar**: T5 adds an env-side model (~6GB for the VLM, less when quantized); it does not affect training VRAM (no gradients). The default T3 backend is extractive and needs no extra weights, so the interpretation loop runs with no hardware beyond the training box.

### Rollout order

```
T1 keyword search ──→ T2 read content ──→ T3 summarize   (interpretation loop, done)
                          └────→ T4 extract → T5 analyze (T4 done; T5 optional, multimodal env)
```

Each time a tool lands: extend the task templates → re-run `run_baselines.py` to re-derive per-category discrimination thresholds → regenerate SFT/DPO data → add matching cases to `eval/eval_cases.jsonl`. Landing T3/T4 also included:

- Task templates: new `paper_reading` (5), `paper_summary` (5), `figure_extraction` (4) and one 4-step interpretation chain; expanded set 62 → 81 tasks
- Splits: new `data/splits/v3_81.json` (v1/v2 left untouched, so historical success rates stay comparable)
- Discrimination: `run_baselines.py`'s per-category gate now covers the new categories (`tests/test_reward_discrimination.py`)
- Bad cases: 8 new `hack/summary-*` / `hack/read-*` / `hack/figure-*` entries in `eval/eval_cases.jsonl`, pinning "wrong budget", "summarize/extract without downloading", "pass an unrequested section", "extract from the wrong paper" and similar point-farming shapes (14 cases total)

---

## 📝 TODO (Development Roadmap)

Ordered by priority. Contributions welcome (see 🤝 Contributing).

### P0 — Toolset expansion (interpretation loop)

T1–T4 are implemented (see "🧰 Toolset Evolution Design"):

- [x] **T1 Keyword search** `search_arxiv_papers`: adds the "find a specific paper" lookup-type retrieval
- [x] **T2 Paper reading** `get_paper_content`: deterministic PDF → text with offline snapshot replay, the prerequisite of all interpretation tasks (critical path)
- [x] **T3 Paper summarization** `summarize_paper`: env-side summarization, turning "interpretation" into a trainable tool-invocation decision (deterministic extractive backend by default; `SUMMARY_BACKEND=local_model` switches to a local model)
- [x] **T4 Figure extraction** `extract_paper_figures`: deterministic extraction of embedded figures + captions, with offline snapshot replay
- [ ] **T5 Figure analysis** `analyze_figure` (optional, multimodal env): the tool, environment integration, task templates, unit tests, and optional parameterised SFT derivation rules are written. A real snapshot and generated dataset are still needed. The VLM stays on the environment side; the policy remains text-only.
  - ⏳ **Snapshot gap**: local offline snapshots are not committed, and no T5 records are available here for verification. During the 2026-09-22 build, arXiv throttled this host to ~5KB/s (a 34MB paper delivered 492KB in 90s), so the full rebuild was not completed. With working network access or cached PDFs, run `python -m AgenticArxiv.rl.build_snapshot --skip-prefetch`; replay fails on missing T5 records.
  - Backends: `extractive` by default (reuses the caption T4 already extracted — deterministic, no weights); `FIGURE_ANALYSIS_BACKEND=vlm VLM_MODEL_PATH=<local VLM dir>` switches to a local VLM (greedy decoding, answers recorded at snapshot-build time).
  - **Generate T5 expert data** (requires a snapshot with T5 records):
    ```bash
    python scripts/generate_parametric_sft_data.py \
      --split-file data/splits/v3_81.json \
      --snapshot data/mock_arxiv_snapshot.json \
      --include-t5
    ```
    The default command still produces the historical v2 dataset; the T5 output goes to `data/sft/sft_v3_t5_parametric_seed.jsonl`.
  - Known boundary: no T5 dataset or model has been generated or trained in this repository; the released model has not learned this tool.

### P1 — Reward curriculum tuning

- [x] **Multi-granular curriculum calibration (measured; verdict: the 30-step ramp does not hold from an SFT start)**

  `scripts/analyze_curriculum.py` reads the real training curves. We ran a two-arm comparison: same SFT starting point (Qwen2.5-1.5B, 2 epochs, loss ≈ 0.08), the same 9 tasks that actually produce within-group gradient, 60 steps each, differing only in `--reward_curriculum_steps`.

  | Component | curriculum 30: suppressed → full | curriculum 0 |
  |---|---|---|
  | `format` | 0.983 → 1.000 | 0.983 → 1.000 |
  | `process` | 0.937 → 1.000 | 0.934 → 1.000 |
  | `tool` | −0.074 → +0.215 | −0.074 → +0.215 |
  | `argument` | −0.545 → −0.335 | −0.530 → −0.319 |
  | `outcome` | −0.127 → +0.028 | −0.127 → +0.030 |

  Two conclusions:

  1. **The curriculum's premise is already satisfied at step one**: `format` starts at 0.983 and `process` at 0.937. The ramp exists to "learn the ReAct protocol first", and SFT has already taught it — the gate is protecting a stage the model has moved past.
  2. **The arms are indistinguishable**: the component curves overlap almost point for point. Suppressing `tool`/`argument`/`outcome` for thirty steps neither improved nor hurt semantic learning; it just spent thirty steps on a down-weighted signal.

  Recommendation: **use `--reward_curriculum_steps 0` for GRPO from an SFT start**; keep the ramp for cold starts / training straight from a base model. Honest caveat: 9 tasks × 60 steps is a small run, and the difference itself sits in the noise — so the finding is not "the curriculum hurts" but "it has not earned those 30 steps".

- [x] **Task-selection defect fixed along the way**: `rl_train` picks the mid success-rate band, but `rates` were measured on the **base model**. After SFT the difficulty band shifts: of 48 train tasks, only **9 (19%)** still yield within-group gradient, and **24 of the 39 zero-variance tasks sit in the "middle band but constant"** case — "mid rate ⇒ group variance" is false for a saturated policy, and those tasks burn steps at `frac_reward_zero_std=1` until `RewardVarianceGuard` aborts. Rates must be re-measured whenever the model changes rather than carried over.

### P2 — Performance & scale

- [x] **Multi-GPU support**: `configs/accelerate/ddp_2gpu.yaml` (DDP) and `fsdp_2gpu.yaml` (FSDP) are in place; training scripts automatically skip the single-GPU pin under `accelerate launch`, and launching QLoRA across processes fails loudly. DDP was verified on a two-GPU machine (both ranks build NCCL, gradients sync, the model saves).
  - The FSDP config needs `torch>=2.6`: `rl/trl_compat.py` only ships a single-GPU placeholder below that, so that path is unverified.
  - Usage: `accelerate launch --config_file configs/accelerate/ddp_2gpu.yaml -m AgenticArxiv.rl.train_sft --no-qlora ...` (the `accelerate` launcher must run under the Python interpreter that has the training dependencies).
- [ ] **vLLM-accelerated sampling**: replace HF generate to raise multi-turn rollout sampling throughput.
  - **Status**: TRL 0.29 requires vLLM 0.10.2–0.12.0, while the only build available on the target platform is a vendor-specific 0.6.2; having both installed makes `trl.trainer.grpo_trainer` fail at import time. Progress depends on a matching platform build of vLLM, so this is an environment dependency rather than a code change, and it is on hold.

### P3 — Long term (algorithmic evolution)

- [x] **DAPO-style improvements (the parts TRL already supports)**: `--loss_type` (token-level loss), `--epsilon_high` (clip-higher) and `--mask_truncated_completions` (overlong filtering) are wired into `train_grpo.py`, with a `--dapo` preset that fills all three at once (`loss_type=dapo`, `epsilon_high=0.28`, `mask_truncated_completions=True`, `beta=0`). Defaults change nothing, so historical experiments stay comparable.
  - **Dynamic sampling (resampling zero-variance groups) is not implemented**: TRL 0.29 exposes no interceptable generation hook, and overriding `_generate_and_score_completions` would drift across versions. The current mitigation is `RewardVarianceGuard` (abort the run after consecutive zero-variance steps) plus the `frac_reward_zero_std` curve, which turns silent idling into a visible signal.
- [ ] **Async training framework**: migrate to verl `fully_async_policy` / AReaL to host SAO (below).

### 🔭 SAO: the next-generation async agentic RL algorithm

> **SAO (Single-Rollout Asynchronous Optimization)** was proposed by Tsinghua University's KEG Lab (2026-07) as an evolution of GRPO for **asynchronous agentic training**. Motivation: rollout is the bottleneck in long-horizon agent tasks, and GRPO's group-wise sampling becomes off-policy and unstable under asynchrony (typically collapsing within 200 steps).
>
> Key technical components:
> 1. **Single-rollout sampling**: one trajectory per prompt, consumed as soon as it arrives, replacing group-wise comparison;
> 2. **DIS (Direct Bilateral Importance Sampling)**: computes `r_t = π_θ / π_rollout` from token log-probs recorded at rollout time and **masks out tokens outside the trust interval `[1−ε_l, 1+ε_h]`** (not PPO-style one-sided clipping);
> 3. **Decoupled value-model updates**: the value model is updated twice per policy update (1:2), with **attention layers frozen** during value training (only MoE projection layers trained);
> 4. **Skip-observation GAE**: advantages propagate only between model-generated tokens, skipping environment observation tokens to filter out environment noise.
>
> Results: stable training for ~1000 steps; **97.3%** on AIME2025 (vs 84.2% for GRPO), 29.8% on SWE-Bench Verified; already used to train GLM-5.2 (750B).
>
> Adoption path: multi-turn rollout has landed → introduce skip-observation masking and DIS bilateral clipping → migrate to verl `fully_async_policy` (`gen_batch_size=1` / `staleness_threshold` / token-level TIS clipping, aligned with SAO) or AReaL v1.0 for fully async training + value model.
>
> 📄 **Paper**: [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning (arXiv:2607.07508)](https://arxiv.org/abs/2607.07508) (Tsinghua KEG; official code not yet open-sourced)

---

**Start your Agentic RL training journey!** 🚀
