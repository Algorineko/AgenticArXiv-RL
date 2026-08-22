<p align="center">
  <a href="README.md">🇨🇳 中文</a> | <a href="README.en.md">🇬🇧 English</a> | <a href="README.es-ES.md">🇪🇸 Español</a>
</p>

# AgenticArXiv-RL — Agentic RL Training Environment

> **An Agentic RL training environment built on ReAct Agent + arXiv tools**  
> Supports a progressive SFT/DPO/GRPO/PPO training pipeline for research on LLM Agent reinforcement learning

---

## Source repository tools

The agent can search and download source code from GitHub and Gitee:

- `search_github_repositories` / `search_gitee_repositories` normalize metadata and remember results in the session.
- `download_github_repository` / `download_gitee_repository` accept a 1-based search-result index, `owner/name`, or a repository URL.
- Downloads are size-limited ZIP archives; they are never executed or automatically extracted.

Public repositories work without credentials where the platform permits it. Set
`GITHUB_TOKEN` or `GITEE_TOKEN` to raise API limits or access resources permitted
by that token. Optional settings are `REPOSITORY_DOWNLOAD_PATH` and
`REPOSITORY_MAX_DOWNLOAD_MB` (default: 100).

After changing repository tasks, regenerate the offline training snapshot:

```bash
cd AgenticArxiv
python -m rl.build_snapshot
```

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

**Expected output**:
```
✅ Task search_01 rollout completed
   Reward: 1.50
   Metrics: task_completed=True, tool_call_accurate=True
   Trajectory saved to: traces/train/rollout_20260621_150000.jsonl
```

---

## 📚 Core Concepts

### MDP Design

| Dimension | Definition |
|-----------|------------|
| **State** | Task description + conversation history + tool results |
| **Action** | 4 tools (arxiv search/download/translate/cache query) + FINISH |
| **Reward** | Five-component multi-granular verifiable reward (format / tool / argument / process / outcome, see below) |
| **Transition** | `execute_tool(action) → observation` (`MockArxivEnv` offline snapshot replay, deterministic & reproducible) |

### Action Space (4 Tools)

1. `get_recently_submitted_cs_papers(aspect, days, max_results)` — Search arXiv papers
2. `download_arxiv_pdf(ref, session_id)` — Download PDF
3. `translate_arxiv_pdf(ref, session_id)` — Translate PDF
4. `get_paper_cache_status(ref, session_id)` — Query cache status

### Verifiable Reward Components

**Multi-granular five-component verifiable reward** (`rl/reward.py`, inspired by LLM-TIR's hierarchical reward). Each component is normalized to `[-1, 1]` and combined as a weighted sum divided by the total weight:

| Component | Default weight | Signal |
|-----------|:---:|--------|
| `format` | 1 | Fraction of steps whose action is a valid JSON tool call or a terminal token |
| `tool` (tool sequence) | 3 | Order-aware **LCS-F1** between predicted and expected tool sequences (`benchmark/metrics.py` strict matching) |
| `argument` (parameters) | 2 | Parameter-key recall × exact value accuracy; automatically skipped when a task has no `expected_tool_args` |
| `process` | 1 | Valid-step credit minus parse/execution-failure and unnecessary-call penalties |
| `outcome` | 3 | Correct completion +1, completion with a wrong tool path +0.25, forced stop −0.5, error −1 |

**Curriculum learning**: for the first 30 training steps the `tool` / `argument` / `outcome` weights are scaled by 1/3 (learn the ReAct protocol first, semantics later); full weights apply from step 30 onward (`RewardCalculator.schedule`).

**Key point**: All rewards are **verifiable** (rule-based), requiring no human annotation — corresponding to the RLVR (Reinforcement Learning with Verifiable Reward) framework. Every trajectory records a `reward_components` breakdown for auditing and reward-hacking analysis.

---

## 🛠️ Training Pipeline (SFT → DPO → GRPO)

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
does not require `LLM_API_KEY`. Use `--model`, `--num_rollouts_per_task`,
`--temperature`, and `--seed` to customize generation. When
`data/mock_arxiv_snapshot.json` exists, tool calls are replayed offline for
reproducibility; otherwise generation falls back to the live tools. Only
different first actions whose reward gap exceeds `--min_reward_gap` are paired.
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

**Reward scoring** (`rl/grpo_reward.py`): the model's single-step completion is parsed into a ReAct action, executed with `MockArxivEnv`, and completed into a "minimal full trajectory" before being scored by the five-component `RewardCalculator` — the same standard used by rollout and benchmark, with no second reward definition.

### Training Quality Guards (auto-verification)

The training pipeline bakes in several layers of automatic validation that turn "silent training failures" into loud errors:

- **Generation-length check**: before training, verifies that `max_completion_length` fits the canonical action, preventing zero-gradient idle runs where the model can never emit a complete action
- **Zero-variance guard**: aborts training (with fix suggestions) when the in-group reward variance stays at 0, meaning all advantages are zero (`RewardVarianceGuard`)
- **Canary evaluation**: samples on fixed tasks every N steps and early-stops if performance degrades below the threshold repeatedly (`CanaryCallback`)
- **Stage verification**: each stage's output model must pass a minimum quality threshold — SFT parse rate ≥ 0.3, DPO mean reward ≥ −0.3, GRPO mean reward ≥ −0.2 (`StageVerifier`, skip with `--no-verify`)
- **Adaptive mixed precision**: bf16 first on CUDA, falling back to fp16, disabled on CPU / MPS (`rl/precision.py`)

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
│  │  ├─ arxiv_tool.py             # arXiv search
│  │  ├─ pdf_download_tool.py      # PDF download
│  │  ├─ pdf_translate_tool.py     # PDF translation
│  │  └─ cache_status_tool.py      # Cache query
│  ├─ benchmark/                     # ⭐ Verifiable Reward source
│  │  ├─ metrics.py               # TaskMetrics, strict tool-sequence & argument matching
│  │  ├─ tasks.py                 # BENCHMARK_TASKS (paper and repository seeds)
│  │  ├─ runner.py                 # Benchmark executor
│  │  ├─ run_benchmark.py          # CLI benchmark entry
│  │  └─ report.py                 # Metrics report
│  ├─ rl/                            # ⭐ RL core
│  │  ├─ train_sft.py              # ⭐ SFT training
│  │  ├─ train_dpo.py              # ⭐ DPO training
│  │  ├─ train_grpo.py             # ⭐ GRPO training (with training guards)
│  │  ├─ env.py                    # RLEnv + MockArxivEnv (offline snapshot env)
│  │  ├─ reward.py                 # RewardCalculator (5-component reward + curriculum)
│  │  ├─ grpo_reward.py            # GRPO reward adapter (single-step completion → trajectory)
│  │  ├─ rollout.py                # Offline rollout data collection
│  │  ├─ trajectory.py             # Trajectory + JSONL read/write
│  │  ├─ build_snapshot.py         # Build arXiv offline snapshot (only network step)
│  │  ├─ canary.py                 # Periodic in-training evaluation (early stop)
│  │  ├─ stage_verifier.py         # Per-stage model quality threshold verification
│  │  └─ precision.py              # Mixed-precision strategy (bf16/fp16/CPU)
│  ├─ models/                        # Storage layer (store_memory for RL, store_mysql for Web)
│  ├─ services/                      # Side-effect services (event_bus / log / runtime)
│  ├─ api/ · mcp_protocol/ · skill_cli/   # Archived Web / MCP / Skill compatibility layers
│  ├─ utils/                         # llm_client, logger, PDF utilities
│  ├─ tests/                         # 16 unit tests (unittest)
│  └─ requirements.txt
├─ scripts/                          # Data generation
│  ├─ generate_sft_data.py          # Expert trajectories via LLM API
│  └─ generate_dpo_data.py          # Preference pairs from local SFT model sampling
├─ docs/
│  ├─ rl_building.md               # Full transformation plan
│  ├─ multigranular_rl.md         # Multi-granular reward design (5 components + curriculum)
│  └─ metric_stats.md            # Metrics/statistics plan
├─ data/                             # Datasets (sft/ & dpo/ are gitignored — generate first)
│  ├─ sft/                           # SFT dataset (JSONL)
│  ├─ dpo/                           # DPO pairs (JSONL)
│  └─ mock_arxiv_snapshot.json       # MockEnv snapshot
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
print(f'Reward: {reward:.2f}')  # Expected: ~1.5
```

---

## 🧪 Test Task Set

From `benchmark/tasks.py`, containing 18 tasks: 8 paper tasks and 10 source-repository tasks.

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
| `github_search_*` | Search GitHub repositories | Code search | `search_github_repositories` |
| `gitee_search_*` | Search Gitee repositories | Code search | `search_gitee_repositories` |
| `*_download_*` | Download a repository/tag | Code download | Platform download tool |
| `*_search_download_*` | Search then download by index | Code composite | Platform search + download |
| `cross_platform_search_01` | Search both platforms | Code composite | Both search tools |

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

**Core dependencies** (`requirements.txt`):
```txt
torch>=2.0.0
transformers>=4.45.0
trl>=0.20.0               # TRL (SFT/DPO/GRPO), verified on 0.29.1
datasets>=2.14.0
accelerate>=0.25.0
arxiv
requests
python-dotenv
loguru
pydantic>=2.0
fire
```

**No longer needed** (removed):
- `fastapi`, `uvicorn` (no web service)
- `sqlalchemy`, `pymysql` (switched to JSONL)
- `pdf2zh` (using mock during training)

---

## 🔗 Related Resources

### Official Documentation
- [TRL Documentation](https://huggingface.co/docs/trl/)
- [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [DPOTrainer](https://huggingface.co/docs/trl/en/dpo_trainer)
- [GRPOTrainer](https://huggingface.co/docs/trl/en/grpo_trainer)

### Papers
- **InstructGPT** (OpenAI, 2022): RLHF three-stage pipeline (SFT → RM → PPO)
- **DPO** (Stanford, 2023): Direct Preference Optimization
- **RLVR**: Reinforcement Learning with Verifiable Reward

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
| **Dependencies** | Heavy (14+ packages) | Lightweight (8 core packages) |

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

---

## 📝 TODO (Development Roadmap)

Ordered by priority. Contributions welcome (see 🤝 Contributing).

### P0 — Near term (close core gaps)

- [ ] **Multi-turn Agentic Rollout**: GRPO currently scores only the model's **single-step** completion as a "synthesized minimal trajectory" — there is no real multi-turn "act → observe → act" interaction. Use TRL's tool-calling / multi-turn support with `MockArxivEnv` as the tool backend and assistant-only loss masking to close the most conspicuous gap against the "Agentic RL" name.
- [ ] **Training observability**: wire up wandb / TensorBoard (currently `report_to=[]`, no monitoring) and log reward / advantage / KL / per-component curves before tuning hyperparameters.

### P1 — Mid term (data & evaluation)

- [ ] **Continue expanding the task set**: `benchmark/tasks.py` now has 18 tasks; grow it to 50+ and auto-derive `expected_tools` / `expected_tool_args`.
- [ ] **eval/ badcase replay**: the `eval/` directory listed in the tree does not exist yet; implement `eval_cases.jsonl` + `badcase_replay.py` to close the bad-case replay loop.
- [ ] **Reward-hacking triage**: build on `RewardVarianceGuard` / `CanaryCallback` with a reward-hacking case library and curriculum weight tuning.

### P2 — Performance & scale

- [ ] **vLLM-accelerated sampling**: replace HF generate to raise rollout throughput (priority rises once multi-turn rollout lands).
- [ ] **Multi-GPU support**: accelerate / FSDP config (accelerate is already a dependency but currently unused).

### P3 — Long term (algorithmic evolution)

- [ ] **DAPO-style improvements**: clip-higher, dynamic sampling, overlong filtering, token-level loss (loss/clip live inside TRL; needs a fork or a `compute_loss` override).
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
> Adoption path: close the P0 multi-turn rollout gap → introduce skip-observation masking and DIS bilateral clipping → migrate to verl `fully_async_policy` (`gen_batch_size=1` / `staleness_threshold` / token-level TIS clipping, aligned with SAO) or AReaL v1.0 for fully async training + value model.
>
> 📄 **Paper**: [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning (arXiv:2607.07508)](https://arxiv.org/abs/2607.07508) (Tsinghua KEG; official code not yet open-sourced)

---

**Start your Agentic RL training journey!** 🚀
