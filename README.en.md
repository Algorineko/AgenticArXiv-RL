<p align="center">
  <a href="README.md">🇨🇳 中文</a> | <a href="README.en.md">🇬🇧 English</a> | <a href="README.es-ES.md">🇪🇸 Español</a>
</p>

# AgenticArXiv-RL — Agentic RL Training Environment

> **An Agentic RL training environment built on ReAct Agent + arXiv tools**  
> Supports a progressive SFT/DPO/GRPO/PPO training pipeline for research on LLM Agent reinforcement learning

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
cd AgenticArxiv
python -m rl.rollout search_01 ../traces/train/
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
| **Reward** | Verifiable (task success +1.0, tool accuracy +0.5, parse error -0.2, etc.) |
| **Transition** | `execute_tool(action) → observation` |

### Action Space (4 Tools)

1. `get_recently_submitted_cs_papers(aspect, days, max_results)` — Search arXiv papers
2. `download_arxiv_pdf(ref, session_id)` — Download PDF
3. `translate_arxiv_pdf(ref, session_id)` — Translate PDF
4. `get_paper_cache_status(ref, session_id)` — Query cache status

### Verifiable Reward Components

| Dimension | Reward | Source |
|-----------|--------|--------|
| Task success | +1.0 | `task_completed` |
| Tool call accurate | +0.5 | `tool_call_accurate` (`_check_tool_sequence`) |
| Parse error | -0.2 | `parse_failures` |
| Tool execution failure | -0.3 | `tool_exec_failures` |
| Timeout | -0.5 | `termination_type == "FORCE_STOP"` |
| Error termination | -1.0 | `termination_type == "ERROR"` |

**Key point**: All rewards are **verifiable** (rule-based), requiring no human annotation — corresponding to the RLVR (Reinforcement Learning with Verifiable Reward) framework.

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
   python -m rl.train_sft
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
   python -m rl.train_dpo
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
python -m rl.train_grpo
```

**Output**: `./outputs/grpo/final` model

**Advantages**:
- No reward model required (DPO's drawback: unable to learn online)
- No value model required (PPO's drawback: high VRAM overhead)
- Suitable for small models (e.g., Qwen2.5-1.5B)

---

## 📂 Directory Structure

```
AgenticArXiv-RL/
├─ AgenticArxiv/                     # Python package
│  ├─ agents/                        # Agent core
│  │  ├─ base_agent.py              # Generic ReAct loop
│  │  ├─ agent_engine.py            # ReActAgent (RL policy)
│  │  ├─ prompt_templates.py
│  │  └─ side_effects.py            # Decoupled side effects interface
│  ├─ tools/                         # Tool layer (action space)
│  │  ├─ tool_registry.py
│  │  ├─ arxiv_tool.py
│  │  ├─ pdf_download_tool.py
│  │  ├─ pdf_translate_tool.py
│  │  └─ cache_status_tool.py
│  ├─ benchmark/                     # ⭐ Verifiable Reward source
│  │  ├─ metrics.py                 # TaskMetrics, _check_tool_sequence
│  │  ├─ tasks.py                   # BENCHMARK_TASKS (task seed set)
│  │  └─ runner.py
│  ├─ rl/                            # ⭐ RL core
│  │  ├─ env.py                     # RLEnv + MockArxivEnv
│  │  ├─ policy.py
│  │  ├─ reward.py                  # RewardCalculator
│  │  ├─ trajectory.py              # Trajectory + JSONL read/write
│  │  ├─ rollout.py
│  │  ├─ tasks.py
│  │  ├─ train_sft.py               # ⭐ SFT training
│  │  ├─ train_dpo.py               # ⭐ DPO training
│  │  └─ train_grpo.py              # ⭐ GRPO training
│  ├─ utils/
│  │  ├─ llm_client.py
│  │  └─ logger.py
│  └─ requirements.txt
├─ traces/                           # Trajectory storage (JSONL)
│  ├─ train/
│  └─ eval/
├─ data/
│  ├─ sft/                           # SFT dataset
│  ├─ dpo/                           # DPO dataset
│  └─ mock_arxiv_snapshot.json       # MockEnv snapshot
├─ eval/
│  ├─ eval_cases.jsonl
│  └─ badcase_replay.py
├─ scripts/
│  ├─ generate_sft_data.py
│  └─ generate_dpo_data.py
├─ archive/                          # Archived (original Web app)
│  ├─ api/
│  ├─ AgenticArxivWeb/
│  ├─ mcp_protocol/
│  └─ skill_cli/
├─ docs/
│  └─ rl_building.md                # Full transformation plan
├─ .venv/                            # Python virtual environment
└─ README.md                         # This document
```

---

## 🔬 Usage Examples

### 1. Rollout (Collect Trajectories)

```bash
cd AgenticArxiv

# Single task
python -m rl.rollout search_01 ../traces/train/

# Batch rollout
python -m rl.rollout --all ../traces/train/
```

### 2. Training Pipeline (SFT → DPO → GRPO)

```bash
# Step 1: Generate SFT data
python scripts/generate_sft_data.py

# Step 2: SFT training
python -m rl.train_sft

# Step 3: Generate DPO data (requires SFT model)
python scripts/generate_dpo_data.py

# Step 4: DPO training
python -m rl.train_dpo

# Step 5: GRPO training
python -m rl.train_grpo
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

From `benchmark/tasks.py`, containing 7 tasks:

| ID | Task | Type | Expected Tool |
|----|------|------|---------------|
| `search_01` | Search for AI papers from the last 7 days | Search | `get_recently_submitted_cs_papers` |
| `search_02` | Get ML papers from the last 3 days | Search | `get_recently_submitted_cs_papers` |
| `search_03` | Search for NLP papers from the last 7 days | Search | `get_recently_submitted_cs_papers` |
| `download_01` | Download the 1st paper as PDF | Download | `download_arxiv_pdf` |
| `translate_01` | Translate the 1st paper | Translation | `translate_arxiv_pdf` |
| `cache_01` | Check cache status of the 1st paper | Cache | `get_paper_cache_status` |
| `composite_01` | Search + Download | Composite | `get_recently_submitted_cs_papers`, `download_arxiv_pdf` |

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
transformers>=4.35.0
trl>=0.8.0                # TRL (SFT/DPO/GRPO/PPO)
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

**Start your Agentic RL training journey!** 🚀
