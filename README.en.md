<p align="center">
  <a href="README.md">🇨🇳 中文</a> | <a href="README.en.md">🇬🇧 English</a> | <a href="README.es-ES.md">🇪🇸 Español</a>
</p>

# AgenticArXiv-RL — Agentic RL Training Environment

> **An Agentic RL training environment built on a ReAct agent + arXiv tools**
> Supports SFT/DPO/GRPO/OPD training paths with verifiable rewards (RLVR), for research on reinforcement learning for LLM agents.
> The ultimate goal: **a lightweight multimodal paper assistant for on-device deployment** — an end-to-end paper-reading model spanning retrieval, download, translation, summarization, and figure understanding.

<p align="center">
  <img src="imgs/AgenticArXiv-RL.jpg" alt="AgenticArXiv-RL project overview" width="800"/>
</p>

---

## 🎯 Project Positioning

This project turns arXiv paper retrieval/download/translation/interpretation tasks into a **trainable reinforcement learning environment**, focused on:

1. **Verifiable Reward**: rule-based rewards (tool-call accuracy, task completion, parsing errors, etc.), no human annotation required
2. **Progressive training**: SFT → DPO → GRPO (an OPD on-policy distillation route is also provided; PPO is unavailable under the current dependencies)
3. **Lightweight engineering**: pure Python + JSONL storage, no database, no frontend — focused on offline training

**Non-goals**: a production-grade arXiv app, a Web UI, a real-time translation service (the original web app has been removed from this repo; see the original [AgenticArXiv](https://github.com/Algorineko/AgenticArXiv)).

---

## 🚀 Quick Start

```bash
# 1. Clone and install (Python 3.9+)
git clone https://github.com/Algorineko/AgenticArXiv-RL.git
cd AgenticArXiv-RL
python3 -m venv .venv && source .venv/bin/activate
pip install -r AgenticArxiv/requirements.txt

# 2. Configure the LLM API (needed for API rollouts; not needed for local training/evaluation)
cat > AgenticArxiv/.env << 'EOF'
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-api-key
MODEL=gpt-4-turbo
EOF

# 3. Test a rollout (reward in [-1, 1], varies with the trajectory)
python -m AgenticArxiv.rl.rollout search_01 traces/train/
# ✅ Task search_01 rollout completed  Reward: 1.00
```

Batch rollouts: `python -m AgenticArxiv.rl.rollout --all --output_dir traces/train/`. Unless noted otherwise, later commands run from the repository root.

---

## 📚 Core Concepts

### MDP Design

| Dimension | Definition |
|------|------|
| **State** | Task description + conversation history + tool results |
| **Action** | 9 tools (see below) + FINISH |
| **Reward** | Five-component multi-granular verifiable reward (format / tool / argument / process / outcome) |
| **Transition** | `execute_tool(action) → observation` (`MockArxivEnv` replays offline snapshots — deterministic and reproducible) |

### Action Space (9 Tools)

1. `get_recently_submitted_cs_papers(aspect, days, max_results)` — browse by subfield + time window
2. `search_arxiv_papers(query, max_results, days=None)` — keyword/title/author search
3. `download_arxiv_pdf(ref, session_id)` — download a PDF
4. `translate_arxiv_pdf(ref, session_id)` — translate a PDF (pdf2zh)
5. `get_paper_cache_status(ref, session_id)` — query cache status
6. `get_paper_content(ref, session_id, section=None)` — read the abstract or a chosen section (deterministic extraction)
7. `summarize_paper(ref, style, max_words)` — env-side summary (tldr / structured / bullet)
8. `extract_paper_figures(ref)` — extract figure files and captions
9. `analyze_figure(ref, figure_no, question=None)` — figure analysis: the env calls a local VLM to read the figure (multimodal environment enabled)

> The full "search → download → read → summarize → extract figures → analyze figures" interpretation loop is now wired end to end: figure analysis is performed env-side by the project's post-trained [FigureQA VLM](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA), which also records the snapshots, and on the policy side the [SFT-T5](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5) checkpoint has learned that four-step chain.
> **Admission rule**: a bigger action space is not automatically a better one — a new tool is admitted only if it "unlocks a new class of tasks", not merely because it "might be useful". See [Toolset Evolution Design](docs/toolset_evolution.md) for the full design decision chain.

### Verifiable Reward (Five Components)

**Multi-granular verifiable reward** (`rl/reward.py`). Five components normalized to `[-1, 1]`, plus diagnostic components and non-compensable failure caps:

| Component | Default weight | Signal |
|------|:---:|------|
| `format` | 1 | Whether each step's action is a valid JSON tool call or a terminator |
| `tool` | 3 | **Order-aware LCS-F1** between the predicted and expected tool sequences |
| `argument` | 2 | Argument key recall × exact-value accuracy |
| `process` | 1 | Bonus for valid steps − penalties for parse failures / execution failures / redundant calls |
| `outcome` | 3 | Correct completion +1, wrong-path completion +0.25, forced stop −0.5, error −1 |
| `result_quality` / `efficiency` | Diagnostic gate | Whether the observation genuinely succeeded; severe redundancy caps the reward |

- **Curriculum learning**: for the first 30 steps the `tool`/`argument`/`outcome` weights are scaled by 1/3 (`RewardCalculator.schedule`). Empirical finding: an SFT starting point already knows ReAct structure (format starts at 0.983), and a controlled comparison of the two curriculum arms showed no measurable difference — **when initializing from SFT, set `--reward_curriculum_steps 0` directly**; keep the schedule for cold-start scenarios.
- **Non-compensable failures**: parse failures, execution failures, and fake FINISH can only earn negative rewards and cannot be offset by format points; when `analyze_figure` lacks a valid answer, the total reward is capped at −0.75.
- All rewards are **rule-based and verifiable** (RLVR); every trajectory logs `reward_components` in detail for auditing. See the [multi-granular reward doc](docs/multigranular_rl.md) for the exact scoring rules.

### Rollout Isolation

`RolloutSandbox` in `rl/sandbox.py` records the environment/store/artifact-directory baselines before each trajectory and restores and cleans up afterwards — multi-turn GRPO, rollouts, and the benchmark all share this reset contract, eliminating state leakage across trajectories.

---

## 🛠️ Training Paths (SFT → DPO → GRPO / OPD)

### Stage 1: SFT

```bash
python scripts/generate_parametric_sft_data.py   # derive expert trajectories parametrically (no API needed)
python -m AgenticArxiv.rl.train_sft              # produces outputs/sft/final
```

> Released: [AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT) (full fine-tune of Qwen2.5-1.5B, 2 epochs / 2628 expert trajectories, loss 0.079; covers the first 8 tools only).

### Stage 2: DPO

```bash
python scripts/generate_dpo_data.py --model outputs/sft/final --num_rollouts_per_task 8
python -m AgenticArxiv.rl.train_dpo              # produces outputs/dpo/final
```

Preference pairs come from local sampling with the SFT model (no `LLM_API_KEY` needed); when snapshots exist they are replayed offline automatically, and pairs are formed only from trajectories whose reward gap exceeds `--min_reward_gap` and whose first tool differs.

### Stage 3: GRPO

```bash
python -m AgenticArxiv.rl.build_snapshot         # build the offline snapshot (the only networking step)
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --max_turns 4

# DAPO preset (loss_type=dapo + clip-higher + overlong filtering + dynamic sampling)
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --dapo

# Sequence-level importance sampling (GSPO) and the Dr.GRPO unbiased variant
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --importance_sampling_level sequence
python -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --loss_type dr_grpo

# Multi-GPU (verified on 2 GPUs with DDP; FSDP requires torch>=2.6)
accelerate launch --config_file configs/accelerate/ddp_2gpu.yaml \
  -m AgenticArxiv.rl.train_grpo --model outputs/sft/final --no-qlora

# Training curves
python -m AgenticArxiv.rl.train_grpo --report_to tensorboard
tensorboard --logdir outputs/grpo/logs
```

> Released: [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO) (33 signal tasks, fully offline snapshot replay). Offline evaluation (seed 45 / 3 repeats, strict success rate pass³), SFT → GRPO: **rl_train 0.081→0.636, dev 0.000→0.375, iid_test 0.056→0.444, ood_test 0.000→0.500**.

**Multi-turn rollout and scoring** (`rl/grpo_reward.py`): each turn, the current policy generates a ReAct action, an independent `MockArxivEnv` executes it and the observation is spliced back into the context; assistant tokens enter the loss while environment tokens only provide context via `env_mask=0`; the finished trajectory is scored by the five-component `RewardCalculator`, under the same standard as rollout / benchmark.

**Training quality guards** (silent failures → loud errors): generation-length health checks, a zero-variance guard (`RewardVarianceGuard`), canary mid-training evaluation with early stopping, stage validation thresholds (SFT parseable rate ≥ 0.3, DPO reward ≥ −0.3, GRPO reward ≥ −0.2), mixed-precision adaptation, and pre-validation of the logging backend. Beyond TRL's built-in metrics, the curves log `reward_components/*` (per-component detail), `reward_weights/*` (curriculum weights), and `rollout/*` (turns / finished / parse_error_rate).

### Stage 3': OPD (on-policy distillation, optional, an alternative to GRPO)

```bash
python -m AgenticArxiv.rl.train_opd --model outputs/sft/final \
  --teacher Qwen/Qwen2.5-7B-Instruct --max_turns 4 --snapshot data/mock_arxiv_snapshot.json
```

The student samples on-policy on task prompts, the teacher provides per-token logprobs, and the loss is reverse-KL (mode-seeking). How OPD relates to GRPO:

| Dimension | GRPO | OPD |
|---|---|---|
| Learning signal | Verifiable reward (sparse, trajectory-level) | Teacher per-token logprobs (dense) |
| Extra model | None | A teacher model (local weights required) |
| Ceiling | Can explore beyond the teacher | Converges to the teacher's behavior |
| Best for | Verifiable rewards available | Strong teacher, exploration budget tight |

### Stage 4: PPO — ⚠️ unavailable under the current dependencies

TRL has removed the classic PPOTrainer (it does not exist on `trl>=0.28.0`), and `train_ppo.py` prints a clear explanation and exits at import time. With verifiable rewards use GRPO; with a strong teacher use OPD — both use less GPU memory than PPO.

### Multimodal post-training (env-side VLM)

```bash
python scripts/build_figure_qa_dataset.py                          # figure QA seed data
python -m AgenticArxiv.rl.train_vlm_figure_qa                      # Qwen3-VL-4B LoRA post-training
FIGURE_ANALYSIS_BACKEND=vlm VLM_MODEL_PATH=<FigureQA 目录> \
  python -m AgenticArxiv.rl.backfill_figure_analysis --snapshot data/mock_arxiv_snapshot.json
```

> Released: [AgenticArXiv-RL-Qwen3-VL-4B-FigureQA](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA) (451 figures / 73 papers, split by paper to prevent leakage; on the 94 held-out figures ROUGE-1 0.105→0.126 and ROUGE-L 0.081→0.102).

---

## 📦 Released Models

| Model | Base | Notes |
|------|------|------|
| [AgenticArXiv-RL-Qwen2.5-1.5B-SFT](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT) | Qwen2.5-1.5B | Stage 1 full-parameter SFT (first 8 tools) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-GRPO](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO) | Qwen2.5-1.5B | Stage 3 GRPO (four-split evaluation above) |
| [AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-SFT-T5) | Qwen2.5-1.5B | SFT + the `analyze_figure` four-step chain (rl_train 0.485 / dev 0.250 / iid 0.278 / ood 0.250, pass³) |
| [AgenticArXiv-RL-Qwen3-VL-4B-FigureQA](https://www.modelscope.cn/models/Algorineko/AgenticArXiv-RL-Qwen3-VL-4B-FigureQA) | Qwen3-VL-4B | env-side figure-analysis VLM |

All models are hosted on ModelScope (`Algorineko/AgenticArXiv-RL-*`); each model card includes the offline evaluation table and an honest statement of limitations.

---

## 🧪 Task Suite and Evaluation

- **Smoke set** (`benchmark/tasks.py`): 8 tasks (search / download / translate / cache / composite)
- **Expanded set** (`benchmark/tasks_expanded.py`): 81 tasks across thirteen template categories (search / keyword_search / ref_form / composite / state / optional / constraint / long_chain / infeasible / paper_reading / paper_summary / figure_extraction / **figure_analysis**), enabled with `run_benchmark.py --task-set expanded`. Both task sets go through the same `TaskSpec`: `expected_tools` / `expected_tool_args` are derived from the same `steps`, so two hand-written lists can never drift apart.
- **Splits**: train / iid_test / ood_test are split by template (`benchmark/splits.py`, frozen under `data/splits/`); `rl_train` keeps only the middle band of success rates (the two ends have zero within-group variance and produce no gradient). After switching models the rates must be re-measured — old bands cannot be reused.
- **Evaluation metrics**: `pass^k` reliability (tau-bench convention), `false_finish` (the degenerate `always_finish` policy scores 91.5% vs `reference` 0%), `ref_score` (compares the `paper_id` rather than the `ref` spelling), and cost normalized by successful runs.
- **Discrimination gate** (`run_baselines.py`): deterministic degenerate policies must clear per-category thresholds — "always search cs.AI" scores 0.833→0.446 on retrieval categories, and "calling tools when none are needed" moves +0.165→−0.235.
- **Bad-case replay** (`eval/badcase_replay.py` + `eval/eval_cases.jsonl`, 14 cases): failed trajectories are frozen into permanent regression cases; replay runs only the scorers, and `pytest` is the gate; `hack/*` records reward-hacking patterns with threshold assertions, doubling as a reward-hacking case library.

```bash
python -m AgenticArxiv.benchmark.run_benchmark --task-set expanded --split iid_test --offline
```

---

## 🛡️ Dependencies

**Core dependencies** (`AgenticArxiv/requirements.txt`): `torch>=2.0`, `transformers>=4.45`, `trl>=0.28.0` (verified on 0.29.1; the lower bound is set by the multi-turn GRPO rollout_func path), `peft`, `bitsandbytes` (QLoRA), `datasets`, `accelerate`, `arxiv`, `requests`, `python-dotenv`, `loguru`, `pydantic>=2`, `fire`.

**Optional dependencies** (`requirements-extra.txt`): `pdf2zh` (real PDF translation), `fastapi`/`uvicorn`/`sqlalchemy`/`pymysql` (the archived web compatibility layer), `mcp` (the archived MCP compatibility layer), `tensorboard`/`wandb` (curve backends), `matplotlib`/`numpy`/`pandas` (plotting scripts).

---

## 🔗 Related Resources

- **Docs**: [Toolset Evolution Design](docs/toolset_evolution.md) · [Multi-granular Reward](docs/multigranular_rl.md) · [Metric Statistics](docs/metric_stats.md) · [Roadmap Research Notes](docs/roadmap_notes.md) · [TRL documentation](https://huggingface.co/docs/trl/)
- **Method papers**: RLVR; DPO (Stanford, 2023); [On-Policy Distillation (Thinking Machines, 2025)](https://thinkingmachines.ai/blog/on-policy-distillation/); [GKD (arXiv:2306.13649)](https://arxiv.org/abs/2306.13649)
- **Original project**: [AgenticArXiv](https://github.com/Algorineko/AgenticArXiv) (web app edition, FastAPI + Vue3 + MySQL)

---

## 🤝 Contributing

Issues and PRs are welcome! Workflow: Fork → feature branch → `pytest AgenticArxiv/tests/` → `feat:`-prefixed commits → PR. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

MIT License

---

## 🙋 FAQ

**Q: How does this differ from the original AgenticArXiv?** The original is a production-grade web app (FastAPI + Vue3 + MySQL, three agent architectures); this project is a pure Python + JSONL RL training environment that keeps only the single-policy ReAct setup, centered on SFT/DPO/GRPO training and verifiable rewards.

**Q: Why GRPO instead of PPO?** GRPO needs no value model (lower memory footprint), suits small models at the 1.5B scale, and is simple to implement and debug; PPO is a better fit for production-grade training of large models.

**Q: Why don't summary/figure-analysis quality enter the reward?** Quality scoring would require an LLM-as-judge, which brings non-determinism and a wider hacking surface. By design, interpretation is reduced to a **tool-call decision problem** (when to call, what to call, and with which arguments) — fully rule-judgeable and reproducible, which is a prerequisite for RLVR.

---

## 📝 Roadmap

> Full rationale and introduction paths in [docs/roadmap_notes.md](docs/roadmap_notes.md). Feel free to pick up an item (see 🤝 Contributing).

### P0 — On-Device Multimodal Paper Assistant (ultimate goal)

- [ ] **Multimodality on the policy side**: move the VLM from the env side into the policy (observations carry figures), reusing the TRL vision-language path already validated in `train_vlm_figure_qa.py`; candidate bases Qwen3-VL-4B / Qwen2.5-VL-2B (on-device tier)
- [ ] **End-to-end reading chain**: retrieval → download → translation → summarization → figure analysis closed-loop within a single model, optimized for on-device inference (quantization + 2-4B scale)
- [ ] **Translated text into the context**: `translate_arxiv_pdf` currently only produces files and the translated text never enters the model's context — add a verifiable "read the translation" tool

### P1 — Next-Generation Agentic RL Algorithms

- [x] **GSPO / Dr.GRPO wiring**: `--importance_sampling_level sequence` (sequence-level importance sampling, GSPO) and `--loss_type dr_grpo` (unbiased length normalization) are wired into `train_grpo.py`, orthogonal to and composable with the `--dapo` preset
- [ ] **Train and compare the new methods**: run GSPO / Dr.GRPO / DAPO variants on the frozen splits and release new checkpoints (GRPO baseline: rl_train 0.636 / dev 0.375 / iid 0.444 / ood 0.500)
- [ ] **SAO-style asynchronous training**: port verl `fully_async_policy` / AReaL, starting with skip-observation masking and DIS two-sided clipping ([arXiv:2607.07508](https://arxiv.org/abs/2607.07508), official code not released)
- [ ] **Cross-step credit assignment**: GiGPO / ARPO-style group-relative cross-step advantages, to ease the trajectory-level sparse signal on long-horizon chains (in-house research item)

### P2 — Jev-Style Discriminative Decision Component

- [ ] **Jev-style discriminative head**: non-autoregressive structured decisions (choice/score, at ~1/400 the cost of an LLM); candidate landing spots: ① a `result_quality` discriminative head replacing the rules ② a reward-component router (replacing the fixed curriculum schedule) ③ structured decoding for tool selection (a closed enumeration is naturally suited to a discriminative model). Constraint: deterministic inference, or diagnostics-only, to keep RLVR reproducible

### P3 — Bounded RSI (restricted self-evolution)

- [ ] **Self-evolving data loop**: held-out evaluation exposes weaknesses → the bad-case library grows automatically → parametric derivation of targeted data → retrain → freeze a new held-out set (reuses the existing evaluation/data/training pipelines; all prerequisites are in place). Acceptance: the four-split pass³ must not regress across rounds, and the bad-case library only grows (the four-step loop of [arXiv:2609.11873](https://arxiv.org/abs/2609.11873))

### ⛔ Environment Blockers

- [ ] **vLLM-accelerated sampling**: TRL 0.29 requires vLLM 0.10.2–0.12.0, and the platform-customized 0.6.2 installed on this machine makes GRPOTrainer fail at import. This is an environment dependency rather than a code change — waiting for the platform to ship a matching build.

---

**Start your Agentic RL training journey!** 🚀
