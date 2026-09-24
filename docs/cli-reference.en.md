# CLI Command Reference

> Scope: the current source tree. Each command lists its working directory,
> inputs, outputs, network/snapshot requirements, and hardware boundary. This
> is a parameter reference, not a claim that the commands were run locally.

## 1. Working directories and prerequisites

| Command family | Run from | Typical requirements |
|---|---|---|
| `python -m benchmark.*` | `AgenticArxiv/` | Python package and registered tools; API runs also need LLM configuration |
| `python -m rl.*` | `AgenticArxiv/` | Python package; snapshot building needs network/arXiv access |
| `python scripts/*.py` | Repository root | The scripts add `AgenticArxiv/` to `sys.path` |
| `python eval/badcase_replay.py` | Repository root | No LLM, network, or model |
| `python draw/plot.py` | Repository root | `draw/` and report data |

Path conventions:

- From `AgenticArxiv/`, refer to root data as `../data/...`.
- From the repository root, use `data/...`.
- `--offline` controls the tool environment; `--backend transformers` only
  controls where the LLM is loaded from. A local model can still use real tools
  unless `--offline` is also specified.
- “Needs GPU” below describes a code-path prerequisite, not a claim about a
  successful training run.

## 2. Build a MockArxivEnv snapshot

Entry point: `AgenticArxiv/rl/build_snapshot.py`.

From `AgenticArxiv/`:

```bash
python -m rl.build_snapshot
```

This is the snapshot pipeline’s only step that normally needs arXiv access. It
records search pools, paper text, summaries, figure extraction, and figure
analysis. Afterward, rollout, rescoring, and most training paths can use
`MockArxivEnv(mode="replay")` offline.

### 2.1 Arguments

| Argument | Default | Purpose | Requirement / note |
|---|---|---|---|
| `--snapshot` | `data/mock_arxiv_snapshot.json` (resolved from the repository root) | Output snapshot path | Parent directories are created |
| `--aspects` | `* AI LG CL CV RO CR` | CS areas to fetch | `nargs="+"` |
| `--keyword-queries` | Three built-in queries | Keyword-search pools | `nargs="+"`; replaces the defaults |
| `--max_results` | `50` | Maximum results per search pool | Not the content prefetch depth |
| `--days` | `30` | Search time window | Part of the search key |
| `--no-pin-references` | off | Do not pin benchmark anchor papers | For raw data research only; not recommended for formal evaluation |
| `--allow-partial` | off | Save after some queries fail | Not recommended for formal training/evaluation |
| `--content-workers` | `8` | PDF prefetch threads | Downloads are parallel; the in-memory store is not shared across workers |
| `--skip-prefetch` | off | Skip parallel PDF prefetch | Useful when PDFs already exist locally |
| `--prefetch-budget` | `900` seconds | Overall prefetch wall-clock budget | Timed-out jobs are retried in the serial stage |
| `--content-max-ref` | `0` | Prefetch content for only the first N papers per pool | `0` means all; search pool size is unchanged |

Example:

```bash
# From AgenticArxiv/: cover common areas and limit content depth
python -m rl.build_snapshot \
  --aspects AI LG CL CV \
  --max_results 30 \
  --content-max-ref 5 \
  --snapshot ../data/mock_arxiv_snapshot.json
```

`--no-pin-references` skips the AI/CV papers needed by benchmark anchors and can
make title/arXiv-id reference tasks unreplayable. `--allow-partial` does not turn
failures into successes; later replay misses still expose the missing records.

## 3. Benchmark runner

Entry point: `AgenticArxiv/benchmark/run_benchmark.py`.

```bash
cd AgenticArxiv
python -m benchmark.run_benchmark --task-set expanded --offline \
  --split ../data/splits/v3_81.json:iid_test \
  --agents regex --output ../data/bench_v3_iid
```

The default task set is the eight-task `benchmark/tasks.py` smoke set. Use
`--task-set expanded` for the 81-task set. The default agents are
`regex mcp skill_cli`, and the default repeat count is 3. For Base/SFT/GRPO
capability comparisons rather than execution-framework comparisons, use
`--agents regex` explicitly.

### 3.1 Arguments

| Argument | Default / choices | Purpose |
|---|---|---|
| `--agents` | `regex mcp skill_cli` | One or more agent types |
| `--repeat` | `3` | Repetitions per task |
| `--tasks` | `None`; choices are `search`, `download`, `translate`, `cache`, `composite`, `keyword_search`, `ref_form`, `optional`, `state`, `long_chain`, `constraint`, `infeasible` | Filter by category; newer reading categories are not in this list |
| `--task-ids` | `None` | Select task ids; takes precedence over split/category |
| `--output` | Root `data/` | Report output directory |
| `--model` | Configured agent model | API model name or local transformers directory |
| `--backend` | `api`; also `transformers` | LLM backend |
| `--local-device` | `auto` | `auto` / `cuda` / `cpu` for transformers |
| `--local-dtype` | `auto` | `auto` / `float16` / `bfloat16` / `float32` |
| `--seed` | `42` | Base seed for local generation |
| `--prefix` | `bench_r<timestamp>` | Session-id prefix |
| `--no-thinking` | off | Pass `enable_thinking=False` to supported chat templates |
| `--offline` | off | Replay tools from `data/mock_arxiv_snapshot.json` |
| `--snapshot` | `None` | Override the snapshot path |
| `--save-traces [PATH]` | off | Save histories; without PATH use `<output>/traces.jsonl` |
| `--split [FILE:]NAME` | `None` | Run one split; requires the expanded task set |
| `--task-set` | `default` | `default` or `expanded` |

The `--tasks` choices are the real argparse boundary. Although
`tasks_expanded.py` contains `paper_reading`, `paper_summary`,
`figure_extraction`, and `figure_analysis`, those names are not valid
`--tasks` values. Use `--task-ids` or a split after selecting `expanded`.

### 3.2 Modes and outputs

| Mode | LLM | Real arXiv | Use |
|---|---|---|---|
| Default API | yes | yes unless `--offline` | API integration or real-tool experiments |
| `--backend transformers` | yes, local | controlled separately by `--offline` | Local-model benchmark |
| `--offline` | yes | no; reads snapshot and uses offline download stubs | Reproducible comparisons and CI checks |
| `--save-traces` | same as above | same as above | Badcase capture or rescoring |

Typical output includes `summary.json`, `report.md`, and `raw_data.csv`, plus
`errors.csv` when errors exist. `traces.jsonl` is created only with
`--save-traces`; it uses the benchmark trace schema, not
`rl.trajectory.Trajectory`.

## 4. Deterministic reward baselines

Entry point: `AgenticArxiv/benchmark/run_baselines.py`.

```bash
python -m benchmark.run_baselines --task-set expanded
python -m benchmark.run_baselines \
  --task-set expanded --seed 42 --random-samples 20 \
  --output ../data/baselines_v3
```

The command calls no model, network, or real tool. It constructs synthetic
histories and runs the same `RewardCalculator` used by rollout/training. The
current `ALL_POLICIES` contains five policies:

| Policy | Behavior |
|---|---|
| `reference` | Replay the declared reference path as the score reference |
| `always_finish` | FINISH immediately |
| `always_search` | Make the same search call for every task |
| `random_tool` | Choose one tool using a stable seed |
| `wrong_args` | Keep reference tool names but use deliberately wrong arguments |

| Argument | Default | Purpose |
|---|---:|---|
| `--task-set` | `expanded` | `default` or `expanded` |
| `--policies` | all policies | Select names |
| `--seed` | `42` | First stable seed for `random_tool` |
| `--random-samples` | `20` | Number of consecutive seeds; at least 1 |
| `--min-reference-gap` | `0.3` | Minimum mean reward gap from `reference` |
| `--min-category-gap` | `0.3` | Minimum per-category gap |
| `--top` | `5` | High-scoring tasks shown per weak policy |
| `--training-step` | `100` | Reward curriculum step; 100 uses full correctness weights |
| `--output` | `None` | Directory for JSON and Markdown reports |

A failed health check returns exit code 1. Synthetic baselines diagnose scorer
sensitivity; they do not measure LLM quality and must not be reported as model
success rates.

## 5. Rescore saved traces

Entry point: `AgenticArxiv/benchmark/rescore_traces.py`. It loads no model. It
combines saved histories with timing/token fields from the old summary and
recomputes metrics for the selected task pool.

```bash
cd AgenticArxiv
python -m benchmark.rescore_traces \
  --traces ../data/bench_v3/traces.jsonl \
  --summary ../data/bench_v3/summary.json \
  --output ../data/bench_v3_rescored \
  --snapshot ../data/mock_arxiv_snapshot.json \
  --task-set expanded \
  --split ../data/splits/v3_81.json:iid_test
```

| Argument | Required | Meaning |
|---|---|---|
| `--traces` | yes | JSONL from `run_benchmark --save-traces` |
| `--summary` | yes | Matching benchmark `summary.json` |
| `--output` | yes | New report directory |
| `--snapshot` | no | Default `data/mock_arxiv_snapshot.json` from `AgenticArxiv/` |
| `--task-set` | no | `default` or `expanded`, default expanded |
| `--split` | no | Further restrict the task pool |

Unknown task ids are reported and skipped. If every row is skipped or no metrics
can be recomputed, the command exits. Rescoring cannot reconstruct model text,
tool time, or missing snapshot results that were not saved in the original run.

## 6. Badcase replay and capture

Entry point: `eval/badcase_replay.py`, from the repository root. Replay needs no
LLM, network, or tools; it only runs the scorer.

```bash
# Replay eval/eval_cases.jsonl
python eval/badcase_replay.py

# Explicit replay with conditions printed
python eval/badcase_replay.py replay \
  --cases eval/eval_cases.jsonl --task-set expanded \
  --training-step 100 --verbose

# Capture candidates from a benchmark trace; --dry-run does not write
python eval/badcase_replay.py capture \
  --traces data/bench_v3/traces.jsonl \
  --cases eval/eval_cases.jsonl \
  --source bench_v3 --dry-run
```

Global arguments are `--cases` (default `eval/eval_cases.jsonl`), `--task-set`
(default `expanded`), `--training-step` (default `100`), and `-v/--verbose`.
`capture` additionally requires `--traces` and accepts `--source` and
`--dry-run`. An `open` case still reproduces the issue; a `newly_fixed` case may
be changed to `fixed` only after confirmation. A regression returns 1.

## 7. SFT data generation

### 7.1 Base or expanded expert samples

Entry point: `scripts/generate_sft_data.py`, from the repository root.

```bash
python scripts/generate_sft_data.py \
  --task_set expanded \
  --split data/splits/v3_81.json:train \
  --snapshot data/mock_arxiv_snapshot.json \
  --output data/sft/sft_v3_train.jsonl
```

| Argument | Default | Purpose |
|---|---|---|
| `--output` | `data/sft/sft_train.jsonl` | JSONL output |
| `--snapshot` | `data/mock_arxiv_snapshot.json` when present | Offline snapshot; otherwise auto environment |
| `--use_llm` | off | Use configured LLM expert; default is deterministic |
| `--task_set` | `basic` | `basic` or `expanded` |
| `--split` | `None` | Expanded data requires versioned `PATH:train` |

The expanded generator rejects bare splits, the full task set, and every held-out
split. Failed or non-strict expert trajectories are not written.

### 7.2 Parametric seed

Entry point: `scripts/generate_parametric_sft_data.py`.

```bash
python scripts/generate_parametric_sft_data.py \
  --split-file data/splits/v2_62.json \
  --snapshot data/mock_arxiv_snapshot.json \
  --output data/sft/sft_v2_parametric_seed.jsonl
```

Arguments: `--split-file` defaults to `data/splits/v2_62.json`, `--snapshot`
defaults to `data/mock_arxiv_snapshot.json`, and `--output` defaults to
`data/sft/sft_v2_parametric_seed.jsonl`. The script validates parents, held-out
overlap, duplicate text, and tool topology before replaying derived tasks.

### 7.3 Linguistic augmentation and mixture

```bash
python scripts/augment_sft_data.py \
  --input data/sft/sft_v1_seed.jsonl \
  --output data/sft/sft_v1_linguistic.jsonl \
  --split-file data/splits/v2_62.json

python scripts/augment_parametric_sft_data.py \
  --input data/sft/sft_v2_parametric_seed.jsonl \
  --split-file data/splits/v2_62.json \
  --output data/sft/sft_v2_parametric_linguistic.jsonl

python scripts/build_sft_train_mix.py \
  --original data/sft/sft_v1_linguistic.jsonl \
  --parametric data/sft/sft_v2_parametric_linguistic.jsonl \
  --output data/sft/sft_v3_train_mix.jsonl \
  --seed 42
```

Augmenters write adjacent `.manifest.json` files. The mixer expects 1020 rows in
the original source and 1908 in the parametric source by default, and rejects
hash, kind, duplicate-message, or input-overwrite errors.

## 8. Training entry points and hardware boundaries

Only the options that change the experiment contract are listed here; the
corresponding `argparse` remains authoritative.

### 8.1 SFT / QLoRA

```bash
python -m AgenticArxiv.rl.train_sft --inspect_only --max_length 4096
python -m AgenticArxiv.rl.train_sft \
  --data data/sft/sft_v3_train_mix.jsonl \
  --data_manifest data/sft/sft_v3_train_mix.jsonl.manifest.json \
  --output_dir outputs/sft_qlora
```

Important defaults are `--model Qwen/Qwen2.5-1.5B-Instruct`,
`--output_dir outputs/sft_qlora`, `--epochs 3`, `--batch_size 1`,
`--grad_accum 8`, `--lr 1e-4`, `--max_length 4096`, `--max_steps -1`, QLoRA
enabled, `--inspect_only` off, manifest checking enabled, verification enabled,
and `--report_to none`. `--inspect_only` audits the manifest and token lengths
without loading a model; it is useful for CPU-only contributors but is not a
training run. QLoRA checks CUDA, BF16, and key package versions.

### 8.2 DPO

Entry point: `python -m AgenticArxiv.rl.train_dpo`. The source defaults are
`--model None`, `--data None`, `--output_dir None`, verification enabled,
`--min_reward -0.3`, `--report_to none`, and no run name. DPO requires a local SFT
model and preference data; the presence of a CLI is not evidence of a completed
training run.

### 8.3 GRPO

Entry point: `python -m AgenticArxiv.rl.train_grpo` (or the package path used by
the repository’s quick-start examples). Key defaults are `--model
outputs/dpo/final`, `--output_dir outputs/grpo`, `--max_steps -1`, `--beta 0.04`,
`--reward_curriculum_steps 30`, `--num_generations 4`,
`--max_completion_length 256`, `--task_set default`, and no split. The parser
also accepts loss types `grpo`, `dapo`, `bnpo`, `dr_grpo`, `cispo`, `sapo`, and
`luspo`, plus `--dapo`, `--snapshot`, `--allow_zero_variance`, rollout-trace
options, and the QLoRA/checkpoint options described in the Chinese reference.

`--split` requires `--task_set expanded`; `rl_train` is derived from split rates.
Constant-reward groups have no GRPO advantage, which is why the zero-variance
guard is enabled by default.

### 8.4 OPD and PPO

OPD defaults include `--model outputs/sft/final`, teacher
`Qwen/Qwen2.5-7B-Instruct`, `--output_dir outputs/opd`, `--epochs 1`,
`--temperature 0.9`, `--lmbda 1.0`, `--max_new_tokens 256`, `--max_turns 1`,
`--max_observation_tokens 256`, `--task_set default`, and canary checks every 50
steps. It requires a local teacher capable of token log-probabilities.

PPO defaults include `--model outputs/grpo/final`, `--output_dir outputs/ppo`,
`--epochs 1`, `--batch_size 4`, `--mini_batch_size 2`, `--grad_accum 1`,
`--lr 1e-6`, `--init_kl_coef 0.05`, `--max_completion_length 256`,
`--temperature 0.7`, snapshot and canary options, verification enabled, and
`--report_to none`. PPO is an experimental entry point; its existence is not a
claim that the historical design has been fully validated.

## 9. Quick troubleshooting

| Symptom | First checks |
|---|---|
| `replay mode snapshot miss` | Snapshot path, tool-key arguments, and whether a new recording is needed; do not treat a miss as a network error |
| Split ids absent from the pool | `--task-set expanded` and `--offline` for snapshot-bound tasks |
| Empty or unavailable `rl_train` | Split `rates` and train coverage; do not silently fall back to all train tasks |
| SFT rejects `source_split` | Use explicit `FILE:train`; check that dev/iid/ood was not mixed in |
| Manifest hash/row mismatch | Regenerate the manifest; do not hand-edit its hash |
| Blocked-task reward is unexpected | Ensure the FINISH Thought names the declared terminal reason |
| Numbers are not comparable | Fix split, snapshot, agent, seed, repeat, backend, and task set |
| Figure analysis is empty/out of range | Check T4 figure extraction first, then `figure_no` and snapshot backend |

## 10. Maintenance rule

This reference copies argparse options, defaults, and call paths from the source.
When an entry point changes, update this file and the English version, then
review `data-formats.md` or `offline-replay.md` if the contract changed. PR
descriptions should mark commands that were checked from source but not run
locally.
