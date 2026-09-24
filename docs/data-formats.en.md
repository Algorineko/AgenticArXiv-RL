# Data Formats and File Contracts

> Scope: the benchmark, offline RL, and SFT data paths in the current `main`.
> This guide documents fields that the source actually reads or writes; example
> values are not model results.

## 1. Establish the working directory

Let `AgenticArXiv-RL/` mean the repository root. Paths below are relative to
that root unless stated otherwise:

- `AgenticArxiv/benchmark/*.py` contains the Python package. Run
  `python -m benchmark...` from `AgenticArxiv/`.
- `scripts/*.py` lives at the repository root and is run from the root.
- `data/splits/*.json` is versioned split input. JSONL and manifests under
  `data/sft/` are normally local artifacts; do not mistake unreviewed data for
  a built-in dataset.
- `data/mock_arxiv_snapshot.json` is an environment snapshot, not a task
  definition and not a benchmark report.

The data path is:

```text
TaskSpec / tasks_expanded.py
        │
        ├── build() ─────────────── benchmark task dict
        │                              │
        │                              ├── benchmark metrics/report
        │                              └── split JSON selects tasks
        │
        ├── generate_sft_data.py ── messages JSONL
        │                              │
        │                              ├── linguistic augmentation
        │                              └── parametric augmentation
        │
        └── MockArxivEnv snapshot ── replayable observations
                                       │
                                       └── rollout / reward / traces
```

## 2. Task declarations: `Step` and `TaskSpec`

Source: `AgenticArxiv/benchmark/task_spec.py`; expanded tasks:
`AgenticArxiv/benchmark/tasks_expanded.py`.

### 2.1 `Step`

`Step` is an immutable dataclass with two fields:

| Field | Type | Meaning | Common misuse |
|---|---|---|---|
| `tool` | `str` | Registered tool name, such as `download_arxiv_pdf` | Writing a natural-language action or an unregistered name |
| `args` | `Dict[str, Any]` | Arguments passed to the tool | Treating framework `session_id` as a required model argument |

`setup` also contains `Step` objects, but the runner uses it to prepare session
state before the task. Setup is not part of the agent trajectory and does not
enter `expected_tools`.

### 2.2 `TaskSpec` fields

| Field | Default | Meaning / source | Common misuse |
|---|---|---|---|
| `id` | none | Stable task identifier; `build()` checks uniqueness | Referencing an id absent from the split |
| `task` | none | Natural-language request shown to the model | Editing text without updating parameters |
| `steps` | `()` | Reference tool-call sequence | Treating an empty sequence as a missing oracle |
| `category` | `misc` | Task category such as `search` or `paper_summary` | Assuming every category is a `--tasks` CLI choice |
| `difficulty` | `medium` | Declared difficulty label | Treating it as a measured success rate |
| `setup` | `()` | State preparation before the task | Counting setup as agent accuracy or an SFT action |
| `template` | `None` | Override for template-level splitting | Randomly splitting parameter variants |
| `requires_offline` | `False` | Task depends on fixed snapshot papers/results | Comparing it without the matching snapshot |
| `note` | `""` | Developer-facing note, not a model answer | Inferring scoring rules that are not declared |
| `termination` | `FINISH` | Expected termination marker | Treating `FINISH` as proof of business success |
| `terminal_mode` | `completed` | `completed` or `blocked` | Adding a blocked reason to ordinary tasks |
| `terminal_reason` | `None` | Verifiable reason for a blocked task | Replacing a specific reason with “cannot do it” |
| `max_iterations` | `None` | ReAct iteration limit | Letting a long chain hit the default limit |
| `depends_on` | `None` | Earlier task id in the same session | Treating it as another serialization of `setup` |

### 2.3 The two reference oracles share one source

`TaskSpec.to_task()` derives both lists from the same `steps` sequence:

```json
{
  "expected_tools": [
    "get_recently_submitted_cs_papers",
    "download_arxiv_pdf",
    "summarize_paper"
  ],
  "expected_tool_args": [
    {"aspect": "CV", "days": 7, "max_results": 5},
    {"ref": 2},
    {"ref": 2, "style": "structured", "max_words": 120}
  ]
}
```

The list lengths therefore cannot drift. Add a `Step` instead of hand-writing
parallel lists in a task dictionary.

The serialized shape of a current task is:

```json
{
  "id": "summary_cv5_structured120",
  "task": "检索最近7天计算机视觉(cs.CV)论文5篇，下载第2篇，用 120 词以内的结构化风格(structured)总结它",
  "expected_tools": [
    "get_recently_submitted_cs_papers",
    "download_arxiv_pdf",
    "summarize_paper"
  ],
  "expected_tool_args": [
    {"aspect": "CV", "days": 7, "max_results": 5},
    {"ref": 2},
    {"ref": 2, "style": "structured", "max_words": 120}
  ],
  "expected_termination": "FINISH",
  "category": "paper_summary",
  "difficulty": "hard",
  "requires_offline": true,
  "note": "T3：奖励只判工具决策与参数，不判摘要文字质量"
}
```

`null` inside `expected_tool_args` has a specific meaning: the step exists, but
its argument is determined by an earlier observation, so argument scoring skips
that item. An empty list `[]` means that no tool call is expected. Do not mix
the two meanings.

### 2.4 The blocked-task termination contract

An infeasible task still has a ground truth. Its reference behavior is “call no
tool and explain the precise blocking reason before FINISH”. The current allowed
`terminal_reason` values are:

| Reason | Meaning | Example |
|---|---|---|
| `missing_context` | The session has no state to resolve a reference | “translate the paper just mentioned” with no setup |
| `invalid_reference` | The ref is out of range or invalid | “download paper 0” |
| `paper_not_found` | The requested paper is absent from the snapshot/session | An unknown arXiv id |
| `unsupported_capability` | The tool set cannot perform the request | Find an author email and send mail |

The minimal shape is:

```json
{
  "id": "infeasible_no_session",
  "task": "把刚才那篇论文翻译一下",
  "expected_tools": [],
  "expected_tool_args": [],
  "expected_termination": "FINISH",
  "category": "infeasible",
  "difficulty": "hard",
  "expected_terminal_mode": "blocked",
  "expected_terminal_reason": "missing_context"
}
```

For a `blocked` task, “completed” is classified as a false completion.
`reference_terminal_thought()` supplies example wording for the declared reason.
Do not treat “no tools + generic FINISH” as a correct blocked answer.

### 2.5 Parameterized task families

`family()` receives the same parameter mapping in its `text` and `steps`
functions, so the natural language and reference calls render from one source.
When adding a family:

1. Put changing values in the parameter object, not separately in the text and `Step`.
2. Generate a stable id for every instance.
3. Let `build()` detect duplicate ids.
4. Use the parameterized SFT generator’s lineage fields instead of copying held-out tasks into train.

## 3. Split files: `train`, `dev`, `iid_test`, `ood_test`

Source: `AgenticArxiv/benchmark/splits.py`; versioned files:
`data/splits/v1.json`, `v2_62.json`, and `v3_81.json`. An explicit file reference
has this form:

```text
data/splits/v3_81.json:train
```

### 3.1 File shape

```json
{
  "version": 3,
  "task_set": "expanded",
  "task_count": 81,
  "derived_from": "data/splits/v2_62.json",
  "split": {
    "train": ["..."],
    "dev": ["..."],
    "iid_test": ["..."],
    "ood_test": ["..."]
  },
  "ood_keys": [["composite", 3], ["composite", 4]],
  "pilot_dev_ids": ["..."],
  "rates": {"search_AI_30d_25": 0.0},
  "rates_metadata": {"metric": "strict_success_rate", "repeat": 3},
  "policy": {"rates": "Derived only from the frozen Base run over train."}
}
```

Current static counts are:

| File | train | dev | iid_test | ood_test | Total |
|---|---:|---:|---:|---:|---:|
| `v1.json` (historical default) | 42 | 0 | 13 | 4 | 59 |
| `v2_62.json` (historical expanded) | 36 | 8 | 14 | 4 | 62 |
| `v3_81.json` (current expanded) | 51 | 8 | 18 | 4 | 81 |

The v3 `rates` still cover only the 36 train tasks inherited from v2. New
`paper_reading`, `paper_summary`, `figure_extraction`, and `figure_analysis`
families are present in the task set and split, but do not receive invented
success rates.

### 3.2 Why split by template

`template_key()` defaults to `(template or category, len(expected_tools))`. This
keeps parameter variants of one tool-chain template on the same side. Otherwise
`search_AI_1d_3` in train and `search_AI_30d_25` in test would measure memorized
wording instead of generalization.

- `iid_test`: unseen parameter instances of a seen template.
- `ood_test`: a held-out template key or chain length.
- `dev`: already inspected in pilots, debugging, or badcase analysis; not a blind test.
- `train`: eligible for SFT; GRPO further filters it by `rates` into `rl_train`.

### 3.3 `rates` and dynamic `rl_train`

`rl_train` is not an array in the JSON. `load_split()` computes it from `rates`:

```text
FLOOR_MAX = 0.2
CEILING_MIN = 0.8
middle = 0.2 <= rate <= 0.8
rl_train = train ∩ {tasks with a rate in the middle band}
```

Six recorded v3 rates are currently in the middle band, so the current v3 file
can derive six `rl_train` ids. That does not mean all 51 train tasks have rates.
Changing the model, snapshot, or evaluation policy requires re-measuring rates;
old difficulty bands must not be reused silently.

Without `rates`, loading `rl_train` raises. If no train task is in the middle
band, it also raises instead of silently creating an empty training set.

### 3.4 Leakage-safe usage

```bash
# From the repository root: generate expanded train-only SFT data
python scripts/generate_sft_data.py \
  --task_set expanded \
  --split data/splits/v3_81.json:train \
  --snapshot data/mock_arxiv_snapshot.json \
  --output data/sft/sft_v3_train.jsonl

# From AgenticArxiv/ for benchmark commands
cd AgenticArxiv
python -m benchmark.run_benchmark \
  --task-set expanded --offline \
  --split ../data/splits/v3_81.json:iid_test
```

The SFT generator rejects a bare `train`, a split without a versioned file, and
`dev`, `iid_test`, or `ood_test` for expanded data. The benchmark checks that all
split ids are in the current task pool. Forgetting `--task-set expanded` or
`--offline` can make the pool incomplete; the command should fail rather than
produce an apparently comparable number.

## 4. Trajectory JSONL

Source: `AgenticArxiv/rl/trajectory.py`. One JSONL line is one `Trajectory`,
whose `steps` list contains `TrajectoryStep` records.

### 4.1 Top-level fields

| Field | Type | Source / meaning | Common misuse |
|---|---|---|---|
| `task_id` | `str` | Task id | Treating it as a split name |
| `task` | `str` | Natural-language task at collection time | Dropping it and losing replay context |
| `session_id` | `str` | Session identifier | Inferring paper identity from it; snapshot keys remove it |
| `steps` | `list` | Multiple `TrajectoryStep` records | Calling benchmark `history` the same schema |
| `final_reward` | `float` | Final scalar reward | Treating it as strict success rate |
| `metrics` | `object` | Dictionary form of `TaskMetrics` | Declaring success from `task_completed` alone |
| `timestamp` | `str` | Timestamp from `create_trajectory()` | Using it as an experiment version |
| `model` | `str` | Model name, possibly empty | Inventing a model result when absent |
| `termination_type` | `str` | `FINISH` / `FORCE_STOP` / `ERROR`, etc. | Treating `FORCE_STOP` as normal completion |
| `reward_components` | `object` | Auditable `RewardBreakdown.to_dict()` | Recording only total and losing reward-hacking evidence |

### 4.2 `TrajectoryStep` fields

| Field | Meaning |
|---|---|
| `step` | 1-based trajectory index |
| `thought` | Current Thought text |
| `action` | JSON string, or `FINISH` / `FORCE_STOP` / `ERROR` |
| `observation` | Tool or environment output string |
| `llm_latency_ms` | Current-step LLM latency, default 0 |
| `tool_latency_ms` | Current-step tool latency, default 0 |
| `parse_failed` | Whether action parsing failed, default `false` |

The following anonymous example uses the same fields and action types as the
dataclasses. The observation text is intentionally shortened:

```json
{
  "task_id": "search_AI_30d_25",
  "task": "检索最近30天人工智能论文，最多25篇",
  "session_id": "bench_r1_task0",
  "steps": [
    {
      "step": 1,
      "thought": "需要先检索论文",
      "action": "{\"name\": \"get_recently_submitted_cs_papers\", \"args\": {\"aspect\": \"AI\", \"days\": 30, \"max_results\": 25}}",
      "observation": "成功获取 25 篇论文",
      "llm_latency_ms": 120,
      "tool_latency_ms": 4,
      "parse_failed": false
    },
    {
      "step": 2,
      "thought": "检索完成，结束任务",
      "action": "FINISH",
      "observation": "任务完成",
      "llm_latency_ms": 80,
      "tool_latency_ms": 0,
      "parse_failed": false
    }
  ],
  "final_reward": 0.91,
  "metrics": {
    "task_id": "search_AI_30d_25",
    "agent_type": "regex",
    "trial": 0,
    "task_completed": true,
    "termination_type": "FINISH",
    "tool_call_accurate": true,
    "arg_score": 1.0,
    "ref_score": 1.0,
    "parse_failures": 0,
    "tool_exec_failures": 0
  },
  "timestamp": "2026-09-24T00:00:00",
  "model": "example-only",
  "termination_type": "FINISH",
  "reward_components": {
    "total": 0.91,
    "format": 1.0,
    "tool": 1.0,
    "argument": 1.0,
    "process": 1.0,
    "outcome": 1.0,
    "result_quality": 1.0,
    "efficiency": 1.0,
    "weights": {"format": 1.0, "tool": 3.0, "argument": 2.0, "process": 1.0, "outcome": 3.0, "result_quality": 0.0, "efficiency": 0.0}
  }
}
```

`load_trajectories()` reconstructs `TrajectoryStep` objects. Older files lacking
`model`, `termination_type`, or `reward_components` still load because the
dataclass fields have defaults. New fields should preserve this compatibility.

### 4.3 Trajectory and benchmark traces are different contracts

`benchmark.run_benchmark --save-traces` writes only:

```json
{
  "task_id": "search_AI_30d_25",
  "agent_type": "regex",
  "trial": 0,
  "session_id": "bench_r1_task0",
  "history": [
    {"thought": "...", "action": "...", "observation": "..."}
  ]
}
```

This format is for `eval/badcase_replay.py` and `rescore_traces.py`. It is not a
`Trajectory` JSONL record: it has no `final_reward`, `metrics`, `timestamp`, or
`reward_components`. The rescoring command also reads old timing/token values
from the companion `summary.json` and combines them with the trace history.

## 5. SFT JSONL and manifests

Sources: `scripts/generate_sft_data.py`,
`scripts/generate_parametric_sft_data.py`, `scripts/augment_sft_data.py`,
`scripts/augment_parametric_sft_data.py`, and
`scripts/build_sft_train_mix.py`.

### 5.1 Base SFT sample

Both deterministic and LLM expert generators use `messages` as the core of a
two-message sample:

```json
{
  "messages": [
    {"role": "user", "content": "当前任务：<task>\n请按照ReAct框架的格式思考和行动:"},
    {"role": "assistant", "content": "Thought: <当前步思考>\nAction: {\"name\": \"<tool>\", \"args\": {}}"}
  ],
  "source_task_id": "<TaskSpec.id>",
  "source_split": "v3_81.json:train",
  "trajectory_step": 0
}
```

Field contract:

| Field | Required | Source | Common misuse |
|---|---|---|---|
| `messages` | yes | Prompt template plus one history decision | Dropping the earlier history |
| `messages[0].role` | yes | Always `user` | Changing it to `system` while assuming the parser still finds task boundaries |
| `messages[1].role` | yes | Always `assistant` | Ending without an Action so the training conversion rejects it |
| `source_task_id` | generator writes it | Spec id | Omitting it and losing semantic provenance |
| `source_split` | generator writes it | Version filename plus `:train` | Writing an absolute path and making lineage non-portable |
| `trajectory_step` | generator writes it | 0-based history index | Treating it as the environment’s 1-based step |

Only non-empty actions other than `PARSE_ERROR`, `ERROR`, and `FORCE_STOP` become
SFT samples. The deterministic expert first executes setup/dependency state and
strictly validates the trajectory. A blocked task produces a reasoned FINISH
sample, not a tool call.

### 5.2 Parametric and linguistic augmentation fields

Parametric seed rows add:

```json
{
  "derived_task_id": "augp1_search_AI_30d_25_2d_4",
  "parent_task_id": "search_AI_30d_25",
  "generation_parameters": {"aspect": "AI", "days": 2, "max_results": 4},
  "dataset_stage": "parametric_v1_expert_seed",
  "sample_sha256": "<sha256(messages canonical JSON)>"
}
```

Validation rejects a parent outside train, a held-out parent, copied benchmark
text, changed tool/setup topology, or duplicate sample fingerprints.

Linguistic augmentation changes only the task wording and current Thought. The
Action, arguments, previous observations, and step order remain unchanged. It
adds:

```json
{
  "parent_sample_sha256": "<seed messages hash>",
  "sample_sha256": "<augmented messages hash>",
  "augmentation": {
    "kind": "linguistic_semantics_preserving",
    "task_variant": 0,
    "thought_variant": 1
  }
}
```

The generator’s six task wrappers × two Thought variants are actual expansion
rules. They add linguistic views, not independent semantic tasks.

### 5.3 Minimum manifest audit fields

Manifest `kind` differs by stage, but a reproducible manifest normally contains:

| Field | Meaning |
|---|---|
| `version` | Manifest format version, not split version |
| `kind` or `augmentation_kind` | Machine-readable dataset stage |
| `input` / `output` | Input and output paths |
| `input_sha256` / `output_sha256` | Content fingerprints |
| `split_file` / `split_sha256` | Task boundary and fingerprint |
| `snapshot` / `snapshot_sha256` | Snapshot used to generate expert trajectories |
| `seed_rows` / `output_rows` | Input and output row counts |
| `unique_sample_fingerprints` | Number of distinct message fingerprints |
| `source_tasks` or `semantic_task_instances` | Semantic task count, not JSONL row count |
| `heldout_overlap` / `heldout_parent_overlap` | Leakage audit; should be 0 |
| `sources` | Source files and manifest fingerprints in a mixture |

The final `sft_v3_train_mix.jsonl.manifest.json` must have `kind`
`qlora_sft_train_mix`. `train_sft.py` checks kind, output SHA-256, row count,
and unique fingerprints by default. `--skip_data_manifest_check` is for temporary
experiments only and cannot support a formal reproducibility claim.

### 5.4 Leakage-safe validation order

1. Select the versioned split’s `train`; do not generate all expanded tasks and delete later.
2. Record `source_split`; do not replace it with a machine-specific absolute path.
3. Check that every parametric parent is in train and that tool-chain topology is unchanged.
4. Check that linguistic source ids equal the complete train set and do not copy held-out text.
5. Validate both input manifests, row counts, hashes, and message de-duplication when mixing.
6. Let `train_sft.py` validate the final manifest again before training.

## 6. Minimal review checklist

Before submitting data or a data-tool change, confirm:

- [ ] Every task id exists in the selected task set.
- [ ] `expected_tools` and `expected_tool_args` come from one `steps` sequence.
- [ ] `setup` is not counted as a model trajectory.
- [ ] `[]` (correctly no call) and `None` (argument oracle not applicable) are distinct.
- [ ] Training references explicit `FILE:train`, not bare `train`.
- [ ] Rate model, agent, backend, repeat count, and snapshot metadata are recorded.
- [ ] Every JSONL row has `source_task_id` and `source_split`.
- [ ] Every augmentation or mixture recomputes `sample_sha256` and has no duplicates.
- [ ] Manifest hashes match the files on disk.
- [ ] Documentation does not turn JSONL row count into semantic-task count or fixtures into results.

## 7. Source map

| Contract | Source |
|---|---|
| Task and terminal semantics | `AgenticArxiv/benchmark/task_spec.py` |
| Expanded tasks | `AgenticArxiv/benchmark/tasks_expanded.py` |
| Split parsing and `rl_train` | `AgenticArxiv/benchmark/splits.py` |
| Trajectory classes and JSONL I/O | `AgenticArxiv/rl/trajectory.py` |
| Benchmark metrics | `AgenticArxiv/benchmark/metrics.py` |
| Base SFT generation | `scripts/generate_sft_data.py` |
| Parametric seed | `scripts/generate_parametric_sft_data.py` |
| Linguistic augmentation | `scripts/augment_sft_data.py`, `scripts/augment_parametric_sft_data.py` |
| Mixture and manifest | `scripts/build_sft_train_mix.py` |
| Pre-training audit | `AgenticArxiv/rl/train_sft.py` |
