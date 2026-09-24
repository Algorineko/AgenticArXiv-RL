# Contributing to AgenticArXiv-RL

Thank you for helping improve AgenticArXiv-RL. The project is a small, rule-
verifiable agentic RL environment around arXiv search, paper operations,
offline snapshots, benchmark tasks, and SFT/DPO/GRPO/OPD/PPO experiments. A
useful contribution does not need a GPU: documentation, schema audits, CPU
tests, deterministic scorer fixes, and better regression fixtures are all in
scope.

This guide describes the current repository, not the original Web application.
The design notes in `docs/rl_building.md` and `docs/metric_stats.md` preserve
history; the live Python modules and their tests are the implementation contract.

## 1. What is in scope

Good contribution areas include:

- the offline RL environment in `AgenticArxiv/rl/`;
- benchmark task declarations, split policies, metrics, reports, and baselines;
- deterministic reward checks and badcase fixtures;
- SFT data generation, provenance, manifests, and leakage guards;
- documentation, examples, CLI help, and reproducibility notes;
- focused tests that do not require an external API key or a model download.

The following are outside the default RL path:

- `archive/`, which preserves the former Web/API/service implementation;
- `AgenticArxivWeb/`, the archived Vue application;
- production arXiv services, real-time translation infrastructure, and database
  deployment unless a change explicitly targets those archived components.

If a change crosses the archive boundary, explain why in the PR before adding
new runtime dependencies.

## 2. Before changing code

1. Read the relevant source and its tests.
2. Check `git status` and keep unrelated user changes out of the commit.
3. Identify the data contract affected by the change: task, split, trajectory,
   snapshot, reward, SFT sample, or manifest.
4. Search for existing users with `rg` before changing a field or CLI option.
5. Decide which validation level below is possible on your machine.

Do not copy a stale README command into a new document. For every command,
confirm its `argparse`, default path, working directory, and output files in the
current entry point.

## 3. Three validation levels

### Level A: documentation-only

Use this level for Markdown, navigation, translations, and explanatory changes.

- Compare every path, option, number, and status with the current source.
- Check Markdown links and code fences.
- If a command was not run, say so in the PR description.
- Do not claim a model result, a successful training run, or an API call merely
  because the command is syntactically plausible.

Documentation changes do not require a GPU or a complete local training stack.
They should not introduce a made-up workflow just to make a page longer.

### Level B: CPU-safe deterministic checks

Use this level for task definitions, reward logic, split parsing, snapshot key
logic, JSONL transforms, and badcase changes.

```bash
pytest AgenticArxiv/tests/ -v --tb=short
```

Prefer focused tests while iterating:

```bash
pytest AgenticArxiv/tests/test_task_spec.py -q
pytest AgenticArxiv/tests/test_splits.py -q
pytest AgenticArxiv/tests/test_multigranular_reward.py -q
pytest AgenticArxiv/tests/test_badcases.py -q
pytest AgenticArxiv/tests/test_generate_sft_data.py -q
```

The exact test filename is authoritative; if a focused path is not present,
choose the nearest existing test rather than adding a command that cannot run.
Fixture-based snapshot tests are preferred to network calls.

### Level C: model, snapshot, or GPU experiments

Use this level only when the change truly needs it. State the prerequisites:

- model name or local path;
- Python/dependency versions;
- snapshot path and whether it was recorded with network access;
- GPU, dtype, and quantization settings;
- split, seed, repeat count, and agent/backend;
- output artifact paths and whether the result is committed or external.

The repository’s CI does not prove GPU training quality. Passing unit tests and
CI means the tested code paths are healthy; it does not prove a checkpoint’s
reward, convergence, or generalization.

## 4. Choosing a small, focused change

Pick one contract per PR where possible:

| If you want to change… | Start with… | Add evidence in… |
|---|---|---|
| task wording or reference calls | `benchmark/task_spec.py`, `tasks_expanded.py` | task and constraint tests |
| reward behavior | `rl/reward.py`, `benchmark/metrics.py` | reward/baseline/badcase tests |
| tool registration or arguments | `tools/`, `tools/bootstrap.py` | tool contract and offline tests |
| train/iid/ood membership | `benchmark/splits.py`, `data/splits/*.json` | split reproducibility tests and policy notes |
| SFT sample shape | `scripts/generate_sft_data.py` | generator and leakage tests |
| SFT lineage or hash | augmentation/mix scripts | manifest and deduplication tests |
| snapshot replay | `rl/env.py`, `rl/build_snapshot.py` | replay/key/miss tests |
| report fields | `benchmark/report.py`, `metrics.py` | serialization and CSV/JSON tests |
| CLI behavior | the module’s `argparse` block | parser-focused or smoke tests |

Do not combine a documentation correction with an unrelated CLI redesign simply
because both appear in the same README section. If a doc reveals a real code
bug, either split the code fix into a separate PR or explain the dependency.

## 5. Data and leakage rules

The expanded benchmark currently has 81 task declarations and a versioned
`v3_81.json` split. Keep these rules when adding data:

1. Build `expected_tools` and `expected_tool_args` from one `steps` sequence.
2. Treat empty `[]` as “the correct behavior uses no tool”; do not confuse it
   with a missing argument oracle (`None`).
3. Keep `setup` out of the agent trajectory and accuracy counts.
4. Use explicit `data/splits/<version>.json:train` for expanded SFT data.
5. Do not put `dev`, `iid_test`, or `ood_test` rows in train-only SFT artifacts.
6. Preserve `source_task_id`, `source_split`, parent ids, and sample hashes.
7. Keep held-out overlap at zero and record it in the manifest.
8. Recompute file SHA-256 after changing JSONL; never hand-edit a stale hash.
9. Do not convert JSONL row count into semantic task count in documentation.
10. Do not publish a success rate without model, snapshot, split, agent, repeat,
    and backend metadata.

## 6. Snapshot and offline rules

`MockArxivEnv(mode="replay")` is intentionally strict. A snapshot miss is a
missing experiment input, not permission to call the network. When documenting
or testing a replay path:

- state whether the snapshot is read-only or being recorded;
- keep `record`, `auto`, and `replay` semantics distinct;
- do not call the offline download stub a real PDF;
- check paper identity after resolving `ref`;
- make T4/T5 figure assumptions explicit;
- keep VLM observations recorded rather than treating extractive reconstruction
  as equivalent to VLM output.

Network access belongs in snapshot preparation, not in a supposedly offline
training or CI command.

## 7. Style and implementation rules

- Follow PEP 8 for Python and use type hints for public interfaces.
- Keep public functions documented when behavior or data shape is non-obvious.
- Prefer small pure helpers for parsing, hashing, and validation.
- Make failure messages actionable: include the path, id, key, or option that
  the contributor can inspect next.
- Preserve deterministic seeds and stable JSON serialization in fixtures.
- Keep compatibility defaults when adding optional trajectory or manifest fields.
- Avoid broad rewrites of unrelated historical documentation.
- Do not add a new dependency for a check that the standard library or existing
  test helper already supports.

## 8. Branches and commits

Create a focused branch from the current `main`. A descriptive branch name such
as `codex/docs-data-contract` or `fix/replay-key` is easier to review than a
generic name. Keep commits small enough that a reviewer can separate source
changes from generated artifacts.

Commit messages should describe the user-visible change, for example:

```text
docs: document v3 benchmark split contract
fix: reject replay fallback as a successful observation
test: cover empty-step blocked task semantics
```

Do not commit local model weights, PDFs, API responses, generated `outputs/`,
or private `.env` files. If a fixture is intentionally added, explain why it is
small, deterministic, and safe to redistribute.

## 9. Pull request checklist

Before opening a PR, confirm:

- [ ] The PR has one clear problem statement.
- [ ] The changed contract and source files are named in the description.
- [ ] New fields have an example, a source, and a common-misuse note where useful.
- [ ] Tests cover changed behavior, or the PR explains why documentation-only
      validation is sufficient.
- [ ] Commands list their working directory and prerequisites.
- [ ] Unrun commands are explicitly marked as unrun.
- [ ] No GPU result, network result, or model metric is implied without evidence.
- [ ] Train/held-out leakage and manifest hashes were considered for data changes.
- [ ] `git diff --check` is clean.
- [ ] The final diff does not include unrelated worktree changes.

### Suggested PR description

```markdown
## Problem
Readers could not tell which split/schema/command was current because ...

## Changes
- Updated ...
- Added ...

## Evidence
- Source checked: `...`
- Tests: `pytest ...` (or “documentation-only; commands not run”)
- CI expectation: ...

## Limits
This change does not claim ...
```

### Documentation PR example

```markdown
Problem: the benchmark guide still called v2_62 the current expanded split.

Changes: describe v1/v2/v3, add explicit `FILE:NAME` examples, and link the
data contract guide.

Evidence: checked `benchmark/splits.py`, `data/splits/v3_81.json`, and the
`run_benchmark.py` argparse choices. No benchmark or training command was run.
```

### Small bugfix example

```markdown
Problem: a replay miss was allowed to look like a successful tool result.

Changes: keep replay strict and add a fixture that asserts the miss is visible.

Evidence: focused offline environment tests pass; no network access was used.
```

## 10. CI

The current workflow is `.github/workflows/ci.yml`. For pull requests to
`main`, it:

1. checks out the repository;
2. installs Python 3.10, 3.11, and 3.12 matrix jobs;
3. installs `AgenticArxiv/requirements.txt`, `pytest`, and `pytest-cov`;
4. runs `pytest AgenticArxiv/tests/ -v --tb=short`;
5. runs the strict E9/F63/F7/F82 flake8 check;
6. runs a non-blocking complexity/line-length flake8 report.

CI does not download model weights, run GPU training, or establish a real-arXiv
success rate. A CI pass is necessary evidence for code paths covered by the
workflow, not a substitute for declaring experiment limitations.

## 11. Documentation maintenance

When a user-facing feature changes, update the smallest relevant set:

- the root README’s short navigation if the entry point is discoverable there;
- `docs/data-formats.md` when a data field or lineage contract changes;
- `docs/cli-reference.md` when an option, default, output, or working directory changes;
- `AgenticArxiv/benchmark/readme.md` when task sets, splits, or metrics change;
- `docs/offline-replay.md` when snapshot modes or tool behavior changes;
- `docs/multigranular_rl.md` when reward components or gates change.

Translated READMEs should be updated when their source content changes, but an
unrelated code PR does not need to enlarge every language file. Keep translation
diffs synchronized with the source section rather than adding a second, subtly
different command contract.

## 12. Questions and issue reports

For an issue, include the command, working directory, Python version, operating
system, selected task set/split, and the smallest relevant error. Redact API
keys, private paths, and private model names. A reproducible fixture or a short
`pytest` reproduction is more useful than a screenshot alone.

Thank you for keeping the project reproducible, reviewable, and honest about
what was actually run.
