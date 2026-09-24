# Multi-granular rewards: current implementation

This document describes the implementation in `AgenticArxiv/rl/reward.py`, not
the earlier five-column design sketch. The public
`RewardCalculator.compute_reward()` API remains compatible with rollout code,
while `compute_reward_breakdown()` exposes the values needed for audit and
tests.

## 1. Overall calculation

For a trajectory `tau` at training step `s`, the legacy weighted score is:

```text
R_legacy(tau, s) = Σ_i w_i(s) r_i(tau) / Σ_i |w_i(s)|
```

The five weighted components are all clipped to `[-1, 1]`. The schedule uses the
following defaults:

| Component | Full weight | Early-course weight | What it measures |
|---|---:|---:|---|
| `format` | 1 | 1 | Valid terminal token or strict tool-action JSON per history step |
| `tool` | 3 | 1 | Order-aware LCS F1 of actual and expected tool names, mapped to `[-1, 1]` |
| `argument` | 2 | 2/3 | Parameter match score mapped from `[0, 1]` to `[-1, 1]` |
| `process` | 1 | 1 | Valid-step credit minus parse, execution, and redundant-call penalties |
| `outcome` | 3 | 1 | Completion, failure, blocked-terminal, and semantic outcome |

The default `RewardCalculator(curriculum_steps=30,
early_correctness_scale=1/3)` reduces tool, argument, and outcome weights before
step 30. `format` and `process` are not reduced. At or after step 30, the full
weights apply. The GRPO CLI can override this with
`--reward_curriculum_steps`; explicitly passing `0` starts with the full
correctness weights.

`expected_tool_args=None` omits the argument component from the denominator. An
empty `expected_tool_args=[]` is different: it is an applicable empty oracle,
usually paired with an infeasible task whose correct tool path is empty.

## 2. Diagnostics are not default weighted objectives

The breakdown also stores two result-grounded diagnostics:

| Diagnostic | Default schedule weight | Purpose |
|---|---:|---|
| `result_quality` | `0.0` | Whether observations show useful, grounded work rather than fallback/empty output |
| `efficiency` | `0.0` | Redundant calls and execution/parse cost relative to expected call count |

They are intentionally outside the legacy weighted average so historical
component curves remain comparable. They still participate in safety gates below.
They are not a hidden sixth and seventh trainable reward by default.

The returned `RewardBreakdown` contains:

```text
total, format, tool, argument, process, outcome,
result_quality, efficiency, weights
```

Every component is rounded to six decimal places before serialization. New
trajectory records store the dictionary under `reward_components`.

## 3. Component semantics

### 3.1 Format

Each history step earns valid-step credit when its action is one of
`FINISH`/`FORCE_STOP`/`ERROR`, or when it parses as a dictionary with a string
tool name and a dictionary in `args` or `parameters`. The component maps the
valid fraction from `[0, 1]` to `[-1, 1]`; an empty history is `-1`.

This is a structural signal. It does not say that the selected tool or its
arguments are correct.

### 3.2 Tool sequence

The scorer computes the longest common subsequence between actual and expected
tool names. Precision and recall are combined as F1, then mapped as:

```text
r_tool = 2 * F1 - 1
```

For an empty expected path, no actual tool call receives `1.0`; any tool call
receives `-1.0`. This makes “do nothing” an explicit contract for infeasible
tasks rather than a missing label.

The tool component checks names and order only. A correct tool name with a wrong
paper reference can therefore still have a high tool score; argument, reference,
outcome, and result-quality components are needed to separate it.

### 3.3 Arguments

`argument_match_score()` lives in `benchmark/metrics.py` and is shared with
benchmark reports. `RewardCalculator` only converts its `[0, 1]` result to
`[-1, 1]`. If the argument oracle is not applicable, the component is omitted
from the weighted denominator instead of being treated as a fake zero.

Exact key/value behavior is task-dependent and can include resolved paper ids.
Documentation should not claim that every free-form argument is scored when the
task declaration leaves it dynamic.

### 3.4 Process

Process starts from valid-step fraction and subtracts penalties for parse
failures, tool execution failures, and extra calls beyond the expected count.
It is clipped to `[-1, 1]`. A well-formed but semantically wrong call can still
look structurally clean here; this is why process is not a replacement for
outcome or result quality.

### 3.5 Outcome

The outcome component is more than “the last action is FINISH”:

| Situation | Outcome |
|---|---:|
| `ERROR` termination | `-1.0` |
| `FORCE_STOP` termination | `-0.5` |
| Any tool execution failure | `-1.0` |
| Blocked task with an explained declared reason | `1.0` |
| Blocked task with false completion | `-0.25` |
| Blocked task with no recognized explanation | `0.0` |
| Completed and tool path accurate | `max(0.25, 2 * min(arg_score, ref_score) - 1)` |
| Completed but tool path not accurate | `0.25` |
| Not completed | `-0.25` |

When `forced_finish` is set, the accurate-completion outcome is capped at `0.25`.
This preserves format/tool/argument differences between single-step rollouts
while withholding the full terminal outcome credit.

For ordinary completed tasks, `FINISH` is only a policy stop decision. It cannot
turn an environment failure into success. For blocked tasks, the final Thought
must match the declared `terminal_reason` family; a generic “done” is a false
completion rather than a valid refusal.

### 3.6 Result quality

Result quality is a deterministic observation contract, not an LLM judge. It
checks markers in the tool observation and the expected tool family:

- empty or missing observations are negative/partial evidence;
- parse failures, tool errors, `offline_fallback`, and explicit fallback markers
  are negative;
- search observations should show papers or ids;
- download/translate/cache observations should show `READY`, `status`, or a
  related successful state;
- content, summary, and figure-analysis observations should contain a paper id
  and a grounded payload marker such as `content`, `summary`, or `answer`;
- figure extraction is useful only when it reports a non-zero figure result;
  `count: 0` is a factual empty extraction and does not count as a useful result
  for a task that expects a figure.

Extra calls extend the diagnostic list with negative entries. This prevents a
policy from turning a correct first result into a full quality score by making
unnecessary calls.

### 3.7 Efficiency

For a task with `n_expected` and an actual tool count `n_actual`, efficiency uses
the excess call ratio plus a quarter-point per parse or execution failure:

```text
excess = max(0, n_actual - n_expected)
penalty = excess / n_expected + 0.25 * error_count
r_efficiency = clip(1 - 2 * penalty)
```

For an expected empty path, no tool call gives `1.0` and any call gives `-1.0`.
Efficiency is a small normalized cost signal; it does not replace exact tool
sequence or argument checks.

## 4. Safety gates

After the weighted score is calculated, `_apply_safety_gates()` applies caps that
cannot be rescued by unrelated positive components:

| Gate | Condition | Total cap |
|---|---|---:|
| hard invalid | `ERROR`, parse failure, tool execution failure, or `result_quality <= -0.75` | `<= -0.75` |
| false finish | metrics reports false completion | `<= -0.25` |
| severe inefficiency | `efficiency <= -0.75` | `<= -0.25` |

The final value is clipped again to `[-1, 1]`. `forced_finish` is handled inside
the outcome component and intentionally does not cap the whole total: doing so
would erase useful parameter/tool differences across one-step GRPO samples and
could make an entire group constant.

The gates are deliberately non-compensable. A trajectory that called the right
tool but failed to execute it must not get a positive total merely from format
and process points.

## 5. Worked qualitative cases

The following are contract examples, not measured scores:

| History shape | Expected interpretation |
|---|---|
| Empty history on a normal task | Format and process are negative; no completion evidence |
| Empty history on an infeasible task | Tool/result-quality can be correct only if the task’s terminal explanation is present in the FINISH step; an empty history alone is not the reference trajectory |
| Correct tool, wrong `ref`, clean FINISH | High format/tool may remain, but argument/ref/outcome and possibly result quality reduce the score |
| Correct search followed by `offline_fallback` observation | Hard result-quality evidence prevents it from looking like a clean success |
| Expected one tool, then two redundant calls | Tool/process/efficiency distinguish it; severe efficiency can cap the total |
| Tool throws, then model FINISHes | Outcome and hard gate keep the result negative |
| Blocked request with precise reason | The terminal semantic classifier can award the blocked outcome |

Do not replace these examples with a fixed score unless the exact trajectory,
task declaration, training step, and evaluator version are all specified.

## 6. Group-relative advantages

GRPO compares multiple samples for the same prompt. The helper
`compute_group_relative_advantages()` computes:

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + epsilon)
```

If a group has zero variance (standard deviation at most `epsilon`), every
advantage is zero. This is stable and honest: a constant group supplies no
relative learning signal. It is also why the split policy avoids tasks whose
measured success rate is too close to 0 or 1 for the selected model.

## 7. Audit and persistence

`Trajectory.reward_components` should be retained with the final reward. A
reviewer can then distinguish:

- correct tool choice versus correct arguments;
- a real grounded observation versus a fallback marker;
- a full terminal outcome versus a forced finish;
- a legacy weighted score versus a safety-gate cap.

Existing trajectory files remain loadable because `reward_components` defaults to
an empty dictionary. New writers should fill it whenever the reward calculator
is used; new tests should assert both total behavior and the relevant breakdown.

## 8. Safe documentation claims

It is accurate to say that the scorer is deterministic, bounded, auditable, and
uses no LLM judge for `result_quality`. It is not accurate to say that the rules
understand the semantic quality of a generated paper summary or VLM answer. They
check structured tool/observation contracts and deterministic markers. A richer
quality judge would be a separate design with a separate validation and reward-
hacking analysis.

When changing reward code, update this document, the baseline guide, and any
badcase fixture that encodes the old behavior. State the training step in tests
and examples whenever curriculum weights affect the expected result.
