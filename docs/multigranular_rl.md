# Multi-granular GRPO rewards

This project uses a hierarchical, fully verifiable reward inspired by
LLM-TIR. It replaces a single sparse success score with feedback at five
levels while preserving the existing rollout API.

For trajectory `tau` at training step `s`, the scalar score is

```text
R(tau, s) = sum_i w_i(s) r_i(tau) / sum_i |w_i(s)|
```

Every active component is bounded to `[-1, 1]`:

| Component | Default weight | Signal |
| --- | ---: | --- |
| `r_format` | 1 | Fraction of steps whose action is a terminal token or strict JSON with a tool name and argument object. |
| `r_tool` | 3 | Order-aware LCS F1 between predicted and expected tool sequences, mapped from `[0, 1]` to `[-1, 1]`. |
| `r_argument` | 2 | Mean of parameter-key recall and exact value accuracy. It is omitted when a task has no `expected_tool_args` oracle. |
| `r_process` | 1 | Dense valid-step credit minus parse, execution, and unnecessary-call penalties. |
| `r_outcome` | 3 | `1` for correct completion, `0.25` for completion with a wrong tool path, `-0.5` for forced stop, and `-1` for error. |

Benchmark tasks provide `expected_tools` for tool selection verification and,
where arguments are statically knowable, `expected_tool_args` for exact
argument verification. A `None` entry skips a dynamic tool step whose arguments
depend on an earlier result.

The curriculum follows LLM-TIR's coarse-to-fine idea. Before step 30, tool,
argument, and outcome weights are multiplied by `1/3`; format and process
weights remain unchanged. From step 30 onward all weights are active. This
lets the model first stabilize the ReAct protocol and later concentrate on
semantic tool/parameter correctness.

GRPO samples multiple trajectories for the same prompt and converts the
scalar scores into group-relative advantages:

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + epsilon)
```

`compute_group_relative_advantages` implements this operation for lightweight
training/evaluation code; TRL's GRPO trainer performs the equivalent grouping
internally. Constant-reward groups receive zero advantage.

The full component dictionary is stored in every new trajectory under
`reward_components`, making reward hacking and curriculum behavior auditable.
Existing trajectory files load unchanged because the field has a default.
