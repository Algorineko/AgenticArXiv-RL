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
| `r_argument` | 2 | Fraction of expected parameter values matched per step, averaged across steps and scaled from `[0, 1]` to `[-1, 1]`. It is omitted when a task has no `expected_tool_args` oracle. |
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

## 诊断分量与安全闸门

`RewardCalculator.compute_reward_breakdown` 除上述五个加权分量外，
还返回 `result_quality`（工具结果质量）与 `efficiency`（调用效率）。
它们在旧版加权平均中的权重均为零，因此不会改变课程学习的分母；
严重失败由加权平均之后的安全闸门处理。

| 条件 | 总奖励上限 | 作用 |
| --- | ---: | --- |
| 终止类型为 `ERROR` | `-0.75` | 错误终止不可由格式分抵消 |
| 出现解析失败或工具执行失败 | `-0.75` | 无效转移不能伪装为完成 |
| `result_quality <= -0.75` | `-0.75` | 工具结果缺失或严重无效 |
| 任一步 `analyze_figure` 缺少有效答案 | `-0.75` | 多步平均值不能掩盖图表分析失败 |
| `false_finish` 且未触发上一组条件 | `-0.25` | 声称完成但任务没有完成 |
| `efficiency <= -0.75` | `-0.25` | 严重冗余的调用受限 |

安全闸门只压低总奖励，不修改各分量的原始记录。
因此审查轨迹时可以同时看 `total`、五个训练分量、
`result_quality`、`efficiency` 和 `TaskMetrics`，判断分数为何被封顶。
单步 rollout 由框架自动补出的 `FINISH` 使用 `forced_finish` 标记；
它只限制 `outcome` 奖励，不把整个组的总分压成同一常数。

## T5 图表分析 observation 的结果质量判定

`analyze_figure` 的工具返回值包含论文标识与答案。
在轨迹里，这个返回值可能被保存成 Python 字典的 `repr`，
也可能是 JSON；部分 rollout 会截短 observation。
只搜索 `paper_id` 和 `answer` 字段名会把空答案误判为有效结果。

现在的判定按以下顺序进行：

1. 尝试将**整个** observation 解析为 JSON；失败时再尝试 Python 字面量。
2. 解析结果必须是顶层对象，不能是列表、字符串或数字。
3. 顶层 `paper_id` 和 `answer` 必须都存在且为字符串。
4. 两个字符串去除首尾空白后都必须有内容。

该工具步骤满足条件时给 `+1`，否则给 `-1`。
这只检查答案是否存在，不检查答案是否正确。
`The caption does not state this.` 这类诚实说明属于非空答案；
第三方 VLM 的文本质量不会作为奖励裁判。

| observation | 图表分析步骤分 | 说明 |
| --- | ---: | --- |
| `{'paper_id': '2601.00004v1', 'answer': 'A rising trend.'}` | `+1` | 有效的 Python `repr` |
| `{"paper_id": "2601.00004v1", "answer": "The caption does not state this."}` | `+1` | 有效的 JSON；未臆造 caption 中不存在的信息 |
| `{'paper_id': '2601.00004v1', 'answer': ''}` | `-1` | 空答案 |
| `{"paper_id": "2601.00004v1", "answer": "   "}` | `-1` | 只有空白 |
| `{'paper_id': '2601.00004v1'}` | `-1` | 缺少答案 |
| `{'answer': 'A rising trend.'}` | `-1` | 缺少论文标识 |
| `{'paper_id': '2601.00004v1', 'meta': {'answer': 'nested'}}` | `-1` | 嵌套字段不能代替顶层答案 |
| `[{'paper_id': '2601.00004v1', 'answer': 'inside a list'}]` | `-1` | 顶层不是对象 |
| `{'paper_id': '2601.00004v1', 'answer': 'cut off` | `-1` | 截断后无法确认完整结果 |

### 单步与多步轨迹

当任务只要求一步 `analyze_figure` 时，该步的 `-1` 就是整条轨迹的
`result_quality=-1`，触发 `<= -0.75` 的严重失败闸门。
即使工具名、参数与 `FINISH` 都正确，总奖励仍最多为 `-0.75`。

多步任务的 `result_quality` 会汇总各工具步骤；多余调用另有负分。
例如四个必要步骤的分数为 `+1, +1, +1, -1` 时，结果质量平均为 `0.5`，
仅检查这个平均值会漏掉图表分析的失败。
安全闸门因此还会逐步检查每次 `analyze_figure` 的答案；
只要有一次无效，即使平均值高于 `-0.75`，总奖励仍最多为 `-0.75`。
这项逐步检查仅针对图表分析，不改变其他工具的评分规则或课程权重。

### 与其他工具的边界

- `get_paper_content` 和 `summarize_paper` 保留原有字段检查。
- 搜索、下载、缓存状态、翻译和图表抽取沿用各自的观察结果规则。
- 图表分析的 `figure_no`、`question` 等参数仍由参数分量检查。
- 格式不完整的 observation 保守记为无效图表分析，不抛出解析异常。

回归用例位于 `AgenticArxiv/tests/test_multigranular_reward.py`，
使用内存中的任务与轨迹；不需要 arXiv、PDF、图像、模型或 GPU。

### 基准参考轨迹

`benchmark.baselines.ReferencePolicy` 也不执行真实工具。
它为普通工具保留确定性的合成 observation；
对于 `analyze_figure`，参考策略需要提供包含非空
`paper_id` 和 `answer` 的合成对象，才能代表一次有效图表分析。
退化策略仍保留各自的合成轨迹，由同一奖励计算器评分。
这样逐类目分差测试检验的是策略质量差异，而不是参考轨迹自身违反
图表结果契约。该基准不声称生成了真实图像或 VLM 答案。
