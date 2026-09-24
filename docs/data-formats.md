# 数据格式与文件契约

> 适用范围：当前 `main` 的 benchmark、离线 RL 和 SFT 数据流水线。本文只描述
> 源码实际读取或写出的字段；示例中的数值不代表模型成绩。

## 1. 先确定工作目录

仓库根目录记作 `AgenticArXiv-RL/`。本文中的路径都是相对于根目录的，除非
特别注明：

- `AgenticArxiv/benchmark/*.py` 是 Python 包代码；从 `AgenticArxiv/` 运行
  `python -m benchmark...`。
- `scripts/*.py` 位于根目录；从根目录运行数据生成器。
- `data/splits/*.json` 是版本化切分输入；`data/sft/` 下生成的 JSONL 和
  manifest 通常是本地实验产物，不应把未审计数据误认为仓库内置数据。
- `data/mock_arxiv_snapshot.json` 是工具环境快照。它不是任务定义，也不是
  benchmark 报告。

整条数据链可以概括为：

```text
TaskSpec / tasks_expanded.py
        │
        ├── build() ─────────────── benchmark task dict
        │                              │
        │                              ├── benchmark metrics/report
        │                              └── split JSON 选任务
        │
        ├── generate_sft_data.py ── messages JSONL
        │                              │
        │                              ├── linguistic augmentation
        │                              └── parametric augmentation
        │
        └── MockArxivEnv snapshot ── 可回放 observation
                                       │
                                       └── rollout / reward / traces
```

## 2. 任务声明：`Step` 和 `TaskSpec`

源码：`AgenticArxiv/benchmark/task_spec.py`；扩展任务：
`AgenticArxiv/benchmark/tasks_expanded.py`。

### 2.1 `Step`

`Step` 是一个不可变 dataclass，只有两个字段：

| 字段 | 类型 | 含义 | 常见误用 |
|---|---|---|---|
| `tool` | `str` | 工具注册名，例如 `download_arxiv_pdf` | 写成自然语言动作，或使用未注册的名字 |
| `args` | `Dict[str, Any]` | 该步传给工具的参数 | 把框架用的 `session_id` 当成模型必须生成的业务参数 |

`setup` 也由 `Step` 组成，但它只是在任务开始前由 runner 建立会话状态；
`setup` 不属于 agent 的轨迹，也不进入 `expected_tools`。

### 2.2 `TaskSpec` 字段

| 字段 | 默认值 | 含义 / 来源 | 常见误用 |
|---|---|---|---|
| `id` | 无 | 稳定的任务标识，`build()` 会检查唯一性 | 在切分文件里写了不存在的 id |
| `task` | 无 | 给模型的自然语言请求 | 手动改文本却忘记同步参数 |
| `steps` | `()` | 标准工具调用序列 | 空序列不是“漏写答案”，而是合法的零工具任务 |
| `category` | `misc` | 任务类别，例如 `search`、`paper_summary` | 把类别名当成 `--tasks` 一定支持的 CLI 选项 |
| `difficulty` | `medium` | 任务声明中的难度标签 | 把它当作实测成功率 |
| `setup` | `()` | 任务开始前的状态准备步骤 | 误计入 agent 的准确率或 SFT 动作 |
| `template` | `None` | 模板级切分键的覆盖值 | 只按 task 文本随机切分而泄漏同模板参数 |
| `requires_offline` | `False` | 是否依赖快照中固定的论文或结果 | 没有快照却把结果当作可比 benchmark |
| `note` | `""` | 给开发者的说明，不是模型答案 | 从 note 推导未声明的评分规则 |
| `termination` | `FINISH` | 期望终止标记 | 把 `FINISH` 等同于业务一定成功 |
| `terminal_mode` | `completed` | `completed` 或 `blocked` | 对普通任务随意添加阻塞原因 |
| `terminal_reason` | `None` | blocked 任务的可验证原因 | 用泛化的“无法执行”替代具体原因 |
| `max_iterations` | `None` | 允许的 ReAct 轮数上限 | 长链仍使用默认上限导致 `FORCE_STOP` |
| `depends_on` | `None` | 同会话中的前置任务 id | 将它当成 `setup` 的另一种序列化形式 |

### 2.3 两份标准答案必须同源

`TaskSpec.to_task()` 从同一个 `steps` 序列生成：

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

这两个列表的长度天然相同。不要直接在任务字典里再手写一份平行列表；
需要新任务时优先写 `Step`，再让 `to_task()` 展开。

一个当前任务的完整序列化形状如下：

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

`expected_tool_args` 中的 `null` 有特殊意义：它表示该步存在，但参数由
前置观察动态决定，参数评分会跳过这一项。空列表 `[]` 则表示“期望没有
任何工具调用”。两者不能混写。

### 2.4 blocked 任务的终止契约

不可行任务不是没有 ground truth。它的标准答案是“不调用工具，并在 FINISH
前说明准确的阻塞原因”。当前允许的 `terminal_reason` 是：

| 原因 | 说明 | 示例 |
|---|---|---|
| `missing_context` | 当前会话没有能解析指代的状态 | “把刚才那篇论文翻译一下”，但没有 setup |
| `invalid_reference` | ref 越界或不合法 | “下载第 0 篇论文” |
| `paper_not_found` | 指定论文不在快照或会话中 | 不存在的 arXiv ID |
| `unsupported_capability` | 工具集合不支持请求 | 找作者邮箱并发邮件 |

对应的最小任务形状是：

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

对于 `blocked` 任务，模型只写“已完成”会被识别为 false completion；
`reference_terminal_thought()` 只生成可验证原因对应的示例措辞。不要在文档、
fixture 或基线中把“零工具 + 泛化 FINISH”当成正确阻塞。

### 2.5 参数化任务族

`family()` 接收同一份参数映射的 `text` 和 `steps` 函数，因此自然语言和
标准调用从同一参数渲染。写参数化任务时：

1. 把会变的值放入参数对象，而不是在字符串和 `Step` 中各写一遍。
2. 为每个实例生成稳定 id。
3. 用 `build()` 检查重复 id。
4. 如果要扩充 SFT，使用参数化生成器的血缘字段，不要把 held-out 任务复制回 train。

## 3. 切分文件：`train`、`dev`、`iid_test`、`ood_test`

源码：`AgenticArxiv/benchmark/splits.py`；版本文件：
`data/splits/v1.json`、`v2_62.json`、`v3_81.json`。显式文件引用格式是：

```text
data/splits/v3_81.json:train
```

### 3.1 文件外形

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

当前 `v3_81.json` 的静态数量是：

| 文件 | train | dev | iid_test | ood_test | 合计 |
|---|---:|---:|---:|---:|---:|
| `v1.json`（历史默认） | 42 | 0 | 13 | 4 | 59 |
| `v2_62.json`（历史扩展） | 36 | 8 | 14 | 4 | 62 |
| `v3_81.json`（当前扩展） | 51 | 8 | 18 | 4 | 81 |

v3 的 `rates` 仍只覆盖由 v2 延续的 36 个 train 任务。新增的
`paper_reading`、`paper_summary`、`figure_extraction`、`figure_analysis`
族出现在任务集和切分中，但没有因此获得伪造的成功率。

### 3.2 为什么按模板切分

`template_key()` 默认用 `(template or category, len(expected_tools))`。这会把
同一个工具链模板的不同参数实例放在同一侧，避免例如 `search_AI_1d_3`
进 train、`search_AI_30d_25` 进 test 后，测试只测出模型是否记住句式。

- `iid_test`：同模板、不同参数实例，检查参数泛化。
- `ood_test`：整个模板键或链长留出，检查形态泛化。
- `dev`：已经用于 pilot、调试或 badcase 分析，不能再冒充盲测。
- `train`：可以用于 SFT；GRPO 还要按 rates 进一步选 `rl_train`。

### 3.3 `rates` 与动态 `rl_train`

`rl_train` 不是 JSON 中的数组，而是 `load_split()` 读取 `rates` 后动态计算：

```text
FLOOR_MAX = 0.2
CEILING_MIN = 0.8
middle = 0.2 <= rate <= 0.8
rl_train = train ∩ {任务有 rate 且属于 middle}
```

v3 当前记录的 rate 中有 6 个任务落在中间带，因此默认的显式 v3 文件可以
算出 6 个 `rl_train` id；这个数字不等于“51 个 train 都有 rate”。换模型、
换快照或换评测策略后，必须重新测量 rates，不能复用旧难度档。

没有 `rates` 时读取 `rl_train` 会报错；train 没有中间带任务时也会报错，
不会静默生成空训练集。

### 3.4 防止泄漏的使用方式

```bash
# 从仓库根目录生成 expanded train-only SFT 数据
python scripts/generate_sft_data.py \
  --task_set expanded \
  --split data/splits/v3_81.json:train \
  --snapshot data/mock_arxiv_snapshot.json \
  --output data/sft/sft_v3_train.jsonl

# 从 AgenticArxiv/ 运行 benchmark；路径要回到仓库根目录
cd AgenticArxiv
python -m benchmark.run_benchmark \
  --task-set expanded --offline \
  --split ../data/splits/v3_81.json:iid_test
```

SFT 生成器会拒绝 expanded 任务集的裸 `train`、没有版本文件的 split、
`dev`、`iid_test` 和 `ood_test`。benchmark 则会检查切分中的 id 是否都在
当前任务池；忘记 `--task-set expanded` 或忘记 `--offline` 都可能造成任务池
不完整，命令应失败而不是输出一个看似可比的数字。

## 4. Trajectory JSONL

源码：`AgenticArxiv/rl/trajectory.py`。一个 JSONL 行对应一个
`Trajectory`，其 `steps` 是 `TrajectoryStep` 对象列表。

### 4.1 顶层字段

| 字段 | 类型 | 来源 / 含义 | 常见误用 |
|---|---|---|---|
| `task_id` | `str` | 任务 id | 将它当作切分名 |
| `task` | `str` | 当时的自然语言任务 | 只保留 id，丢掉复现所需文本 |
| `session_id` | `str` | 一次会话标识 | 用 session id 推断论文身份；快照 key 会剔除它 |
| `steps` | `list` | 多个 `TrajectoryStep` | 把 benchmark 的 `history` 原样宣称为同一 schema |
| `final_reward` | `float` | 最终标量奖励 | 当成严格成功率 |
| `metrics` | `object` | `TaskMetrics` 的字典形式 | 只看 `task_completed` 就判业务成功 |
| `timestamp` | `str` | `create_trajectory()` 生成时间 | 用它作为实验版本或确定性证据 |
| `model` | `str` | 模型名，可为空 | 缺失时编造模型成绩 |
| `termination_type` | `str` | `FINISH` / `FORCE_STOP` / `ERROR` 等 | 把 `FORCE_STOP` 当成正常完成 |
| `reward_components` | `object` | `RewardBreakdown.to_dict()` 的可审计分解 | 只记录 total，无法追查 reward hacking |

### 4.2 `TrajectoryStep` 字段

| 字段 | 含义 |
|---|---|
| `step` | 从 1 开始的轨迹序号 |
| `thought` | 模型当前步的 Thought 文本 |
| `action` | JSON 字符串，或终止标记 `FINISH` / `FORCE_STOP` / `ERROR` |
| `observation` | 工具或环境返回的字符串 |
| `llm_latency_ms` | 当前步 LLM 延迟，默认 0 |
| `tool_latency_ms` | 当前步工具延迟，默认 0 |
| `parse_failed` | 是否无法解析动作，默认 `false` |

一个短的匿名轨迹形状如下；观察内容被刻意缩短，但字段和动作类型与类定义一致：

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

`load_trajectories()` 会把 `steps` 重建为 `TrajectoryStep`；旧文件没有
`model`、`termination_type` 或 `reward_components` 时，dataclass 默认值仍使
它们可加载。新增字段时应保留向后兼容默认值。

### 4.3 Trajectory 与 benchmark trace 不是一个契约

`benchmark.run_benchmark --save-traces` 写出的每行只包括：

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

它服务于 `eval/badcase_replay.py` 和 `rescore_traces.py`，不是
`Trajectory` JSONL：没有 `final_reward`、`metrics`、`timestamp` 或
`reward_components`。重评分器还会从同目录的 `summary.json` 取旧 timing/token
字段，再与 trace 的 history 合成新的 metrics。

## 5. SFT JSONL 与 manifest

源码：`scripts/generate_sft_data.py`、
`scripts/generate_parametric_sft_data.py`、`scripts/augment_sft_data.py`、
`scripts/augment_parametric_sft_data.py`、`scripts/build_sft_train_mix.py`。

### 5.1 基础 SFT 样本

确定性和 LLM 专家生成器都会写以 `messages` 为核心的两轮样本：

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

字段契约：

| 字段 | 必须性 | 来源 | 常见误用 |
|---|---|---|---|
| `messages` | 必须 | prompt 模板 + 一条历史决策 | 把完整 history 丢掉，只保留最后一轮 |
| `messages[0].role` | 必须 | 固定为 `user` | 改成 `system` 后仍假设生成器能解析任务边界 |
| `messages[1].role` | 必须 | 固定为 `assistant` | 让 assistant 不以 Action 结尾，训练转换会拒绝 |
| `source_task_id` | 生成器会写 | spec id | 省略后无法审计语义来源 |
| `source_split` | 生成器会写 | 版本文件名 + `:train` | 写绝对路径，导致跨机器血缘不可移植 |
| `trajectory_step` | 生成器会写 | history 中的 0-based 索引 | 当作环境 step 的 1-based 序号 |

只有动作非空且不是 `PARSE_ERROR`、`ERROR`、`FORCE_STOP` 的 history 步才会
形成 SFT 样本；确定性专家还会先执行 setup/depends_on 前置状态，并对轨迹做
严格成功校验。blocked 任务产生的是说明原因的 FINISH 样本，而不是工具调用。

### 5.2 参数化和语言扩增字段

参数化 seed 的每行会额外写出：

```json
{
  "derived_task_id": "augp1_search_AI_30d_25_2d_4",
  "parent_task_id": "search_AI_30d_25",
  "generation_parameters": {"aspect": "AI", "days": 2, "max_results": 4},
  "dataset_stage": "parametric_v1_expert_seed",
  "sample_sha256": "<sha256(messages 的规范 JSON)>"
}
```

参数化校验会拒绝：父任务不在 train、父任务在 held-out、复制已有
benchmark 文本、改变父任务工具链或 setup 拓扑、重复 sample 指纹。

语言扩增只改变当前任务措辞和 Thought；Action、参数、历史 Observation 和
步骤顺序必须保持不变。它会添加：

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

默认的 6 个任务措辞包装 × 2 个 Thought 版本是生成器的实际展开规则；
它增加的是语言视图，不是独立语义任务数。

### 5.3 manifest 的最低审计字段

不同阶段的 `kind` 不同，但可复现 manifest 通常包含以下字段：

| 字段 | 含义 |
|---|---|
| `version` | manifest 格式版本，而不是 split 版本 |
| `kind` 或 `augmentation_kind` | 当前数据阶段的机器可读名称 |
| `input` / `output` | 输入和输出路径 |
| `input_sha256` / `output_sha256` | 文件内容指纹 |
| `split_file` / `split_sha256` | 任务边界及其指纹 |
| `snapshot` / `snapshot_sha256` | 生成专家轨迹使用的快照及指纹 |
| `seed_rows` / `output_rows` | 输入与输出行数 |
| `unique_sample_fingerprints` | 去重后的 messages 数量 |
| `source_tasks` 或 `semantic_task_instances` | 语义任务数，不等于 JSONL 行数 |
| `heldout_overlap` / `heldout_parent_overlap` | 泄漏审计结果，应为 0 |
| `sources` | 混合集引用的两个来源及其 manifest 指纹 |

最终混合数据 `sft_v3_train_mix.jsonl.manifest.json` 的 `kind` 必须是
`qlora_sft_train_mix`。`train_sft.py` 默认检查 kind、输出 SHA-256、行数和
唯一指纹；`--skip_data_manifest_check` 只适合临时实验，不能作为正式可复现
结果。

### 5.4 防止 train / held-out 泄漏的检查顺序

1. 先选择版本化 split 的 `train`，不要先生成全量 expanded 再删除。
2. 记录 `source_split`，不要用机器绝对路径替代版本文件名。
3. 对参数化派生任务检查 parent 是否在 train，并检查工具链拓扑未变。
4. 对语言扩增检查 source id 集合等于完整 train 集，且文本不等于 held-out 原文。
5. 混合数据时验证两个输入 manifest、行数、hash 和 messages 去重。
6. 训练前让 `train_sft.py` 再次验证最终文件 manifest。

## 6. 最小核对清单

提交数据或数据工具改动前，逐项确认：

- [ ] 所有任务 id 都能在对应任务集找到。
- [ ] `expected_tools` 和 `expected_tool_args` 来自同一份 `steps`。
- [ ] `setup` 没有被计入模型轨迹。
- [ ] `[]`（正确不调用）和 `None`（不适用参数 oracle）没有混淆。
- [ ] 训练只引用显式的 `FILE:train`，而不是裸 `train`。
- [ ] `rates` 的模型、agent、backend、repeat 和快照信息写入 metadata。
- [ ] 每条 JSONL 样本都能追溯到 `source_task_id` 和 `source_split`。
- [ ] 每次扩增或混合都重算 `sample_sha256`，没有重复指纹。
- [ ] manifest 的 hash 与实际文件一致。
- [ ] 文档没有把 JSONL 行数写成语义任务数，也没有把 fixture 当成训练结果。

## 7. 相关源码定位

| 契约 | 源码 |
|---|---|
| 任务与终止语义 | `AgenticArxiv/benchmark/task_spec.py` |
| expanded 任务 | `AgenticArxiv/benchmark/tasks_expanded.py` |
| split 解析和 `rl_train` | `AgenticArxiv/benchmark/splits.py` |
| 轨迹类和 JSONL 读写 | `AgenticArxiv/rl/trajectory.py` |
| benchmark metrics | `AgenticArxiv/benchmark/metrics.py` |
| 基础 SFT 生成 | `scripts/generate_sft_data.py` |
| 参数化 seed | `scripts/generate_parametric_sft_data.py` |
| 语言扩增 | `scripts/augment_sft_data.py`、`scripts/augment_parametric_sft_data.py` |
| 混合数据与 manifest | `scripts/build_sft_train_mix.py` |
| 训练前审计 | `AgenticArxiv/rl/train_sft.py` |
