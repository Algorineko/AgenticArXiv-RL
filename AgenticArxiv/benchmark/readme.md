# Benchmark 模块

> 本文按当前扩展任务集和 CLI 源码维护。当前正式扩展切分是
> `data/splits/v3_81.json`；`v2_62.json` 和 `v1.json` 保留为历史实验输入，
> 不要在新实验中把它们写成“当前版本”。

Benchmark 用同一套任务声明驱动三种执行模式（`regex` / `mcp` / `skill_cli`），
再从轨迹中提取工具序列、参数、终止状态、时间和 token 指标。任务声明位于
`tasks.py` 和 `tasks_expanded.py`；`TaskSpec` 的 `steps` 同源派生
`expected_tools` 与 `expected_tool_args`。

## 1. 快速运行

以下命令从 `AgenticArxiv/` 目录运行：

```bash
cd AgenticArxiv

# 基础 smoke set：8 条任务 × 3 种 Agent × 3 次重复 = 72 次运行
python -m benchmark.run_benchmark

# 只跑一种 Agent；适合比较 Base/SFT/GRPO 的策略能力
python -m benchmark.run_benchmark --agents regex

# 当前 v3 的开发集：8 条任务，结果不能当盲测
python -m benchmark.run_benchmark \
  --task-set expanded \
  --offline \
  --split ../data/splits/v3_81.json:dev

# 当前 v3 的 IID 留出集
python -m benchmark.run_benchmark \
  --task-set expanded \
  --offline \
  --split ../data/splits/v3_81.json:iid_test \
  --agents regex

# 当前 v3 的 OOD 留出集
python -m benchmark.run_benchmark \
  --task-set expanded \
  --offline \
  --split ../data/splits/v3_81.json:ood_test \
  --agents regex

# 按任务 id 运行，不把新解读类别误写成 --tasks choice
python -m benchmark.run_benchmark \
  --task-set expanded --offline \
  --task-ids summary_cv5_structured120 analyze_cv5_ref1_desc
```

默认 `task-set=default` 读取 `benchmark/tasks.py` 的 8 条任务；
`task-set=expanded` 才读取 `tasks_expanded.py` 的 81 条任务。expanded 任务中
有一部分 ground truth 绑定快照；没有 `--offline` 时，运行器会按
`offline_only_ids()` 跳过它们，并提示任务池已经缩小。要做正式的阶段对比，
必须固定 `task-set`、split 文件、snapshot、agent、seed、repeat 和 backend。

## 2. CLI 参数与选择边界

`run_benchmark.py` 的实际 argparse 选项如下：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--agents` | `regex mcp skill_cli` | 可重复传入一个或多个 agent |
| `--repeat` | `3` | 每个任务的重复次数 |
| `--tasks` | `None` | 按类别筛选 |
| `--task-ids` | `None` | 按 id 筛选，优先于 split/category |
| `--output` | 仓库根目录 `data/` | 报告输出目录 |
| `--model` | settings 中的模型 | API 模型名或 transformers 本地路径 |
| `--backend` | `api` | `api` 或 `transformers` |
| `--local-device` | `auto` | transformers 的 `auto` / `cuda` / `cpu` |
| `--local-dtype` | `auto` | `auto` / `float16` / `bfloat16` / `float32` |
| `--seed` | `42` | 本地生成基础随机种子 |
| `--prefix` | `bench_r<timestamp>` | session id 前缀 |
| `--no-thinking` | 关闭 | 关闭支持模型的 thinking 模式 |
| `--offline` | 关闭 | 使用快照回放，不请求真实 arXiv |
| `--snapshot` | `None` | 覆盖默认快照路径 |
| `--save-traces [PATH]` | 关闭 | 保存每条 history 的 JSONL |
| `--split [FILE:]NAME` | `None` | 使用显式或默认切分 |
| `--task-set` | `default` | `default` 或 `expanded` |

`--tasks` 的 choices 是源码硬边界：

```text
search, download, translate, cache, composite,
keyword_search, ref_form, optional, state,
long_chain, constraint, infeasible
```

因此 `paper_reading`、`paper_summary`、`figure_extraction`、`figure_analysis`
虽然是当前 expanded 任务的 category，却不能写成
`--tasks paper_summary`。请改用 `--task-ids` 或显式 split。

### 本地模型和离线工具是两个开关

```bash
python -m benchmark.run_benchmark \
  --backend transformers \
  --model /path/to/Qwen2.5-1.5B-Instruct \
  --local-device cuda \
  --local-dtype bfloat16 \
  --agents regex \
  --task-set expanded \
  --offline \
  --split ../data/splits/v3_81.json:iid_test
```

`--backend transformers` 只控制 LLM 加载方式；`--offline` 才控制工具环境。
本地模型仍可能访问真实 arXiv，除非同时加 `--offline`。反过来，API 模型也
可以使用固定 snapshot。

## 3. v1、v2、v3 的版本区别

当前仓库中三份 JSON 的静态切分如下：

| 文件 | train | dev | iid_test | ood_test | 合计 | 角色 |
|---|---:|---:|---:|---:|---:|---|
| `data/splits/v1.json` | 42 | 0 | 13 | 4 | 59 | 历史默认文件，保留旧实验复现 |
| `data/splits/v2_62.json` | 36 | 8 | 14 | 4 | 62 | 历史扩展版，仍是部分 SFT 血缘的来源 |
| `data/splits/v3_81.json` | 51 | 8 | 18 | 4 | 81 | 当前 expanded 版 |

### v1：历史默认路径

`benchmark/splits.py` 的 `DEFAULT_SPLIT_PATH` 仍然指向 `v1.json`。所以只写
`--split iid_test` 会读取历史默认文件；这是为了保持旧实验可复现，不是
“自动选择最新切分”。正式报告和阶段比较必须写完整的 `FILE:NAME`。

### v2：历史 62 条扩展集

v2 的 train=36、dev=8、iid=14、ood=4。旧数据生成器、已有 manifest 和部分
历史 SFT 方案使用 `v2_62.json:train`；不要为了“统一命名”把 v2 全局替换成
v3，因为这会改变既有实验的任务集合。

### v3：当前 81 条扩展集

v3 在 v2 的基础上加入并重新安排了解读任务族。当前 train=51、dev=8、
iid=18、ood=4；`rates` 仍只覆盖延续自 v2 的 36 个 train id：

- `paper_reading`：正文读取和章节选择；
- `paper_summary`：`style` 与预算 bucket 的选择；
- `figure_extraction`：先下载，再抽出图表；
- `figure_analysis`：抽图后选择 `figure_no` 和 `describe/axes/trend`；
- 既有 `search`、`ref_form`、`state`、`constraint`、`long_chain` 等族的
  参数和留出实例也重新安排。

v3 的 `rates_metadata.note` 明确写出新增族尚无实测成功率。缺 rate 的任务仍
在 train 中用于任务集边界，但不会进入动态 `rl_train`。

## 4. split、rates 和 `rl_train`

切分按 template key，而不是按单条 task 随机切：默认 key 是
`(template or category, len(expected_tools))`。这样同一模板的参数变体不会
一条进 train、另一条进 test。

| 名称 | 语义 | 能否用于正式训练 |
|---|---|---|
| `train` | 训练语义来源；v3 有 51 条 | 可以用于 train-only SFT；GRPO 仍建议筛选 |
| `dev` | 已检查的 pilot/开发任务；v3 有 8 条 | 不应进入正式 SFT 或被当盲测 |
| `iid_test` | 同模板、不同参数；v3 有 18 条 | 留作参数泛化评测 |
| `ood_test` | 留出的模板或链长；v3 有 4 条 | 留作形态泛化评测 |
| `rl_train` | train 中有 rate 且 `0.2 <= rate <= 0.8` | 动态计算，不能手写成固定数组 |

`rl_train` 的中间带规则来自 `splits.py`：成功率接近 0 或 1 时，同一 prompt
的组内奖励容易零方差，GRPO 不产生有效优势。v3 当前 36 个有 rate 的任务中，
有 6 个落在中间带；这只是当前冻结 Base run 的结果，不是 51 条 train 的
完整实测覆盖。

```bash
# 从 AgenticArxiv/ 显式引用 v3 的 train
python -m benchmark.run_benchmark \
  --task-set expanded --offline \
  --split ../data/splits/v3_81.json:train

# GRPO 读取动态 rl_train 时同样带上版本文件
python -m rl.train_grpo \
  --task_set expanded \
  --split ../data/splits/v3_81.json:rl_train \
  --snapshot ../data/mock_arxiv_snapshot.json
```

如果 split 没有 `rates`，读取 `rl_train` 会报错；如果 train 没有中间带任务，
也会报错，而不是悄悄返回空列表。换模型、换快照、换 agent 或换 repeat 后，
应重新测量并记录新的 rates metadata。

## 5. SFT 使用边界

expanded SFT 必须显式使用某一版本文件的 train：

```bash
cd ..
python scripts/generate_sft_data.py \
  --task_set expanded \
  --split data/splits/v3_81.json:train \
  --snapshot data/mock_arxiv_snapshot.json \
  --output data/sft/sft_v3_train.jsonl
cd AgenticArxiv
```

生成器会拒绝：

- 未传 `--split`；
- 只有 `train` 而没有 `PATH:train`；
- 把 `dev`、`iid_test` 或 `ood_test` 作为 expanded SFT 来源；
- split 中存在 expanded 任务集没有的 id。

确定性专家还会在离线环境中执行工具，轨迹不通过严格成功校验就不会写出。
这意味着“能生成文件”不等于“所有任务都生成了样本”；应读取 manifest 或
日志中的行数、语义任务数和失败信息。

## 6. 退化策略基线

```bash
python -m benchmark.run_baselines --task-set expanded

python -m benchmark.run_baselines \
  --task-set expanded \
  --seed 42 \
  --random-samples 20 \
  --output ../data/agentic-arxiv-baselines
```

当前基线不是四种而是五种：

| 策略 | 用途 |
|---|---|
| `reference` | 任务声明的标准工具路径 |
| `always_finish` | 检查立即 FINISH 是否被误读为完成 |
| `always_search` | 检查无视任务、固定搜索是否刷分 |
| `random_tool` | 检查随机合法动作的基线分数 |
| `wrong_args` | 检查工具名正确但参数错误时的扣分 |

默认需要 reference gap 和逐类别 gap 都至少达到 `0.3`；失败返回退出码 1。
基线构造的是 synthetic trajectory，不执行工具，也不测量模型质量。

## 7. 轨迹、重评分和 badcase

### 保存 benchmark traces

```bash
python -m benchmark.run_benchmark \
  --task-set expanded --offline \
  --split ../data/splits/v3_81.json:dev \
  --save-traces ../data/bench_v3_dev/traces.jsonl
```

输出每行包含 `task_id`、`agent_type`、`trial`、`session_id` 和 `history`。
它不是 `rl/trajectory.py` 的 `Trajectory`：没有 `final_reward`、
`reward_components` 或 `timestamp`。

### 用新规则重评分

```bash
python -m benchmark.rescore_traces \
  --traces ../data/bench_v3_dev/traces.jsonl \
  --summary ../data/bench_v3_dev/summary.json \
  --output ../data/bench_v3_dev/rescored \
  --snapshot ../data/mock_arxiv_snapshot.json \
  --task-set expanded \
  --split ../data/splits/v3_81.json:dev
```

重评分不加载模型，只读取已有 history 和 summary 的 timing/token 字段，并
在选定 task pool 上重新调用 `extract_metrics()`。trace 中未知的任务会被跳过；
不能从缺失的 trace 中恢复模型文本或工具调用。

### 回放和捕获 badcase

```bash
cd ..
python eval/badcase_replay.py replay \
  --cases eval/eval_cases.jsonl --task-set expanded \
  --training-step 100 --verbose

python eval/badcase_replay.py capture \
  --traces data/bench_v3_dev/traces.jsonl \
  --cases eval/eval_cases.jsonl \
  --source v3_dev --dry-run
cd AgenticArxiv
```

`open` case 仍复现，`newly_fixed` 表示当前不再复现，`fixed` 再次出现则是
regression。`training_step` 要固定，否则奖励课程会改变“是否复现”的条件。

## 8. 代表性任务与类别

不要在文档中把 81 条任务完整复制成第二份目录；完整定义在
`tasks_expanded.py`，版本归属在 `v3_81.json`。下面只列出可解释边界的代表：

| 类别 | 代表 id | 工具链 / 约束 | 关键检查 |
|---|---|---|---|
| `search` | `search_AI_30d_25` | 单次方向检索 | aspect、days、max_results |
| `keyword_search` | `search_kw_llm` | 关键词检索 | query 规范化与快照 query hash |
| `ref_form` | `ref_ctrl_id_download` | 同一论文的 id/ref/title 说法 | 解析出的 paper id |
| `state` | `state_ref_last_active` | 依赖会话最近活动论文 | session state 是否同步 |
| `optional` | `opt_threads` | 翻译可选 threads 等参数 | 可选字段是否原样传递 |
| `composite` | `multi_cr5_cache1` | 多步搜索/下载/缓存 | 步骤顺序和多余调用 |
| `long_chain` | `chain_ai5_read_then_summary` | 读内容再总结 | `max_iterations` 和前置状态 |
| `constraint` | `constraint_search_only` | 只做目标动作，不额外下载 | 负向工具约束 |
| `infeasible` | `infeasible_no_session` | 正确行为是无工具并解释阻塞 | terminal reason |
| `paper_reading` | `read_cv5_method` | 搜索→下载→读章节 | `section` 是否正确 |
| `paper_summary` | `summary_cv5_structured120` | 搜索→下载→总结 | style 与 120 budget |
| `figure_extraction` | `figure_cv5_ref1` | 搜索→下载→抽图 | T4 count 和 ref |
| `figure_analysis` | `analyze_cv5_ref1_desc` | 搜索→下载→抽图→分析 | figure_no 和 question |

## 9. 输出文件

`BenchmarkReport.save_all()` 通常写入：

```text
data/<output>/
├─ raw_data.csv       # 每条明细，含 session_id 等字段
├─ report.md          # Markdown 汇总
├─ summary.json       # 汇总、details 和 errors
└─ errors.csv         # 有异常时的会话记录
```

如果调用 `--save-traces`，另有：

```text
data/<output>/traces.jsonl
```

`draw/plot.py` 读取的是报告/CSV 数据，不会自动运行 benchmark：

```bash
cd ..
python draw/plot.py --data data/raw_data.csv --output draw/images
```

## 10. 指标边界

### 性能字段

| 字段 | 含义 |
|---|---|
| `total_time_ms` | 端到端时间 |
| `total_llm_ms` | LLM 调用累计时间 |
| `total_tool_ms` | 工具执行累计时间 |
| `framework_overhead_ms` | `total - llm - tool` 的框架开销 |
| `iteration_count` | ReAct 迭代次数 |
| `prompt_tokens` / `completion_tokens` / `total_tokens` | token 用量 |

### 准确性字段

| 字段 | 含义和限制 |
|---|---|
| `task_completed` | 以 FINISH 结束；不单独证明业务完成 |
| `termination_type` | `FINISH` / `FORCE_STOP` / `ERROR` 等 |
| `tool_call_accurate` | 工具名序列严格相等，含顺序和多余调用 |
| `arg_score` | 参数匹配度；没有 oracle 时保持中性并标记不适用 |
| `ref_score` | 解析后的 paper id 匹配度，不只看 ref 写法 |
| `false_finish` | FINISH 但少做了期望工具的情况 |
| `parse_failures` | 动作解析失败次数 |
| `tool_exec_failures` | 工具执行失败次数 |
| `terminal_semantics_accurate` | blocked 任务是否解释了声明原因 |

“严格成功”需要正常结束、工具和参数正确、指代正确、无解析/执行失败，且
blocked 任务的终止语义准确；不能只用 `task_completed=True` 统计成功率。

## 11. 维护原则

新增任务或工具时同步核对：

1. `task_spec.py` 是否仍由 `steps` 派生两个 oracle。
2. split 是否记录版本、rates、pilot 和 held-out policy。
3. `--tasks` choices 是否真的包含要写进命令的类别。
4. snapshot-bound 任务是否在离线环境有完整记录。
5. trace、Trajectory、manifest 三种 JSONL/JSON schema 是否没有被混称。
6. 文档是否把“源码可解释”与“实际运行结果”分开。
