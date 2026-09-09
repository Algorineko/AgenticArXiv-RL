# Benchmark 模块

三种 Agent 模式（regex / mcp / skill_cli）的性能与健壮性对比测试。

## 运行 Benchmark

```bash
cd AgenticArxiv

# 全部任务、全部 Agent、重复 3 次（默认）
python -m benchmark.run_benchmark

# 指定 Agent 类型
python -m benchmark.run_benchmark --agents regex mcp

# 指定任务类别: search / download / translate / cache / composite
python -m benchmark.run_benchmark --tasks search

# 指定任务 ID
python -m benchmark.run_benchmark --task-ids search_01 cache_01

# 调整重复次数
python -m benchmark.run_benchmark --repeat 5

# 指定 LLM 模型
python -m benchmark.run_benchmark --model gpt-4-turbo

# 使用本地 Hugging Face 模型（不读取 LLM_API_KEY）
python -m benchmark.run_benchmark \
  --backend transformers \
  --model /path/to/Qwen2.5-1.5B-Instruct \
  --local-device cuda \
  --local-dtype bfloat16 \
  --agents regex \
  --offline

# 指定输出目录（默认 ../data）
python -m benchmark.run_benchmark --output /path/to/output

# 指定 session 前缀（用于区分不同测试轮次，默认 bench_r<timestamp>）
python -m benchmark.run_benchmark --prefix bench_r1

# 当前 62 条任务的开发集（从 AgenticArxiv/ 目录运行）
python -m benchmark.run_benchmark \
  --task-set expanded \
  --offline \
  --split ../data/splits/v2_62.json:dev
```

默认 8 个任务 x 3 种 Agent x 3 次重复 = 72 次运行。

### 本地模型后端

`--backend api` 保持原来的 OpenAI-compatible API 行为，也是默认值。
`--backend transformers` 直接通过 `AutoModelForCausalLM` 加载 `--model`
指定的本地目录，不依赖 API key。模型在同一次 Benchmark 中只加载一次，供所有
任务和 Agent 复用。

本地后端专用参数：

| 参数 | 含义 |
|---|---|
| `--local-device` | `auto` / `cuda` / `cpu`，默认 `auto` |
| `--local-dtype` | `auto` / `float16` / `bfloat16` / `float32` |
| `--seed` | 生成随机种子；每次调用使用 seed 加调用序号 |

本地模型仍应配合 `--offline` 使用：前者控制 LLM 从哪里加载，后者控制工具是否
访问真实 arXiv，两者解决的是不同问题。为了测模型能力而不是三种执行框架的差异，
Base/SFT/GRPO 阶段默认只跑 `--agents regex`；三种 Agent 的对比实验再单独运行。

### 训练/开发/留出集切分

当前 62 条扩展任务使用 `../data/splits/v2_62.json`：

| 名字 | 条数 | 是什么 |
|---|---:|---|
| `train` | 36 | 可用于构建 SFT/后续 RL 数据的任务 |
| `dev` | 8 | 已在 pilot 中反复查看轨迹的任务，用于调试和 Bad Case 分析 |
| `iid_test` | 14 | 与 train 同模板、但参数不同的盲测实例——「换个参数还会不会」 |
| `ood_test` | 4 | train/dev/iid 均未出现的模板或链长——「换个形态还会不会」 |

这 8 条 pilot 任务不能再放进最终测试集：我们已经根据其轨迹决定了后续数据方向，继续把
它们称作“盲测”会产生人为调参泄漏。它们仍很有价值，但角色是 `dev`，不是整个 Benchmark。

`v2_62.json` 的 `rates` 来自冻结环境中当前 Qwen Base 对 train 的三次重复，以严格成功率
统计；`rl_train` 由此计算为 6 条 20%～80% 的中间难度任务。rates 只覆盖 train，不使用
iid/ood 的任务级结果做训练选择。沿用旧模型或旧评测环境的 rates，会把过时难度带进 GRPO。

切分在**模板**层面进行：`search_AI_1d_3` 与 `search_AI_30d_25` 是同一模板换参数，
按任务随机切会让它们分居两侧，测出来的「泛化」其实是记忆。

两个留出集都要看：只有 iid 提升可能是记住了模板，ood 也提升才更像基础能力变强。

`rl_train` 供 `rl/train_grpo.py --split rl_train` 使用——GRPO 的优势是组内相对的，
成功率贴近 0 或 1 的任务每条采样奖励一致，方差为零、不产生梯度，放进训练集是空转。

从 `AgenticArxiv/` 目录运行时，应显式指定：

```bash
--split ../data/splits/v2_62.json:train
--split ../data/splits/v2_62.json:dev
--split ../data/splits/v2_62.json:iid_test
--split ../data/splits/v2_62.json:ood_test
```

只写 `--split iid_test` 会继续读取历史默认文件 `v1.json`，这是为了让旧实验可复现，不能用于
当前 62 条任务的正式对比。阶段间对比必须引用同一份显式切分文件。

历史 `v1.json` 固定保存原来的 59 条任务（train=42、iid=13、ood=4，以及由旧 rates 计算的
rl_train=13）。新增关键词检索任务后不回写 v1，否则同一个版本名会在不同时间代表不同实验。

正式 SFT 数据同样必须使用显式的 v2 train，不能把全部 expanded 任务交给生成器：

```bash
cd ..  # 从 AgenticArxiv/ 回到仓库根目录
python scripts/generate_sft_data.py \
  --task_set expanded \
  --split data/splits/v2_62.json:train \
  --snapshot data/mock_arxiv_snapshot.json \
  --output data/sft/sft_v0_train.jsonl
```

生成器会拒绝未指定 split、裸 `train` 以及 dev/iid/ood，并在专家工具执行失败时终止，避免
把测试题或失败轨迹写进监督数据。

## 退化策略基线

```bash
cd AgenticArXiv

# 不调用 LLM、网络或真实工具；用扩展任务集检查评分器能否区分弱策略
python -m benchmark.run_baselines --task-set expanded

# 保存逐任务 JSON 和 Markdown 报告；random_tool 默认从 seed 起采样 20 次
python -m benchmark.run_baselines --task-set expanded --seed 42 --random-samples 20 --output /tmp/agentic-arxiv-baselines
```

该命令会并排评分四种确定性策略：

| 策略 | 用途 |
|---|---|
| `reference` | 回放任务声明的标准工具路径，作为评分上限 |
| `always_finish` | 立即终止，检查“正常 FINISH”会不会被误读为完成 |
| `always_search` | 无视任务、固定调用一次搜索 |
| `random_tool` | 对每条任务以固定 seed 选择一次合法工具调用 |

报告单独显示 `finish_rate` 与 `exact_tool_path_rate`：前者只表示轨迹以 `FINISH` 结束，不是业务任务已完成。`random_tool` 汇总多个 seed 并报告标准差；没有参数标准答案的任务不参与平均参数分。报告还会列出每种退化策略中未完整匹配参考工具和参数、但仍获得高分的任务，便于直接定位奖励漏洞。

默认健康门槛要求每种退化策略的平均奖励比 `reference` 至少低 `0.3`；不满足时命令返回非零状态，可直接接入 CI。可用 `--min-reference-gap` 调整门槛，或用 `--top` 调整每种策略展示的高分任务数。

此外还有一道**逐类目**门槛 `--min-category-gap`（默认同为 `0.3`）。总体均值会把单个类目的漏洞摊平：`always_search` 曾经在 search 类目上距参考仅 `0.167`（无视任务、永远发同一个 cs.AI 查询，在「检索 cs.CL」任务上拿 `0.933`），而总体均值差有 `0.832`，总体闸照样 PASS。逐类目闸会剔除「策略恰好复现了参考解法」的那些行——infeasible 任务上 `always_finish` **就是**参考解法（正确行为是一次工具都不调），那不是漏洞。

## 绘图

```bash
cd ..  # 回到仓库根目录（draw/ 与 data/ 位于根目录，不在 AgenticArxiv/ 内；已在根目录时可跳过）

# 使用默认路径（读 data/raw_data.csv，输出到 draw/images/）
python draw/plot.py

# 自定义路径
python draw/plot.py --data data/raw_data.csv --output draw/images
```

生成 5 张图表：

| 文件 | 内容 |
|---|---|
| `time_breakdown.png` | 堆叠条形图：各 Agent 平均 LLM/Tool/Overhead 时间 |
| `accuracy_comparison.png` | 分组条形图：任务完成率 + 工具调用准确率 |
| `iteration_boxplot.png` | 箱线图：迭代次数分布 |
| `per_task_time.png` | 分组条形图：每个任务在不同 Agent 下的耗时 |
| `token_usage.png` | 条形图：平均 Token 用量 |

## 输出文件

```
data/
  raw_data.csv      # 逐条明细（含 session_id 列），可用于论文绘图
  report.md         # Markdown 对比表格
  summary.json      # JSON 格式汇总 + 明细 + errors
  errors.csv        # 异常会话记录（session_id + error），仅在有异常时生成

draw/images/
  time_breakdown.png
  accuracy_comparison.png
  iteration_boxplot.png
  per_task_time.png
  token_usage.png
```

## 测试任务

| ID | 类别 | 任务描述 | 预期工具 |
|---|---|---|---|
| search_01 | search | 检索 cs.AI 论文 | get_recently_submitted_cs_papers |
| search_02 | search | 检索 cs.LG 论文 | get_recently_submitted_cs_papers |
| search_03 | search | 检索 cs.CL 论文 | get_recently_submitted_cs_papers |
| search_04 | search | 检索全部计算机科学论文 | get_recently_submitted_cs_papers |
| download_01 | download | 下载第 1 篇论文 PDF | download_arxiv_pdf |
| translate_01 | translate | 翻译第 1 篇论文 | translate_arxiv_pdf |
| cache_01 | cache | 查看缓存状态 | get_paper_cache_status |
| composite_01 | composite | 搜索 + 下载（多步骤） | get_recently_submitted_cs_papers, download_arxiv_pdf |

有依赖关系的任务（download_01 → search_01, translate_01 → download_01 等）会自动先执行依赖。

## 指标体系

### 性能指标

| 指标 | 说明 |
|---|---|
| total_time_ms | 端到端总耗时 |
| total_llm_ms | 累计 LLM 调用时间 |
| total_tool_ms | 累计工具执行时间 |
| framework_overhead_ms | 框架开销 (= total - llm - tool) |
| iteration_count | ReAct 迭代次数 |
| tokens | Token 消耗量 |

### 准确性指标

| 指标 | 说明 |
|---|---|
| task_completed | 轨迹是否以 `FINISH` 正常结束；不验证业务终态 |
| termination_type | 终止类型: FINISH / FORCE_STOP / ERROR / INCOMPLETE |
| tool_call_accurate | 实际工具调用是否与预期工具序列完全相等（顺序严格、无多余/重复调用），只比工具名 |
| arg_score | 参数级匹配度 `[0,1]`：逐步比对期望键的**取值**；未声明 `expected_tool_args` 时为 1.0 |
| ref_score | 指代解析准确率 `[0,1]`：比对**解析出的 `paper_id`** 而非 `ref` 的写法；未声明 `expected_paper` 时为 1.0 |
| false_finish | 以 `FINISH` 结束、但期望工具没做全。只抓「做少了」，绕路多调不算 |
| parse_failures | LLM 响应解析失败次数 |
| tool_exec_failures | 工具执行失败次数 |

## 模块结构

```
benchmark/
  __init__.py
  task_spec.py        # TaskSpec/Step：expected_tools 与 expected_tool_args 同源派生
  tasks.py           # 8 条冒烟任务 (BENCHMARK_TASKS)
  tasks_expanded.py   # 62 条完整基准集 (--task-set expanded)
  runner.py           # BenchmarkRunner：驱动 Agent 执行测试集
  metrics.py          # TaskMetrics：从 run() 结果提取指标
  baselines.py        # 确定性退化策略与评分敏感性汇总
  splits.py           # 模板层 train/iid/ood 切分与 load_split
  badcases.py         # 坏例用例的判定与回放（CLI 在 eval/badcase_replay.py）
  report.py           # BenchmarkReport：生成 Markdown/CSV/JSON 报告
  run_benchmark.py    # CLI 入口
  run_baselines.py    # 离线退化策略诊断 CLI
```
