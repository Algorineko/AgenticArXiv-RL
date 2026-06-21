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

# 指定输出目录（默认 ../data）
python -m benchmark.run_benchmark --output /path/to/output

# 指定 session 前缀（用于区分不同测试轮次，默认 bench_r<timestamp>）
python -m benchmark.run_benchmark --prefix bench_r1
```

默认 7 个任务 x 3 种 Agent x 3 次重复 = 63 次运行。

## 绘图

```bash
cd AgenticArXiv  # 项目根目录

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
| task_completed | 任务是否正常完成 (FINISH) |
| termination_type | 终止类型: FINISH / FORCE_STOP / ERROR / INCOMPLETE |
| tool_call_accurate | 实际工具调用是否包含全部预期工具（顺序子序列匹配） |
| parse_failures | LLM 响应解析失败次数 |
| tool_exec_failures | 工具执行失败次数 |

## 模块结构

```
benchmark/
  __init__.py
  tasks.py           # 测试任务定义 (BENCHMARK_TASKS)
  runner.py           # BenchmarkRunner：驱动 Agent 执行测试集
  metrics.py          # TaskMetrics：从 run() 结果提取指标
  report.py           # BenchmarkReport：生成 Markdown/CSV/JSON 报告
  run_benchmark.py    # CLI 入口
```
