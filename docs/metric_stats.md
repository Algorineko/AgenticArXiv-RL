# 指标统计
毕设的实验部分，旨在验证工程部分实现的三种agent工具调用模式的指标，以对比表示其性能和健壮性，主要包括：
- 统计上的时间，准确率等性能
- 一些极端/边界case对agent/LLM的冲击（这部分在之后要分析，暂不考虑）
当前功能代码基本完善，在开发指标统计代码的时候，应避免过度修改功能代码，但为了便于统计，或是要增加新的指标可以变动

---

## 实施 Plan

### 一、指标体系设计

#### 1.1 性能指标（Performance）

| 指标 | 说明 | 数据来源 |
|---|---|---|
| `total_time_ms` | 任务端到端耗时 | run() 前后计时 |
| `iteration_count` | ReAct 迭代次数 | len(history) |
| `total_llm_time_ms` | 累计 LLM 调用时间 | sum(steps.llm_latency_ms) |
| `total_tool_time_ms` | 累计工具执行时间 | sum(steps.tool_latency_ms) |
| `avg_llm_latency_ms` | 平均单次 LLM 延迟 | total_llm / iteration_count |
| `avg_tool_latency_ms` | 平均单次工具延迟 | total_tool / iteration_count |
| `framework_overhead_ms` | 框架开销 = total - llm - tool | 计算得出 |
| `token_usage` | Token 消耗（输入+输出） | LLM API 响应 `usage` 字段 |

**现有基础**：`agent_steps` 表已有 `llm_latency_ms` 和 `tool_latency_ms`，需要补充端到端时间和聚合统计。

#### 1.2 准确性/健壮性指标（Accuracy & Robustness）

| 指标 | 说明 | 判定方式 |
|---|---|---|
| `task_completed` | 任务是否完成 | 终止 action = FINISH |
| `termination_type` | 终止类型 | FINISH / FORCE_STOP / ERROR |
| `tool_call_accuracy` | 是否调用了正确的工具 | 对比预期工具序列 |
| `parse_success_rate` | LLM 响应解析成功率 | action_dict != None 的比例 |
| `tool_exec_success_rate` | 工具执行成功率 | observation 不含"错误"/"失败" |
| `correct_params` | 参数是否有效 | 关键参数值校验 |

---

### 二、标准化测试集

设计 5 类测试任务，覆盖不同工具调用场景：

| 类型 | 测试任务示例 | 预期工具 | 说明 |
|---|---|---|---|
| **搜索** | "检索最近7天AI(cs.AI)方向的论文，最多5篇" | `get_recently_submitted_cs_papers` | 单工具 |
| **搜索** | "获取最近3天机器学习(cs.LG)方向的论文，最多10篇" | `get_recently_submitted_cs_papers` | 参数变体 |
| **下载** | "下载第1篇论文的PDF" | `download_arxiv_pdf` | 依赖搜索结果 |
| **翻译** | "翻译第1篇论文" | `translate_arxiv_pdf` | 异步任务 |
| **缓存** | "查看第1篇论文的缓存状态" | `get_paper_cache_status` | 查询操作 |
| **复合** | "搜索CV论文(最多3篇)，然后下载第1篇" | `search` + `download` | 多步骤 |

#### 执行规模

每个任务 x 3 种 Agent x 10 次重复 = **约 270 次** Agent 运行（考虑 LLM API 成本）。

---

### 三、实现方案

#### 3.1 新增文件结构

```
AgenticArxiv/
  benchmark/                     # 新增 benchmark 模块
    __init__.py
    tasks.py                     # 测试任务定义（BENCHMARK_TASKS 列表）
    runner.py                    # Benchmark 运行器（驱动三种 Agent 执行测试集）
    metrics.py                   # 指标提取和计算
    report.py                    # 统计报告生成（Markdown + CSV + JSON）
    run_benchmark.py             # CLI 入口
```

#### 3.2 对功能代码的最小改动

**仅修改 `agents/base_agent.py` 的 `run()` 返回值**，增加：

```python
# 新增返回字段（不影响现有逻辑）
return {
    "task": task,
    "msg_id": msg_id,
    "history": history,
    "final_observation": final_observation,
    # ---- 新增 ----
    "total_time_ms": int((time.time() - run_start) * 1000),
    "iteration_count": len(history),
    "agent_type": self.agent_type,
    "timing": {
        "total_llm_ms": sum(s["llm_ms"] for s in step_timings),
        "total_tool_ms": sum(s["tool_ms"] for s in step_timings),
        "steps": step_timings,
    },
    "token_usage": token_usage,  # 从 LLM 响应的 usage 字段提取
}
```

需要在 run() 循环中追加 `step_timings` 列表收集每步时间。

**不需要改动**：API 端点、ORM、工具、前端。

#### 3.3 各模块职责

##### `benchmark/tasks.py`

```python
BENCHMARK_TASKS = [
    {
        "id": "search_01",
        "task": "检索最近7天内人工智能(cs.AI)方向的论文，最多5篇",
        "expected_tools": ["get_recently_submitted_cs_papers"],
        "expected_termination": "FINISH",
        "category": "search",
    },
    ...
]
```

##### `benchmark/runner.py`

```python
class BenchmarkRunner:
    def __init__(self, agent_types, repeat=3): ...
    
    def run_all(self, tasks) -> List[BenchmarkResult]:
        """对每个任务 x 每种 Agent x 重复 N 次执行"""
        for task in tasks:
            for agent_type in self.agent_types:
                for trial in range(self.repeat):
                    session_id = f"bench_{task['id']}_{agent_type}_{trial}"
                    # 处理依赖（如需先搜索再下载）
                    self._ensure_dependencies(task, session_id, agent_type)
                    agent = self._create_agent(agent_type)
                    result = agent.run(task=task["task"], session_id=session_id)
                    results.append(BenchmarkResult(...))
        return results
```

##### `benchmark/metrics.py`

```python
@dataclass
class TaskMetrics:
    task_id: str
    agent_type: str
    trial: int
    # 性能
    total_time_ms: int
    iteration_count: int
    total_llm_ms: int
    total_tool_ms: int
    framework_overhead_ms: int
    # 准确性
    task_completed: bool
    termination_type: str
    tool_call_sequence: List[str]
    tool_call_accurate: bool
    parse_failures: int
    tool_exec_failures: int

def extract_metrics(task_def, result, agent_type, trial) -> TaskMetrics: ...
```

##### `benchmark/report.py`

```python
class BenchmarkReport:
    def summary_by_agent(self) -> Dict:
        """按 Agent 类型聚合：平均时间、完成率、准确率"""
    
    def comparison_table(self) -> str:
        """生成 Markdown 对比表格（可直接复制到论文）"""
    
    def to_csv(self, path):
        """导出 CSV（用于画图）"""
    
    def to_json(self, path):
        """导出 JSON"""
    
    def print_report(self):
        """终端输出"""
```

##### `benchmark/run_benchmark.py`

```bash
# 用法
python -m benchmark.run_benchmark                          # 全部
python -m benchmark.run_benchmark --agents regex mcp # 指定 Agent
python -m benchmark.run_benchmark --repeat 5               # 重复次数
python -m benchmark.run_benchmark --tasks search           # 按类别
python -m benchmark.run_benchmark --output benchmark/output/ # 输出目录
```

---

### 四、预期输出

#### 4.1 终端对比表

```
=== AgenticArxiv Benchmark Report ===
模型: claude-sonnet-4.6 | 重复: 3 | 任务: 5

### 性能对比（平均值）

| 指标               | regex | mcp   | skill_cli |
|--------------------|-------------|-------|-----------|
| 总耗时(ms)         | 3200        | 4100  | 3800      |
| LLM 时间(ms)       | 2800        | 2850  | 2900      |
| 工具时间(ms)       | 200         | 350   | 500       |
| 框架开销(ms)       | 200         | 900   | 400       |
| 迭代次数           | 2.1         | 2.3   | 2.4       |

### 准确性对比

| 指标               | regex | mcp   | skill_cli |
|--------------------|-------------|-------|-----------|
| 任务完成率         | 93%         | 87%   | 80%       |
| 工具调用准确率     | 90%         | 87%   | 83%       |
| 解析成功率         | 95%         | 93%   | 85%       |
```

#### 4.2 CSV 输出（论文画图）

```csv
task_id,agent_type,trial,total_time_ms,llm_ms,tool_ms,overhead_ms,iterations,completed,termination,tool_accurate
search_01,regex,0,3150,2800,150,200,2,true,FINISH,true
search_01,mcp,0,4200,2900,400,900,2,true,FINISH,true
...
```

---

### 五、执行步骤

1. 创建 `benchmark/` 目录和文件结构
2. **小幅修改 `base_agent.py`**：扩展 run() 返回值
3. 实现 `benchmark/tasks.py`：测试集
4. 实现 `benchmark/metrics.py`：指标提取
5. 实现 `benchmark/runner.py`：运行器
6. 实现 `benchmark/report.py`：报告生成
7. 实现 `benchmark/run_benchmark.py`：CLI 入口
8. 端到端测试运行

### 六、验证方法

```bash
# 快速验证（单任务、单 Agent、1 次）
python -m benchmark.run_benchmark --tasks search --agents regex --repeat 1

# 完整 benchmark
python -m benchmark.run_benchmark --repeat 3 --output benchmark/output/

# 检查输出
cat benchmark/output/report.md
cat benchmark/output/raw_data.csv

# 验证功能代码不受影响
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","message":"搜索AI论文"}'
```
