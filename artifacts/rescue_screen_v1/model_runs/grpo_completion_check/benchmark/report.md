## Benchmark 对比报告
模型: Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO | 样本数: 24 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 4967.8 |
| LLM 时间(ms) | 4965.7 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 2.2 |
| 迭代次数 | 1.8 |
| Token 用量 | 3966 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 46% |
| pass^2 | n/a |
| pass^3 | n/a |

> 24 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 100% |
| 严格成功率 | 46% |
| 工具调用准确率 | 46% |
| 参数准确率 | 48% |
| 指代解析准确率 | 38% |
| 阻塞终止语义准确率 | n/a |
| 假完成率 | 54% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 8653.1 |
| 工具调用 / 成功 | 1.7 |
| 耗时(ms) / 成功 | 10838.9 |
| 迭代数中位·成功 | 3 |
| 迭代数中位·失败 | 1 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 0 |
| 声称完成但错 | 13 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 13 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 11 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| rescue_v1_search_dev_cache_0 | 1 | 2157 | 100% | 0% | 0% |
| rescue_v1_search_dev_cache_1 | 1 | 2056 | 100% | 0% | 0% |
| rescue_v1_search_dev_cache_2 | 1 | 2301 | 100% | 0% | 0% |
| rescue_v1_search_dev_cache_3 | 1 | 2266 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_0 | 1 | 2607 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_1 | 1 | 2604 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_2 | 1 | 2706 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_3 | 1 | 2645 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_translate_0 | 1 | 8443 | 100% | 100% | 100% |
| rescue_v1_search_dev_download_translate_1 | 1 | 8229 | 100% | 100% | 100% |
| rescue_v1_search_dev_download_translate_2 | 1 | 8195 | 100% | 100% | 100% |
| rescue_v1_search_dev_download_translate_3 | 1 | 4977 | 100% | 0% | 0% |
| rescue_v1_search_dev_search_0 | 1 | 6840 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_1 | 1 | 5922 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_2 | 1 | 5887 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_3 | 1 | 5978 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_download_0 | 1 | 9309 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_download_1 | 1 | 9303 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_download_2 | 1 | 9059 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_download_3 | 1 | 9037 | 100% | 100% | 100% |
| rescue_v1_search_dev_translate_0 | 1 | 2172 | 100% | 0% | 0% |
| rescue_v1_search_dev_translate_1 | 1 | 2203 | 100% | 0% | 0% |
| rescue_v1_search_dev_translate_2 | 1 | 2174 | 100% | 0% | 0% |
| rescue_v1_search_dev_translate_3 | 1 | 2158 | 100% | 0% | 0% |