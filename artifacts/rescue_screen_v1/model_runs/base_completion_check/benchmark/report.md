## Benchmark 对比报告
模型: Qwen/Qwen2.5-1.5B-Instruct | 样本数: 24 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 26487.5 |
| LLM 时间(ms) | 26478.4 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 9.1 |
| 迭代次数 | 6.7 |
| Token 用量 | 15149.8 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 0% |
| pass^2 | n/a |
| pass^3 | n/a |

> 24 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 8% |
| 严格成功率 | 0% |
| 工具调用准确率 | 0% |
| 参数准确率 | 42% |
| 指代解析准确率 | 20% |
| 阻塞终止语义准确率 | n/a |
| 假完成率 | 4% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | n/a |
| 工具调用 / 成功 | n/a |
| 耗时(ms) / 成功 | n/a |
| 迭代数中位·成功 | n/a |
| 迭代数中位·失败 | 7 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 22 |
| 声称完成但错 | 2 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 24 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 0 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| rescue_v1_search_dev_cache_0 | 1 | 26151 | 0% | 0% | 0% |
| rescue_v1_search_dev_cache_1 | 1 | 25086 | 0% | 0% | 0% |
| rescue_v1_search_dev_cache_2 | 1 | 23964 | 0% | 0% | 0% |
| rescue_v1_search_dev_cache_3 | 1 | 24204 | 0% | 0% | 0% |
| rescue_v1_search_dev_download_0 | 1 | 30308 | 0% | 0% | 0% |
| rescue_v1_search_dev_download_1 | 1 | 30807 | 0% | 0% | 0% |
| rescue_v1_search_dev_download_2 | 1 | 30967 | 0% | 0% | 0% |
| rescue_v1_search_dev_download_3 | 1 | 24666 | 0% | 0% | 0% |
| rescue_v1_search_dev_download_translate_0 | 1 | 26094 | 0% | 0% | 0% |
| rescue_v1_search_dev_download_translate_1 | 1 | 29755 | 0% | 0% | 0% |
| rescue_v1_search_dev_download_translate_2 | 1 | 17155 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_translate_3 | 1 | 31362 | 0% | 0% | 0% |
| rescue_v1_search_dev_search_0 | 1 | 28391 | 0% | 0% | 0% |
| rescue_v1_search_dev_search_1 | 1 | 27077 | 0% | 0% | 0% |
| rescue_v1_search_dev_search_2 | 1 | 27970 | 0% | 0% | 0% |
| rescue_v1_search_dev_search_3 | 1 | 27353 | 0% | 0% | 0% |
| rescue_v1_search_dev_search_download_0 | 1 | 31412 | 0% | 0% | 0% |
| rescue_v1_search_dev_search_download_1 | 1 | 30687 | 0% | 0% | 0% |
| rescue_v1_search_dev_search_download_2 | 1 | 31659 | 0% | 0% | 0% |
| rescue_v1_search_dev_search_download_3 | 1 | 30931 | 0% | 0% | 0% |
| rescue_v1_search_dev_translate_0 | 1 | 6284 | 100% | 0% | 0% |
| rescue_v1_search_dev_translate_1 | 1 | 23950 | 0% | 0% | 0% |
| rescue_v1_search_dev_translate_2 | 1 | 26113 | 0% | 0% | 0% |
| rescue_v1_search_dev_translate_3 | 1 | 23354 | 0% | 0% | 0% |