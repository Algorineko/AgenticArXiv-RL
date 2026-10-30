## Benchmark 对比报告
模型: Qwen/Qwen2.5-1.5B-Instruct | 样本数: 24 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 15724.9 |
| LLM 时间(ms) | 15719 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 5.9 |
| 迭代次数 | 4.2 |
| Token 用量 | 9196.9 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 38% |
| pass^2 | n/a |
| pass^3 | n/a |

> 24 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 58% |
| 严格成功率 | 38% |
| 工具调用准确率 | 42% |
| 参数准确率 | 42% |
| 指代解析准确率 | 25% |
| 阻塞终止语义准确率 | n/a |
| 假完成率 | 8% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.1 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 24525.1 |
| 工具调用 / 成功 | 8.7 |
| 耗时(ms) / 成功 | 41933.1 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 7 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 10 |
| 声称完成但错 | 5 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 15 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 9 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| rescue_v1_search_dev_cache_0 | 1 | 5230 | 100% | 100% | 100% |
| rescue_v1_search_dev_cache_1 | 1 | 4872 | 100% | 100% | 100% |
| rescue_v1_search_dev_cache_2 | 1 | 4970 | 100% | 100% | 100% |
| rescue_v1_search_dev_cache_3 | 1 | 5169 | 100% | 100% | 100% |
| rescue_v1_search_dev_download_0 | 1 | 9832 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_1 | 1 | 6578 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_2 | 1 | 24618 | 0% | 0% | 0% |
| rescue_v1_search_dev_download_3 | 1 | 26941 | 0% | 0% | 0% |
| rescue_v1_search_dev_download_translate_0 | 1 | 28057 | 0% | 0% | 0% |
| rescue_v1_search_dev_download_translate_1 | 1 | 30437 | 0% | 0% | 0% |
| rescue_v1_search_dev_download_translate_2 | 1 | 25267 | 0% | 0% | 0% |
| rescue_v1_search_dev_download_translate_3 | 1 | 29160 | 0% | 0% | 0% |
| rescue_v1_search_dev_search_0 | 1 | 6933 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_1 | 1 | 6350 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_2 | 1 | 6337 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_3 | 1 | 6178 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_download_0 | 1 | 10389 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_download_1 | 1 | 11329 | 100% | 0% | 100% |
| rescue_v1_search_dev_search_download_2 | 1 | 33822 | 0% | 0% | 0% |
| rescue_v1_search_dev_search_download_3 | 1 | 32895 | 0% | 0% | 0% |
| rescue_v1_search_dev_translate_0 | 1 | 5852 | 100% | 0% | 0% |
| rescue_v1_search_dev_translate_1 | 1 | 23914 | 0% | 0% | 0% |
| rescue_v1_search_dev_translate_2 | 1 | 21687 | 0% | 0% | 0% |
| rescue_v1_search_dev_translate_3 | 1 | 10581 | 100% | 0% | 0% |