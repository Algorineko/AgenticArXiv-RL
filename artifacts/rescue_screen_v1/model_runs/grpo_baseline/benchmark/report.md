## Benchmark 对比报告
模型: Algorineko/AgenticArXiv-RL-Qwen2.5-1.5B-GRPO | 样本数: 24 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 5033.4 |
| LLM 时间(ms) | 5031 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 2.5 |
| 迭代次数 | 1.8 |
| Token 用量 | 3955.3 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 50% |
| pass^2 | n/a |
| pass^3 | n/a |

> 24 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 100% |
| 严格成功率 | 50% |
| 工具调用准确率 | 50% |
| 参数准确率 | 50% |
| 指代解析准确率 | 40% |
| 阻塞终止语义准确率 | n/a |
| 假完成率 | 50% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 7910.7 |
| 工具调用 / 成功 | 1.7 |
| 耗时(ms) / 成功 | 10066.8 |
| 迭代数中位·成功 | 3 |
| 迭代数中位·失败 | 1 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 0 |
| 声称完成但错 | 12 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 12 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 12 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| rescue_v1_search_dev_cache_0 | 1 | 2304 | 100% | 0% | 0% |
| rescue_v1_search_dev_cache_1 | 1 | 2158 | 100% | 0% | 0% |
| rescue_v1_search_dev_cache_2 | 1 | 2202 | 100% | 0% | 0% |
| rescue_v1_search_dev_cache_3 | 1 | 2227 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_0 | 1 | 2611 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_1 | 1 | 2533 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_2 | 1 | 2558 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_3 | 1 | 2465 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_translate_0 | 1 | 8046 | 100% | 100% | 100% |
| rescue_v1_search_dev_download_translate_1 | 1 | 8101 | 100% | 100% | 100% |
| rescue_v1_search_dev_download_translate_2 | 1 | 8219 | 100% | 100% | 100% |
| rescue_v1_search_dev_download_translate_3 | 1 | 9275 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_0 | 1 | 6326 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_1 | 1 | 5723 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_2 | 1 | 5837 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_3 | 1 | 5983 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_download_0 | 1 | 9279 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_download_1 | 1 | 8862 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_download_2 | 1 | 8615 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_download_3 | 1 | 8830 | 100% | 100% | 100% |
| rescue_v1_search_dev_translate_0 | 1 | 2130 | 100% | 0% | 0% |
| rescue_v1_search_dev_translate_1 | 1 | 2200 | 100% | 0% | 0% |
| rescue_v1_search_dev_translate_2 | 1 | 2076 | 100% | 0% | 0% |
| rescue_v1_search_dev_translate_3 | 1 | 2242 | 100% | 0% | 0% |