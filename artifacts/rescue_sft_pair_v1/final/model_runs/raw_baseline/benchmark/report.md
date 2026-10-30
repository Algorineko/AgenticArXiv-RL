## Benchmark 对比报告
模型: Qwen2.5-1.5B-Instruct | 样本数: 24 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 3916.9 |
| LLM 时间(ms) | 3912.8 |
| 工具时间(ms) | 0.2 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 3.9 |
| 迭代次数 | 4.3 |
| Token 用量 | 9940.6 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 33% |
| pass^2 | n/a |
| pass^3 | n/a |

> 24 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 67% |
| 严格成功率 | 33% |
| 工具调用准确率 | 38% |
| 参数准确率 | 49% |
| 指代解析准确率 | 45% |
| 阻塞终止语义准确率 | n/a |
| 假完成率 | 21% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.1 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 29821.8 |
| 工具调用 / 成功 | 10 |
| 耗时(ms) / 成功 | 11750.8 |
| 迭代数中位·成功 | 3.5 |
| 迭代数中位·失败 | 4.5 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 6 |
| 声称完成但错 | 8 |
| 异常终止 | 2 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 16 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 8 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| rescue_v1_final_test_active_cache_translate_0 | 1 | 2305 | 100% | 100% | 100% |
| rescue_v1_final_test_active_cache_translate_1 | 1 | 2318 | 100% | 100% | 100% |
| rescue_v1_final_test_active_cache_translate_2 | 1 | 2337 | 100% | 100% | 100% |
| rescue_v1_final_test_active_cache_translate_3 | 1 | 2304 | 100% | 100% | 100% |
| rescue_v1_final_test_double_download_0 | 1 | 2364 | 100% | 0% | 100% |
| rescue_v1_final_test_double_download_1 | 1 | 1394 | 100% | 0% | 0% |
| rescue_v1_final_test_double_download_2 | 1 | 7150 | 0% | 0% | 0% |
| rescue_v1_final_test_double_download_3 | 1 | 6227 | 0% | 0% | 0% |
| rescue_v1_final_test_download_cache_translate_0 | 1 | 4261 | 100% | 0% | 0% |
| rescue_v1_final_test_download_cache_translate_1 | 1 | 7254 | 0% | 0% | 0% |
| rescue_v1_final_test_download_cache_translate_2 | 1 | 3392 | 0% | 0% | 0% |
| rescue_v1_final_test_download_cache_translate_3 | 1 | 6469 | 0% | 0% | 0% |
| rescue_v1_final_test_search_download_cache_0 | 1 | 2467 | 100% | 0% | 0% |
| rescue_v1_final_test_search_download_cache_1 | 1 | 2635 | 100% | 0% | 0% |
| rescue_v1_final_test_search_download_cache_2 | 1 | 3455 | 100% | 100% | 100% |
| rescue_v1_final_test_search_download_cache_3 | 1 | 3610 | 100% | 100% | 100% |
| rescue_v1_final_test_search_download_translate_0 | 1 | 3834 | 100% | 100% | 100% |
| rescue_v1_final_test_search_download_translate_1 | 1 | 3490 | 100% | 100% | 100% |
| rescue_v1_final_test_search_download_translate_2 | 1 | 6797 | 0% | 0% | 0% |
| rescue_v1_final_test_search_download_translate_3 | 1 | 2972 | 100% | 0% | 0% |
| rescue_v1_final_test_translate_cache_0 | 1 | 2447 | 100% | 0% | 0% |
| rescue_v1_final_test_translate_cache_1 | 1 | 2594 | 0% | 0% | 0% |
| rescue_v1_final_test_translate_cache_2 | 1 | 6894 | 0% | 0% | 0% |
| rescue_v1_final_test_translate_cache_3 | 1 | 5036 | 100% | 0% | 0% |