## Benchmark 对比报告
模型: new-SFT-visible-16steps | 样本数: 24 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 1310.4 |
| LLM 时间(ms) | 1309 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 1.5 |
| 迭代次数 | 2 |
| Token 用量 | 4284.2 |

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
| 正常结束率(FINISH) | 100% |
| 严格成功率 | 0% |
| 工具调用准确率 | 0% |
| 参数准确率 | 31% |
| 指代解析准确率 | 31% |
| 阻塞终止语义准确率 | n/a |
| 假完成率 | 100% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.3 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | n/a |
| 工具调用 / 成功 | n/a |
| 耗时(ms) / 成功 | n/a |
| 迭代数中位·成功 | n/a |
| 迭代数中位·失败 | 2 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 0 |
| 声称完成但错 | 24 |
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
| rescue_v1_final_test_active_cache_translate_0 | 1 | 1302 | 100% | 0% | 0% |
| rescue_v1_final_test_active_cache_translate_1 | 1 | 1294 | 100% | 0% | 0% |
| rescue_v1_final_test_active_cache_translate_2 | 1 | 1287 | 100% | 0% | 0% |
| rescue_v1_final_test_active_cache_translate_3 | 1 | 1295 | 100% | 0% | 0% |
| rescue_v1_final_test_double_download_0 | 1 | 1288 | 100% | 0% | 0% |
| rescue_v1_final_test_double_download_1 | 1 | 1301 | 100% | 0% | 0% |
| rescue_v1_final_test_double_download_2 | 1 | 1290 | 100% | 0% | 0% |
| rescue_v1_final_test_double_download_3 | 1 | 1334 | 100% | 0% | 0% |
| rescue_v1_final_test_download_cache_translate_0 | 1 | 1320 | 100% | 0% | 0% |
| rescue_v1_final_test_download_cache_translate_1 | 1 | 1300 | 100% | 0% | 0% |
| rescue_v1_final_test_download_cache_translate_2 | 1 | 1301 | 100% | 0% | 0% |
| rescue_v1_final_test_download_cache_translate_3 | 1 | 1308 | 100% | 0% | 0% |
| rescue_v1_final_test_search_download_cache_0 | 1 | 1302 | 100% | 0% | 0% |
| rescue_v1_final_test_search_download_cache_1 | 1 | 1298 | 100% | 0% | 0% |
| rescue_v1_final_test_search_download_cache_2 | 1 | 1289 | 100% | 0% | 0% |
| rescue_v1_final_test_search_download_cache_3 | 1 | 1287 | 100% | 0% | 0% |
| rescue_v1_final_test_search_download_translate_0 | 1 | 1604 | 100% | 0% | 0% |
| rescue_v1_final_test_search_download_translate_1 | 1 | 1290 | 100% | 0% | 0% |
| rescue_v1_final_test_search_download_translate_2 | 1 | 1284 | 100% | 0% | 0% |
| rescue_v1_final_test_search_download_translate_3 | 1 | 1290 | 100% | 0% | 0% |
| rescue_v1_final_test_translate_cache_0 | 1 | 1300 | 100% | 0% | 0% |
| rescue_v1_final_test_translate_cache_1 | 1 | 1298 | 100% | 0% | 0% |
| rescue_v1_final_test_translate_cache_2 | 1 | 1295 | 100% | 0% | 0% |
| rescue_v1_final_test_translate_cache_3 | 1 | 1293 | 100% | 0% | 0% |