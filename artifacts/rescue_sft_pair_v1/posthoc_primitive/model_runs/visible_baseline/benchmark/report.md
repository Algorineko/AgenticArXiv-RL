## Benchmark 对比报告
模型: new-SFT-visible-16steps | 样本数: 24 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 1341.6 |
| LLM 时间(ms) | 1339.9 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 1.7 |
| 迭代次数 | 2 |
| Token 用量 | 4285.7 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 67% |
| pass^2 | n/a |
| pass^3 | n/a |

> 24 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 100% |
| 严格成功率 | 67% |
| 工具调用准确率 | 67% |
| 参数准确率 | 75% |
| 指代解析准确率 | 70% |
| 阻塞终止语义准确率 | n/a |
| 假完成率 | 33% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.2 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 6428.5 |
| 工具调用 / 成功 | 1.5 |
| 耗时(ms) / 成功 | 2012.4 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 2 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 0 |
| 声称完成但错 | 8 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 8 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 16 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| rescue_v1_search_dev_cache_0 | 1 | 1305 | 100% | 100% | 100% |
| rescue_v1_search_dev_cache_1 | 1 | 1308 | 100% | 100% | 100% |
| rescue_v1_search_dev_cache_2 | 1 | 1311 | 100% | 100% | 100% |
| rescue_v1_search_dev_cache_3 | 1 | 1307 | 100% | 100% | 100% |
| rescue_v1_search_dev_download_0 | 1 | 1302 | 100% | 100% | 100% |
| rescue_v1_search_dev_download_1 | 1 | 1311 | 100% | 100% | 100% |
| rescue_v1_search_dev_download_2 | 1 | 1301 | 100% | 100% | 100% |
| rescue_v1_search_dev_download_3 | 1 | 1326 | 100% | 100% | 100% |
| rescue_v1_search_dev_download_translate_0 | 1 | 1286 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_translate_1 | 1 | 1288 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_translate_2 | 1 | 1292 | 100% | 0% | 0% |
| rescue_v1_search_dev_download_translate_3 | 1 | 1303 | 100% | 0% | 0% |
| rescue_v1_search_dev_search_0 | 1 | 1746 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_1 | 1 | 1467 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_2 | 1 | 1469 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_3 | 1 | 1460 | 100% | 100% | 100% |
| rescue_v1_search_dev_search_download_0 | 1 | 1296 | 100% | 0% | 0% |
| rescue_v1_search_dev_search_download_1 | 1 | 1294 | 100% | 0% | 0% |
| rescue_v1_search_dev_search_download_2 | 1 | 1296 | 100% | 0% | 0% |
| rescue_v1_search_dev_search_download_3 | 1 | 1287 | 100% | 0% | 0% |
| rescue_v1_search_dev_translate_0 | 1 | 1338 | 100% | 100% | 100% |
| rescue_v1_search_dev_translate_1 | 1 | 1297 | 100% | 100% | 100% |
| rescue_v1_search_dev_translate_2 | 1 | 1305 | 100% | 100% | 100% |
| rescue_v1_search_dev_translate_3 | 1 | 1304 | 100% | 100% | 100% |