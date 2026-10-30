## Benchmark 对比报告
模型: Qwen2.5-1.5B-Instruct | 样本数: 12 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 2674.5 |
| LLM 时间(ms) | 2672.2 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 2.3 |
| 迭代次数 | 3.1 |
| Token 用量 | 7266.8 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 25% |
| pass^2 | n/a |
| pass^3 | n/a |

> 12 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 92% |
| 严格成功率 | 25% |
| 工具调用准确率 | 25% |
| 参数准确率 | 25% |
| 指代解析准确率 | 25% |
| 阻塞终止语义准确率 | n/a |
| 假完成率 | 50% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.2 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 29067.3 |
| 工具调用 / 成功 | 8.3 |
| 耗时(ms) / 成功 | 10698 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 3 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 0 |
| 声称完成但错 | 8 |
| 异常终止 | 1 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 9 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 3 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| receipt_v1_promotion_force_download_4_downloaded | 1 | 1290 | 100% | 100% | 100% |
| receipt_v1_promotion_force_download_4_listed | 1 | 1511 | 100% | 0% | 0% |
| receipt_v1_promotion_force_download_5_downloaded | 1 | 1318 | 100% | 100% | 100% |
| receipt_v1_promotion_force_download_5_listed | 1 | 1988 | 0% | 0% | 0% |
| receipt_v1_promotion_force_download_6_downloaded | 1 | 4139 | 100% | 0% | 0% |
| receipt_v1_promotion_force_download_6_listed | 1 | 1484 | 100% | 0% | 0% |
| receipt_v1_promotion_translate_4_downloaded | 1 | 1588 | 100% | 100% | 100% |
| receipt_v1_promotion_translate_4_listed | 1 | 7122 | 100% | 0% | 0% |
| receipt_v1_promotion_translate_5_downloaded | 1 | 1532 | 100% | 0% | 0% |
| receipt_v1_promotion_translate_5_listed | 1 | 5191 | 100% | 0% | 0% |
| receipt_v1_promotion_translate_6_downloaded | 1 | 2396 | 100% | 0% | 0% |
| receipt_v1_promotion_translate_6_listed | 1 | 2535 | 100% | 0% | 0% |