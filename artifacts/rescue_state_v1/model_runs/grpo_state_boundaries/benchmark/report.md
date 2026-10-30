## Benchmark 对比报告
模型: published-SFT-GRPO-1.5B | 样本数: 12 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 3993.7 |
| LLM 时间(ms) | 3991.3 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 2.3 |
| 迭代次数 | 1.7 |
| Token 用量 | 3596.5 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 33% |
| pass^2 | n/a |
| pass^3 | n/a |

> 12 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 100% |
| 严格成功率 | 33% |
| 工具调用准确率 | 67% |
| 参数准确率 | 50% |
| 指代解析准确率 | 67% |
| 阻塞终止语义准确率 | n/a |
| 假完成率 | 33% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 10789.5 |
| 工具调用 / 成功 | 2 |
| 耗时(ms) / 成功 | 11981 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 1.5 |

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
| ceiling | > 80% | 4 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| state_v1_force_download_1_downloaded | 1 | 5055 | 100% | 0% | 100% |
| state_v1_force_download_1_listed | 1 | 5020 | 100% | 0% | 100% |
| state_v1_force_download_2_downloaded | 1 | 2038 | 100% | 0% | 0% |
| state_v1_force_download_2_listed | 1 | 4712 | 100% | 0% | 100% |
| state_v1_force_download_3_downloaded | 1 | 2109 | 100% | 0% | 0% |
| state_v1_force_download_3_listed | 1 | 4919 | 100% | 0% | 100% |
| state_v1_translate_1_downloaded | 1 | 4719 | 100% | 100% | 100% |
| state_v1_translate_1_listed | 1 | 5463 | 100% | 100% | 100% |
| state_v1_translate_2_downloaded | 1 | 2251 | 100% | 0% | 0% |
| state_v1_translate_2_listed | 1 | 4606 | 100% | 100% | 100% |
| state_v1_translate_3_downloaded | 1 | 2097 | 100% | 0% | 0% |
| state_v1_translate_3_listed | 1 | 4935 | 100% | 100% | 100% |