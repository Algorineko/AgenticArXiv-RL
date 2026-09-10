## Benchmark 对比报告
模型: /mnt/disk4/gaojiayu/models/Qwen2.5-1.5B-Instruct | 样本数: 108 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 2809.9 |
| LLM 时间(ms) | 2807 |
| 工具时间(ms) | 0 |
| 框架开销(ms) | 2.9 |
| 迭代次数 | 3.9 |
| Token 用量 | 5700.1 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 15% |
| pass^2 | 9% |
| pass^3 | 8% |

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 60% |
| 严格成功率 | 15% |
| 工具调用准确率 | 22% |
| 参数准确率 | 42% |
| 指代解析准确率 | n/a |
| 假完成率 | 27% |
| 平均解析失败 | 0 |
| 平均工具失败 | 1.6 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 38475.9 |
| 工具调用 / 成功 | 19.8 |
| 耗时(ms) / 成功 | 18966.8 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 4 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 43 |
| 声称完成但错 | 49 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 27 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 6 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 3 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| chain_ai5_dl2_tr2 | 3 | 4320.3 | 0% | 0% | 0% |
| chain_cv5_cache_dl_tr_cache | 3 | 4560.7 | 100% | 33% | 33% |
| chain_lg10_dl_tr_last | 3 | 3135.3 | 67% | 0% | 0% |
| chain_ro5_dl_three | 3 | 2394 | 100% | 0% | 0% |
| constraint_cache_only | 3 | 1093 | 100% | 100% | 100% |
| constraint_download_only | 3 | 2048.3 | 100% | 0% | 0% |
| constraint_search_only | 3 | 5311.3 | 0% | 0% | 0% |
| constraint_two_downloads_only | 3 | 1983.3 | 100% | 0% | 0% |
| infeasible_no_session | 3 | 1457 | 100% | 0% | 0% |
| infeasible_unknown_id | 3 | 4382.3 | 0% | 0% | 0% |
| infeasible_zero_index | 3 | 4209.3 | 0% | 0% | 0% |
| multi_cr5_cache1 | 3 | 2077.3 | 100% | 33% | 100% |
| multi_lg10_dl3 | 3 | 4569.7 | 0% | 0% | 0% |
| opt_force_tr | 3 | 1257.7 | 100% | 100% | 100% |
| opt_keep_dual | 3 | 1203.7 | 100% | 0% | 33% |
| opt_threads | 3 | 1125 | 100% | 0% | 0% |
| ref_ctrl_null_translate | 3 | 1156.3 | 100% | 0% | 0% |
| ref_ctrl_word_handover | 3 | 4350 | 0% | 0% | 0% |
| ref_ctrl_word_marionette | 3 | 3756 | 33% | 0% | 0% |
| ref_stress_image_restoration | 3 | 3976.7 | 0% | 0% | 0% |
| ref_stress_state_across | 3 | 1175.3 | 100% | 0% | 100% |
| ref_stress_uncertainty_aware | 3 | 2484.7 | 67% | 0% | 67% |
| search_AI_30d_25 | 3 | 4241.3 | 0% | 0% | 0% |
| search_CL_30d_10 | 3 | 3505.7 | 33% | 33% | 33% |
| search_CL_7d_5 | 3 | 3169.3 | 33% | 33% | 33% |
| search_CR_7d_5 | 3 | 4087.3 | 0% | 0% | 0% |
| search_CV_14d_3 | 3 | 3526.3 | 67% | 0% | 0% |
| search_LG_14d_1 | 3 | 3160.3 | 67% | 0% | 0% |
| search_LG_3d_10 | 3 | 2661 | 67% | 67% | 67% |
| search_RO_3d_8 | 3 | 3424.3 | 33% | 33% | 33% |
| search_kw_llm | 3 | 4395.3 | 0% | 0% | 0% |
| state_cache_before_dl | 3 | 1062.3 | 100% | 0% | 0% |
| state_ref_last_active | 3 | 2051.3 | 100% | 0% | 0% |
| state_ref_last_of_10 | 3 | 1142.3 | 100% | 100% | 100% |
| state_ref_ordinal_cn | 3 | 1624.7 | 100% | 0% | 0% |
| state_translate_active | 3 | 1077.3 | 100% | 0% | 0% |