## Benchmark 对比报告
模型: [LOCAL_PATH_WITHHELD] | 样本数: 81 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 13267.7 |
| LLM 时间(ms) | 13263.9 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 3.8 |
| 迭代次数 | 3.1 |
| Token 用量 | 7015.7 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 73% |
| pass^2 | n/a |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 95% |
| 严格成功率 | 73% |
| 工具调用准确率 | 79% |
| 参数准确率 | 80% |
| 指代解析准确率 | 78% |
| 阻塞终止语义准确率 | 40% |
| 假完成率 | 10% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.2 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 9631.7 |
| 工具调用 / 成功 | 2.8 |
| 耗时(ms) / 成功 | 18215.0 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 3 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 4 |
| 声称完成但错 | 18 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 22 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 59 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 1 | 23422 | 100% | 100% | 100% |
| analyze_cv5_ref1_desc | 1 | 20261 | 100% | 100% | 100% |
| analyze_cv5_ref3_trend | 1 | 20037 | 100% | 100% | 100% |
| analyze_ro5_ref2_axes | 1 | 19958 | 100% | 100% | 100% |
| chain_ai5_dl2_tr2 | 1 | 20378 | 100% | 100% | 100% |
| chain_ai5_read_then_summary | 1 | 28946 | 100% | 100% | 100% |
| chain_cl5_dl3_tr3 | 1 | 19863 | 100% | 100% | 100% |
| chain_cv5_cache_dl_tr_cache | 1 | 28370 | 100% | 100% | 100% |
| chain_lg10_dl_tr_last | 1 | 15353 | 100% | 0% | 0% |
| chain_ro5_dl_three | 1 | 26623 | 100% | 100% | 100% |
| constraint_cache_only | 1 | 5482 | 100% | 100% | 100% |
| constraint_download_only | 1 | 5844 | 100% | 100% | 100% |
| constraint_search_no_file | 1 | 9456 | 100% | 0% | 100% |
| constraint_search_only | 1 | 21199 | 100% | 0% | 0% |
| constraint_translate_only | 1 | 11106 | 100% | 100% | 100% |
| constraint_two_downloads_only | 1 | 8905 | 100% | 100% | 100% |
| figure_ai5_ref4 | 1 | 3297 | 100% | 0% | 0% |
| figure_cv5_ref1 | 1 | 16016 | 100% | 100% | 100% |
| figure_cv5_ref3 | 1 | 8823 | 100% | 0% | 0% |
| figure_ro5_ref2 | 1 | 15608 | 100% | 100% | 100% |
| infeasible_index_out_of_range | 1 | 3685 | 100% | 0% | 100% |
| infeasible_no_session | 1 | 41422 | 0% | 0% | 0% |
| infeasible_unknown_id | 1 | 11764 | 100% | 0% | 0% |
| infeasible_unsupported_action | 1 | 3417 | 100% | 100% | 100% |
| infeasible_zero_index | 1 | 26543 | 0% | 0% | 0% |
| multi_ai3_tr1 | 1 | 17442 | 100% | 0% | 0% |
| multi_ai5_dl1_tr | 1 | 28835 | 100% | 100% | 100% |
| multi_cl5_dl2_cache | 1 | 15222 | 100% | 100% | 100% |
| multi_cr5_cache1 | 1 | 13953 | 100% | 0% | 100% |
| multi_cv3_dl1 | 1 | 11634 | 100% | 100% | 100% |
| multi_cv5_dl1_cache_tr | 1 | 22385 | 100% | 100% | 100% |
| multi_lg10_dl3 | 1 | 13921 | 100% | 100% | 100% |
| multi_ro3_dl_two | 1 | 22652 | 100% | 100% | 100% |
| opt_force_dl | 1 | 5908 | 100% | 100% | 100% |
| opt_force_tr | 1 | 6505 | 100% | 100% | 100% |
| opt_keep_dual | 1 | 6673 | 100% | 100% | 100% |
| opt_service | 1 | 6704 | 100% | 100% | 100% |
| opt_threads | 1 | 6198 | 100% | 100% | 100% |
| qa_cr5_abstract | 1 | 16229 | 100% | 0% | 100% |
| read_ai5_default | 1 | 17636 | 100% | 100% | 100% |
| read_cl5_result | 1 | 16142 | 100% | 100% | 100% |
| read_cv5_method | 1 | 18784 | 100% | 100% | 100% |
| read_lg5_conclusion | 1 | 14640 | 100% | 100% | 100% |
| ref_ctrl_id_cache | 1 | 6407 | 100% | 100% | 100% |
| ref_ctrl_id_download | 1 | 8956 | 100% | 100% | 100% |
| ref_ctrl_null_cache | 1 | 6510 | 100% | 100% | 100% |
| ref_ctrl_null_translate | 1 | 11029 | 100% | 100% | 100% |
| ref_ctrl_word_handover | 1 | 8407 | 100% | 0% | 0% |
| ref_ctrl_word_marionette | 1 | 21919 | 0% | 0% | 0% |
| ref_stress_image_restoration | 1 | 8356 | 100% | 0% | 0% |
| ref_stress_learning_state | 1 | 12710 | 100% | 0% | 0% |
| ref_stress_state_across | 1 | 20098 | 100% | 0% | 0% |
| ref_stress_uncertainty_aware | 1 | 14135 | 100% | 0% | 0% |
| search_AI_1d_3 | 1 | 13889 | 100% | 100% | 100% |
| search_AI_30d_25 | 1 | 7513 | 100% | 100% | 100% |
| search_CL_30d_10 | 1 | 6815 | 100% | 100% | 100% |
| search_CL_7d_5 | 1 | 6057 | 100% | 100% | 100% |
| search_CR_14d_10 | 1 | 6026 | 100% | 100% | 100% |
| search_CR_7d_5 | 1 | 7689 | 100% | 100% | 100% |
| search_CV_14d_3 | 1 | 6205 | 100% | 100% | 100% |
| search_CV_1d_1 | 1 | 7430 | 100% | 100% | 100% |
| search_LG_14d_1 | 1 | 8172 | 100% | 100% | 100% |
| search_LG_3d_10 | 1 | 7697 | 100% | 100% | 100% |
| search_RO_30d_25 | 1 | 7912 | 100% | 100% | 100% |
| search_RO_3d_8 | 1 | 7396 | 100% | 100% | 100% |
| search_kw_agentic_rl | 1 | 8478 | 100% | 100% | 100% |
| search_kw_llm | 1 | 8163 | 100% | 100% | 100% |
| search_kw_rag | 1 | 8263 | 100% | 100% | 100% |
| state_cache_before_dl | 1 | 8977 | 100% | 100% | 100% |
| state_cache_two | 1 | 8290 | 100% | 100% | 100% |
| state_dl_then_cache | 1 | 30197 | 0% | 0% | 0% |
| state_ref_cache_active | 1 | 6597 | 100% | 100% | 100% |
| state_ref_last_active | 1 | 7482 | 100% | 100% | 100% |
| state_ref_last_of_10 | 1 | 5969 | 100% | 100% | 100% |
| state_ref_ordinal_cn | 1 | 12304 | 100% | 0% | 0% |
| state_translate_active | 1 | 7861 | 100% | 100% | 100% |
| summary_ai5_tldr60 | 1 | 3294 | 100% | 0% | 0% |
| summary_cl5_bullet250 | 1 | 16236 | 100% | 100% | 100% |
| summary_cv5_structured120 | 1 | 19004 | 100% | 100% | 100% |
| summary_lg5_tldr120 | 1 | 16518 | 100% | 0% | 100% |
| summary_ro5_structured250 | 1 | 18484 | 100% | 100% | 100% |