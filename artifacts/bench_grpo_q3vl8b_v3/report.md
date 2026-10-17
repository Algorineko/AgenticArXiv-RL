## Benchmark 对比报告
模型: ~/data/models/Qwen3-VL-8B-Instruct | 样本数: 81 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 9563 |
| LLM 时间(ms) | 9559.3 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 3.7 |
| 迭代次数 | 3 |
| Token 用量 | 6853 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 72% |
| pass^2 | n/a |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 99% |
| 严格成功率 | 72% |
| 工具调用准确率 | 84% |
| 参数准确率 | 81% |
| 指代解析准确率 | 83% |
| 阻塞终止语义准确率 | 0% |
| 假完成率 | 7% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.1 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 9570.5 |
| 工具调用 / 成功 | 2.8 |
| 耗时(ms) / 成功 | 13355.2 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 2 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 0 |
| 声称完成但错 | 22 |
| 异常终止 | 1 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 23 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 58 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 1 | 16791 | 100% | 100% | 100% |
| analyze_cv5_ref1_desc | 1 | 16915 | 100% | 100% | 100% |
| analyze_cv5_ref3_trend | 1 | 15324 | 100% | 100% | 100% |
| analyze_ro5_ref2_axes | 1 | 16801 | 100% | 100% | 100% |
| chain_ai5_dl2_tr2 | 1 | 15797 | 100% | 100% | 100% |
| chain_ai5_read_then_summary | 1 | 21212 | 100% | 0% | 0% |
| chain_cl5_dl3_tr3 | 1 | 12823 | 100% | 0% | 0% |
| chain_cv5_cache_dl_tr_cache | 1 | 19720 | 100% | 100% | 100% |
| chain_lg10_dl_tr_last | 1 | 15023 | 100% | 0% | 100% |
| chain_ro5_dl_three | 1 | 16301 | 100% | 100% | 100% |
| constraint_cache_only | 1 | 5040 | 100% | 100% | 100% |
| constraint_download_only | 1 | 5037 | 100% | 100% | 100% |
| constraint_search_no_file | 1 | 6228 | 100% | 0% | 100% |
| constraint_search_only | 1 | 6425 | 100% | 100% | 100% |
| constraint_translate_only | 1 | 5122 | 100% | 100% | 100% |
| constraint_two_downloads_only | 1 | 7557 | 100% | 100% | 100% |
| figure_ai5_ref4 | 1 | 13569 | 100% | 100% | 100% |
| figure_cv5_ref1 | 1 | 12637 | 100% | 100% | 100% |
| figure_cv5_ref3 | 1 | 12156 | 100% | 100% | 100% |
| figure_ro5_ref2 | 1 | 12871 | 100% | 100% | 100% |
| infeasible_index_out_of_range | 1 | 5225 | 100% | 0% | 0% |
| infeasible_no_session | 1 | 4765 | 100% | 0% | 0% |
| infeasible_unknown_id | 1 | 4753 | 100% | 0% | 0% |
| infeasible_unsupported_action | 1 | 17958 | 0% | 0% | 0% |
| infeasible_zero_index | 1 | 4773 | 100% | 0% | 0% |
| multi_ai3_tr1 | 1 | 9070 | 100% | 100% | 100% |
| multi_ai5_dl1_tr | 1 | 12294 | 100% | 100% | 100% |
| multi_cl5_dl2_cache | 1 | 12506 | 100% | 100% | 100% |
| multi_cr5_cache1 | 1 | 9532 | 100% | 100% | 100% |
| multi_cv3_dl1 | 1 | 9071 | 100% | 100% | 100% |
| multi_cv5_dl1_cache_tr | 1 | 15822 | 100% | 100% | 100% |
| multi_lg10_dl3 | 1 | 9461 | 100% | 100% | 100% |
| multi_ro3_dl_two | 1 | 12869 | 100% | 100% | 100% |
| opt_force_dl | 1 | 5277 | 100% | 100% | 100% |
| opt_force_tr | 1 | 4928 | 100% | 100% | 100% |
| opt_keep_dual | 1 | 5535 | 100% | 100% | 100% |
| opt_service | 1 | 5221 | 100% | 0% | 100% |
| opt_threads | 1 | 5428 | 100% | 100% | 100% |
| qa_cr5_abstract | 1 | 12720 | 100% | 100% | 100% |
| read_ai5_default | 1 | 12518 | 100% | 100% | 100% |
| read_cl5_result | 1 | 13016 | 100% | 100% | 100% |
| read_cv5_method | 1 | 13236 | 100% | 100% | 100% |
| read_lg5_conclusion | 1 | 13052 | 100% | 100% | 100% |
| ref_ctrl_id_cache | 1 | 8188 | 100% | 100% | 100% |
| ref_ctrl_id_download | 1 | 4734 | 100% | 0% | 100% |
| ref_ctrl_null_cache | 1 | 5488 | 100% | 100% | 100% |
| ref_ctrl_null_translate | 1 | 5917 | 100% | 100% | 100% |
| ref_ctrl_word_handover | 1 | 5882 | 100% | 0% | 100% |
| ref_ctrl_word_marionette | 1 | 6083 | 100% | 0% | 100% |
| ref_stress_image_restoration | 1 | 6182 | 100% | 100% | 100% |
| ref_stress_learning_state | 1 | 5904 | 100% | 0% | 100% |
| ref_stress_state_across | 1 | 13023 | 100% | 0% | 0% |
| ref_stress_uncertainty_aware | 1 | 4930 | 100% | 100% | 100% |
| search_AI_1d_3 | 1 | 12919 | 100% | 100% | 100% |
| search_AI_30d_25 | 1 | 6527 | 100% | 100% | 100% |
| search_CL_30d_10 | 1 | 6519 | 100% | 100% | 100% |
| search_CL_7d_5 | 1 | 6186 | 100% | 100% | 100% |
| search_CR_14d_10 | 1 | 6282 | 100% | 100% | 100% |
| search_CR_7d_5 | 1 | 6773 | 100% | 100% | 100% |
| search_CV_14d_3 | 1 | 5796 | 100% | 100% | 100% |
| search_CV_1d_1 | 1 | 6941 | 100% | 100% | 100% |
| search_LG_14d_1 | 1 | 6223 | 100% | 100% | 100% |
| search_LG_3d_10 | 1 | 6568 | 100% | 100% | 100% |
| search_RO_30d_25 | 1 | 6576 | 100% | 100% | 100% |
| search_RO_3d_8 | 1 | 6175 | 100% | 100% | 100% |
| search_kw_agentic_rl | 1 | 6569 | 100% | 0% | 100% |
| search_kw_llm | 1 | 6511 | 100% | 0% | 100% |
| search_kw_rag | 1 | 6588 | 100% | 0% | 100% |
| state_cache_before_dl | 1 | 7416 | 100% | 0% | 0% |
| state_cache_two | 1 | 9592 | 100% | 100% | 100% |
| state_dl_then_cache | 1 | 8586 | 100% | 100% | 100% |
| state_ref_cache_active | 1 | 5917 | 100% | 100% | 100% |
| state_ref_last_active | 1 | 5261 | 100% | 100% | 100% |
| state_ref_last_of_10 | 1 | 4767 | 100% | 100% | 100% |
| state_ref_ordinal_cn | 1 | 5351 | 100% | 100% | 100% |
| state_translate_active | 1 | 5280 | 100% | 100% | 100% |
| summary_ai5_tldr60 | 1 | 16819 | 100% | 0% | 0% |
| summary_cl5_bullet250 | 1 | 13043 | 100% | 100% | 100% |
| summary_cv5_structured120 | 1 | 16913 | 100% | 0% | 0% |
| summary_lg5_tldr120 | 1 | 17051 | 100% | 0% | 0% |
| summary_ro5_structured250 | 1 | 16741 | 100% | 0% | 0% |