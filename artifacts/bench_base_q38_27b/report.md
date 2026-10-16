## Benchmark 对比报告
模型: ~/data/models/Qwen3.8-27B | 样本数: 81 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 160273.4 |
| LLM 时间(ms) | 160265 |
| 工具时间(ms) | 0.2 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 8.3 |
| 迭代次数 | 3.5 |
| Token 用量 | 8849.1 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 31% |
| pass^2 | n/a |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 80% |
| 严格成功率 | 31% |
| 工具调用准确率 | 36% |
| 参数准确率 | 54% |
| 指代解析准确率 | 45% |
| 阻塞终止语义准确率 | 40% |
| 假完成率 | 22% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.3 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 28671.0 |
| 工具调用 / 成功 | 8.2 |
| 耗时(ms) / 成功 | 519286.0 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 4 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 16 |
| 声称完成但错 | 40 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 56 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 25 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 1 | 30727 | 100% | 0% | 0% |
| analyze_cv5_ref1_desc | 1 | 302544 | 100% | 100% | 100% |
| analyze_cv5_ref3_trend | 1 | 66127 | 100% | 0% | 0% |
| analyze_ro5_ref2_axes | 1 | 384387 | 100% | 0% | 0% |
| chain_ai5_dl2_tr2 | 1 | 179554 | 100% | 100% | 100% |
| chain_ai5_read_then_summary | 1 | 34609 | 100% | 0% | 0% |
| chain_cl5_dl3_tr3 | 1 | 298194 | 0% | 0% | 0% |
| chain_cv5_cache_dl_tr_cache | 1 | 110211 | 100% | 0% | 0% |
| chain_lg10_dl_tr_last | 1 | 186509 | 100% | 0% | 0% |
| chain_ro5_dl_three | 1 | 238445 | 0% | 0% | 0% |
| constraint_cache_only | 1 | 98958 | 100% | 0% | 0% |
| constraint_download_only | 1 | 63016 | 100% | 100% | 100% |
| constraint_search_no_file | 1 | 81228 | 100% | 0% | 0% |
| constraint_search_only | 1 | 257285 | 100% | 0% | 0% |
| constraint_translate_only | 1 | 120160 | 100% | 100% | 100% |
| constraint_two_downloads_only | 1 | 196877 | 100% | 0% | 0% |
| figure_ai5_ref4 | 1 | 114659 | 100% | 0% | 0% |
| figure_cv5_ref1 | 1 | 262411 | 100% | 100% | 100% |
| figure_cv5_ref3 | 1 | 109910 | 100% | 0% | 0% |
| figure_ro5_ref2 | 1 | 287290 | 0% | 0% | 0% |
| infeasible_index_out_of_range | 1 | 77665 | 100% | 0% | 100% |
| infeasible_no_session | 1 | 32702 | 100% | 100% | 100% |
| infeasible_unknown_id | 1 | 163758 | 100% | 0% | 0% |
| infeasible_unsupported_action | 1 | 45873 | 100% | 0% | 100% |
| infeasible_zero_index | 1 | 181705 | 100% | 0% | 0% |
| multi_ai3_tr1 | 1 | 291896 | 0% | 0% | 0% |
| multi_ai5_dl1_tr | 1 | 265758 | 100% | 100% | 100% |
| multi_cl5_dl2_cache | 1 | 96867 | 100% | 0% | 0% |
| multi_cr5_cache1 | 1 | 272611 | 0% | 0% | 0% |
| multi_cv3_dl1 | 1 | 123958 | 100% | 0% | 0% |
| multi_cv5_dl1_cache_tr | 1 | 252466 | 0% | 0% | 0% |
| multi_lg10_dl3 | 1 | 283795 | 0% | 0% | 0% |
| multi_ro3_dl_two | 1 | 284746 | 0% | 0% | 0% |
| opt_force_dl | 1 | 341338 | 0% | 0% | 0% |
| opt_force_tr | 1 | 81943 | 100% | 0% | 100% |
| opt_keep_dual | 1 | 253345 | 100% | 0% | 0% |
| opt_service | 1 | 185952 | 0% | 0% | 0% |
| opt_threads | 1 | 247994 | 100% | 0% | 0% |
| qa_cr5_abstract | 1 | 318360 | 0% | 0% | 0% |
| read_ai5_default | 1 | 234286 | 100% | 0% | 100% |
| read_cl5_result | 1 | 179095 | 100% | 0% | 0% |
| read_cv5_method | 1 | 375268 | 0% | 0% | 0% |
| read_lg5_conclusion | 1 | 184500 | 100% | 100% | 100% |
| ref_ctrl_id_cache | 1 | 56998 | 100% | 100% | 100% |
| ref_ctrl_id_download | 1 | 131408 | 100% | 0% | 0% |
| ref_ctrl_null_cache | 1 | 97472 | 100% | 0% | 0% |
| ref_ctrl_null_translate | 1 | 191899 | 100% | 0% | 0% |
| ref_ctrl_word_handover | 1 | 173641 | 100% | 0% | 0% |
| ref_ctrl_word_marionette | 1 | 222948 | 0% | 0% | 0% |
| ref_stress_image_restoration | 1 | 88815 | 100% | 100% | 100% |
| ref_stress_learning_state | 1 | 309897 | 100% | 0% | 0% |
| ref_stress_state_across | 1 | 56273 | 100% | 100% | 100% |
| ref_stress_uncertainty_aware | 1 | 271575 | 100% | 0% | 0% |
| search_AI_1d_3 | 1 | 76702 | 100% | 100% | 100% |
| search_AI_30d_25 | 1 | 42234 | 100% | 100% | 100% |
| search_CL_30d_10 | 1 | 83100 | 100% | 100% | 100% |
| search_CL_7d_5 | 1 | 54504 | 100% | 100% | 100% |
| search_CR_14d_10 | 1 | 262105 | 100% | 0% | 0% |
| search_CR_7d_5 | 1 | 178058 | 100% | 0% | 0% |
| search_CV_14d_3 | 1 | 74380 | 100% | 100% | 100% |
| search_CV_1d_1 | 1 | 98433 | 100% | 100% | 100% |
| search_LG_14d_1 | 1 | 121688 | 100% | 0% | 0% |
| search_LG_3d_10 | 1 | 65588 | 100% | 100% | 100% |
| search_RO_30d_25 | 1 | 204800 | 0% | 0% | 0% |
| search_RO_3d_8 | 1 | 35496 | 100% | 100% | 100% |
| search_kw_agentic_rl | 1 | 59275 | 100% | 100% | 100% |
| search_kw_llm | 1 | 48768 | 100% | 100% | 100% |
| search_kw_rag | 1 | 50892 | 100% | 100% | 100% |
| state_cache_before_dl | 1 | 107994 | 100% | 0% | 0% |
| state_cache_two | 1 | 152323 | 0% | 0% | 0% |
| state_dl_then_cache | 1 | 100297 | 100% | 100% | 100% |
| state_ref_cache_active | 1 | 84594 | 100% | 0% | 0% |
| state_ref_last_active | 1 | 234710 | 100% | 0% | 0% |
| state_ref_last_of_10 | 1 | 55274 | 100% | 100% | 100% |
| state_ref_ordinal_cn | 1 | 45783 | 100% | 100% | 100% |
| state_translate_active | 1 | 140538 | 100% | 0% | 0% |
| summary_ai5_tldr60 | 1 | 213131 | 0% | 0% | 0% |
| summary_cl5_bullet250 | 1 | 102378 | 100% | 0% | 0% |
| summary_cv5_structured120 | 1 | 179299 | 100% | 0% | 0% |
| summary_lg5_tldr120 | 1 | 141670 | 100% | 0% | 0% |
| summary_ro5_structured250 | 1 | 168227 | 100% | 0% | 0% |