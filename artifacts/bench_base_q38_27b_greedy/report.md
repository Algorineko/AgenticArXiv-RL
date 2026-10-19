## Benchmark 对比报告
模型: ~/data/models/Qwen3.8-27B | 样本数: 81 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 168615.8 |
| LLM 时间(ms) | 168606.7 |
| 工具时间(ms) | 0.1 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 9 |
| 迭代次数 | 3.6 |
| Token 用量 | 9111.6 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 25% |
| pass^2 | n/a |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 80% |
| 严格成功率 | 25% |
| 工具调用准确率 | 32% |
| 参数准确率 | 48% |
| 指代解析准确率 | 36% |
| 阻塞终止语义准确率 | 40% |
| 假完成率 | 22% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.2 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 36901.8 |
| 工具调用 / 成功 | 10.4 |
| 耗时(ms) / 成功 | 682893.8 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 4 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 16 |
| 声称完成但错 | 45 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 61 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 20 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 1 | 326436 | 0% | 0% | 0% |
| analyze_cv5_ref1_desc | 1 | 256567 | 100% | 0% | 0% |
| analyze_cv5_ref3_trend | 1 | 403342 | 0% | 0% | 0% |
| analyze_ro5_ref2_axes | 1 | 105194 | 100% | 0% | 0% |
| chain_ai5_dl2_tr2 | 1 | 154814 | 100% | 0% | 0% |
| chain_ai5_read_then_summary | 1 | 275053 | 0% | 0% | 0% |
| chain_cl5_dl3_tr3 | 1 | 403541 | 0% | 0% | 0% |
| chain_cv5_cache_dl_tr_cache | 1 | 119527 | 100% | 0% | 0% |
| chain_lg10_dl_tr_last | 1 | 359437 | 100% | 0% | 0% |
| chain_ro5_dl_three | 1 | 246754 | 0% | 0% | 0% |
| constraint_cache_only | 1 | 51684 | 100% | 100% | 100% |
| constraint_download_only | 1 | 72654 | 100% | 100% | 100% |
| constraint_search_no_file | 1 | 67980 | 100% | 100% | 100% |
| constraint_search_only | 1 | 226004 | 100% | 0% | 0% |
| constraint_translate_only | 1 | 39445 | 100% | 0% | 0% |
| constraint_two_downloads_only | 1 | 177507 | 100% | 0% | 0% |
| figure_ai5_ref4 | 1 | 270288 | 100% | 0% | 0% |
| figure_cv5_ref1 | 1 | 182474 | 100% | 100% | 100% |
| figure_cv5_ref3 | 1 | 274597 | 100% | 0% | 100% |
| figure_ro5_ref2 | 1 | 69936 | 100% | 0% | 0% |
| infeasible_index_out_of_range | 1 | 95184 | 100% | 0% | 100% |
| infeasible_no_session | 1 | 30934 | 100% | 100% | 100% |
| infeasible_unknown_id | 1 | 153504 | 100% | 0% | 0% |
| infeasible_unsupported_action | 1 | 56213 | 100% | 100% | 100% |
| infeasible_zero_index | 1 | 265928 | 0% | 0% | 0% |
| multi_ai3_tr1 | 1 | 237154 | 100% | 0% | 0% |
| multi_ai5_dl1_tr | 1 | 160107 | 100% | 100% | 100% |
| multi_cl5_dl2_cache | 1 | 155306 | 0% | 0% | 0% |
| multi_cr5_cache1 | 1 | 208776 | 100% | 0% | 0% |
| multi_cv3_dl1 | 1 | 148458 | 100% | 0% | 0% |
| multi_cv5_dl1_cache_tr | 1 | 173214 | 100% | 0% | 0% |
| multi_lg10_dl3 | 1 | 189367 | 100% | 0% | 100% |
| multi_ro3_dl_two | 1 | 98602 | 100% | 0% | 0% |
| opt_force_dl | 1 | 54292 | 100% | 100% | 100% |
| opt_force_tr | 1 | 99133 | 100% | 0% | 100% |
| opt_keep_dual | 1 | 122899 | 100% | 0% | 0% |
| opt_service | 1 | 67255 | 100% | 0% | 0% |
| opt_threads | 1 | 247628 | 100% | 0% | 0% |
| qa_cr5_abstract | 1 | 93481 | 100% | 0% | 0% |
| read_ai5_default | 1 | 398329 | 0% | 0% | 0% |
| read_cl5_result | 1 | 128285 | 100% | 0% | 0% |
| read_cv5_method | 1 | 306366 | 0% | 0% | 0% |
| read_lg5_conclusion | 1 | 342727 | 0% | 0% | 0% |
| ref_ctrl_id_cache | 1 | 57305 | 100% | 100% | 100% |
| ref_ctrl_id_download | 1 | 134910 | 100% | 0% | 0% |
| ref_ctrl_null_cache | 1 | 61286 | 100% | 100% | 100% |
| ref_ctrl_null_translate | 1 | 204085 | 0% | 0% | 0% |
| ref_ctrl_word_handover | 1 | 492494 | 100% | 0% | 0% |
| ref_ctrl_word_marionette | 1 | 141913 | 100% | 0% | 100% |
| ref_stress_image_restoration | 1 | 273425 | 100% | 0% | 0% |
| ref_stress_learning_state | 1 | 312888 | 100% | 0% | 0% |
| ref_stress_state_across | 1 | 59621 | 100% | 100% | 100% |
| ref_stress_uncertainty_aware | 1 | 371168 | 0% | 0% | 0% |
| search_AI_1d_3 | 1 | 176025 | 100% | 0% | 0% |
| search_AI_30d_25 | 1 | 83155 | 100% | 0% | 0% |
| search_CL_30d_10 | 1 | 69687 | 100% | 100% | 100% |
| search_CL_7d_5 | 1 | 64959 | 100% | 0% | 100% |
| search_CR_14d_10 | 1 | 234489 | 100% | 0% | 0% |
| search_CR_7d_5 | 1 | 244282 | 0% | 0% | 0% |
| search_CV_14d_3 | 1 | 260037 | 100% | 0% | 0% |
| search_CV_1d_1 | 1 | 100433 | 100% | 0% | 0% |
| search_LG_14d_1 | 1 | 98982 | 100% | 100% | 100% |
| search_LG_3d_10 | 1 | 131392 | 100% | 100% | 100% |
| search_RO_30d_25 | 1 | 88539 | 100% | 0% | 0% |
| search_RO_3d_8 | 1 | 52522 | 100% | 100% | 100% |
| search_kw_agentic_rl | 1 | 65061 | 100% | 0% | 0% |
| search_kw_llm | 1 | 56094 | 100% | 100% | 100% |
| search_kw_rag | 1 | 55004 | 100% | 100% | 100% |
| state_cache_before_dl | 1 | 60022 | 100% | 0% | 0% |
| state_cache_two | 1 | 83218 | 100% | 0% | 0% |
| state_dl_then_cache | 1 | 116217 | 100% | 100% | 100% |
| state_ref_cache_active | 1 | 99422 | 100% | 0% | 0% |
| state_ref_last_active | 1 | 112151 | 100% | 0% | 0% |
| state_ref_last_of_10 | 1 | 54140 | 100% | 100% | 100% |
| state_ref_ordinal_cn | 1 | 48206 | 100% | 100% | 100% |
| state_translate_active | 1 | 204586 | 100% | 0% | 0% |
| summary_ai5_tldr60 | 1 | 310722 | 0% | 0% | 0% |
| summary_cl5_bullet250 | 1 | 216880 | 100% | 0% | 0% |
| summary_cv5_structured120 | 1 | 79932 | 100% | 0% | 0% |
| summary_lg5_tldr120 | 1 | 226574 | 0% | 0% | 0% |
| summary_ro5_structured250 | 1 | 239675 | 0% | 0% | 0% |