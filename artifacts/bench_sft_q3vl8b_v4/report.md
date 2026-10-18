## Benchmark 对比报告
模型: ~/data/models/Qwen3-VL-8B-Instruct | 样本数: 81 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 15578.4 |
| LLM 时间(ms) | 15573.5 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 4.9 |
| 迭代次数 | 3.9 |
| Token 用量 | 8824.4 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 53% |
| pass^2 | n/a |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 81% |
| 严格成功率 | 53% |
| 工具调用准确率 | 56% |
| 参数准确率 | 76% |
| 指代解析准确率 | 76% |
| 阻塞终止语义准确率 | 0% |
| 假完成率 | 4% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.3 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 16622.7 |
| 工具调用 / 成功 | 5.5 |
| 耗时(ms) / 成功 | 29345.4 |
| 迭代数中位·成功 | 3 |
| 迭代数中位·失败 | 5 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 13 |
| 声称完成但错 | 23 |
| 异常终止 | 2 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 38 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 43 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 1 | 17008 | 100% | 0% | 100% |
| analyze_cv5_ref1_desc | 1 | 29338 | 0% | 0% | 0% |
| analyze_cv5_ref3_trend | 1 | 21400 | 100% | 100% | 100% |
| analyze_ro5_ref2_axes | 1 | 29521 | 0% | 0% | 0% |
| chain_ai5_dl2_tr2 | 1 | 18087 | 100% | 100% | 100% |
| chain_ai5_read_then_summary | 1 | 18174 | 100% | 100% | 100% |
| chain_cl5_dl3_tr3 | 1 | 14519 | 100% | 0% | 0% |
| chain_cv5_cache_dl_tr_cache | 1 | 21813 | 100% | 100% | 100% |
| chain_lg10_dl_tr_last | 1 | 15669 | 100% | 100% | 100% |
| chain_ro5_dl_three | 1 | 22181 | 0% | 0% | 0% |
| constraint_cache_only | 1 | 6800 | 100% | 100% | 100% |
| constraint_download_only | 1 | 5927 | 100% | 100% | 100% |
| constraint_search_no_file | 1 | 6063 | 100% | 0% | 100% |
| constraint_search_only | 1 | 12191 | 100% | 100% | 100% |
| constraint_translate_only | 1 | 6930 | 100% | 100% | 100% |
| constraint_two_downloads_only | 1 | 12506 | 100% | 100% | 100% |
| figure_ai5_ref4 | 1 | 14885 | 100% | 100% | 100% |
| figure_cv5_ref1 | 1 | 13249 | 100% | 100% | 100% |
| figure_cv5_ref3 | 1 | 13045 | 100% | 100% | 100% |
| figure_ro5_ref2 | 1 | 13022 | 100% | 100% | 100% |
| infeasible_index_out_of_range | 1 | 17275 | 100% | 0% | 0% |
| infeasible_no_session | 1 | 16033 | 100% | 0% | 0% |
| infeasible_unknown_id | 1 | 33892 | 0% | 0% | 0% |
| infeasible_unsupported_action | 1 | 18802 | 0% | 0% | 0% |
| infeasible_zero_index | 1 | 22436 | 100% | 0% | 0% |
| multi_ai3_tr1 | 1 | 12016 | 100% | 100% | 100% |
| multi_ai5_dl1_tr | 1 | 14068 | 100% | 100% | 100% |
| multi_cl5_dl2_cache | 1 | 15096 | 100% | 100% | 100% |
| multi_cr5_cache1 | 1 | 10831 | 100% | 100% | 100% |
| multi_cv3_dl1 | 1 | 17610 | 100% | 0% | 0% |
| multi_cv5_dl1_cache_tr | 1 | 17374 | 100% | 100% | 100% |
| multi_lg10_dl3 | 1 | 11941 | 100% | 100% | 100% |
| multi_ro3_dl_two | 1 | 17236 | 100% | 0% | 0% |
| opt_force_dl | 1 | 21996 | 0% | 0% | 0% |
| opt_force_tr | 1 | 6865 | 100% | 100% | 100% |
| opt_keep_dual | 1 | 6611 | 100% | 100% | 100% |
| opt_service | 1 | 6216 | 100% | 100% | 100% |
| opt_threads | 1 | 7500 | 100% | 100% | 100% |
| qa_cr5_abstract | 1 | 20310 | 100% | 0% | 0% |
| read_ai5_default | 1 | 12969 | 100% | 100% | 100% |
| read_cl5_result | 1 | 18121 | 100% | 0% | 0% |
| read_cv5_method | 1 | 13595 | 100% | 100% | 100% |
| read_lg5_conclusion | 1 | 16518 | 100% | 100% | 100% |
| ref_ctrl_id_cache | 1 | 7592 | 100% | 100% | 100% |
| ref_ctrl_id_download | 1 | 69143 | 100% | 0% | 0% |
| ref_ctrl_null_cache | 1 | 5584 | 100% | 100% | 100% |
| ref_ctrl_null_translate | 1 | 7047 | 100% | 100% | 100% |
| ref_ctrl_word_handover | 1 | 17192 | 100% | 0% | 0% |
| ref_ctrl_word_marionette | 1 | 9118 | 100% | 0% | 0% |
| ref_stress_image_restoration | 1 | 12480 | 100% | 0% | 0% |
| ref_stress_learning_state | 1 | 20132 | 100% | 0% | 0% |
| ref_stress_state_across | 1 | 17073 | 100% | 0% | 0% |
| ref_stress_uncertainty_aware | 1 | 8872 | 100% | 0% | 0% |
| search_AI_1d_3 | 1 | 32910 | 0% | 0% | 0% |
| search_AI_30d_25 | 1 | 12745 | 100% | 0% | 0% |
| search_CL_30d_10 | 1 | 6847 | 100% | 100% | 100% |
| search_CL_7d_5 | 1 | 13177 | 100% | 0% | 0% |
| search_CR_14d_10 | 1 | 23017 | 0% | 0% | 0% |
| search_CR_7d_5 | 1 | 20717 | 0% | 0% | 0% |
| search_CV_14d_3 | 1 | 12327 | 100% | 0% | 0% |
| search_CV_1d_1 | 1 | 21348 | 0% | 0% | 0% |
| search_LG_14d_1 | 1 | 20400 | 0% | 0% | 0% |
| search_LG_3d_10 | 1 | 29344 | 0% | 0% | 0% |
| search_RO_30d_25 | 1 | 20197 | 0% | 0% | 0% |
| search_RO_3d_8 | 1 | 5689 | 100% | 100% | 100% |
| search_kw_agentic_rl | 1 | 6114 | 100% | 100% | 100% |
| search_kw_llm | 1 | 7314 | 100% | 100% | 100% |
| search_kw_rag | 1 | 22680 | 0% | 0% | 0% |
| state_cache_before_dl | 1 | 10006 | 100% | 100% | 100% |
| state_cache_two | 1 | 20563 | 0% | 0% | 0% |
| state_dl_then_cache | 1 | 16035 | 100% | 0% | 0% |
| state_ref_cache_active | 1 | 6655 | 100% | 100% | 100% |
| state_ref_last_active | 1 | 7955 | 100% | 100% | 100% |
| state_ref_last_of_10 | 1 | 6655 | 100% | 100% | 100% |
| state_ref_ordinal_cn | 1 | 6301 | 100% | 100% | 100% |
| state_translate_active | 1 | 7300 | 100% | 100% | 100% |
| summary_ai5_tldr60 | 1 | 14499 | 100% | 100% | 100% |
| summary_cl5_bullet250 | 1 | 14885 | 100% | 100% | 100% |
| summary_cv5_structured120 | 1 | 15409 | 100% | 100% | 100% |
| summary_lg5_tldr120 | 1 | 20179 | 100% | 0% | 0% |
| summary_ro5_structured250 | 1 | 18713 | 100% | 0% | 0% |