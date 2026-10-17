## Benchmark 对比报告
模型: ~/data/models/Qwen3-VL-8B-Instruct | 样本数: 81 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 10520.1 |
| LLM 时间(ms) | 10516.1 |
| 工具时间(ms) | 0.1 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 4 |
| 迭代次数 | 3.2 |
| Token 用量 | 7206 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 67% |
| pass^2 | n/a |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 95% |
| 严格成功率 | 67% |
| 工具调用准确率 | 73% |
| 参数准确率 | 76% |
| 指代解析准确率 | 77% |
| 阻塞终止语义准确率 | 20% |
| 假完成率 | 9% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.2 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 10809.1 |
| 工具调用 / 成功 | 3.3 |
| 耗时(ms) / 成功 | 15780.2 |
| 迭代数中位·成功 | 3 |
| 迭代数中位·失败 | 3 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 3 |
| 声称完成但错 | 23 |
| 异常终止 | 1 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 27 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 54 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 1 | 16699 | 100% | 100% | 100% |
| analyze_cv5_ref1_desc | 1 | 15879 | 100% | 100% | 100% |
| analyze_cv5_ref3_trend | 1 | 16877 | 100% | 100% | 100% |
| analyze_ro5_ref2_axes | 1 | 16235 | 100% | 100% | 100% |
| chain_ai5_dl2_tr2 | 1 | 15322 | 100% | 100% | 100% |
| chain_ai5_read_then_summary | 1 | 16838 | 100% | 100% | 100% |
| chain_cl5_dl3_tr3 | 1 | 12099 | 100% | 0% | 0% |
| chain_cv5_cache_dl_tr_cache | 1 | 20979 | 100% | 100% | 100% |
| chain_lg10_dl_tr_last | 1 | 13360 | 100% | 100% | 100% |
| chain_ro5_dl_three | 1 | 15687 | 100% | 100% | 100% |
| constraint_cache_only | 1 | 9117 | 100% | 0% | 0% |
| constraint_download_only | 1 | 8948 | 100% | 0% | 0% |
| constraint_search_no_file | 1 | 5712 | 100% | 0% | 100% |
| constraint_search_only | 1 | 6446 | 100% | 100% | 100% |
| constraint_translate_only | 1 | 5202 | 100% | 100% | 100% |
| constraint_two_downloads_only | 1 | 8945 | 100% | 100% | 100% |
| figure_ai5_ref4 | 1 | 12813 | 100% | 100% | 100% |
| figure_cv5_ref1 | 1 | 12353 | 100% | 100% | 100% |
| figure_cv5_ref3 | 1 | 13315 | 100% | 100% | 100% |
| figure_ro5_ref2 | 1 | 12266 | 100% | 100% | 100% |
| infeasible_index_out_of_range | 1 | 4919 | 100% | 0% | 0% |
| infeasible_no_session | 1 | 8624 | 100% | 0% | 0% |
| infeasible_unknown_id | 1 | 23526 | 100% | 0% | 0% |
| infeasible_unsupported_action | 1 | 18708 | 0% | 0% | 0% |
| infeasible_zero_index | 1 | 6304 | 100% | 0% | 0% |
| multi_ai3_tr1 | 1 | 9111 | 100% | 100% | 100% |
| multi_ai5_dl1_tr | 1 | 13837 | 100% | 100% | 100% |
| multi_cl5_dl2_cache | 1 | 12374 | 100% | 100% | 100% |
| multi_cr5_cache1 | 1 | 9171 | 100% | 100% | 100% |
| multi_cv3_dl1 | 1 | 9177 | 100% | 100% | 100% |
| multi_cv5_dl1_cache_tr | 1 | 17155 | 100% | 100% | 100% |
| multi_lg10_dl3 | 1 | 10366 | 100% | 100% | 100% |
| multi_ro3_dl_two | 1 | 12604 | 100% | 100% | 100% |
| opt_force_dl | 1 | 5619 | 100% | 100% | 100% |
| opt_force_tr | 1 | 5657 | 100% | 100% | 100% |
| opt_keep_dual | 1 | 5801 | 100% | 100% | 100% |
| opt_service | 1 | 20812 | 0% | 0% | 0% |
| opt_threads | 1 | 6940 | 100% | 100% | 100% |
| qa_cr5_abstract | 1 | 12470 | 100% | 100% | 100% |
| read_ai5_default | 1 | 12155 | 100% | 100% | 100% |
| read_cl5_result | 1 | 12547 | 100% | 100% | 100% |
| read_cv5_method | 1 | 12813 | 100% | 100% | 100% |
| read_lg5_conclusion | 1 | 12996 | 100% | 100% | 100% |
| ref_ctrl_id_cache | 1 | 6779 | 100% | 100% | 100% |
| ref_ctrl_id_download | 1 | 9375 | 100% | 0% | 0% |
| ref_ctrl_null_cache | 1 | 8820 | 100% | 100% | 100% |
| ref_ctrl_null_translate | 1 | 6890 | 100% | 100% | 100% |
| ref_ctrl_word_handover | 1 | 6089 | 100% | 0% | 0% |
| ref_ctrl_word_marionette | 1 | 5653 | 100% | 0% | 0% |
| ref_stress_image_restoration | 1 | 8748 | 100% | 0% | 0% |
| ref_stress_learning_state | 1 | 8970 | 100% | 0% | 0% |
| ref_stress_state_across | 1 | 7694 | 100% | 0% | 100% |
| ref_stress_uncertainty_aware | 1 | 6399 | 100% | 0% | 0% |
| search_AI_1d_3 | 1 | 12737 | 100% | 100% | 100% |
| search_AI_30d_25 | 1 | 6473 | 100% | 100% | 100% |
| search_CL_30d_10 | 1 | 6469 | 100% | 100% | 100% |
| search_CL_7d_5 | 1 | 6087 | 100% | 100% | 100% |
| search_CR_14d_10 | 1 | 6168 | 100% | 100% | 100% |
| search_CR_7d_5 | 1 | 6373 | 100% | 100% | 100% |
| search_CV_14d_3 | 1 | 6155 | 100% | 100% | 100% |
| search_CV_1d_1 | 1 | 6392 | 100% | 100% | 100% |
| search_LG_14d_1 | 1 | 6130 | 100% | 100% | 100% |
| search_LG_3d_10 | 1 | 6417 | 100% | 100% | 100% |
| search_RO_30d_25 | 1 | 6522 | 100% | 100% | 100% |
| search_RO_3d_8 | 1 | 6080 | 100% | 100% | 100% |
| search_kw_agentic_rl | 1 | 6327 | 100% | 0% | 100% |
| search_kw_llm | 1 | 7068 | 100% | 0% | 100% |
| search_kw_rag | 1 | 7339 | 100% | 0% | 100% |
| state_cache_before_dl | 1 | 17825 | 0% | 0% | 0% |
| state_cache_two | 1 | 11440 | 100% | 0% | 0% |
| state_dl_then_cache | 1 | 20273 | 0% | 0% | 0% |
| state_ref_cache_active | 1 | 5516 | 100% | 100% | 100% |
| state_ref_last_active | 1 | 5405 | 100% | 100% | 100% |
| state_ref_last_of_10 | 1 | 4825 | 100% | 100% | 100% |
| state_ref_ordinal_cn | 1 | 4775 | 100% | 100% | 100% |
| state_translate_active | 1 | 5366 | 100% | 100% | 100% |
| summary_ai5_tldr60 | 1 | 16225 | 100% | 0% | 0% |
| summary_cl5_bullet250 | 1 | 16635 | 100% | 0% | 0% |
| summary_cv5_structured120 | 1 | 12865 | 100% | 100% | 100% |
| summary_lg5_tldr120 | 1 | 17320 | 100% | 0% | 0% |
| summary_ro5_structured250 | 1 | 16727 | 100% | 0% | 0% |