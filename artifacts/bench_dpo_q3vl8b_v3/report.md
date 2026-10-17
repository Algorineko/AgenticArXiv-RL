## Benchmark 对比报告
模型: ~/data/models/Qwen3-VL-8B-Instruct | 样本数: 81 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 10600.5 |
| LLM 时间(ms) | 10597 |
| 工具时间(ms) | 0.1 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 3.5 |
| 迭代次数 | 3.2 |
| Token 用量 | 7178 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 65% |
| pass^2 | n/a |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 1 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 98% |
| 严格成功率 | 65% |
| 工具调用准确率 | 74% |
| 参数准确率 | 76% |
| 指代解析准确率 | 75% |
| 阻塞终止语义准确率 | 0% |
| 假完成率 | 11% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.3 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 10970.2 |
| 工具调用 / 成功 | 3.3 |
| 耗时(ms) / 成功 | 16200.8 |
| 迭代数中位·成功 | 3 |
| 迭代数中位·失败 | 2.5 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 1 |
| 声称完成但错 | 26 |
| 异常终止 | 1 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 28 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 0 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 53 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 1 | 15475 | 100% | 100% | 100% |
| analyze_cv5_ref1_desc | 1 | 17218 | 100% | 100% | 100% |
| analyze_cv5_ref3_trend | 1 | 16915 | 100% | 100% | 100% |
| analyze_ro5_ref2_axes | 1 | 17204 | 100% | 100% | 100% |
| chain_ai5_dl2_tr2 | 1 | 15321 | 100% | 100% | 100% |
| chain_ai5_read_then_summary | 1 | 17504 | 100% | 100% | 100% |
| chain_cl5_dl3_tr3 | 1 | 16779 | 100% | 100% | 100% |
| chain_cv5_cache_dl_tr_cache | 1 | 19064 | 100% | 100% | 100% |
| chain_lg10_dl_tr_last | 1 | 13959 | 100% | 100% | 100% |
| chain_ro5_dl_three | 1 | 15971 | 100% | 100% | 100% |
| constraint_cache_only | 1 | 9040 | 100% | 0% | 0% |
| constraint_download_only | 1 | 9759 | 100% | 0% | 0% |
| constraint_search_no_file | 1 | 6180 | 100% | 0% | 100% |
| constraint_search_only | 1 | 6102 | 100% | 100% | 100% |
| constraint_translate_only | 1 | 5167 | 100% | 100% | 100% |
| constraint_two_downloads_only | 1 | 8514 | 100% | 100% | 100% |
| figure_ai5_ref4 | 1 | 11692 | 100% | 100% | 100% |
| figure_cv5_ref1 | 1 | 13566 | 100% | 100% | 100% |
| figure_cv5_ref3 | 1 | 12868 | 100% | 100% | 100% |
| figure_ro5_ref2 | 1 | 12691 | 100% | 100% | 100% |
| infeasible_index_out_of_range | 1 | 4805 | 100% | 0% | 0% |
| infeasible_no_session | 1 | 6386 | 100% | 0% | 100% |
| infeasible_unknown_id | 1 | 21019 | 100% | 0% | 0% |
| infeasible_unsupported_action | 1 | 17496 | 0% | 0% | 0% |
| infeasible_zero_index | 1 | 4661 | 100% | 0% | 0% |
| multi_ai3_tr1 | 1 | 8982 | 100% | 100% | 100% |
| multi_ai5_dl1_tr | 1 | 13765 | 100% | 100% | 100% |
| multi_cl5_dl2_cache | 1 | 12551 | 100% | 100% | 100% |
| multi_cr5_cache1 | 1 | 9273 | 100% | 100% | 100% |
| multi_cv3_dl1 | 1 | 9552 | 100% | 100% | 100% |
| multi_cv5_dl1_cache_tr | 1 | 16888 | 100% | 100% | 100% |
| multi_lg10_dl3 | 1 | 9542 | 100% | 100% | 100% |
| multi_ro3_dl_two | 1 | 12813 | 100% | 100% | 100% |
| opt_force_dl | 1 | 5640 | 100% | 100% | 100% |
| opt_force_tr | 1 | 8073 | 100% | 100% | 100% |
| opt_keep_dual | 1 | 5588 | 100% | 100% | 100% |
| opt_service | 1 | 7900 | 100% | 0% | 100% |
| opt_threads | 1 | 9847 | 100% | 100% | 100% |
| qa_cr5_abstract | 1 | 14305 | 100% | 100% | 100% |
| read_ai5_default | 1 | 12011 | 100% | 100% | 100% |
| read_cl5_result | 1 | 13149 | 100% | 100% | 100% |
| read_cv5_method | 1 | 12713 | 100% | 100% | 100% |
| read_lg5_conclusion | 1 | 13089 | 100% | 100% | 100% |
| ref_ctrl_id_cache | 1 | 6066 | 100% | 100% | 100% |
| ref_ctrl_id_download | 1 | 8723 | 100% | 0% | 0% |
| ref_ctrl_null_cache | 1 | 8861 | 100% | 100% | 100% |
| ref_ctrl_null_translate | 1 | 5740 | 100% | 100% | 100% |
| ref_ctrl_word_handover | 1 | 22684 | 0% | 0% | 0% |
| ref_ctrl_word_marionette | 1 | 6136 | 100% | 0% | 0% |
| ref_stress_image_restoration | 1 | 7507 | 100% | 0% | 0% |
| ref_stress_learning_state | 1 | 8678 | 100% | 0% | 0% |
| ref_stress_state_across | 1 | 7153 | 100% | 0% | 100% |
| ref_stress_uncertainty_aware | 1 | 14584 | 100% | 0% | 0% |
| search_AI_1d_3 | 1 | 12518 | 100% | 100% | 100% |
| search_AI_30d_25 | 1 | 6537 | 100% | 100% | 100% |
| search_CL_30d_10 | 1 | 6505 | 100% | 100% | 100% |
| search_CL_7d_5 | 1 | 6119 | 100% | 100% | 100% |
| search_CR_14d_10 | 1 | 6204 | 100% | 100% | 100% |
| search_CR_7d_5 | 1 | 6405 | 100% | 100% | 100% |
| search_CV_14d_3 | 1 | 6177 | 100% | 100% | 100% |
| search_CV_1d_1 | 1 | 13729 | 100% | 0% | 0% |
| search_LG_14d_1 | 1 | 6159 | 100% | 100% | 100% |
| search_LG_3d_10 | 1 | 6443 | 100% | 100% | 100% |
| search_RO_30d_25 | 1 | 6502 | 100% | 100% | 100% |
| search_RO_3d_8 | 1 | 6111 | 100% | 100% | 100% |
| search_kw_agentic_rl | 1 | 7695 | 100% | 0% | 100% |
| search_kw_llm | 1 | 6000 | 100% | 0% | 100% |
| search_kw_rag | 1 | 7155 | 100% | 0% | 100% |
| state_cache_before_dl | 1 | 6210 | 100% | 0% | 0% |
| state_cache_two | 1 | 11349 | 100% | 0% | 0% |
| state_dl_then_cache | 1 | 10327 | 100% | 0% | 0% |
| state_ref_cache_active | 1 | 5456 | 100% | 100% | 100% |
| state_ref_last_active | 1 | 7004 | 100% | 100% | 100% |
| state_ref_last_of_10 | 1 | 4781 | 100% | 100% | 100% |
| state_ref_ordinal_cn | 1 | 4738 | 100% | 100% | 100% |
| state_translate_active | 1 | 5498 | 100% | 100% | 100% |
| summary_ai5_tldr60 | 1 | 16947 | 100% | 0% | 0% |
| summary_cl5_bullet250 | 1 | 16955 | 100% | 0% | 0% |
| summary_cv5_structured120 | 1 | 16798 | 100% | 0% | 0% |
| summary_lg5_tldr120 | 1 | 17377 | 100% | 0% | 0% |
| summary_ro5_structured250 | 1 | 17776 | 100% | 0% | 0% |