## Benchmark 对比报告
模型: /home/tione/notebook/home/arianliu/data/models/Qwen3-VL-8B-Instruct | 样本数: 162 | 异常: 0

### 性能对比（平均值）

| 指标 | regex |
|---|---|
| 总耗时(ms) | 15187.2 |
| LLM 时间(ms) | 15183 |
| 工具时间(ms) | 0 |
| 路由时间(ms) | 0 |
| 框架开销(ms) | 4.2 |
| 迭代次数 | 3.5 |
| Token 用量 | 7848.9 |

### 可靠性（pass^k：k 次试验全部成功）

| 指标 | regex |
|---|---|
| pass^1 | 59% |
| pass^2 | 54% |
| pass^3 | n/a |

> 81 个任务的试验次数不足 3，未计入 pass^3（最少的任务只有 2 次）。

### 准确性对比

| 指标 | regex |
|---|---|
| 正常结束率(FINISH) | 85% |
| 严格成功率 | 59% |
| 工具调用准确率 | 67% |
| 参数准确率 | 72% |
| 指代解析准确率 | 69% |
| 阻塞终止语义准确率 | 40% |
| 假完成率 | 8% |
| 平均解析失败 | 0 |
| 平均工具失败 | 0.6 |

### 代价（按成功次数归一化）

| 指标 | regex |
|---|---|
| Token / 成功 | 13245.0 |
| 工具调用 / 成功 | 4.3 |
| 耗时(ms) / 成功 | 25628.4 |
| 迭代数中位·成功 | 2 |
| 迭代数中位·失败 | 4 |

失败形态：

| 指标 | regex |
|---|---|
| 撞迭代上限 | 24 |
| 声称完成但错 | 42 |
| 异常终止 | 0 |

### 任务难度分档（跨 Agent 合并成功率）

| 档位 | 成功率 | 任务数 | 用途 |
|---|---|---|---|
| floor | < 20% | 29 | 评测下限；GRPO 无梯度 |
| middle | 20~80% | 8 | **GRPO 训练集**；奖励方差最大 |
| ceiling | > 80% | 44 | 评测上限；GRPO 无梯度 |

### 按任务对比

| 任务 | 样本 | 平均耗时(ms) | 正常结束率 | 严格成功率 | 工具准确率 |
|---|---|---|---|---|---|
| analyze_ai5_ref4_desc | 2 | 21188 | 100% | 0% | 100% |
| analyze_cv5_ref1_desc | 2 | 25723.5 | 50% | 0% | 50% |
| analyze_cv5_ref3_trend | 2 | 32322 | 0% | 0% | 0% |
| analyze_ro5_ref2_axes | 2 | 31098.5 | 50% | 0% | 0% |
| chain_ai5_dl2_tr2 | 2 | 22468 | 100% | 100% | 100% |
| chain_ai5_read_then_summary | 2 | 22034 | 100% | 100% | 100% |
| chain_cl5_dl3_tr3 | 2 | 14366.5 | 100% | 0% | 0% |
| chain_cv5_cache_dl_tr_cache | 2 | 36234.5 | 100% | 50% | 50% |
| chain_lg10_dl_tr_last | 2 | 27230 | 50% | 50% | 50% |
| chain_ro5_dl_three | 2 | 18204 | 100% | 100% | 100% |
| constraint_cache_only | 2 | 8887 | 100% | 100% | 100% |
| constraint_download_only | 2 | 6269.5 | 100% | 100% | 100% |
| constraint_search_no_file | 2 | 9683.5 | 100% | 0% | 100% |
| constraint_search_only | 2 | 20969 | 50% | 0% | 0% |
| constraint_translate_only | 2 | 24969 | 50% | 50% | 50% |
| constraint_two_downloads_only | 2 | 13270 | 100% | 100% | 100% |
| figure_ai5_ref4 | 2 | 15149.5 | 100% | 0% | 100% |
| figure_cv5_ref1 | 2 | 14761 | 100% | 50% | 50% |
| figure_cv5_ref3 | 2 | 16539 | 100% | 100% | 100% |
| figure_ro5_ref2 | 2 | 14846.5 | 100% | 100% | 100% |
| infeasible_index_out_of_range | 2 | 15585 | 100% | 0% | 0% |
| infeasible_no_session | 2 | 19395.5 | 50% | 0% | 0% |
| infeasible_unknown_id | 2 | 27787 | 50% | 0% | 0% |
| infeasible_unsupported_action | 2 | 4202 | 100% | 100% | 100% |
| infeasible_zero_index | 2 | 17052 | 50% | 0% | 0% |
| multi_ai3_tr1 | 2 | 17790 | 100% | 0% | 0% |
| multi_ai5_dl1_tr | 2 | 14950.5 | 100% | 100% | 100% |
| multi_cl5_dl2_cache | 2 | 14805.5 | 100% | 0% | 100% |
| multi_cr5_cache1 | 2 | 19506 | 100% | 0% | 0% |
| multi_cv3_dl1 | 2 | 12738.5 | 100% | 100% | 100% |
| multi_cv5_dl1_cache_tr | 2 | 22108.5 | 50% | 0% | 0% |
| multi_lg10_dl3 | 2 | 12857.5 | 100% | 100% | 100% |
| multi_ro3_dl_two | 2 | 15394.5 | 100% | 50% | 100% |
| opt_force_dl | 2 | 5571 | 100% | 100% | 100% |
| opt_force_tr | 2 | 6195.5 | 100% | 100% | 100% |
| opt_keep_dual | 2 | 11176.5 | 100% | 100% | 100% |
| opt_service | 2 | 6193.5 | 100% | 100% | 100% |
| opt_threads | 2 | 6508 | 100% | 100% | 100% |
| qa_cr5_abstract | 2 | 15463.5 | 100% | 100% | 100% |
| read_ai5_default | 2 | 16395 | 100% | 0% | 100% |
| read_cl5_result | 2 | 14182 | 100% | 100% | 100% |
| read_cv5_method | 2 | 27504 | 0% | 0% | 0% |
| read_lg5_conclusion | 2 | 15878 | 100% | 100% | 100% |
| ref_ctrl_id_cache | 2 | 7595.5 | 100% | 100% | 100% |
| ref_ctrl_id_download | 2 | 6907.5 | 100% | 100% | 100% |
| ref_ctrl_null_cache | 2 | 5787.5 | 100% | 100% | 100% |
| ref_ctrl_null_translate | 2 | 10189.5 | 100% | 100% | 100% |
| ref_ctrl_word_handover | 2 | 26516.5 | 0% | 0% | 0% |
| ref_ctrl_word_marionette | 2 | 27281 | 0% | 0% | 0% |
| ref_stress_image_restoration | 2 | 20643.5 | 0% | 0% | 0% |
| ref_stress_learning_state | 2 | 23057.5 | 50% | 0% | 0% |
| ref_stress_state_across | 2 | 22836.5 | 100% | 0% | 0% |
| ref_stress_uncertainty_aware | 2 | 10217 | 100% | 0% | 0% |
| search_AI_1d_3 | 2 | 23975 | 100% | 0% | 0% |
| search_AI_30d_25 | 2 | 7523 | 100% | 100% | 100% |
| search_CL_30d_10 | 2 | 8763 | 100% | 100% | 100% |
| search_CL_7d_5 | 2 | 7375 | 100% | 100% | 100% |
| search_CR_14d_10 | 2 | 11156 | 100% | 50% | 50% |
| search_CR_7d_5 | 2 | 7482 | 100% | 100% | 100% |
| search_CV_14d_3 | 2 | 6857 | 100% | 100% | 100% |
| search_CV_1d_1 | 2 | 24221 | 100% | 0% | 0% |
| search_LG_14d_1 | 2 | 11631 | 100% | 0% | 0% |
| search_LG_3d_10 | 2 | 7731 | 100% | 100% | 100% |
| search_RO_30d_25 | 2 | 8164.5 | 100% | 100% | 100% |
| search_RO_3d_8 | 2 | 6725.5 | 100% | 100% | 100% |
| search_kw_agentic_rl | 2 | 7336 | 100% | 100% | 100% |
| search_kw_llm | 2 | 6491 | 100% | 100% | 100% |
| search_kw_rag | 2 | 21417 | 100% | 50% | 50% |
| state_cache_before_dl | 2 | 9725 | 100% | 100% | 100% |
| state_cache_two | 2 | 9145 | 100% | 100% | 100% |
| state_dl_then_cache | 2 | 9479 | 100% | 100% | 100% |
| state_ref_cache_active | 2 | 7162 | 100% | 100% | 100% |
| state_ref_last_active | 2 | 10772.5 | 100% | 100% | 100% |
| state_ref_last_of_10 | 2 | 5678 | 100% | 100% | 100% |
| state_ref_ordinal_cn | 2 | 5764 | 100% | 100% | 100% |
| state_translate_active | 2 | 9323.5 | 100% | 100% | 100% |
| summary_ai5_tldr60 | 2 | 16172.5 | 100% | 100% | 100% |
| summary_cl5_bullet250 | 2 | 19169 | 50% | 0% | 50% |
| summary_cv5_structured120 | 2 | 24671.5 | 0% | 0% | 0% |
| summary_lg5_tldr120 | 2 | 16061 | 100% | 100% | 100% |
| summary_ro5_structured250 | 2 | 23706 | 50% | 50% | 50% |